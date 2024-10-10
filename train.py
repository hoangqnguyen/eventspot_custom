# train.py

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Import your custom modules
from dataset import (
    VolleyballVideoDataset,
    custom_collate_fn,
    visualize_batch,
)
from model import SimpleVideoTFModel
from loss import HungarianMatcher, SetCriterion
from transform import get_transforms

from easydict import EasyDict as edict
from utils import store_json


def load_config(config_path):
    """
    Dynamically loads a Python config file and returns the config dictionary.

    Args:
        config_path (str): Path to the config Python file.

    Returns:
        EasyDict: Configuration dictionary.
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def update_config(config, overrides):
    """
    Updates the configuration dictionary with overrides.

    Args:
        config (EasyDict): Original configuration.
        overrides (dict): Overrides to apply.

    Returns:
        EasyDict: Updated configuration.
    """
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"Overriding config parameter: {key} = {value}")
        else:
            print(f"Warning: Config has no attribute named '{key}'. Skipping override.")
    return config


def get_num_classes(classes_file):
    """
    Reads the classes file and returns the number of classes.

    Args:
        classes_file (str): Path to the classes file.

    Returns:
        tuple: (num_classes, class_names)
    """
    with open(classes_file, "r") as f:
        class_names = f.read().splitlines()
    # Include 'background' as the first class
    class_names = ["background"] + class_names
    num_classes = len(class_names)
    return num_classes, class_names


def create_experiment_dir(config, run_name=None):
    """
    Creates a unique experiment directory to store checkpoints and config.

    Args:
        config (EasyDict): Configuration dictionary.
        run_name (str, optional): Custom run name. If None, use timestamp.

    Returns:
        str: Path to the experiment directory.
    """
    exp_dir = Path("exp")
    exp_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        from datetime import datetime

        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_dir = exp_dir / run_name
    checkpoints_dir = experiment_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Update config checkpoint_dir to point to the new checkpoints directory
    config.checkpoint_dir = str(checkpoints_dir)

    # Save the config file into the experiment directory for reference

    # Save the config as a JSON file into the experiment directory for reference
    config_save_path = experiment_dir / "config.json"
    # Convert EasyDict to regular dict
    store_json(config_save_path, config, pretty=True)
    # config_dict = edict_to_dict(config)
    # with open(config_save_path, "w") as f:
    #     json.dump(config, f, indent=4)

    return str(experiment_dir)


def train(config_path="configs/base_kovo.py", **overrides):
    """
    Trains the SimpleVideoTFModel using the specified configuration and overrides.

    Args:
        config_path (str): Path to the config Python file.
        **overrides: Key-value pairs to override the configuration.
    """
    # Load configuration
    config = load_config(config_path)

    # Apply overrides
    config = update_config(config, overrides)

    # Infer num_classes from the classes_file
    num_classes, class_names = get_num_classes(config.classes_file)
    config.num_classes = num_classes
    config.class_names = class_names  # Add class_names to config for visualization

    # Create a unique experiment directory
    experiment_dir = create_experiment_dir(config)
    print(f"Experiment directory created at: {experiment_dir}")

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    # Define transforms for training and validation
    train_transforms = get_transforms(
        split="train",
        frame_size=tuple(config.frame_size),
    )
    val_transforms = get_transforms(
        split="val",
        frame_size=tuple(config.frame_size),
    )

    # Instantiate training and validation datasets
    train_dataset = VolleyballVideoDataset(
        json_file=config.train_json,
        frames_dir=config.frames_dir,
        classes_file=config.classes_file,
        transform=train_transforms,
        window_size=config.window_size,
        stride=config.stride,
        num_events=config.num_events,
        frame_size=tuple(config.frame_size),
    )

    val_dataset = VolleyballVideoDataset(
        json_file=config.val_json,
        frames_dir=config.frames_dir,
        classes_file=config.classes_file,
        transform=val_transforms,
        window_size=config.window_size,
        stride=config.stride,
        num_events=config.num_events,
        frame_size=tuple(config.frame_size),
    )

    # Print dataset statistics
    print("\nTraining Dataset Statistics:")
    train_dataset.print_stats()

    print("\nValidation Dataset Statistics:")
    val_dataset.print_stats()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        # pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        # pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    # Initialize Matcher and Criterion
    matcher = HungarianMatcher(
        cost_class=config.cost_class,
        cost_frame=config.cost_frame,
        cost_coord=config.cost_coord,
    )

    weight_dict = {
        "loss_ce": config.weight_dict.get("loss_ce", 1.0),
        "loss_frame": config.weight_dict.get("loss_frame", 1.0),
        "loss_xy": config.weight_dict.get("loss_xy", 1.0),
    }

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config.eos_coef,
        num_classes=config.num_classes,
        losses=["labels", "frames", "xy"],
    )

    # Instantiate the model
    model = SimpleVideoTFModel(
        num_classes=config.num_classes,
        num_queries=config.num_queries,
        backbone_name=config.backbone_name,
        pretrained=config.pretrained,
        learning_rate=config.learning_rate,
        hidden_dim=config.hidden_dim,
        nheads=config.nheads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=config.eos_coef,
    )

    # Setup TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=str(Path(config.log_dir) / "video_model"),
        name="",
        version=None,  # Let Lightning handle versioning
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="loss",
        dirpath=config.checkpoint_dir,
        filename="best-checkpoint-{epoch:02d}-{loss:.4f}",
        save_top_k=config.save_top_k,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="loss",
        patience=config.early_stop_patience,
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Instantiate the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        # precision="bf16",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=50,
        gradient_clip_val=0.1,  # Optional: gradient clipping
        accumulate_grad_batches=config.gradient_accumulation_steps,
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    # Optionally, test the model on the validation set
    # trainer.test(model, dataloaders=val_loader)

    # Save the final model
    final_model_path = os.path.join(experiment_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")


def main():
    fire.Fire(train)


if __name__ == "__main__":
    main()
