import os
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

# Import your custom modules
from dataset import get_dataloaders
from model import SimpleVideoTFModel
from loss import HungarianMatcher, SetCriterion
from utils import load_config, update_config, get_num_classes, create_experiment_dir, DoNothing


def train(config_path="configs/base_kovo.py", resume_exp_dir=None, **overrides):
    """
    Trains or resumes the SimpleVideoTFModel using the specified configuration and overrides.

    Args:
        config_path (str): Path to the config Python file.
        exp_dir (str): Path to the experiment directory to resume from (optional). Example: resume_exp_dir="exp/20241011_191033"
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

    # Determine if we are resuming from an existing experiment
    ckpt_path = None
    if exp_resume_dir:
        # Find the latest or best checkpoint to resume from
        ckpt_path = find_latest_checkpoint(exp_resume_dir)
        if ckpt_path:
            print(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            print(f"No valid checkpoint found in {exp_resume_dir}, starting fresh.")

    # Create a unique experiment directory if not resuming
    experiment_dir = create_experiment_dir(config) if not exp_resume_dir else exp_resume_dir
    print(f"Experiment directory: {experiment_dir}")

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    train_loader, val_loader, test_loader = get_dataloaders(config)

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
        criterion=criterion,
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
    ) if config.early_stop_patience > 0 else DoNothing()

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model.print_model_stats()

    # Instantiate the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=50,
        gradient_clip_val=0.1,  # Optional: gradient clipping
        accumulate_grad_batches=config.gradient_accumulation_steps,
    )

    # Start training from the checkpoint if found, otherwise start from scratch
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Save the final model
    final_model_path = os.path.join(experiment_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")


def find_latest_checkpoint(exp_dir):
    """
    Finds the latest or best checkpoint in the given experiment directory.

    Args:
        exp_dir (str): Path to the experiment directory.

    Returns:
        str: Path to the latest or best checkpoint file, or None if not found.
    """
    checkpoints_dir = Path(exp_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"No checkpoint directory found in {exp_dir}.")
        return None

    # Search for final model or best model checkpoint
    final_ckpt = checkpoints_dir.parent / "final_model.ckpt"
    if final_ckpt.exists():
        return str(final_ckpt)

    # If no final model, return the latest or best checkpoint based on modification time
    checkpoint_files = sorted(checkpoints_dir.glob("*.ckpt"), key=os.path.getmtime)
    if checkpoint_files:
        return str(checkpoint_files[-1])  # Return the most recent checkpoint

    print(f"No checkpoints found in {checkpoints_dir}.")
    return None


def main():
    fire.Fire(train)


if __name__ == "__main__":
    main()
