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
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Import your custom modules
from dataset import get_dataloaders
from model import SimpleVideoTFModel
from loss import HungarianMatcher, SetCriterion
from utils import (
    load_config,
    update_config,
    get_num_classes,
    create_experiment_dir,
    find_latest_checkpoint,
    DoNothing,
)


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
    if resume_exp_dir:
        # Find the latest or best checkpoint to resume from
        ckpt_path = find_latest_checkpoint(resume_exp_dir)
        if ckpt_path:
            print(f"Resuming training from checkpoint: {ckpt_path}")
        else:
            print(f"No valid checkpoint found in {resume_exp_dir}, starting fresh.")

    # Create a unique experiment directory if not resuming
    experiment_dir = (
        create_experiment_dir(config) if not resume_exp_dir else resume_exp_dir
    )
    print(f"Experiment directory: {experiment_dir}")

    # Set random seed for reproducibility
    pl.seed_everything(config.seed)

    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Instantiate the model
    model = SimpleVideoTFModel(**config)

    # Setup TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=str(Path(config.log_dir) / "video_model"),
        name="",
        version=None,  # Let Lightning handle versioning
    )

    # Setup CSV logger
    csv_logger = CSVLogger(
        save_dir=str(Path(config.log_dir) / "csv_logs"),
        name="csv_logs",
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.checkpoint_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.4f}",
        save_top_k=config.save_top_k,
        mode="min",
    )

    early_stop_callback = (
        EarlyStopping(
            monitor="val_loss",
            patience=config.early_stop_patience,
            verbose=True,
            mode="min",
        )
        if config.early_stop_patience > 0
        else DoNothing()
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model.print_model_stats()

    # Instantiate the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        logger=[tb_logger, csv_logger],  # Add both TensorBoard and CSV loggers
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=50,
        gradient_clip_val=0.1,  # Optional: gradient clipping
        accumulate_grad_batches=config.gradient_accumulation_steps,
    )

    model = (
        torch.compile(model, mode="reduce-overhead") if config.compile else model
    )  # torch.compile is failing, so don't enable it for now
    # Start training from the checkpoint if found, otherwise start from scratch
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Save the final model
    final_model_path = os.path.join(experiment_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")


def main():
    fire.Fire(train)


if __name__ == "__main__":
    main()
