"""Utilities for resuming training from checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
from omegaconf import DictConfig


def load_checkpoint_info(checkpoint_path: str | Path) -> dict:
    """
    Load metadata from a training checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint .keras file or directory
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    # If it's a .keras file, look for metadata in parent directory
    if checkpoint_path.suffix == '.keras':
        checkpoint_dir = checkpoint_path.parent
    else:
        checkpoint_dir = checkpoint_path
    
    # Try to load training metadata
    metadata_path = checkpoint_dir / 'checkpoint_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    return {}


def save_checkpoint_metadata(
    checkpoint_dir: Path,
    epoch: int,
    history: dict,
    config: DictConfig = None,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
):
    metadata = {
        "epoch": epoch,
        "best_val_loss": min(history.get("val_loss", [float("inf")])),
        "final_train_loss": history.get("loss", [])[-1] if history.get("loss") else None,
        "total_epochs": len(history.get("loss", [])),
        "wandb_run_id": wandb_run_id if wandb_run_id is not None else None,
        "wandb_run_name": wandb_run_name if wandb_run_name is not None else None,
    }

    if config is not None:
        from omegaconf import OmegaConf
        metadata["config"] = OmegaConf.to_container(config, resolve=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / "checkpoint_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)



def resume_from_checkpoint(
    model: tf.keras.Model,
    checkpoint_path: str | Path,
    resume_cfg: DictConfig,
    verbose: bool = True
) -> Tuple[tf.keras.Model, int, dict]:
    """
    Resume training from a checkpoint.
    
    Args:
        model: Keras model to load weights into
        checkpoint_path: Path to checkpoint
        resume_cfg: Resume configuration from config.resume
        verbose: Print loading information
        
    Returns:
        Tuple of (model, starting_epoch, checkpoint_info)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint metadata
    checkpoint_info = load_checkpoint_info(checkpoint_path)
    
    if verbose:
        print("\n" + "="*60)
        print("RESUMING FROM CHECKPOINT")
        print("="*60)
        print(f"Checkpoint: {checkpoint_path}")
        if checkpoint_info:
            print(f"Previous epoch: {checkpoint_info.get('epoch', 'unknown')}")
            print(f"Best val loss: {checkpoint_info.get('best_val_loss', 'unknown')}")
    
    # Determine what to restore
    restore = resume_cfg.restore
    
    # Load the model
    if checkpoint_path.suffix == '.keras':
        # Full model save
        loaded_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        
        if restore.model_weights:
            model.set_weights(loaded_model.get_weights())
            if verbose:
                print("✓ Model weights loaded")
        
        if restore.optimizer_state and not resume_cfg.adjust.reset_optimizer:
            # Transfer optimizer state (requires same architecture)
            try:
                model.optimizer.set_weights(loaded_model.optimizer.get_weights())
                if verbose:
                    print("✓ Optimizer state loaded")
            except:
                if verbose:
                    print("⚠ Could not load optimizer state (continuing without it)")
    else:
        # Just weights
        model.load_weights(checkpoint_path)
        if verbose:
            print("✓ Model weights loaded")
    
    # Determine starting epoch
    starting_epoch = 0
    if restore.epoch_counter and 'epoch' in checkpoint_info:
        starting_epoch = checkpoint_info['epoch'] + 1
        if verbose:
            print(f"✓ Resuming from epoch {starting_epoch}")
    
    if verbose:
        print("="*60 + "\n")
    
    return model, starting_epoch, checkpoint_info


def calculate_epochs_for_resume(
    resume_cfg: DictConfig,
    original_epochs: int,
    starting_epoch: int
) -> int:
    """
    Calculate how many epochs to train when resuming.
    
    Args:
        resume_cfg: Resume configuration
        original_epochs: Original number of epochs in config
        starting_epoch: Epoch we're resuming from
        
    Returns:
        Total number of epochs to train to
    """
    if resume_cfg.adjust.additional_epochs is not None:
        # Train for N additional epochs
        return starting_epoch + resume_cfg.adjust.additional_epochs
    else:
        # Train to original total
        return original_epochs