"""
Train PilotNet using Hydra configuration with resume support.

Usage:
    # Fresh training
    python src/pilot_academy/train_pilotnet.py
    
    # Resume from checkpoint
    python src/pilot_academy/train_pilotnet.py \
        resume.enabled=true \
        resume.checkpoint_path=outputs/2024-01-28/12-30-45_run/checkpoint_best.keras
    
    # Resume and train for 50 more epochs
    python src/pilot_academy/train_pilotnet.py \
        resume.enabled=true \
        resume.checkpoint_path=outputs/2024-01-28/12-30-45_run/checkpoint_best.keras \
        resume.adjust.additional_epochs=50
"""

from __future__ import annotations

from pathlib import Path

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from pilot_academy.data.build_datasets import build_datasets_from_config
from pilot_academy.models.registry import build_model
from pilot_academy.training.factory import training_session_from_cfg
from pilot_academy.training.resume_utils import (
    resume_from_checkpoint,
    calculate_epochs_for_resume,
    save_checkpoint_metadata,
)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra configuration."""
    
    # Get Hydra output directory for this run
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    # Build datasets with new sampling system
    train_ds, val_ds, extras = build_datasets_from_config(cfg)
    
    # Print dataset info
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Training samples: {len(extras['df_train'])}")
    print(f"Validation samples: {len(extras['df_val'])}")
    print(f"Validation steps: {extras['val_steps']}")
    print(f"Cache size: {extras['cache_size']}")
    print(f"\nSampling Configuration:")
    print(f"  Strategy: {extras['sampling_strategy']}")
    for key, value in extras['sampling_config'].items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # Build model
    model = build_model(cfg.model)
    print(f"Model: {model.name}")
    print(f"Parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}\n")
    
    # Check if resuming from checkpoint
    starting_epoch = 0
    checkpoint_info = {}
    
    if cfg.resume.enabled:
        if cfg.resume.checkpoint_path is None:
            raise ValueError("resume.enabled=true but resume.checkpoint_path is not set")
        
        model, starting_epoch, checkpoint_info = resume_from_checkpoint(
            model=model,
            checkpoint_path=cfg.resume.checkpoint_path,
            resume_cfg=cfg.resume,
            verbose=True
        )
    
    # Initialize WandB if enabled
    callbacks_cfg = cfg.train.callbacks
    if callbacks_cfg.wandb.enabled:
        import wandb
        
        # Add sampling config to wandb
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config['sampling_info'] = extras['sampling_config']
        
        # Add resume info if applicable
        if cfg.resume.enabled:
            wandb_config['resumed_from'] = str(cfg.resume.checkpoint_path)
            wandb_config['starting_epoch'] = starting_epoch
        
        wandb_kwargs = {
            "project": callbacks_cfg.wandb.project,
            "config": wandb_config,
            "resume": "allow" if cfg.resume.enabled else False,
        }

        if cfg.resume.enabled:
            wandb_kwargs["name"] = checkpoint_info['wandb_run_name']
            wandb_kwargs["id"] = checkpoint_info['wandb_run_id']

        wandb.init(**wandb_kwargs)

        print("✓ WandB initialized\n")
    
    # Build training session
    session = training_session_from_cfg(cfg.train, run_dir=run_dir)
    
    # Adjust epochs if resuming
    original_epochs = session.epochs
    if cfg.resume.enabled:
        session.epochs = calculate_epochs_for_resume(
            resume_cfg=cfg.resume,
            original_epochs=original_epochs,
            starting_epoch=starting_epoch
        )
    
    # Compile model (after potential weight loading)
    session.compile_model(model)
    
    # Reset learning rate schedule if requested
    if cfg.resume.enabled and cfg.resume.adjust.reset_lr_schedule:
        print("Resetting learning rate schedule")
        # Rebuild optimizer with fresh learning rate
        session.compile_model(model)
    
    # Print training configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Run name: {cfg.run.name}")
    if cfg.resume.enabled:
        print(f"Resuming from epoch: {starting_epoch}")
        print(f"Training to epoch: {session.epochs}")
        print(f"Additional epochs: {session.epochs - starting_epoch}")
    else:
        print(f"Starting from: epoch 0")
        print(f"Total epochs: {session.epochs}")
    print(f"Batch size (train): {cfg.dataset.train.batch_size}")
    print(f"Batch size (val): {cfg.dataset.val.batch_size}")
    print(f"Output directory: {run_dir}")
    print("="*60 + "\n")
    
    # Train
    print(f"Starting training...\n")
    history = session.fit(
        model, 
        train_ds, 
        val_ds,
        initial_epoch=starting_epoch  # Start from correct epoch
    )
    
    # Save final model
    final_model_path = run_dir / f"{wandb.run.name}-final.keras"
    model.save(final_model_path)
    
    # Save checkpoint metadata
    save_checkpoint_metadata(
        checkpoint_dir=run_dir / "checkpoints",
        epoch=session.epochs - 1,
        history=history.history,
        config=cfg,
        wandb_run_id=wandb.run.id if callbacks_cfg.wandb.enabled else None,
        wandb_run_name=wandb.run.name if callbacks_cfg.wandb.enabled else None,
    )

    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if cfg.resume.enabled:
        print(f"Resumed from: {cfg.resume.checkpoint_path}")
        print(f"Started at epoch: {starting_epoch}")
    
    if "val_loss" in history.history:
        best_val_loss = min(history.history["val_loss"])
        best_epoch = history.history["val_loss"].index(best_val_loss) + starting_epoch + 1
        print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    final_train_loss = history.history["loss"][-1]
    print(f"Final training loss: {final_train_loss:.4f}")
    
    if "val_loss" in history.history:
        final_val_loss = history.history["val_loss"][-1]
        print(f"Final validation loss: {final_val_loss:.4f}")
    
    print(f"\nModel saved to: {final_model_path}")
    print(f"Metadata saved to: {run_dir / 'checkpoint_metadata.json'}")
    print("="*60)
    
    # Log final metrics to WandB if enabled
    if callbacks_cfg.wandb.enabled:
        wandb.finish()
        print("\n✓ WandB run finished")


if __name__ == "__main__":
    main()