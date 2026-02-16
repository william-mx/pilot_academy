from __future__ import annotations

from pathlib import Path

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from pilot_academy.data.build_datasets import build_datasets_from_config
from pilot_academy.models.registry import build_model
from pilot_academy.training.factory import training_session_from_cfg


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
    
    # Initialize WandB if enabled (must be done before training_spec builds callbacks)
    callbacks_cfg = cfg.train.callbacks
    if callbacks_cfg.wandb.enabled:
        import wandb
        
        # Add sampling config to wandb
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config['sampling_info'] = extras['sampling_config']
        
        wandb.init(
            project=callbacks_cfg.wandb.project,
            name=cfg.run.name,
            config=wandb_config,
        )
        print("✓ WandB initialized\n")
    
    # Build training spec (this will add WandbMetricsLogger to callbacks if wandb.enabled=True)
    session = training_session_from_cfg(cfg.train, run_dir=run_dir)
    session.compile_model(model)
    
    # Print training configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Run name: {cfg.run.name}")
    print(f"Epochs: {session.epochs}")
    print(f"Batch size (train): {cfg.dataset.train.batch_size}")
    print(f"Batch size (val): {cfg.dataset.val.batch_size}")
    print(f"Output directory: {run_dir}")
    print("="*60 + "\n")
    
    # Train
    print(f"Starting training...\n")
    history = session.fit(model, train_ds, val_ds)
    
    # Save final model
    final_model_path = run_dir / f"{cfg.run.name}_final.keras"
    model.save(final_model_path)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if "val_loss" in history.history:
        best_val_loss = min(history.history["val_loss"])
        best_epoch = history.history["val_loss"].index(best_val_loss) + 1
        print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    final_train_loss = history.history["loss"][-1]
    print(f"Final training loss: {final_train_loss:.4f}")
    
    if "val_loss" in history.history:
        final_val_loss = history.history["val_loss"][-1]
        print(f"Final validation loss: {final_val_loss:.4f}")
    
    print(f"\nModel saved to: {final_model_path}")
    print("="*60)
    
    # Log final metrics to WandB if enabled
    if callbacks_cfg.wandb.enabled:
        wandb.finish()
        print("\n✓ WandB run finished")


if __name__ == "__main__":
    main()