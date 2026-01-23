"""
Train PilotNet using Hydra configuration.

Usage:
    python src/pilot_academy/train_pilotnet.py
    python src/pilot_academy/train_pilotnet.py train.epochs=50
    python src/pilot_academy/train_pilotnet.py model=pilotnet_custom dataset=my_dataset
"""

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
    
    # Build datasets
    train_ds, val_ds, info = build_datasets_from_config(cfg)
    print(f"Datasets loaded: {len(info['df_train'])} train, {len(info['df_val'])} val")
    
    # Build model
    model = build_model(cfg)
    print(f"Model built: {model.name} ({model.count_params():,} parameters)")
    
    # Initialize WandB if enabled (must be done before training_spec builds callbacks)
    callbacks_cfg = cfg.train.callbacks
    if hasattr(cfg, "callbacks") and hasattr(callbacks_cfg, "wandb") and callbacks_cfg.wandb.enabled:
        import wandb
        
        wandb.init(
            project=callbacks_cfg.wandb.project,
            name=cfg.run.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print("WandB initialized")
    
    # Build training spec (this will add WandbMetricsLogger to callbacks if wandb.enabled=True)
    training_spec = training_session_from_cfg(cfg.train, run_dir=run_dir)
    training_spec.compile_model(model)
    
    # Train
    print(f"Training {cfg.run.name} for {training_spec.epochs} epochs...")
    history = training_spec.fit(model, train_ds, val_ds)
    
    # Save final model
    final_model_path = run_dir / f"{cfg.run.name}_final.keras"
    model.save(final_model_path)
    
    # Print summary
    best_val_loss = min(history.history.get("val_loss", [float("inf")]))
    final_train_loss = history.history["loss"][-1]
    print(f"Training complete: best_val_loss={best_val_loss:.4f}, final_train_loss={final_train_loss:.4f}")
    print(f"Model saved to: {final_model_path}")


if __name__ == "__main__":
    main()