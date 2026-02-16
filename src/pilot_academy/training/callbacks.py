from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf


def build_callbacks(
    cfg: DictConfig, 
    run_dir: Optional[Path]
) -> List[tf.keras.callbacks.Callback]:
    """
    Build Keras callbacks from Hydra/OmegaConf callbacks config.

    Expected config structure:
        callbacks:
            wandb:
                enabled: false
                project: pilotnet
            checkpoint:
                enabled: true
                monitor: val_loss
                save_best_only: true
                verbose: 1
            early_stopping:
                enabled: true
                monitor: val_loss
                patience: 10
                restore_best_weights: true
                verbose: 1
            reduce_lr:
                enabled: false
                monitor: val_loss
                factor: 0.5
                patience: 60
                min_lr: 1e-7
                verbose: 1
                cooldown: 1
    
    Args:
        cfg: Callbacks configuration (DictConfig)
        run_dir: Optional run directory for saving checkpoints
        
    Returns:
        List of configured Keras callbacks
    """
    callbacks: List[tf.keras.callbacks.Callback] = []

    # WandB callback
    if hasattr(cfg, "wandb") and cfg.wandb.enabled:
        from wandb.integration.keras import WandbMetricsLogger
        callbacks.append(WandbMetricsLogger())

    # ModelCheckpoint callback
    if hasattr(cfg, "checkpoint") and cfg.checkpoint.enabled:
        if run_dir is None:
            raise ValueError("Checkpoint callback enabled but run_dir is None")

        ckpt_cfg = cfg.checkpoint
        ckpt_path = run_dir / "checkpoints" / "model.keras"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor=ckpt_cfg.monitor,
                save_best_only=ckpt_cfg.save_best_only,
                verbose=int(ckpt_cfg.verbose),
            )
        )

    # ReduceLROnPlateau callback
    if hasattr(cfg, "reduce_lr") and cfg.reduce_lr.enabled:
        lr_cfg = cfg.reduce_lr
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=lr_cfg.monitor,
                factor=float(lr_cfg.factor),
                patience=int(lr_cfg.patience),
                min_lr=float(lr_cfg.min_lr),
                verbose=int(lr_cfg.verbose),
                cooldown=int(lr_cfg.get("cooldown", 0)),
            )
        )

    # EarlyStopping callback
    if hasattr(cfg, "early_stopping") and cfg.early_stopping.enabled:
        es_cfg = cfg.early_stopping
        
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg.monitor,
                patience=int(es_cfg.patience),
                restore_best_weights=es_cfg.restore_best_weights,
                verbose=int(es_cfg.verbose),
            )
        )

    return callbacks