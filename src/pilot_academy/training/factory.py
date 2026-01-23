from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from omegaconf import DictConfig, OmegaConf

from pilot_academy.training.spec import TrainingSpec
from pilot_academy.training.callbacks import build_callbacks
from pilot_academy.training.optimizers import build_optimizer


def training_session_from_cfg(
    cfg: DictConfig,
    *,
    run_dir: Optional[Union[str, Path]] = None,
) -> TrainingSpec:
    """
    Build TrainingSpec from Hydra configuration.
    
    Expected config structure:
        train:
            train:
                loss: mse
                metrics: [mae]
                epochs: 100
                steps_per_epoch: null
                val_steps: null
            optim:
                type: adam
                lr: 0.001
                schedule:
                    type: exponential_decay
                    decay_steps: 1000
                    decay_rate: 0.96
            callbacks:
                ...
    
    Args:
        cfg: Hydra/OmegaConf configuration
        run_dir: Optional run directory for callbacks
        
    Returns:
        TrainingSpec instance ready for model.compile() and model.fit()
    """
    # Resolve interpolations
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    
    fit_cfg = cfg.train
    optim_cfg = cfg.optim
    cb_cfg = cfg.callbacks

    # Build optimizer
    optimizer = build_optimizer(optim_cfg)

    # Loss function
    loss = fit_cfg.loss

    # Metrics
    metrics_raw = fit_cfg.metrics
    if metrics_raw is None or metrics_raw == []:
        metrics = []
    elif isinstance(metrics_raw, str):
        # Single metric as string
        metrics = [metrics_raw]
    else:
        # Already a list or tuple
        metrics = list(metrics_raw)

    # Training parameters
    epochs = int(fit_cfg.epochs)
    steps_per_epoch = fit_cfg.steps_per_epoch
    val_steps = fit_cfg.val_steps

    # Build callbacks
    run_path = Path(run_dir) if run_dir is not None else None
    callbacks = build_callbacks(cb_cfg, run_path)

    return TrainingSpec(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        epochs=epochs,
        steps_per_epoch=int(steps_per_epoch) if steps_per_epoch is not None else None,
        validation_steps=int(val_steps) if val_steps is not None else None,
        callbacks=callbacks,
    )