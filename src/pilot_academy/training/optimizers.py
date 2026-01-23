# pilot_academy/training/optimizers.py

"""Optimizer and learning rate schedule builders."""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf


def _as_float(x: Any) -> float:
    """Convert various types to float."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise TypeError(f"Cannot convert to float: {x!r}")


def build_lr_schedule(
    schedule_cfg: DictConfig,
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Build learning rate schedule from config.
    
    Args:
        schedule_cfg: Schedule configuration with fields:
            - type: Schedule type (e.g., 'exponential_decay')
            - initial_lr: Initial learning rate
            - decay_steps: Steps between decay applications
            - decay_rate: Decay rate
            - staircase: Whether to use staircase decay (optional)
    
    Returns:
        TensorFlow learning rate schedule
    
    Raises:
        ValueError: If schedule type is not supported
    """
    schedule_type = schedule_cfg.type.lower()
    
    if schedule_type in {"exponential_decay", "exponentialdecay"}:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=_as_float(schedule_cfg.initial_lr),
            decay_steps=int(schedule_cfg.decay_steps),
            decay_rate=_as_float(schedule_cfg.decay_rate),
            staircase=schedule_cfg.get("staircase", False),
        )
    
    raise ValueError(f"Unsupported LR schedule type: {schedule_type}")


def build_optimizer(optim_cfg: DictConfig) -> tf.keras.optimizers.Optimizer:
    """
    Build Keras optimizer from config.
    
    Args:
        optim_cfg: Optimizer configuration with fields:
            - type: Optimizer type ('adam' or 'sgd')
            - lr: Learning rate (default: 1e-3)
            - schedule: Optional learning rate schedule config
            - momentum: SGD momentum (default: 0.0)
            - nesterov: Whether to use Nesterov momentum for SGD (default: False)
    
    Returns:
        Configured Keras optimizer
    
    Raises:
        ValueError: If optimizer type is not supported
    """
    # Set defaults using OmegaConf
    OmegaConf.set_struct(optim_cfg, False)
    optim_cfg.type = optim_cfg.get("type", "adam")
    optim_cfg.lr = optim_cfg.get("lr", 1e-3)
    
    opt_type = optim_cfg.type.lower()
    lr = _as_float(optim_cfg.lr)

    # Build learning rate schedule if configured
    if "schedule" in optim_cfg and optim_cfg.schedule:
        schedule_cfg = OmegaConf.create(optim_cfg.schedule)
        # Set initial_lr from optimizer lr if not specified in schedule
        if "initial_lr" not in schedule_cfg:
            schedule_cfg.initial_lr = lr
        lr = build_lr_schedule(schedule_cfg)

    if opt_type == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)

    if opt_type == "sgd":
        optim_cfg.momentum = optim_cfg.get("momentum", 0.0)
        optim_cfg.nesterov = optim_cfg.get("nesterov", False)
        
        return tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=_as_float(optim_cfg.momentum),
            nesterov=optim_cfg.nesterov,
        )

    raise ValueError(f"Unsupported optimizer type: {opt_type}")