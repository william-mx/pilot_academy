"""Model registry for building models from configuration."""

from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf
from omegaconf import DictConfig

from .pilotnet import build_pilotnet
from .vit_sf_navbc_steer import build_vit_nav


# Type alias for model builder functions
ModelBuilder = Callable[[DictConfig], tf.keras.Model]


_MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    "pilotnet-sf-bc-steer": lambda cfg: build_pilotnet(
        input_shape=tuple(cfg.model.input_shape)
    ),
    "vit-sf-navbc-steer": lambda cfg: build_vit_nav(
        input_shape=tuple(cfg.model.input_shape),
        channels=tuple(cfg.model.get("channels", (64, 128))),
        heads=cfg.model.get("heads", 4),
        mlp_ratio=cfg.model.get("mlp_ratio", 2.0),
        dropout=cfg.model.get("dropout", 0.0),
        patch_size=cfg.model.get("patch_size", 16),
        num_commands=cfg.model.get("num_commands", 6),
    ),
}


def register_model(name: str, builder: ModelBuilder) -> None:
    """
    Register a new model builder.
    
    Args:
        name: Model name/identifier
        builder: Function that takes DictConfig and returns a Keras model
    
    Example:
        >>> def my_model_builder(cfg: DictConfig) -> tf.keras.Model:
        ...     return build_my_model(cfg.model.hidden_dims)
        >>> register_model("my-model", my_model_builder)
    """
    if name in _MODEL_BUILDERS:
        raise ValueError(f"Model '{name}' is already registered")
    _MODEL_BUILDERS[name] = builder


def build_model(cfg: DictConfig) -> tf.keras.Model:
    """
    Build a model from configuration.
    
    Expected config structure:
        model:
            name: pilotnet-sf-bc-steer
            input_shape: [66, 200, 3]
            # ... other model-specific params
    
    Args:
        name: Model name/identifier
        cfg: Hydra/OmegaConf configuration
        
    Returns:
        Configured Keras model
        
    Raises:
        ValueError: If model name is not registered
        
    Example:
        >>> cfg = OmegaConf.create({
        ...     "model": {
        ...         "name": "pilotnet-sf-bc-steer",
        ...         "input_shape": [66, 200, 3]
        ...     }
        ... })
        >>> model = build_model("pilotnet-sf-bc-steer", cfg)
    """
    name = cfg.name
    if name not in _MODEL_BUILDERS:
        available = ", ".join(_MODEL_BUILDERS.keys())
        raise ValueError(
            f"Unknown model name: '{name}'. "
            f"Available models: {available}"
        )
    
    return _MODEL_BUILDERS[name](cfg)


def list_models() -> list[str]:
    """
    List all registered model names.
    
    Returns:
        List of available model names
    """
    return list(_MODEL_BUILDERS.keys())