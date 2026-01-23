"""Model registry for building models from configuration."""

from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf
from omegaconf import DictConfig

from .pilotnet import build_pilotnet


# Type alias for model builder functions
ModelBuilder = Callable[[DictConfig], tf.keras.Model]


_MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    "pilotnet-sf-bc-steer": lambda cfg: build_pilotnet(
        input_shape=tuple(cfg.model.model.input_shape)
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
        ...     return build_my_model(cfg.models.hidden_dims)
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

    name = cfg.model.name

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