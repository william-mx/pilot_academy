# tests/test_pilotnet.py

from pathlib import Path

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from pilot_academy.models.registry import build_model


def main():
    # Paths
    PROJECT_DIR = Path("/workspaces/pilot_academy")
    MODEL_CFG_PATH = PROJECT_DIR / "config" / "models" / "pilotnet-sf-bc-steer.yaml"
    
    assert MODEL_CFG_PATH.exists(), f"Missing model config: {MODEL_CFG_PATH}"

    # Load config using OmegaConf
    cfg = OmegaConf.load(MODEL_CFG_PATH)

    # Provide missing interpolations for testing
    defaults = OmegaConf.create({"image_shape": [200, 200, 3]})
    cfg = OmegaConf.merge(defaults, cfg)

    model_name = cfg.name
    input_shape = tuple(cfg.model.input_shape)

    # Build model
    model = build_model(model_name, cfg)
    
    # Create dummy input (B, H, W, C)
    batch_size = 2
    x = np.random.rand(batch_size, *input_shape).astype(np.float32)

    # Forward pass
    y = model(x, training=False)

    # Print results
    print("TensorFlow:", tf.__version__)
    print("Model:", model.name)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output dtype:", y.dtype)

    # Sanity checks
    assert y.shape[0] == batch_size, "Batch dimension mismatch"
    assert y.dtype.is_floating, "Output should be floating-point"
    assert len(y.shape) == 2, "Output should be 2D (batch, features)"


if __name__ == "__main__":
    main()