from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

from pilot_academy.training.factory import training_session_from_cfg


def create_dummy_data():
    """Create simple dummy dataset for testing."""
    # Generate random data: 100 samples, 10 features -> 1 output
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100, 1).astype(np.float32)
    
    train_ds = tf.data.Dataset.from_tensor_slices((X[:80], y[:80])).batch(16)
    val_ds = tf.data.Dataset.from_tensor_slices((X[80:], y[80:])).batch(16)
    
    return train_ds, val_ds


def create_dummy_model():
    """Create simple model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model



# Load config
cfg = OmegaConf.load("config/train/default.yaml")

# Create training spec
session = training_session_from_cfg(cfg, run_dir="runs/experiment_001")

# Create and train model
model = create_dummy_model()
train_ds, val_ds = create_dummy_data()

session.compile_model(model)
history = session.fit(model, train_ds, val_ds)

print(f"Basic config passed! Final loss: {history.history['loss'][-1]:.4f}")
