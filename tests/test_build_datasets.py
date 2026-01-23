import tensorflow as tf
from pathlib import Path

from pilot_academy.data.build_datasets import build_datasets_from_config

PROJECT_DIR = Path("/workspaces/pilot_academy")
CONFIG_PATH = PROJECT_DIR / "config/datasets/all_towns_with_weather_tfdata.yaml"

train_ds, val_ds, info = build_datasets_from_config(CONFIG_PATH)

print("TF:", tf.__version__)
print("train_ds element_spec:", train_ds.element_spec)
print("val_ds element_spec:", val_ds.element_spec)

def _fmt_leaf(t):
    shape = getattr(t, "shape", None)
    dtype = getattr(t, "dtype", None)
    return f"{shape} {dtype}"

def _print_structure(struct, prefix=""):
    if isinstance(struct, (tuple, list)):
        for j, v in enumerate(struct):
            _print_structure(v, prefix=f"{prefix}[{j}]")
    elif isinstance(struct, dict):
        for k, v in struct.items():
            _print_structure(v, prefix=f"{prefix}['{k}']")
    else:
        print(f"  {prefix}: {_fmt_leaf(struct)}")

# ---- smoke test: a couple of train batches
for i, batch in enumerate(train_ds.take(2)):
    print(f"[train batch {i}] structure:")
    _print_structure(batch)
    _ = tf.nest.map_structure(lambda t: t.numpy() if hasattr(t, "numpy") else t, batch)

# ---- smoke test: a couple of val batches
for i, batch in enumerate(val_ds.take(2)):
    print(f"[val batch {i}] structure:")
    _print_structure(batch)
    _ = tf.nest.map_structure(lambda t: t.numpy() if hasattr(t, "numpy") else t, batch)

print("OK: tf.data pipelines run and materialize correctly.")