import tensorflow as tf
from pathlib import Path
import hydra
from omegaconf import DictConfig

from pilot_academy.data.build_datasets import (
    build_datasets_from_config,
)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # ------------------------------------------------------------
    # Build datasets with new sampling system
    # ------------------------------------------------------------
    train_ds, val_ds, extras = build_datasets_from_config(cfg)

    print("TF:", tf.__version__)
    print("train_ds element_spec:", train_ds.element_spec)
    print("val_ds element_spec:", val_ds.element_spec)
    
    # Print sampling info
    print("\nSampling Configuration:")
    print(f"  Strategy: {extras['sampling_strategy']}")
    print(f"  Config: {extras['sampling_config']}")
    print(f"  Cache size: {extras['cache_size']}")

    # ------------------------------------------------------------
    # Helpers to inspect structure
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Smoke test: train batches
    # ------------------------------------------------------------
    print("\nTraining batches:")
    for i, batch in enumerate(train_ds.take(2)):
        print(f"[train batch {i}] structure:")
        _print_structure(batch)
        _ = tf.nest.map_structure(
            lambda t: t.numpy() if hasattr(t, "numpy") else t,
            batch,
        )

    # ------------------------------------------------------------
    # Smoke test: val batches
    # ------------------------------------------------------------
    print("\nValidation batches:")
    for i, batch in enumerate(val_ds.take(2)):
        print(f"[val batch {i}] structure:")
        _print_structure(batch)
        _ = tf.nest.map_structure(
            lambda t: t.numpy() if hasattr(t, "numpy") else t,
            batch,
        )

    print("\n✓ OK: tf.data pipelines run and materialize correctly.")


if __name__ == "__main__":
    main()