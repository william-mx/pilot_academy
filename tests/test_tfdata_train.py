# scripts/test_tfdata_train.py
from pathlib import Path
import json

import pandas as pd
import tensorflow as tf

from pilot_academy.data.cache import cache_events
from pilot_academy.data.tfdata import make_train_dataset
from pilot_academy.data.io import read_fn


# -------------------------
# CONFIG
# -------------------------

PROJECT_DIR = Path("/workspaces/pilot_academy")

DATASET = "all_towns_with_weather"

DATASET_DIR = PROJECT_DIR / "data" / DATASET
SPLITS_ROOT = DATASET_DIR / "splits"
IMAGES_DIR = DATASET_DIR / "raw/images"
LABEL_TO_ID_PATH = DATASET_DIR / "raw/label_to_id.json"
ACTION_TO_ID_PATH = DATASET_DIR / "raw/action_to_id.json"

IMAGE_SHAPE = (220, 220, 3)
BATCH_SIZE = 3
H, W, C = IMAGE_SHAPE

N_EVENTS = 17
N_STRAIGHT = 100
SHUFFLE_BUFFER = 600

# Use your existing split directory name
SPLIT = "split_20260122_122820"  # <-- update if needed


def load_label_to_id(path: Path) -> dict:
    with open(path, "r") as f:
        raw = json.load(f)
    return {tuple(k.split("|")): int(v) for k, v in raw.items()}


def main():
    # ---- paths
    split_dir = SPLITS_ROOT / SPLIT
    train_path = split_dir / "df_train.csv"
    val_path = split_dir / "df_val.csv"
    all_path = split_dir / "df_full.csv"

    for p in [split_dir, train_path, val_path, all_path, IMAGES_DIR, LABEL_TO_ID_PATH, ACTION_TO_ID_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    # ---- image paths (sorted)
    image_paths = sorted(IMAGES_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNGs found in {IMAGES_DIR}")

    # ---- mappings
    with open(ACTION_TO_ID_PATH, "r") as f:
        action_to_id = json.load(f)

    label_to_id = load_label_to_id(LABEL_TO_ID_PATH)
    STRAIGHT_CLASS_ID = label_to_id[("straight", "follow_road")]

    # ---- load dfs
    df_train = pd.read_csv(train_path, low_memory=False)
    df_val = pd.read_csv(val_path, low_memory=False)   # not required, but useful for sanity checks
    df_all = pd.read_csv(all_path, low_memory=False)

    print("Loaded split:", split_dir)
    print("Rows:", {"train": len(df_train), "val": len(df_val), "full": len(df_all)})
    print("Num images:", len(image_paths))
    print("STRAIGHT_CLASS_ID:", STRAIGHT_CLASS_ID)

    # ---- cache events
    cache = cache_events(df_all, image_paths, STRAIGHT_CLASS_ID, read_fn)
    print(f"Cached {len(cache)} images.")

    # ---- build train dataset (your pattern)
    train_ds = make_train_dataset(
        df_train_base=df_train,
        cache=cache,
        image_paths=image_paths,
        action_to_id=action_to_id,
        read_fn=read_fn,
        straight_class_id=STRAIGHT_CLASS_ID,
        H=H,
        W=W,
        C=C,
        n_events=N_EVENTS,
        n_straight=N_STRAIGHT,
        batch_size=BATCH_SIZE,
        shuffle_buffer=SHUFFLE_BUFFER
    )

    print("TF:", tf.__version__)
    print("train_ds element_spec:", train_ds.element_spec)

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

    # ---- smoke test: take a few batches
    for i, batch in enumerate(train_ds.take(3)):
        print(f"[batch {i}] structure:")
        _print_structure(batch, prefix="")

        # Force materialization (will raise if something is wrong)
        _ = tf.nest.map_structure(lambda t: t.numpy() if hasattr(t, "numpy") else t, batch)

    print("OK: train tf.data pipeline runs.")


if __name__ == "__main__":
    main()
