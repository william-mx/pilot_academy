# scripts/make_split.py
import json
from pathlib import Path

import pandas as pd

from pilot_academy.data.split import make_val_train_split


# =========================
# CONFIG (edit here)
# =========================
DATASET = "all_towns_with_weather"

PROJECT_DIR = Path("/workspaces/pilot_academy")
DATASET_DIR = PROJECT_DIR / "data" / DATASET
EXPORT_DIR = DATASET_DIR / "splits"

IMAGES_DIR = DATASET_DIR / "raw/images"
LABEL_TO_ID_PATH = DATASET_DIR / "raw/label_to_id.json"   # keys like "straight|follow_road"
ACTION_TO_ID_PATH = DATASET_DIR / "raw/action_to_id.json"
CSV_PATH = DATASET_DIR / "raw/df_annotations.csv"

N_VAL = 3
N_STRAIGHT = 100
SEED = 42
# =========================


def load_label_to_id(path: Path) -> dict:
    """Load label_to_id from JSON where keys are 'road_context|driver_action'."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {tuple(k.split("|")): int(v) for k, v in raw.items()}


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # image paths (index-aligned with your frame_index logic)
    image_paths = sorted(IMAGES_DIR.glob("*.png"))

    # mappings
    with open(ACTION_TO_ID_PATH, "r") as f:
        action_to_id = json.load(f)

    label_to_id = load_label_to_id(LABEL_TO_ID_PATH)
    STRAIGHT_CLASS_ID = label_to_id[("straight", "follow_road")]

    # annotations
    df_annotations = pd.read_csv(CSV_PATH, low_memory=False)

    # create split
    df_val, df_train, out_dir = make_val_train_split(
        df_annotations=df_annotations,
        n_val=N_VAL,
        seed=SEED,
        n_straight=N_STRAIGHT,
        straight_class_id=STRAIGHT_CLASS_ID,
        save_dir=EXPORT_DIR,
    )

    print("Rows:", {"val": len(df_val), "train": len(df_train)})


if __name__ == "__main__":
    main()
