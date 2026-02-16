# scripts/make_split.py
import json
from pathlib import Path

import pandas as pd

from pilot_academy.data.split import balanced_random_train_val_split


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

VAL_SPLIT = 0.2
SEED = 42

# Optional class filtering (set BOTH to None to use all class_ids)
CLASSES = None                # e.g. [0, 1, 2, 3]
EXCLUDE_CLASSES = None        # e.g. [3]
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

    # mappings (kept here because you had them; not used by the new split directly)
    with open(ACTION_TO_ID_PATH, "r") as f:
        action_to_id = json.load(f)

    label_to_id = load_label_to_id(LABEL_TO_ID_PATH)
    straight_class_id = label_to_id[("straight", "follow_road")]

    # annotations
    df_annotations = pd.read_csv(CSV_PATH, low_memory=False)

    # OPTIONAL: if you use frame_index as index, set it here (only if the column exists)
    if "frame_index" in df_annotations.columns:
        df_annotations = df_annotations.set_index("frame_index", drop=True)

    # create split (new function)
    df_train, df_val, max_train_samples_per_class, out_dir = balanced_random_train_val_split(
        df=df_annotations,
        classes=CLASSES,
        exclude_classes=EXCLUDE_CLASSES,
        val_split=VAL_SPLIT,
        class_col="class_id",
        seed=SEED,
        save_dir=EXPORT_DIR,
    )

    print("Rows:", {"val": len(df_val), "train": len(df_train)})
    print("max_train_samples_per_class:", max_train_samples_per_class)
    print("export:", str(out_dir) if out_dir is not None else None)


if __name__ == "__main__":
    main()
