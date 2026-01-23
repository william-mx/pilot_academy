# tests/test_image_caching.py

import json
from pathlib import Path

import pandas as pd

from pilot_academy.data.io import read_fn
from pilot_academy.data.cache import cache_events

# =========================
# CONFIG (edit here)
# =========================
DATASET = "all_towns_with_weather"

PROJECT_DIR = Path("/workspaces/pilot_academy")
DATASET_DIR = PROJECT_DIR / "data" / DATASET

IMAGES_DIR = DATASET_DIR / "raw/images"
LABEL_TO_ID_PATH = DATASET_DIR / "raw/label_to_id.json"   # keys like "straight|follow_road"
CSV_PATH = DATASET_DIR / "raw/df_annotations.csv"
# =========================


def load_label_to_id(path: Path) -> dict:
    """Load label_to_id from JSON where keys are 'road_context|driver_action'."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {tuple(k.split("|")): int(v) for k, v in raw.items()}


def main():

    # image paths (index-aligned with your frame_index logic)
    image_paths = sorted(IMAGES_DIR.glob("*.png"))

    label_to_id = load_label_to_id(LABEL_TO_ID_PATH)
    STRAIGHT_CLASS_ID = label_to_id[("straight", "follow_road")]

    # annotations
    df_annotations = pd.read_csv(CSV_PATH, low_memory=False)

    cache = cache_events(df_annotations, image_paths, STRAIGHT_CLASS_ID, read_fn)

    print(f"Cached {len(cache)} images in memory.")

if __name__ == "__main__":
    main()
