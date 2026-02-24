# dataset_builder.py
from __future__ import annotations

import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple

from data_splitter import (
    SplitConfig,
    make_splits,
    write_split_artifacts,
)


# -----------------------------
# CONFIG (EDIT THESE)
# -----------------------------
OUT_ROOT = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\outputs\out_yolo"
SPLIT_OUT_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\data\splits"
DATASET_OUT_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\data\dataset_ctr_gcn"

SEED = 12345
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# If None → dynamically scan and print stats only
FIXED_T = 100        # e.g. 100 once decided
NUM_KPTS = 17
NUM_CHANNELS = 3
NUM_PERSON = 1

PADDING_MODE = "zeros"   # locked choice
# -----------------------------


def uniform_sample_indices(T_orig: int, T: int) -> np.ndarray:
    """Uniformly sample T indices from [0, T_orig)."""
    return np.linspace(0, T_orig - 1, T).astype(np.int64)


def process_pose_npz(
    npz_path: str,
    T: int,
) -> Tuple[np.ndarray, int, Dict]:
    """
    Returns:
      data: (C, T, V, M)
      label: int
      meta: dict
    """
    with np.load(npz_path, allow_pickle=True) as z:
        poses = z["poses"]            # (T_orig, V, 3)
        meta_json = json.loads(z["meta_json"].item())

    T_orig, V, C = poses.shape
    if V != NUM_KPTS:
        raise RuntimeError(
            f"Unexpected number of keypoints (V={V}, expected {NUM_KPTS}) in {npz_path}"
        )
    assert C == 3

    width = meta_json.get("width")
    height = meta_json.get("height")
    if width is None or height is None:
        raise RuntimeError(f"Missing width/height in meta_json for {npz_path}")

    # Normalize x, y
    poses = poses.copy()
    poses[..., 0] /= float(width)
    poses[..., 1] /= float(height)

    # Temporal sampling / padding
    if T_orig >= T:
        idx = uniform_sample_indices(T_orig, T)
        poses_T = poses[idx]
    else:
        poses_T = np.zeros((T, V, C), dtype=np.float32)
        poses_T[:T_orig] = poses

    # Reformat: (T, V, C) → (C, T, V, M)
    data = poses_T.transpose(2, 0, 1)[..., None]  # add M dim

    # Label is encoded by filename (DatumMeta), not pose extractor
    basename = os.path.basename(npz_path)
    label_match = re.search(r"_([DU][PFU])_", basename)
    if not label_match:
        raise RuntimeError(f"Could not infer label_code from filename: {npz_path}")

    label_code = label_match.group(1).upper()
    label = 1 if label_code.endswith("F") else 0  # fail=1, pass=0

    angle_match = re.search(r"_([0-9]+(?:\.[0-9]+)?)_", basename)
    angle = float(angle_match.group(1)) if angle_match else -1.0

    meta = {
        "npz_path": npz_path.replace("\\", "/"),
        "participant_dir": os.path.basename(os.path.dirname(npz_path)),
        "angle": angle,
        "label_code": label_code,
        "T_orig": int(T_orig),
    }

    return data.astype(np.float32), label, meta


def scan_frame_lengths(items: List) -> None:
    lengths = [np.load(it.npz_path)["poses"].shape[0] for it in items]
    arr = np.array(lengths)

    print("\n[dataset_builder] Frame count statistics:")
    print(f"  min: {arr.min()}")
    print(f"  mean: {arr.mean():.1f}")
    print(f"  median: {np.median(arr)}")
    print(f"  90%: {np.percentile(arr, 90):.1f}")
    print(f"  95%: {np.percentile(arr, 95):.1f}")
    print(f"  max: {arr.max()}")
    print("\nChoose FIXED_T accordingly (e.g., 100 or 120).")


def build_split_dataset(
    items: List,
    T: int,
    split_name: str,
    out_dir: str,
) -> None:
    data_list = []
    labels = []
    meta_list = []

    for it in items:
        d, y, m = process_pose_npz(it.npz_path, T)
        data_list.append(d)
        labels.append(y)
        meta_list.append(m)

    data_arr = np.stack(data_list, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split_name}.npz")

    np.savez_compressed(
        out_path,
        data=data_arr,
        labels=labels_arr,
        meta=np.array(meta_list, dtype=object),
    )

    print(f"[dataset_builder] Saved {split_name}: {data_arr.shape} → {out_path}")


def main():
    cfg = SplitConfig(
        seed=SEED,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    split_result, split_items = make_splits(OUT_ROOT, cfg)
    write_split_artifacts(SPLIT_OUT_DIR, split_result, split_items)

    # Dynamic scan mode
    if FIXED_T is None:
        all_items = (
            split_items["train"]
            + split_items["val"]
            + split_items["test"]
        )
        scan_frame_lengths(all_items)
        return

    # Build datasets
    for split_name in ("train", "val", "test"):
        build_split_dataset(
            split_items[split_name],
            T=FIXED_T,
            split_name=split_name,
            out_dir=DATASET_OUT_DIR,
        )

    print("\n[dataset_builder] Dataset build complete.")


if __name__ == "__main__":
    main()
