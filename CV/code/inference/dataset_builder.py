# dataset_builder.py
from __future__ import annotations

import json
import os
import re
import numpy as np
from typing import Dict, List, Tuple

from data_splitter import (
    KFoldConfig,
    make_kfold_splits,
    write_kfold_artifacts,
    normalize_participant_key,
)

# -----------------------------
# CONFIG (EDIT THESE)
# -----------------------------
OUT_ROOT = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\outputs\out_yolo"

CV_SPLIT_OUT_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\data\cv_splits"
DATASET_OUT_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\data\dataset_ctr_gcn"

SEED = 12345
K_FOLDS = 5
VAL_STRATEGY = "next_fold"  # val = next fold after test

# If None → dynamically scan and print stats only
FIXED_T = 100        # e.g. 100 once decided
NUM_KPTS = 17
NUM_CHANNELS = 3
NUM_PERSON = 1

PADDING_MODE = "zeros"   # locked choice
# -----------------------------


def uniform_sample_indices(T_orig: int, T: int) -> np.ndarray:
    """Uniformly sample T indices from [0, T_orig)."""
    if T_orig <= 0:
        return np.zeros((T,), dtype=np.int64)
    return np.linspace(0, T_orig - 1, T).astype(np.int64)


def process_pose_npz(npz_path: str, T: int) -> Tuple[np.ndarray, int, Dict]:
    """
    Load a pose npz (produced by your pipeline) and convert to CTR-GCN format.

    Returns:
      data: (C, T, V, M)
      label: int (fail=1, pass=0)
      meta: dict
    """
    with np.load(npz_path, allow_pickle=True) as z:
        poses = z["poses"]  # (T_orig, V, 3)
        meta_json = json.loads(z["meta_json"].item())

    if poses.ndim != 3:
        raise RuntimeError(f"Expected poses shape (T,V,3), got {poses.shape} in {npz_path}")

    T_orig, V, C = poses.shape
    if V != NUM_KPTS:
        raise RuntimeError(f"Unexpected keypoints V={V}, expected {NUM_KPTS} in {npz_path}")
    if C != NUM_CHANNELS:
        raise RuntimeError(f"Unexpected channels C={C}, expected {NUM_CHANNELS} in {npz_path}")

    width = meta_json.get("width")
    height = meta_json.get("height")
    if width is None or height is None:
        raise RuntimeError(f"Missing width/height in meta_json for {npz_path}")

    poses = poses.astype(np.float32, copy=True)

    # Normalize x, y to [0,1]
    poses[..., 0] /= float(width)
    poses[..., 1] /= float(height)

    # Temporal sampling / padding
    if T_orig >= T:
        idx = uniform_sample_indices(T_orig, T)
        poses_T = poses[idx]
    else:
        poses_T = np.zeros((T, V, C), dtype=np.float32)
        poses_T[:T_orig] = poses

    # (T, V, C) -> (C, T, V, M)
    data = poses_T.transpose(2, 0, 1)[..., None]  # add M dim => M=1

    # Label from filename
    basename = os.path.basename(npz_path)
    label_match = re.search(r"_([DU][PFU])_", basename)
    if not label_match:
        raise RuntimeError(f"Could not infer label_code from filename: {npz_path}")

    label_code = label_match.group(1).upper()
    label = 1 if label_code.endswith("F") else 0  # fail=1, pass=0

    # Optional angle from filename
    angle_match = re.search(r"_([0-9]+(?:\.[0-9]+)?)_", basename)
    angle = float(angle_match.group(1)) if angle_match else -1.0

    meta = {
        "npz_path": npz_path.replace("\\", "/"),
        "participant_dir": os.path.basename(os.path.dirname(npz_path)),
        "participant_key": normalize_participant_key(os.path.basename(os.path.dirname(npz_path))),
        "angle": angle,
        "label_code": label_code,
        "T_orig": int(T_orig),
        "width": int(width),
        "height": int(height),
    }

    return data.astype(np.float32), int(label), meta


def scan_frame_lengths(npz_paths: List[str]) -> None:
    lengths = []
    for p in npz_paths:
        with np.load(p, allow_pickle=True) as z:
            lengths.append(int(z["poses"].shape[0]))
    arr = np.array(lengths, dtype=np.int64)

    print("\n[dataset_builder] Frame count statistics:")
    print(f"  min: {arr.min()}")
    print(f"  mean: {arr.mean():.1f}")
    print(f"  median: {np.median(arr)}")
    print(f"  90%: {np.percentile(arr, 90):.1f}")
    print(f"  95%: {np.percentile(arr, 95):.1f}")
    print(f"  max: {arr.max()}")
    print("\nChoose FIXED_T accordingly (e.g., 100 or 120).")


def build_split_dataset_from_items(
    items: List,
    T: int,
    split_name: str,
    out_dir: str,
    out_prefix: str,
) -> str:
    """
    items: list[DatumMeta] from data_splitter
    saves: {out_prefix}_{split_name}.npz inside out_dir
    """
    data_list: List[np.ndarray] = []
    labels: List[int] = []
    meta_list: List[Dict] = []

    for it in items:
        d, y, m = process_pose_npz(it.npz_path, T)
        data_list.append(d)
        labels.append(y)
        meta_list.append(m)

    if not data_list:
        raise RuntimeError(f"No items for {out_prefix} {split_name}")

    data_arr = np.stack(data_list, axis=0)          # (N, C, T, V, M)
    labels_arr = np.array(labels, dtype=np.int64)   # (N,)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_prefix}_{split_name}.npz")

    np.savez_compressed(
        out_path,
        data=data_arr,
        labels=labels_arr,
        meta=np.array(meta_list, dtype=object),
    )

    print(f"[dataset_builder] Saved {out_prefix}_{split_name}: {data_arr.shape} -> {out_path}")
    return out_path


def main():
    cfg = KFoldConfig(seed=SEED, k=K_FOLDS, val_strategy=VAL_STRATEGY)

    kfold_result, fold_items = make_kfold_splits(OUT_ROOT, cfg)
    write_kfold_artifacts(CV_SPLIT_OUT_DIR, kfold_result, fold_items)

    # Dynamic scan mode
    if FIXED_T is None:
        all_npz = []
        for fold_idx, splits in fold_items.items():
            for split_name in ("train", "val", "test"):
                all_npz.extend([it.npz_path for it in splits[split_name]])
        scan_frame_lengths(all_npz)
        return

    # Build datasets for each fold
    for fold_idx, splits in fold_items.items():
        out_prefix = f"fold_{fold_idx}"
        for split_name in ("train", "val", "test"):
            build_split_dataset_from_items(
                splits[split_name],
                T=FIXED_T,
                split_name=split_name,
                out_dir=DATASET_OUT_DIR,
                out_prefix=out_prefix,
            )

    print("\n[dataset_builder] K-fold dataset build complete.")


if __name__ == "__main__":
    main()