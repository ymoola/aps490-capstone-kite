# dataset_builder.py
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from code.inference.data_splitter import (
        KFoldConfig,
        make_kfold_splits,
        write_kfold_artifacts,
        normalize_participant_key,
        DatumMeta,
    )
except ModuleNotFoundError:
    from data_splitter import (
        KFoldConfig,
        make_kfold_splits,
        write_kfold_artifacts,
        normalize_participant_key,
        DatumMeta,
    )

# -----------------------------
# CONFIG (relative to project root)
# -----------------------------
OUT_ROOT = str(_PROJECT_ROOT / "outputs" / "out_yolo")

CV_SPLIT_OUT_DIR = str(_PROJECT_ROOT / "data" / "cv_splits")
DATASET_OUT_DIR = str(_PROJECT_ROOT / "data" / "dataset_ctr_gcn")

SEED = 12345
K_FOLDS = 5
VAL_STRATEGY = "next_fold"  # val = next fold after test

# If None -> dynamically scan and print stats only
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


# -----------------------------
# Cache: process each raw NPZ exactly once
# -----------------------------
class ProcessedPoseCache:
    """
    Processes each raw pose NPZ once and caches the result in memory.
    Fold assembly just indexes into this cache instead of re-loading from disk.
    """
    def __init__(self, T: int):
        self.T = T
        self._cache: Dict[str, Tuple[np.ndarray, int, Dict]] = {}

    def get(self, npz_path: str) -> Tuple[np.ndarray, int, Dict]:
        key = os.path.normpath(npz_path)
        if key not in self._cache:
            self._cache[key] = process_pose_npz(npz_path, self.T)
        return self._cache[key]

    def get_frame_length(self, npz_path: str) -> int:
        """Return T_orig from cached meta (avoids a second disk read for scan mode)."""
        _, _, meta = self.get(npz_path)
        return meta["T_orig"]

    @property
    def size(self) -> int:
        return len(self._cache)


def scan_frame_lengths_from_cache(cache: ProcessedPoseCache, npz_paths: List[str]) -> None:
    """Print frame-length statistics using already-cached processed data."""
    lengths = np.array(
        [cache.get_frame_length(p) for p in npz_paths], dtype=np.int64
    )

    print("\n[dataset_builder] Frame count statistics:")
    print(f"  min: {lengths.min()}")
    print(f"  mean: {lengths.mean():.1f}")
    print(f"  median: {np.median(lengths)}")
    print(f"  90%: {np.percentile(lengths, 90):.1f}")
    print(f"  95%: {np.percentile(lengths, 95):.1f}")
    print(f"  max: {lengths.max()}")
    print("\nChoose FIXED_T accordingly (e.g., 100 or 120).")


def build_split_dataset_from_items(
    items: List,
    cache: ProcessedPoseCache,
    split_name: str,
    out_dir: str,
    out_prefix: str,
) -> str:
    """
    items: list[DatumMeta] from data_splitter
    cache: ProcessedPoseCache (avoids redundant disk reads)
    saves: {out_prefix}_{split_name}.npz inside out_dir
    """
    data_list: List[np.ndarray] = []
    labels: List[int] = []
    meta_list: List[Dict] = []

    for it in items:
        d, y, m = cache.get(it.npz_path)
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


# Standalone helper for production or any caller that already has items
def build_dataset_npz_from_items(
    items: List[DatumMeta],
    T: int,
    out_path: str,
    split_name: str,
    cache: Optional[ProcessedPoseCache] = None,
) -> str:
    """Build a CTR-GCN format NPZ from DatumMeta items, reusing cache if provided."""
    if cache is None:
        cache = ProcessedPoseCache(T)

    data_list: List[np.ndarray] = []
    labels: List[int] = []
    meta_list: List[Dict] = []

    for it in items:
        d, y, m = cache.get(it.npz_path)
        data_list.append(d)
        labels.append(y)
        meta_list.append(m)

    if not data_list:
        raise RuntimeError(f"No items for {split_name}")

    data_arr = np.stack(data_list, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        data=data_arr,
        labels=labels_arr,
        meta=np.array(meta_list, dtype=object),
    )
    print(f"[dataset_builder] Saved {split_name}: {data_arr.shape} -> {out_path}")
    return out_path


# -----------------------------
# Split fingerprint: skip rebuild when nothing changed
# -----------------------------
def _compute_splits_fingerprint(
    out_root: str, cfg: KFoldConfig, fixed_t: Optional[int],
) -> str:
    """
    Deterministic hash of (sorted NPZ file list + config + FIXED_T).
    If the fingerprint matches the last build, we can skip everything.
    """
    out_root_abs = os.path.abspath(os.path.expanduser(out_root))
    npz_files = sorted(
        os.path.join(r, f)
        for r, _, files in os.walk(out_root_abs)
        for f in files
        if f.lower().endswith(".npz")
    )
    h = hashlib.sha256()
    for p in npz_files:
        h.update(p.encode())
        h.update(str(os.path.getmtime(p)).encode())
    h.update(json.dumps({"seed": cfg.seed, "k": cfg.k, "val": cfg.val_strategy, "T": fixed_t}).encode())
    return h.hexdigest()[:16]


def _read_saved_fingerprint(dataset_out_dir: str) -> Optional[str]:
    fp_path = os.path.join(dataset_out_dir, ".build_fingerprint")
    if os.path.isfile(fp_path):
        return open(fp_path, "r").read().strip()
    return None


def _write_fingerprint(dataset_out_dir: str, fp: str) -> None:
    os.makedirs(dataset_out_dir, exist_ok=True)
    with open(os.path.join(dataset_out_dir, ".build_fingerprint"), "w") as f:
        f.write(fp)


def main():
    cfg = KFoldConfig(seed=SEED, k=K_FOLDS, val_strategy=VAL_STRATEGY)

    # Check if we can skip the entire build
    fingerprint = _compute_splits_fingerprint(OUT_ROOT, cfg, FIXED_T)
    saved_fp = _read_saved_fingerprint(DATASET_OUT_DIR)
    if saved_fp == fingerprint:
        print("[dataset_builder] Inputs unchanged (fingerprint match) - skipping rebuild.")
        return

    kfold_result, fold_items = make_kfold_splits(OUT_ROOT, cfg)
    write_kfold_artifacts(CV_SPLIT_OUT_DIR, kfold_result, fold_items)

    # Collect all unique NPZ paths across all folds
    all_npz_paths: List[str] = []
    seen: set = set()
    for fold_idx, splits in fold_items.items():
        for split_name in ("train", "val", "test"):
            for it in splits[split_name]:
                if it.npz_path not in seen:
                    seen.add(it.npz_path)
                    all_npz_paths.append(it.npz_path)

    # Dynamic scan mode
    if FIXED_T is None:
        cache = ProcessedPoseCache(T=100)  # T doesn't matter for length scan
        scan_frame_lengths_from_cache(cache, all_npz_paths)
        return

    # Process every unique NPZ exactly once
    cache = ProcessedPoseCache(T=FIXED_T)
    print(f"[dataset_builder] Processing {len(all_npz_paths)} unique pose NPZs...")
    for p in all_npz_paths:
        cache.get(p)
    print(f"[dataset_builder] Cache populated: {cache.size} samples processed once.")

    # Assemble fold NPZs by indexing into cache (no re-processing)
    for fold_idx, splits in fold_items.items():
        out_prefix = f"fold_{fold_idx}"
        for split_name in ("train", "val", "test"):
            build_split_dataset_from_items(
                splits[split_name],
                cache=cache,
                split_name=split_name,
                out_dir=DATASET_OUT_DIR,
                out_prefix=out_prefix,
            )

    _write_fingerprint(DATASET_OUT_DIR, fingerprint)
    print("\n[dataset_builder] K-fold dataset build complete.")


if __name__ == "__main__":
    main()
