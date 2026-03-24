# production/train_production.py
"""
Production model training script.

1. Reads HPO summary to auto-select best hyperparameters.
2. Creates an 80/20 participant-level train/val split.
3. Builds CTR-GCN format dataset NPZs for the split.
4. Trains for all 250 epochs (saves best checkpoint by val_balanced_acc).
5. Outputs model, training history, and split info to CV/production/.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# ============================================================
# PATHS (edit as needed)
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CV_ROOT = os.path.join(PROJECT_ROOT, "CV")

HPO_SUMMARY_PATH = os.path.join(CV_ROOT, "runs", "ctr_gcn_kfold_hpo", "summary_by_hparams.json")
POSE_NPZ_ROOT = os.path.join(CV_ROOT, "outputs", "out_yolo")
CTR_GCN_REPO = os.path.join(CV_ROOT, "frameworks", "CTR-GCN")
PRODUCTION_OUT_DIR = os.path.join(CV_ROOT, "production")

# Model/data shape (must match HPO training)
NUM_CLASS = 2
NUM_POINT = 17
NUM_PERSON = 1
IN_CHANNELS = 3
GRAPH = "graph.coco17.Graph"
GRAPH_ARGS: Dict[str, Any] = {}
DROPOUT = 0.4

# Dataset
FIXED_T = 100
NUM_CHANNELS = 3

# Training
EPOCHS = 250
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTED_LOSS = True
BEST_METRIC = "val_balanced_acc"
NUM_WORKERS = 4
DEVICE = "cuda"

# Split
SPLIT_SEED = 42
VAL_RATIO = 0.2  # 80/20

# ============================================================

from CV.code.inference.ctr_gcn import TrainConfig, train_validate_test  
from CV.code.inference.data_splitter import build_index, normalize_participant_key 
from CV.code.inference.dataset_builder import process_pose_npz 

def load_best_hparams(summary_path: str) -> Dict[str, Any]:
    """Load best hyperparameters from HPO summary."""
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    best = summary["best"]
    print(f"[production] Best HPs from k-fold CV:")
    print(f"  batch_size: {best['batch_size']}")
    print(f"  lr: {best['lr']}")
    print(f"  weight_decay: {best['weight_decay']}")
    print(f"  val_balanced_acc mean: {best['val_mean']:.4f} +/- {best['val_std']:.4f}")
    return best


def make_participant_split(
    pose_root: str,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], Dict[str, List]]:
    """
    80/20 participant-level random split.

    Returns:
        train_participants, val_participants, items_by_participant
    """
    items, warnings = build_index(pose_root)
    if warnings:
        print(f"[production] {len(warnings)} warnings during indexing (see split_info.json)")

    # Group by participant
    by_p: Dict[str, List] = {}
    for it in items:
        key = normalize_participant_key(it.participant_dir)
        by_p.setdefault(key, []).append(it)

    participants = sorted(by_p.keys())
    n = len(participants)
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_val

    rng = random.Random(seed)
    shuffled = participants[:]
    rng.shuffle(shuffled)

    train_p = sorted(shuffled[:n_train])
    val_p = sorted(shuffled[n_train:])

    train_samples = sum(len(by_p[p]) for p in train_p)
    val_samples = sum(len(by_p[p]) for p in val_p)

    print(f"[production] Split: {n_train} train / {n_val} val participants (seed={seed})")
    print(f"  Train participants ({n_train}): {train_p}")
    print(f"  Val participants ({n_val}): {val_p}")
    print(f"  Train samples: {train_samples}, Val samples: {val_samples}")

    return train_p, val_p, by_p


def build_dataset_npz(
    items: List,
    out_path: str,
    T: int,
    split_name: str,
) -> str:
    """Build a CTR-GCN format NPZ from DatumMeta items."""
    data_list: List[np.ndarray] = []
    labels: List[int] = []
    meta_list: List[Dict] = []

    for it in items:
        d, y, m = process_pose_npz(it.npz_path, T)
        data_list.append(d)
        labels.append(y)
        meta_list.append(m)

    if not data_list:
        raise RuntimeError(f"No items for {split_name}")

    data_arr = np.stack(data_list, axis=0)        # (N, C, T, V, M)
    labels_arr = np.array(labels, dtype=np.int64)  # (N,)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        data=data_arr,
        labels=labels_arr,
        meta=np.array(meta_list, dtype=object),
    )
    print(f"[production] Saved {split_name}: {data_arr.shape} -> {out_path}")
    return out_path


def main():
    os.makedirs(PRODUCTION_OUT_DIR, exist_ok=True)

    # 1. Load best HPs
    best_hp = load_best_hparams(HPO_SUMMARY_PATH)
    bs = best_hp["batch_size"]
    lr = best_hp["lr"]
    wd = best_hp["weight_decay"]

    # 2. Participant-level 80/20 split
    train_p, val_p, by_p = make_participant_split(POSE_NPZ_ROOT, VAL_RATIO, SPLIT_SEED)

    train_items = []
    for p in train_p:
        train_items.extend(sorted(by_p[p], key=lambda x: x.rel_path))
    val_items = []
    for p in val_p:
        val_items.extend(sorted(by_p[p], key=lambda x: x.rel_path))

    # 3. Build dataset NPZs
    dataset_dir = os.path.join(PRODUCTION_OUT_DIR, "dataset")
    train_npz = build_dataset_npz(train_items, os.path.join(dataset_dir, "train.npz"), FIXED_T, "train")
    val_npz = build_dataset_npz(val_items, os.path.join(dataset_dir, "val.npz"), FIXED_T, "val")

    # 4. Save split info
    split_info = {
        "seed": SPLIT_SEED,
        "val_ratio": VAL_RATIO,
        "train_participants": train_p,
        "val_participants": val_p,
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "total_participants": len(train_p) + len(val_p),
        "total_samples": len(train_items) + len(val_items),
    }
    split_info_path = os.path.join(PRODUCTION_OUT_DIR, "split_info.json")
    with open(split_info_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"[production] Saved split info: {split_info_path}")

    # 5. Train
    model_kwargs = dict(
        num_class=NUM_CLASS,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        in_channels=IN_CHANNELS,
        graph=GRAPH,
        graph_args=GRAPH_ARGS,
        drop_out=DROPOUT,
    )

    run_dir = os.path.join(PRODUCTION_OUT_DIR, "run")
    cfg = TrainConfig(
        device=DEVICE,
        epochs=EPOCHS,
        batch_size=bs,
        num_workers=NUM_WORKERS,
        lr=lr,
        weight_decay=wd,
        use_weighted_sampler=USE_WEIGHTED_SAMPLER,
        use_class_weighted_loss=USE_CLASS_WEIGHTED_LOSS,
        out_dir=run_dir,
        save_best=True,
        best_metric=BEST_METRIC,
        patience=40,
    )

    # Save run config
    run_config = {
        "best_hparams": best_hp,
        "model_kwargs": model_kwargs,
        "train_config": asdict(cfg),
        "split_info": split_info,
        "paths": {
            "train_npz": train_npz,
            "val_npz": val_npz,
            "ctr_gcn_repo": CTR_GCN_REPO,
            "production_out": PRODUCTION_OUT_DIR,
        },
        "time_started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    run_config_path = os.path.join(PRODUCTION_OUT_DIR, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n[production] Starting training: bs={bs}, lr={lr}, wd={wd}, epochs={EPOCHS}")
    t0 = time.time()

    result = train_validate_test(
        ctr_repo_root=CTR_GCN_REPO,
        train_npz=train_npz,
        val_npz=val_npz,
        test_npz=val_npz,  # no separate test set; val is used for monitoring
        model_kwargs=model_kwargs,
        cfg=cfg,
    )

    elapsed = time.time() - t0

    # 6. Copy best model to production root for easy access
    best_ckpt_src = os.path.join(run_dir, "best.pt")
    best_ckpt_dst = os.path.join(PRODUCTION_OUT_DIR, "best_model.pt")
    if os.path.isfile(best_ckpt_src):
        import shutil
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        print(f"[production] Best model copied to: {best_ckpt_dst}")

    # 7. Save final summary
    final_summary = {
        "best_hparams": best_hp,
        "split_info": split_info,
        "elapsed_sec": round(elapsed, 2),
        "elapsed_min": round(elapsed / 60, 1),
        "best_model_path": best_ckpt_dst,
        "time_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if isinstance(result, dict):
        final_summary["val_metrics"] = result.get("final_test")
        final_summary["history_epochs"] = len(result.get("history", []))

    summary_path = os.path.join(PRODUCTION_OUT_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\n[production] Training complete in {elapsed/60:.1f} min")
    print(f"[production] Best model: {best_ckpt_dst}")
    print(f"[production] Summary: {summary_path}")


if __name__ == "__main__":
    main()
