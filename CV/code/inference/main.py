# inference/main.py
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from ctr_gcn import TrainConfig, train_validate_test


# ============================================================
# EDIT THESE VARIABLES
# ============================================================

DATASET_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\data\dataset_ctr_gcn"
CTR_GCN_REPO = r"CV\frameworks\CTR-GCN"

# Root folder where ALL runs go (each run gets its own subdir)
RUNS_ROOT = r"CV\runs\ctr_gcn_kfold_hpo"

# 5-fold by default
K_FOLDS = 5

# Model/data shape
NUM_CLASS = 2
NUM_POINT = 17
NUM_PERSON = 1
IN_CHANNELS = 3

GRAPH = "graph.coco17.Graph"
GRAPH_ARGS: Dict[str, Any] = {}
DROPOUT = 0.4

# Loss/sampling
USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTED_LOSS = True

# HPO grid (your request)
EPOCHS = 250
BATCH_SIZES = [16, 32, 64]
LRS = [1e-3, 1e-2, 1e-4, 1e-5]
WEIGHT_DECAYS = [1e-4, 1e-5]
PATIENCE = 25

NUM_WORKERS = 4
DEVICE = "cuda"  # "cpu" or "cuda:0"

BEST_METRIC = "val_balanced_acc"  # what to select best checkpoint by


# ============================================================


def fold_paths(dataset_dir: str, fold: int) -> Tuple[str, str, str]:
    train_npz = os.path.join(dataset_dir, f"fold_{fold}_train.npz")
    val_npz = os.path.join(dataset_dir, f"fold_{fold}_val.npz")
    test_npz = os.path.join(dataset_dir, f"fold_{fold}_test.npz")
    return train_npz, val_npz, test_npz


def ensure_exists(*paths: str) -> None:
    for p in paths:
        if not os.path.isfile(p):
            raise RuntimeError(f"Missing file: {os.path.abspath(p)}")


def make_run_name(fold: int, bs: int, lr: float, wd: float) -> str:
    def fmt(x: float) -> str:
        # 1e-3 -> 1e-03 to be filesystem-friendly
        return f"{x:.0e}".replace("+", "")
    return f"fold{fold}_bs{bs}_lr{fmt(lr)}_wd{fmt(wd)}"


def safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def summarize_history(history):
    """
    Supports:
      1) dict[str, list[float]]  (old expected format)
      2) list[dict]              (your current format: one dict per epoch)
    Returns a dict of best/last values + best epoch for each scalar metric.
    """
    if not history:
        return {"available": False}

    # ---- Case A: list of epoch dicts (your history.json) ----
    if isinstance(history, list):
        # Collect keys that look like scalar metrics (ignore epoch, confusion counts okay)
        keys = set()
        for row in history:
            if isinstance(row, dict):
                keys.update(row.keys())

        # remove non-metric keys you don't want summarized
        keys.discard("epoch")

        out = {"available": True, "format": "list_of_dicts", "num_epochs": len(history)}

        # helper: get series for a key
        def series(key: str):
            vals = []
            for row in history:
                if not isinstance(row, dict):
                    continue
                v = row.get(key, None)
                if v is None:
                    vals.append(None)
                else:
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(None)
            return vals

        for k in sorted(keys):
            vals = series(k)
            # drop missing
            valid = [(i, v) for i, v in enumerate(vals) if v is not None]
            if not valid:
                continue

            idxs, vs = zip(*valid)

            # best rule: loss -> min, everything else -> max (heuristic)
            if "loss" in k.lower():
                best_i = idxs[int(min(range(len(vs)), key=lambda j: vs[j]))]
                best_v = float(min(vs))
            else:
                best_i = idxs[int(max(range(len(vs)), key=lambda j: vs[j]))]
                best_v = float(max(vs))

            # epoch number (prefer explicit epoch field if present)
            best_epoch = history[best_i].get("epoch", best_i + 1) if isinstance(history[best_i], dict) else best_i + 1

            # last valid
            last_i, last_v = valid[-1]
            last_epoch = history[last_i].get("epoch", last_i + 1) if isinstance(history[last_i], dict) else last_i + 1

            out[f"{k}_best"] = best_v
            out[f"{k}_best_epoch"] = int(best_epoch)
            out[f"{k}_last"] = float(last_v)
            out[f"{k}_last_epoch"] = int(last_epoch)

        return out

    # ---- Case B: dict of metric -> list ----
    if isinstance(history, dict):
        out = {"available": True, "format": "dict_of_lists"}
        for k, v in history.items():
            if not isinstance(v, list) or len(v) == 0:
                continue

            vals = []
            for x in v:
                try:
                    vals.append(float(x))
                except Exception:
                    vals.append(None)

            valid = [(i, x) for i, x in enumerate(vals) if x is not None]
            if not valid:
                continue

            idxs, vs = zip(*valid)

            if "loss" in k.lower():
                best_i = idxs[int(min(range(len(vs)), key=lambda j: vs[j]))]
                best_v = float(min(vs))
            else:
                best_i = idxs[int(max(range(len(vs)), key=lambda j: vs[j]))]
                best_v = float(max(vs))

            out[f"{k}_best"] = best_v
            out[f"{k}_best_epoch"] = int(best_i + 1)
            out[f"{k}_last"] = float(vs[-1])
        return out

    # Unknown format
    return {"available": False, "format": str(type(history))}

def main():
    os.makedirs(RUNS_ROOT, exist_ok=True)

    model_kwargs = dict(
        num_class=NUM_CLASS,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        in_channels=IN_CHANNELS,
        graph=GRAPH,
        graph_args=GRAPH_ARGS,
        drop_out=DROPOUT,
    )

    # All experiments recorded here
    all_rows: List[Dict[str, Any]] = []

    # Grid search across hyperparams
    grid = list(itertools.product(BATCH_SIZES, LRS, WEIGHT_DECAYS))
    print(f"[HPO] grid size = {len(grid)} combos x {K_FOLDS} folds = {len(grid)*K_FOLDS} runs")

    for (bs, lr, wd) in grid:
        for fold in range(K_FOLDS):
            run_name = make_run_name(fold=fold, bs=bs, lr=lr, wd=wd)
            out_dir = os.path.join(RUNS_ROOT, run_name)
            os.makedirs(out_dir, exist_ok=True)

            train_npz, val_npz, test_npz = fold_paths(DATASET_DIR, fold)
            ensure_exists(train_npz, val_npz, test_npz)

            cfg = TrainConfig(
                device=DEVICE,
                epochs=EPOCHS,
                batch_size=bs,
                num_workers=NUM_WORKERS,
                lr=lr,
                weight_decay=wd,
                use_weighted_sampler=USE_WEIGHTED_SAMPLER,
                use_class_weighted_loss=USE_CLASS_WEIGHTED_LOSS,
                out_dir=out_dir,
                save_best=True,
                best_metric=BEST_METRIC,
                patience=PATIENCE,

            )

            # Persist run config for reproducibility
            safe_write_json(
                os.path.join(out_dir, "run_config.json"),
                {
                    "run_name": run_name,
                    "fold": fold,
                    "paths": {
                        "train_npz": train_npz,
                        "val_npz": val_npz,
                        "test_npz": test_npz,
                    },
                    "train_config": asdict(cfg),
                    "model_kwargs": model_kwargs,
                    "time_started": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

            print(f"\n[RUN] {run_name}")
            t0 = time.time()

            # ------------------------------------------------------------------
            # IMPORTANT:
            # For best results, train_validate_test should RETURN a dict like:
            #
            # {
            #   "best_ckpt": "...",
            #   "history": { "train_loss":[...], "val_balanced_acc":[...], ... },
            #   "val_metrics": {...},   # metrics at best checkpoint
            #   "test_metrics": {...},  # metrics on test set using best checkpoint
            # }
            #
            # If your current function returns None, you can still rely on its logs,
            # but you won't get programmatic summaries (see section 2 below).
            # ------------------------------------------------------------------
            result = train_validate_test(
                ctr_repo_root=CTR_GCN_REPO,
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=test_npz,
                model_kwargs=model_kwargs,
                cfg=cfg,
            )

            dt = time.time() - t0

            # Normalize result shape
            if isinstance(result, dict):
                history = result.get("history")
                val_metrics = result.get("val_metrics")
                test_metrics = result.get("test_metrics")
                best_ckpt = result.get("best_ckpt")
            else:
                history = None
                val_metrics = None
                test_metrics = None
                best_ckpt = None

            # Save full history for plotting
            if history is not None:
                safe_write_json(os.path.join(out_dir, "history.json"), history)

            # Save metrics snapshots
            if val_metrics is not None:
                safe_write_json(os.path.join(out_dir, "val_metrics.json"), val_metrics)
            if test_metrics is not None:
                safe_write_json(os.path.join(out_dir, "test_metrics.json"), test_metrics)

            # One row per run for easy aggregation
            row: Dict[str, Any] = {
                "run_name": run_name,
                "fold": fold,
                "batch_size": bs,
                "lr": lr,
                "weight_decay": wd,
                "epochs": EPOCHS,
                "device": DEVICE,
                "duration_sec": round(dt, 2),
                "best_ckpt": best_ckpt,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "history_summary": summarize_history(history),
            }
            all_rows.append(row)

            # Also write a rolling table so you can monitor progress
            safe_write_jsonl(os.path.join(RUNS_ROOT, "all_runs.jsonl"), all_rows)

            print(f"[DONE] {run_name} in {dt/60:.1f} min")

    # Final aggregation: group by hyperparams, average over folds
    summary = aggregate_across_folds(all_rows, metric_key=BEST_METRIC)
    safe_write_json(os.path.join(RUNS_ROOT, "summary_by_hparams.json"), summary)
    print("\n[HPO] Wrote summary_by_hparams.json")
    print("[HPO] All runs complete.")


def _get_metric(metrics: Dict[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    v = metrics.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def aggregate_across_folds(all_rows: List[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    """
    Aggregate per-hparam across folds.
    Produces mean/std for val metric + test metric if available.
    """
    # group by (bs, lr, wd)
    groups: Dict[Tuple[int, float, float], List[Dict[str, Any]]] = {}
    for r in all_rows:
        groups.setdefault((r["batch_size"], r["lr"], r["weight_decay"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for (bs, lr, wd), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        # collect val metric (prefer val_metrics[metric_key]; fallback to history_summary)
        vals: List[float] = []
        tests: List[float] = []

        for r in rows:
            v = _get_metric(r.get("val_metrics"), metric_key)
            if v is None:
                # fallback: history_summary value if present
                hs = r.get("history_summary") or {}
                v = hs.get(f"{metric_key}_best")
            if v is not None:
                vals.append(float(v))

            # Optional: also track test balanced acc if present
            t = _get_metric(r.get("test_metrics"), "test_balanced_acc")
            if t is None:
                t = _get_metric(r.get("test_metrics"), "balanced_acc")
            if t is not None:
                tests.append(float(t))

        item: Dict[str, Any] = {
            "batch_size": bs,
            "lr": lr,
            "weight_decay": wd,
            "num_folds": len(rows),
            "val_metric_key": metric_key,
            "val_mean": float(np.mean(vals)) if vals else None,
            "val_std": float(np.std(vals)) if vals else None,
            "test_mean": float(np.mean(tests)) if tests else None,
            "test_std": float(np.std(tests)) if tests else None,
            "missing_val_folds": int(len(rows) - len(vals)),
            "missing_test_folds": int(len(rows) - len(tests)),
        }
        out.append(item)

    # rank by val_mean descending
    ranked = sorted(out, key=lambda d: (d["val_mean"] is not None, d["val_mean"]), reverse=True)

    return {
        "metric_key": metric_key,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }


if __name__ == "__main__":
    main()