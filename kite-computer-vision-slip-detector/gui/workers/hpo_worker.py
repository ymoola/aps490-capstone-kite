"""Worker for k-fold HPO grid search training (stage 4)."""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from PySide6.QtCore import Signal

from gui.workers.base_worker import BaseWorker
from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager, STATUS_COMPLETED, STATUS_FAILED, STATUS_IN_PROGRESS

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class HPOTrainingWorker(BaseWorker):
    """Run HPO grid search across k folds."""

    run_started = Signal(str)               # run_name
    run_completed = Signal(str, dict)        # run_name, summary_dict

    def __init__(
        self,
        config: ProjectConfig,
        checkpoint: CheckpointManager,
        run_name_filter: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._config = config
        self._checkpoint = checkpoint
        self._run_name_filter = run_name_filter

    def run(self) -> None:
        config = self._config
        self.log.emit("[hpo] Starting HPO grid search...")

        try:
            from code.inference.ctr_gcn import TrainConfig, TrainingCancelled, train_validate_test
            from code.inference.main import (
                fold_paths, make_run_name, aggregate_across_folds,
                safe_write_json, safe_write_jsonl, summarize_history,
            )

            dataset_dir = os.path.join(os.path.dirname(config.pose_output_root), "data", "dataset_ctr_gcn")
            runs_root = config.runs_root
            os.makedirs(runs_root, exist_ok=True)

            model_kwargs = dict(
                num_class=config.num_class,
                num_point=config.num_point,
                num_person=config.num_person,
                in_channels=config.in_channels,
                graph=config.graph,
                graph_args={},
                drop_out=config.dropout,
            )

            grid = list(itertools.product(
                config.batch_sizes,
                config.learning_rates,
                config.weight_decays,
            ))
            completed_runs = self._checkpoint.get_completed_hpo_runs()
            all_rows = self._load_existing_run_rows(runs_root)
            rows_by_name = {row.get("run_name"): row for row in all_rows if row.get("run_name")}

            ordered_runs: List[tuple[int, int, int, float, float, str]] = []
            incomplete_runs: List[tuple[int, int, int, float, float, str]] = []
            new_runs: List[tuple[int, int, int, float, float, str]] = []

            for (bs, lr, wd) in grid:
                for fold in range(config.k_folds):
                    run_name = make_run_name(fold=fold, bs=bs, lr=lr, wd=wd)
                    if self._run_name_filter and run_name != self._run_name_filter:
                        continue
                    if not self._run_name_filter and run_name in completed_runs:
                        ordered_runs.append((fold, bs, lr, wd, run_name, "completed"))
                        continue

                    state = self._get_run_resume_state(runs_root, run_name)
                    item = (fold, bs, lr, wd, run_name, state)
                    if state == "incomplete":
                        incomplete_runs.append(item)
                    else:
                        new_runs.append(item)

            runnable_runs = incomplete_runs + new_runs
            total_runs = 1 if self._run_name_filter else len(grid) * config.k_folds

            if self._run_name_filter:
                total_runs = 1
                self.log.emit(f"[hpo] Restarting selected run: {self._run_name_filter}")
            else:
                self.log.emit(
                    f"[hpo] Resume order: {len(incomplete_runs)} incomplete runs first, then {len(new_runs)} new runs"
                )

            self.log.emit(f"[hpo] Grid: {len(grid)} combos x {config.k_folds} folds = {total_runs} runs")
            self.log.emit(f"[hpo] Already completed: {len(completed_runs)} runs")

            self._checkpoint.set_stage_status("hpo_training", STATUS_IN_PROGRESS)

            done = 0

            for (fold, bs, lr, wd, run_name, state) in ordered_runs:
                self.log.emit(f"[hpo] Skipping {run_name} (already completed)")
                done += 1
                self.progress.emit(done, total_runs)

            for (fold, bs, lr, wd, run_name, state) in runnable_runs:
                if self.is_cancelled:
                    self.log.emit("[hpo] Cancelled.")
                    self.finished.emit()
                    return

                out_dir = os.path.join(runs_root, run_name)
                if state == "incomplete":
                    self.log.emit(f"[hpo] Re-running incomplete run: {run_name}")
                else:
                    self.log.emit(f"[hpo] Starting new run: {run_name}")

                train_npz, val_npz, test_npz = fold_paths(dataset_dir, fold)
                for p in (train_npz, val_npz, test_npz):
                    if not os.path.isfile(p):
                        self.error.emit(f"Missing dataset file: {p}")
                        self.finished.emit()
                        return

                os.makedirs(out_dir, exist_ok=True)

                cfg = TrainConfig(
                    device=config.device,
                    epochs=config.epochs,
                    batch_size=bs,
                    num_workers=config.num_workers,
                    lr=lr,
                    weight_decay=wd,
                    use_weighted_sampler=config.use_weighted_sampler,
                    use_class_weighted_loss=config.use_class_weighted_loss,
                    out_dir=out_dir,
                    save_best=True,
                    best_metric=config.best_metric,
                    patience=config.patience,
                )

                # Save run config
                safe_write_json(
                    os.path.join(out_dir, "run_config.json"),
                    {
                        "run_name": run_name,
                        "fold": fold,
                        "paths": {"train_npz": train_npz, "val_npz": val_npz, "test_npz": test_npz},
                        "train_config": asdict(cfg),
                        "model_kwargs": model_kwargs,
                        "time_started": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )

                self.log.emit(f"[hpo] Starting {run_name}...")
                self.run_started.emit(run_name)
                t0 = time.time()

                try:
                    result = train_validate_test(
                        ctr_repo_root=config.ctr_gcn_repo_path,
                        train_npz=train_npz,
                        val_npz=val_npz,
                        test_npz=test_npz,
                        model_kwargs=model_kwargs,
                        cfg=cfg,
                        should_stop=lambda: self.is_cancelled,
                    )
                    dt = time.time() - t0

                    if isinstance(result, dict):
                        history = result.get("history")
                        val_metrics = result.get("val_metrics")
                        test_metrics = result.get("test_metrics") or result.get("final_test")
                        best_ckpt = result.get("best_ckpt")
                        if best_ckpt is None:
                            best_ckpt = (result.get("paths") or {}).get("best_ckpt")
                    else:
                        history = val_metrics = test_metrics = best_ckpt = None

                    if history is not None:
                        safe_write_json(os.path.join(out_dir, "history.json"), history)
                    if val_metrics is not None:
                        safe_write_json(os.path.join(out_dir, "val_metrics.json"), val_metrics)
                    if test_metrics is not None:
                        safe_write_json(os.path.join(out_dir, "test_metrics.json"), test_metrics)

                    row = {
                        "run_name": run_name,
                        "fold": fold,
                        "batch_size": bs,
                        "lr": lr,
                        "weight_decay": wd,
                        "epochs": config.epochs,
                        "device": config.device,
                        "duration_sec": round(dt, 2),
                        "best_ckpt": best_ckpt,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "history_summary": summarize_history(history),
                    }
                    rows_by_name[run_name] = row
                    safe_write_jsonl(
                        os.path.join(runs_root, "all_runs.jsonl"),
                        self._sorted_rows(rows_by_name),
                    )

                    self._checkpoint.set_hpo_run_status(
                        run_name, STATUS_COMPLETED,
                        fold=fold, batch_size=bs, lr=lr, weight_decay=wd,
                    )
                    self.run_completed.emit(run_name, row)
                    self.log.emit(f"[hpo] {run_name} done in {dt/60:.1f} min")

                except TrainingCancelled:
                    self._checkpoint.set_hpo_run_status(
                        run_name, "pending",
                        fold=fold, batch_size=bs, lr=lr, weight_decay=wd,
                    )
                    self._checkpoint.refresh_stage_state(config)
                    self.log.emit(f"[hpo] {run_name} cancelled during training.")
                    self.finished.emit()
                    return

                except Exception as e:
                    self._checkpoint.set_hpo_run_status(
                        run_name, STATUS_FAILED,
                        fold=fold, batch_size=bs, lr=lr, weight_decay=wd,
                    )
                    self.log.emit(f"[hpo] {run_name} FAILED: {e}")

                done += 1
                self.progress.emit(done, total_runs)

            # Final aggregation
            final_rows = self._sorted_rows(rows_by_name)
            if final_rows:
                summary = aggregate_across_folds(final_rows, metric_key=config.best_metric)
                safe_write_json(os.path.join(runs_root, "summary_by_hparams.json"), summary)
                self.log.emit("[hpo] Wrote summary_by_hparams.json")

            self._checkpoint.set_stage_status("hpo_training", STATUS_COMPLETED)
            self.log.emit("[hpo] HPO grid search complete.")

        except Exception as e:
            self._checkpoint.set_stage_status("hpo_training", STATUS_FAILED)
            self.error.emit(f"HPO training failed: {e}")

        self.finished.emit()

    @staticmethod
    def _load_existing_run_rows(runs_root: str) -> List[Dict[str, Any]]:
        path = os.path.join(runs_root, "all_runs.jsonl")
        if not os.path.isfile(path):
            return []

        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and row.get("run_name"):
                    rows.append(row)
        return rows

    @staticmethod
    def _sorted_rows(rows_by_name: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [rows_by_name[name] for name in sorted(rows_by_name.keys())]

    @staticmethod
    def _get_run_resume_state(runs_root: str, run_name: str) -> str:
        run_dir = os.path.join(runs_root, run_name)
        history_path = os.path.join(run_dir, "history.json")
        summary_path = os.path.join(run_dir, "summary.json")
        return "incomplete" if os.path.isfile(history_path) and not os.path.isfile(summary_path) else "new"
