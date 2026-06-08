"""Worker for production model training (stage 5)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from gui.workers.base_worker import BaseWorker
from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager, STATUS_COMPLETED, STATUS_FAILED, STATUS_IN_PROGRESS

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class ProductionTrainingWorker(BaseWorker):
    """Train the final production model using best HPs from HPO."""

    def __init__(self, config: ProjectConfig, checkpoint: CheckpointManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._checkpoint = checkpoint

    def run(self) -> None:
        config = self._config
        self.log.emit("[production] Starting production training...")

        try:
            from code.inference.ctr_gcn import TrainConfig, TrainingCancelled, train_validate_test
            from code.production.train_production import load_best_hparams, make_participant_split
            from code.inference.dataset_builder import ProcessedPoseCache, build_dataset_npz_from_items

            out_dir = config.production_output_dir
            os.makedirs(out_dir, exist_ok=True)

            # 1. Load best HPs from HPO summary
            summary_path = os.path.join(config.runs_root, "summary_by_hparams.json")
            if not os.path.isfile(summary_path):
                self.error.emit(f"HPO summary not found: {summary_path}\nRun HPO training first.")
                self.finished.emit()
                return

            best_hp = load_best_hparams(summary_path)
            bs = best_hp["batch_size"]
            lr = best_hp["lr"]
            wd = best_hp["weight_decay"]

            self.log.emit(f"[production] Best HPs: bs={bs}, lr={lr}, wd={wd}")
            self.log.emit(f"[production] Val BAcc: {best_hp.get('val_mean', 0):.4f} +/- {best_hp.get('val_std', 0):.4f}")

            if self.is_cancelled:
                self.finished.emit()
                return

            # 2. Participant-level split
            self.log.emit("[production] Building participant split...")
            train_p, val_p, by_p = make_participant_split(
                config.pose_output_root,
                config.production_val_ratio,
                config.production_split_seed,
            )

            train_items = []
            for p in train_p:
                train_items.extend(sorted(by_p[p], key=lambda x: x.rel_path))
            val_items = []
            for p in val_p:
                val_items.extend(sorted(by_p[p], key=lambda x: x.rel_path))

            self.log.emit(f"[production] Split: {len(train_p)} train / {len(val_p)} val participants")
            self.log.emit(f"[production] Samples: {len(train_items)} train / {len(val_items)} val")

            if self.is_cancelled:
                self.finished.emit()
                return

            # 3. Build dataset NPZs
            self.log.emit("[production] Building dataset NPZs...")
            cache = ProcessedPoseCache(T=config.fixed_t)
            dataset_dir = os.path.join(out_dir, "dataset")

            train_npz = build_dataset_npz_from_items(
                train_items, config.fixed_t,
                os.path.join(dataset_dir, "train.npz"), "train", cache=cache,
            )
            val_npz = build_dataset_npz_from_items(
                val_items, config.fixed_t,
                os.path.join(dataset_dir, "val.npz"), "val", cache=cache,
            )

            if self.is_cancelled:
                self.finished.emit()
                return

            # 4. Save split info
            split_info = {
                "seed": config.production_split_seed,
                "val_ratio": config.production_val_ratio,
                "train_participants": train_p,
                "val_participants": val_p,
                "train_samples": len(train_items),
                "val_samples": len(val_items),
            }
            with open(os.path.join(out_dir, "split_info.json"), "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=2)

            # 5. Train
            self._checkpoint.set_stage_status("production_training", STATUS_IN_PROGRESS)

            model_kwargs = dict(
                num_class=config.num_class,
                num_point=config.num_point,
                num_person=config.num_person,
                in_channels=config.in_channels,
                graph=config.graph,
                graph_args={},
                drop_out=config.dropout,
            )

            run_dir = os.path.join(out_dir, "run")
            cfg = TrainConfig(
                device=config.device,
                epochs=config.epochs,
                batch_size=bs,
                num_workers=config.num_workers,
                lr=lr,
                weight_decay=wd,
                use_weighted_sampler=config.use_weighted_sampler,
                use_class_weighted_loss=config.use_class_weighted_loss,
                out_dir=run_dir,
                save_best=True,
                best_metric=config.best_metric,
                patience=config.production_patience,
            )

            self.log.emit(f"[production] Training: bs={bs}, lr={lr}, wd={wd}, epochs={config.epochs}")
            t0 = time.time()

            result = train_validate_test(
                ctr_repo_root=config.ctr_gcn_repo_path,
                train_npz=train_npz,
                val_npz=val_npz,
                test_npz=val_npz,
                model_kwargs=model_kwargs,
                cfg=cfg,
                should_stop=lambda: self.is_cancelled,
            )

            elapsed = time.time() - t0

            # 6. Copy best model
            best_src = os.path.join(run_dir, "best.pt")
            best_dst = os.path.join(out_dir, "best_model.pt")
            if os.path.isfile(best_src):
                shutil.copy2(best_src, best_dst)
                self.log.emit(f"[production] Best model: {best_dst}")

            # 7. Save summary
            final_summary = {
                "best_hparams": best_hp,
                "split_info": split_info,
                "elapsed_sec": round(elapsed, 2),
                "best_model_path": best_dst,
                "time_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if isinstance(result, dict):
                final_summary["val_metrics"] = result.get("final_test")
                final_summary["history_epochs"] = len(result.get("history", []))

            with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
                json.dump(final_summary, f, indent=2)

            self._checkpoint.set_stage_status("production_training", STATUS_COMPLETED)
            self.log.emit(f"[production] Training complete in {elapsed/60:.1f} min")

        except TrainingCancelled:
            self._checkpoint.refresh_stage_state(config)
            self.log.emit("[production] Training cancelled.")

        except Exception as e:
            self._checkpoint.set_stage_status("production_training", STATUS_FAILED)
            self.error.emit(f"Production training failed: {e}")

        self.finished.emit()
