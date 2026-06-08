"""Worker for dataset building (stage 3: k-fold split + CTR-GCN format conversion)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from gui.workers.base_worker import BaseWorker
from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager, STATUS_COMPLETED, STATUS_FAILED, STATUS_IN_PROGRESS

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class DatasetBuildWorker(BaseWorker):
    """Build k-fold CTR-GCN format datasets from extracted pose NPZs."""

    def __init__(self, config: ProjectConfig, checkpoint: CheckpointManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._checkpoint = checkpoint

    def run(self) -> None:
        config = self._config
        self.log.emit("[dataset] Starting dataset build...")
        self._checkpoint.set_stage_status("dataset_building", STATUS_IN_PROGRESS)

        try:
            from code.inference.data_splitter import KFoldConfig, make_kfold_splits, write_kfold_artifacts
            from code.inference.dataset_builder import (
                ProcessedPoseCache, build_split_dataset_from_items,
                _compute_splits_fingerprint, _read_saved_fingerprint, _write_fingerprint,
            )

            pose_root = config.pose_output_root
            cv_split_dir = os.path.join(os.path.dirname(pose_root), "data", "cv_splits")
            dataset_dir = os.path.join(os.path.dirname(pose_root), "data", "dataset_ctr_gcn")

            kfold_cfg = KFoldConfig(
                seed=config.cv_seed,
                k=config.k_folds,
                val_strategy=config.val_strategy,
            )

            # Fingerprint check
            fp = _compute_splits_fingerprint(pose_root, kfold_cfg, config.fixed_t)
            saved_fp = _read_saved_fingerprint(dataset_dir)
            if saved_fp == fp:
                self.log.emit("[dataset] Inputs unchanged (fingerprint match) - skipping rebuild.")
                self._checkpoint.set_stage_status("dataset_building", STATUS_COMPLETED)
                self.finished.emit()
                return

            if self.is_cancelled:
                return

            # Build splits
            self.log.emit("[dataset] Computing k-fold splits...")
            kfold_result, fold_items = make_kfold_splits(pose_root, kfold_cfg)
            write_kfold_artifacts(cv_split_dir, kfold_result, fold_items)

            if self.is_cancelled:
                return

            # Collect all unique NPZ paths
            all_npz = []
            seen = set()
            for fold_idx, splits in fold_items.items():
                for split_name in ("train", "val", "test"):
                    for it in splits[split_name]:
                        if it.npz_path not in seen:
                            seen.add(it.npz_path)
                            all_npz.append(it.npz_path)

            # Process each NPZ once
            cache = ProcessedPoseCache(T=config.fixed_t)
            self.log.emit(f"[dataset] Processing {len(all_npz)} unique pose NPZs...")
            for i, p in enumerate(all_npz):
                if self.is_cancelled:
                    return
                cache.get(p)
                self.progress.emit(i + 1, len(all_npz))

            self.log.emit(f"[dataset] Cache populated: {cache.size} samples.")

            # Assemble fold datasets
            total_folds = len(fold_items) * 3  # 3 splits per fold
            done = 0
            for fold_idx, splits in fold_items.items():
                for split_name in ("train", "val", "test"):
                    if self.is_cancelled:
                        return
                    build_split_dataset_from_items(
                        splits[split_name],
                        cache=cache,
                        split_name=split_name,
                        out_dir=dataset_dir,
                        out_prefix=f"fold_{fold_idx}",
                    )
                    done += 1
                    self.log.emit(f"[dataset] Built fold_{fold_idx}_{split_name}")

            _write_fingerprint(dataset_dir, fp)
            self._checkpoint.set_stage_status("dataset_building", STATUS_COMPLETED)
            self.log.emit("[dataset] K-fold dataset build complete.")

        except Exception as e:
            self._checkpoint.set_stage_status("dataset_building", STATUS_FAILED)
            self.error.emit(f"Dataset build failed: {e}")

        self.finished.emit()
