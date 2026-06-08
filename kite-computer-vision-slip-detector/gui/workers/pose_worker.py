"""Worker for pose extraction (stage 1+2: extract + preprocess)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List

import numpy as np

from PySide6.QtCore import Signal

from gui.workers.base_worker import BaseWorker
from gui.config import ProjectConfig
from gui.checkpoint import CheckpointManager, STATUS_COMPLETED, STATUS_FAILED, STATUS_IN_PROGRESS
from gui.core.pipeline_adapter import (
    build_pose_extractor,
    build_interp_config,
    build_smoother,
    mirrored_output_path,
    stage_tag,
)


class PoseExtractionWorker(BaseWorker):
    """Extract skeleton poses from videos and save as NPZ files."""

    video_completed = Signal(str, str)   # (video_path, output_npz_path)
    video_failed = Signal(str, str)      # (video_path, error_message)

    def __init__(
        self,
        config: ProjectConfig,
        video_paths: List[str],
        checkpoint: CheckpointManager,
        parent=None,
    ):
        super().__init__(parent)
        self._config = config
        self._video_paths = video_paths
        self._checkpoint = checkpoint

    def run(self) -> None:
        config = self._config
        total = len(self._video_paths)
        if total == 0:
            self.log.emit("[pose] No videos to process.")
            self.finished.emit()
            return

        self._checkpoint.set_stage_status("pose_extraction", STATUS_IN_PROGRESS)

        try:
            extractor = build_pose_extractor(config, gpu_id=0)
        except Exception as e:
            self.error.emit(f"Failed to load pose model: {e}")
            return

        interp_cfg = build_interp_config(config) if config.do_interp else None
        smoother = build_smoother(config) if config.do_smooth else None
        tag = stage_tag(config.do_interp, config.do_smooth)

        # Lazy import for interpolation function
        if config.do_interp:
            from code.preprocessing.pose_interpolation import interpolate_pose_sequence

        data_root_abs = os.path.abspath(config.video_root)
        out_root_abs = os.path.abspath(config.pose_output_root)
        os.makedirs(out_root_abs, exist_ok=True)

        config_hash = CheckpointManager.pipeline_config_hash(
            config.pose_backend, config.do_interp, config.do_smooth,
            config.fps_scale, config.interp_mode, config.ema_alpha, config.conf_thr,
        )

        for i, video_path in enumerate(self._video_paths):
            if self.is_cancelled:
                self.log.emit("[pose] Cancelled by user.")
                break

            video_abs = os.path.abspath(video_path)
            out_path = mirrored_output_path(video_abs, data_root_abs, out_root_abs, tag)
            basename = os.path.basename(video_path)

            # Skip if already done with same config
            vs = self._checkpoint.get_video_status(video_path)
            if vs and vs.status == STATUS_COMPLETED and vs.config_hash == config_hash:
                if os.path.isfile(vs.output_npz):
                    self.log.emit(f"[pose] Skip (cached): {basename}")
                    self.progress.emit(i + 1, total)
                    continue

            self.log.emit(f"[pose] Processing ({i+1}/{total}): {basename}")

            try:
                poses_raw, meta = extractor.extract_pose_from_video(
                    video_abs, conf_thr=config.conf_thr,
                )

                poses_stage = poses_raw

                if config.do_interp:
                    poses_stage = interpolate_pose_sequence(poses_stage, interp_cfg)

                if config.do_smooth and smoother is not None:
                    poses_stage = smoother.smooth_sequence(poses_stage)

                payload_meta = {
                    **meta,
                    "video_relpath": os.path.relpath(video_abs, data_root_abs),
                    "pipeline": {
                        "do_interp": config.do_interp,
                        "do_smooth": config.do_smooth,
                    },
                    "interp_config": asdict(interp_cfg) if interp_cfg else None,
                    "smooth_config": asdict(smoother.config) if smoother else None,
                    "poses_raw_shape": list(np.asarray(poses_raw).shape),
                    "poses_saved_stage": tag,
                }

                np.savez_compressed(
                    out_path,
                    poses=np.asarray(poses_stage, dtype=np.float32),
                    meta_json=json.dumps(payload_meta),
                )

                self._checkpoint.set_video_status(
                    video_path, STATUS_COMPLETED,
                    output_npz=out_path, config_hash=config_hash,
                )
                self.video_completed.emit(video_path, out_path)
                self.log.emit(f"[pose] Done: {basename} -> {os.path.basename(out_path)}")

            except Exception as e:
                err_msg = str(e)
                self._checkpoint.set_video_status(
                    video_path, STATUS_FAILED, error=err_msg, config_hash=config_hash,
                )
                self.video_failed.emit(video_path, err_msg)
                self.log.emit(f"[pose] FAIL: {basename}: {err_msg}")

            self.progress.emit(i + 1, total)

        # Persist manifest
        self._checkpoint.save_pose_manifest()
        self._checkpoint.set_stage_status("pose_extraction", STATUS_COMPLETED)
        self.finished.emit()
