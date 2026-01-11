# pose.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yolo as yolo_backend
import mp_pose_landmarker as mp_backend


@dataclass
class PoseBackendConfig:
    """
    Backend-agnostic config. Only fields relevant to the chosen backend are used.
    """
    name: str  # "yolo" or "mediapipe"

    # -------- YOLO backend options --------
    yolo_model_path: str = "CV/yolo11x-pose.pt"
    yolo_num_kpts: int = 17
    yolo_device: int | str | None = None
    yolo_batch_size: int = 8
    yolo_verbose: bool = False

    # -------- MediaPipe Pose Landmarker options --------
    # NOTE: This is a .task model file (Pose Landmarker model)
    mp_model_path: str = "CV/models/pose_landmarker.task"
    mp_num_poses: int = 1
    mp_min_det_conf: float = 0.5
    mp_min_presence_conf: float = 0.5
    mp_min_track_conf: float = 0.5


class PoseExtractor:
    def __init__(self, cfg: PoseBackendConfig):
        self.cfg = cfg
        self.backend_name = cfg.name.lower()

        # NEW: running timestamp for mediapipe VIDEO mode
        self._mp_next_ts_ms = 0

        if self.backend_name == "yolo":
            self.model = yolo_backend.load_model(cfg.yolo_model_path)

        elif self.backend_name == "mediapipe":
            self.model = mp_backend.load_model(
                model_path=cfg.mp_model_path,
                num_poses=cfg.mp_num_poses,
                min_pose_detection_confidence=cfg.mp_min_det_conf,
                min_pose_presence_confidence=cfg.mp_min_presence_conf,
                min_tracking_confidence=cfg.mp_min_track_conf,
            )
        else:
            raise ValueError(f"Unknown backend: {cfg.name!r}")

    def extract_pose_from_video(self, video_path: str):
        if self.backend_name == "yolo":
            return yolo_backend.extract_pose_from_video(
                video_path,
                self.model,
                num_kpts=self.cfg.yolo_num_kpts,
                device=self.cfg.yolo_device,
                batch_size=self.cfg.yolo_batch_size,
                verbose=self.cfg.yolo_verbose,
            )

        if self.backend_name == "mediapipe":
            poses, meta, next_ts = mp_backend.extract_pose_from_video(
                video_path,
                self.model,
                num_kpts=33,
                timestamp_base_ms=self._mp_next_ts_ms,  # <<< NEW
            )
            self._mp_next_ts_ms = next_ts            # <<< NEW
            return poses, meta

        raise RuntimeError("Unreachable")


def iter_videos(root: str):
    """
    Keep one canonical video iterator.
    """
    return yolo_backend.iter_videos(root)


def extract_pose_from_video(
    video_path: str,
    extractor: PoseExtractor,
    *args: Any,
    **kwargs: Any,
):
    """
    Stable API for main.py. Additional args/kwargs are ignored intentionally
    to prevent backend-specific leakage into pipeline code.
    """
    return extractor.extract_pose_from_video(video_path)
