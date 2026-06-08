"""
Thin wrappers that call existing code/* functions with ProjectConfig arguments
instead of module-level constants.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Tuple

import numpy as np

from gui.config import ProjectConfig


def iter_videos(video_root: str):
    """Iterate video files under video_root (delegates to existing code)."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from code.data_population.pose import iter_videos as _iter
    return list(_iter(video_root))


def build_pose_extractor(config: ProjectConfig, gpu_id: int = 0):
    """Build a PoseExtractor from ProjectConfig (bypasses module-level globals)."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from code.data_population.pose import PoseExtractor, PoseBackendConfig

    name = config.pose_backend.lower()
    if name == "yolo":
        device = config.device if config.device != "cuda" else f"cuda:{gpu_id}"
        cfg = PoseBackendConfig(
            name="yolo",
            yolo_model_path=config.yolo_model_path,
            yolo_device=device,
            yolo_batch_size=8,
            yolo_verbose=False,
            yolo_num_kpts=17,
        )
    elif name == "mediapipe":
        mp_model = os.path.join(
            os.path.dirname(config.yolo_model_path),
            "pose_landmarker_heavy.task",
        )
        cfg = PoseBackendConfig(
            name="mediapipe",
            mp_model_path=mp_model,
            mp_num_poses=1,
            mp_min_det_conf=0.5,
            mp_min_presence_conf=0.5,
            mp_min_track_conf=0.5,
        )
    elif name == "openpose":
        project_root_str = str(Path(__file__).resolve().parents[2])
        cfg = PoseBackendConfig(
            name="openpose",
            op_exe_path=os.path.join(project_root_str, "OpenPose", "bin", "OpenPoseDemo.exe"),
            op_model_folder=os.path.join(project_root_str, "OpenPose", "models"),
            op_model_pose="BODY_25",
            op_number_people_max=1,
        )
    else:
        raise ValueError(f"Unknown pose backend: {name!r}")

    return PoseExtractor(cfg)


def build_interp_config(config: ProjectConfig):
    """Build InterpolationConfig from ProjectConfig."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from code.preprocessing.pose_interpolation import InterpolationConfig

    return InterpolationConfig(
        scale_factor=config.fps_scale,
        mode=config.interp_mode,
        conf_thr=config.conf_thr,
        frame_min_kpts=8,
        frame_min_frac=0.0,
        clip_to_frame=None,
    )


def build_smoother(config: ProjectConfig):
    """Build EMAPoseSmoother from ProjectConfig."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from code.preprocessing.pose_smoothing import EMAPoseSmoother, SmoothingConfig

    return EMAPoseSmoother(SmoothingConfig(
        alpha=config.ema_alpha,
        conf_thr=config.conf_thr,
        smooth_conf=False,
        missing_policy="hold",
        clip_to_frame=None,
    ))


def mirrored_output_path(
    video_abs: str, data_root_abs: str, out_root_abs: str, tag: str,
) -> str:
    """Mirror video path structure into output directory."""
    rel_path = os.path.relpath(video_abs, data_root_abs)
    rel_no_ext, _ = os.path.splitext(rel_path)
    out_path = os.path.join(out_root_abs, rel_no_ext + f"_{tag}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def stage_tag(do_interp: bool, do_smooth: bool) -> str:
    if do_interp and do_smooth:
        return "raw_interp_smooth"
    if do_interp:
        return "raw_interp"
    if do_smooth:
        return "raw_smooth"
    return "raw"
