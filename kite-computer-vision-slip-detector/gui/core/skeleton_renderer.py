"""Wraps visualize.py draw_frame() for GUI skeleton previews."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from code.data_population.visualize import (
    load_npz, to_TNKC, pick_edges, draw_frame, render_full_video_to_tmp,
)


def render_skeleton_preview(
    npz_path: str,
    frame_index: Optional[int] = None,
    width: int = 640,
    height: int = 480,
    conf_thr: float = 0.05,
) -> Optional[QPixmap]:
    """
    Render a single skeleton frame from a pose NPZ and return as QPixmap.
    If frame_index is None, uses the midpoint frame.
    """
    try:
        poses_raw, meta = load_npz(Path(npz_path))
        poses = to_TNKC(poses_raw)
        T, N, K, C = poses.shape
        edges = pick_edges(K)

        if frame_index is None:
            frame_index = T // 2

        frame_index = max(0, min(frame_index, T - 1))

        img = draw_frame(
            t=frame_index,
            poses=poses,
            W=width,
            H=height,
            conf_thr=conf_thr,
            person_index=0,
            edges=edges,
            draw_on_black=True,
            label_indices=False,
        )

        # BGR to RGB for Qt
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    except Exception:
        return None


def get_npz_info(npz_path: str) -> dict:
    """Extract basic info from a pose NPZ without full processing."""
    try:
        poses_raw, meta = load_npz(Path(npz_path))
        poses = to_TNKC(poses_raw)
        T, N, K, C = poses.shape
        return {
            "T": T,
            "N": N,
            "K": K,
            "C": C,
            "meta": meta,
        }
    except Exception as e:
        return {"error": str(e)}


def render_full_skeleton_video(
    npz_path: str,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> str:
    """Render full skeleton overlay video. Returns output path."""
    poses_raw, meta = load_npz(Path(npz_path))
    poses = to_TNKC(poses_raw)
    edges = pick_edges(poses.shape[2])

    render_full_video_to_tmp(poses, Path(output_path), edges)
    return output_path
