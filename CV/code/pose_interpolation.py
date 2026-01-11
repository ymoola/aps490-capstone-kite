# pose_interpolation.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

InterpolationMode = Literal["linear", "catmull_rom"]


@dataclass(frozen=True)
class InterpolationConfig:
    """Configuration for pose interpolation / temporal upsampling."""
    scale_factor: int = 4                      # e.g. 4 means 30fps -> ~120fps pose timeline
    mode: InterpolationMode = "linear"         # "linear" or "catmull_rom"
    conf_thr: float = 0.05                     # keypoint scores below this are treated as missing
    clip_to_frame: Optional[Tuple[int, int]] = None  # (width,height) to clip x/y after interpolation


def output_length(num_frames: int, scale_factor: int) -> int:
    """
    If you insert (scale_factor-1) in-betweens between each original pair:
      T_out = (T_in - 1) * scale_factor + 1
    """
    if num_frames <= 0:
        return 0
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")
    if num_frames == 1:
        return 1
    return (num_frames - 1) * int(scale_factor) + 1


def _fill_missing_forward_backward(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    arr: (T,) float
    valid: (T,) bool
    Fill invalid entries by nearest valid (forward fill then backward fill).
    If nothing valid exists, returns all zeros.
    """
    T = arr.shape[0]
    if T == 0:
        return arr.copy()
    if not valid.any():
        return np.zeros_like(arr)

    out = arr.copy()

    # forward fill
    last = None
    for t in range(T):
        if valid[t]:
            last = out[t]
        elif last is not None:
            out[t] = last

    # backward fill
    last = None
    for t in range(T - 1, -1, -1):
        if valid[t]:
            last = out[t]
        elif last is not None:
            out[t] = last

    return out


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, u: float) -> float:
    """
    Standard (uniform) Catmull-Rom spline.
    u in [0,1]
    """
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def _upsample_1d(x: np.ndarray, scale: int, mode: InterpolationMode) -> np.ndarray:
    """
    Upsample a 1D series x of length T by integer scale factor.
    Output length: (T-1)*scale + 1
    """
    x = np.asarray(x, dtype=np.float32)
    T = x.shape[0]
    if T == 0:
        return x.copy()
    if T == 1 or scale == 1:
        return x.copy()

    out_len = (T - 1) * scale + 1
    out = np.empty((out_len,), dtype=np.float32)

    if mode == "linear":
        idx = 0
        for t in range(T - 1):
            a = float(x[t])
            b = float(x[t + 1])
            for k in range(scale):
                u = k / float(scale)
                out[idx] = (1.0 - u) * a + u * b
                idx += 1
        out[idx] = float(x[-1])
        return out

    if mode == "catmull_rom":
        idx = 0
        for t in range(T - 1):
            p0 = float(x[t - 1]) if t - 1 >= 0 else float(x[t])
            p1 = float(x[t])
            p2 = float(x[t + 1])
            p3 = float(x[t + 2]) if (t + 2) < T else float(x[t + 1])

            for k in range(scale):
                u = k / float(scale)
                out[idx] = _catmull_rom(p0, p1, p2, p3, u)
                idx += 1
        out[idx] = float(x[-1])
        return out

    raise ValueError(f"Unknown mode: {mode}")


def interpolate_pose_sequence(
    poses: np.ndarray,
    config: InterpolationConfig = InterpolationConfig(),
) -> np.ndarray:
    """
    Temporal upsampling of single-person pose data.

    Input:
      poses: (T, V, C) where C is 2 (x,y) or 3 (x,y,score)

    Output:
      poses_up: (T_out, V, C) where T_out = (T-1)*scale_factor + 1

    Behavior with confidence:
      - If C==3, points with score < conf_thr are treated as missing.
      - Missing points are filled (nearest valid) before interpolation.
      - Confidence channel is interpolated linearly (even if mode is catmull_rom).
    """
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3:
        raise ValueError(f"poses must be 3D (T,V,C). Got shape {poses.shape}")

    T, V, C = poses.shape
    if C not in (2, 3):
        raise ValueError(f"Expected C=2 or C=3 channels, got C={C}")

    if config.scale_factor < 1 or int(config.scale_factor) != config.scale_factor:
        raise ValueError("scale_factor must be an integer >= 1")
    scale = int(config.scale_factor)

    if T == 0:
        return poses.copy()
    if T == 1 or scale == 1:
        # Nothing to interpolate
        return poses.copy()

    T_out = output_length(T, scale)
    out = np.zeros((T_out, V, C), dtype=np.float32)

    # For each keypoint v, interpolate x and y (and conf if present)
    for v in range(V):
        x = poses[:, v, 0].copy()
        y = poses[:, v, 1].copy()

        if C == 3:
            conf = poses[:, v, 2].copy()
            valid = conf >= float(config.conf_thr)

            x_f = _fill_missing_forward_backward(x, valid)
            y_f = _fill_missing_forward_backward(y, valid)
            c_f = _fill_missing_forward_backward(conf, valid)

            x_u = _upsample_1d(x_f, scale, config.mode)
            y_u = _upsample_1d(y_f, scale, config.mode)
            c_u = _upsample_1d(c_f, scale, "linear")  # safest for confidence

            # If no valid points ever, keep as zeros
            if not valid.any():
                x_u[:] = 0.0
                y_u[:] = 0.0
                c_u[:] = 0.0

            out[:, v, 0] = x_u
            out[:, v, 1] = y_u
            out[:, v, 2] = np.clip(c_u, 0.0, 1.0)

        else:
            # No confidence channel provided: just interpolate x,y directly
            x_u = _upsample_1d(x, scale, config.mode)
            y_u = _upsample_1d(y, scale, config.mode)
            out[:, v, 0] = x_u
            out[:, v, 1] = y_u

    # Optionally clip x/y to frame bounds
    if config.clip_to_frame is not None:
        w, h = config.clip_to_frame
        out[..., 0] = np.clip(out[..., 0], 0.0, max(0.0, float(w - 1)))
        out[..., 1] = np.clip(out[..., 1], 0.0, max(0.0, float(h - 1)))

    return out


def interpolate_between_two_frames(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    num_inbetweens: int,
    mode: InterpolationMode = "linear",
) -> np.ndarray:
    """
    Convenience helper: interpolate pose data between two frames.

    kpts_a, kpts_b: (V,C) with C=2 or 3
    num_inbetweens: how many frames to generate between them
      - 0 returns empty array with shape (0,V,C)
      - 3 returns 3 frames at u = 1/4, 2/4, 3/4

    Returns: (num_inbetweens, V, C)
    """
    a = np.asarray(kpts_a, dtype=np.float32)
    b = np.asarray(kpts_b, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Expected kpts_a and kpts_b to have same shape (V,C). Got {a.shape} and {b.shape}")

    V, C = a.shape
    if C not in (2, 3):
        raise ValueError("C must be 2 or 3")

    if num_inbetweens <= 0:
        return np.zeros((0, V, C), dtype=np.float32)

    out = np.zeros((num_inbetweens, V, C), dtype=np.float32)

    if mode == "linear":
        for i in range(num_inbetweens):
            u = (i + 1) / float(num_inbetweens + 1)
            out[i] = (1.0 - u) * a + u * b
        return out

    if mode == "catmull_rom":
        # Catmull-Rom needs 4 points; for a two-frame interpolation, it degenerates.
        # We approximate using linear here (still returns correct count).
        for i in range(num_inbetweens):
            u = (i + 1) / float(num_inbetweens + 1)
            out[i] = (1.0 - u) * a + u * b
        return out

    raise ValueError(f"Unknown mode: {mode}")
