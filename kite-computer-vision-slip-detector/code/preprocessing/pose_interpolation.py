# pose_interpolation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

InterpolationMode = Literal["linear", "catmull_rom"]


@dataclass(frozen=True)
class InterpolationConfig:
    """
    Temporal upsampling config.

    Key behavior (what you asked for):
      - We ONLY interpolate between two consecutive original frames (t, t+1)
        if BOTH endpoint frames are "valid".
      - If either endpoint is invalid -> ALL in-betweens are blank (zeros).
      - Additionally, we only interpolate an individual keypoint v if that keypoint
        is confident in BOTH endpoints; otherwise that keypoint is blank for in-betweens.

    "Blank" means:
      x=0, y=0, conf=0  (so downstream drawing logic won't render anything)
    """
    scale_factor: int = 4
    mode: InterpolationMode = "linear"     # you can keep "linear" for now
    conf_thr: float = 0.05

    # frame validity gate (prevents interpolating when person is basically gone)
    frame_min_kpts: int = 8                # BODY_25: try 8-12; COCO17: try 5-8
    frame_min_frac: float = 0.0            # optional; 0 disables

    clip_to_frame: Optional[Tuple[int, int]] = None  # (width,height)


def output_length(num_frames: int, scale_factor: int) -> int:
    if num_frames <= 0:
        return 0
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")
    if num_frames == 1:
        return 1
    return (num_frames - 1) * int(scale_factor) + 1


def _catmull_rom(p0: float, p1: float, p2: float, p3: float, u: float) -> float:
    u2 = u * u
    u3 = u2 * u
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
    )


def _compute_frame_ok(conf_all: np.ndarray, conf_thr: float, min_kpts: int, min_frac: float) -> np.ndarray:
    """
    conf_all: (T,V)
    frame_ok[t] True if frame has enough confident keypoints.
    """
    good = conf_all >= float(conf_thr)
    good_count = good.sum(axis=1)
    ok = good_count >= int(min_kpts)
    if min_frac and float(min_frac) > 0:
        ok = ok & ((good_count / max(1, conf_all.shape[1])) >= float(min_frac))
    return ok


def interpolate_pose_sequence(
    poses: np.ndarray,
    config: InterpolationConfig = InterpolationConfig(),
) -> np.ndarray:
    """
    Input:
      poses: (T,V,C) where C is 2 or 3.
        If C==3: channel order [x,y,conf].

    Output:
      (T_out,V,C) where T_out = (T-1)*scale + 1

    Pairwise interpolation rule:
      - For each interval t->t+1 we generate (scale-1) in-betweens.
      - If either endpoint frame is invalid => in-betweens are blank (zeros).
      - If endpoints are valid but a specific keypoint is not confident in both endpoints,
        that keypoint is blank for the in-betweens.
      - Original frames that are invalid (frame_ok False) are also blanked.
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
        # Still apply blanking on invalid frames if C==3
        out = poses.copy()
        if C == 3:
            frame_ok = _compute_frame_ok(out[:, :, 2], config.conf_thr, config.frame_min_kpts, config.frame_min_frac)
            for t in range(T):
                if not frame_ok[t]:
                    out[t, :, :] = 0.0
        return out

    T_out = output_length(T, scale)
    out = np.zeros((T_out, V, C), dtype=np.float32)

    # Determine which original frames are "valid" (only if C==3)
    if C == 3:
        conf_all = poses[:, :, 2]  # (T,V)
        frame_ok = _compute_frame_ok(conf_all, config.conf_thr, config.frame_min_kpts, config.frame_min_frac)
    else:
        frame_ok = np.ones((T,), dtype=bool)

    # Helper to write an original frame (blank it if invalid)
    def write_original_frame(t: int, out_idx: int) -> None:
        if C == 3 and not frame_ok[t]:
            out[out_idx, :, :] = 0.0
        else:
            out[out_idx, :, :] = poses[t, :, :]

    # Fill timeline interval-by-interval
    for t in range(T - 1):
        base = t * scale

        # place original frame t
        write_original_frame(t, base)

        # in-betweens
        if scale > 1:
            # If either endpoint frame invalid -> blank the whole in-between block
            if not (frame_ok[t] and frame_ok[t + 1]):
                # already zeros by default, but keep explicit
                out[base + 1: base + scale, :, :] = 0.0
            else:
                # endpoint frames valid: interpolate per keypoint if kp confident in both endpoints
                a = poses[t]       # (V,C)
                b = poses[t + 1]

                # per-keypoint endpoint validity (only for C==3)
                if C == 3:
                    kp_ok = (a[:, 2] >= config.conf_thr) & (b[:, 2] >= config.conf_thr)
                else:
                    kp_ok = np.ones((V,), dtype=bool)

                for k in range(1, scale):
                    u = k / float(scale)  # u in (0,1)
                    out_idx = base + k

                    if C == 2:
                        out[out_idx, :, 0:2] = (1.0 - u) * a[:, 0:2] + u * b[:, 0:2]
                    else:
                        # start as blank
                        out[out_idx, :, :] = 0.0

                        # only fill joints that are good at BOTH endpoints
                        good = kp_ok
                        if good.any():
                            if config.mode == "linear":
                                out[out_idx, good, 0] = (1.0 - u) * a[good, 0] + u * b[good, 0]
                                out[out_idx, good, 1] = (1.0 - u) * a[good, 1] + u * b[good, 1]
                            elif config.mode == "catmull_rom":
                                # For pairwise Catmull-Rom you need context points; we can approximate using local frames
                                # If you want true CR, you must ensure t-1 and t+2 are also valid frames.
                                # Here we fall back to linear unless the 4-frame window is valid.
                                can_cr = (t - 1 >= 0) and (t + 2 < T) and frame_ok[t - 1] and frame_ok[t + 2]
                                if not can_cr:
                                    out[out_idx, good, 0] = (1.0 - u) * a[good, 0] + u * b[good, 0]
                                    out[out_idx, good, 1] = (1.0 - u) * a[good, 1] + u * b[good, 1]
                                else:
                                    p0 = poses[t - 1]
                                    p1 = poses[t]
                                    p2 = poses[t + 1]
                                    p3 = poses[t + 2]
                                    # only CR-interpolate for joints that are good at all 4 endpoints
                                    good4 = good & (p0[:, 2] >= config.conf_thr) & (p3[:, 2] >= config.conf_thr)
                                    # linear for those that fail 4-point validity
                                    lin = good & ~good4
                                    if lin.any():
                                        out[out_idx, lin, 0] = (1.0 - u) * a[lin, 0] + u * b[lin, 0]
                                        out[out_idx, lin, 1] = (1.0 - u) * a[lin, 1] + u * b[lin, 1]
                                    if good4.any():
                                        # scalar CR per joint
                                        for j in np.where(good4)[0]:
                                            out[out_idx, j, 0] = _catmull_rom(
                                                float(p0[j, 0]), float(p1[j, 0]), float(p2[j, 0]), float(p3[j, 0]), u
                                            )
                                            out[out_idx, j, 1] = _catmull_rom(
                                                float(p0[j, 1]), float(p1[j, 1]), float(p2[j, 1]), float(p3[j, 1]), u
                                            )
                            else:
                                raise ValueError(f"Unknown mode: {config.mode}")

                            # confidence interpolated linearly between endpoints for the good joints
                            out[out_idx, good, 2] = np.clip((1.0 - u) * a[good, 2] + u * b[good, 2], 0.0, 1.0)

    # place last original frame at end
    write_original_frame(T - 1, (T - 1) * scale)

    # Optional clip
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

    IMPORTANT: This helper assumes BOTH endpoints are valid.
    If you want "blank if invalid", do that check outside this function.
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
        # With only two frames, Catmull-Rom degenerates; fall back to linear.
        for i in range(num_inbetweens):
            u = (i + 1) / float(num_inbetweens + 1)
            out[i] = (1.0 - u) * a + u * b
        return out

    raise ValueError(f"Unknown mode: {mode}")
