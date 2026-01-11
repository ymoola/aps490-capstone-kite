# pose_smoothing.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


MissingPolicy = Literal["hold", "zero"]
# hold: if confidence is low, keep previous smoothed x/y (no update)
# zero: if confidence is low, force x/y to 0 (generally not recommended, but sometimes useful)


@dataclass(frozen=True)
class SmoothingConfig:
    """
    Configuration for EMA pose smoothing.

    alpha:
      0.0 -> very heavy smoothing (laggy)
      1.0 -> no smoothing
    conf_thr:
      keypoints with confidence < conf_thr are treated as missing.
    smooth_conf:
      if True, EMA-smooth the confidence channel too (usually False).
    missing_policy:
      - "hold": keep previous smoothed x/y when missing
      - "zero": set missing x/y to 0 (and conf to 0) in the output
    """
    alpha: float = 0.7
    conf_thr: float = 0.05
    smooth_conf: bool = False
    missing_policy: MissingPolicy = "hold"
    clip_to_frame: Optional[Tuple[int, int]] = None  # (width,height) clip x/y after smoothing


class EMAPoseSmoother:
    """
    EMA smoother for single-person pose sequences.

    Supports:
      - frame-by-frame smoothing via smooth_frame((V,C))
      - sequence smoothing via smooth_sequence((T,V,C))

    Expected input shape:
      - (V,2) or (V,3) for a frame
      - (T,V,2) or (T,V,3) for a sequence
    """

    def __init__(self, config: SmoothingConfig = SmoothingConfig()):
        if not (0.0 <= config.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.config = config
        self._prev: Optional[np.ndarray] = None  # (V,C)

    def reset(self):
        """Reset state (call when a new video starts)."""
        self._prev = None

    def smooth_frame(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Smooth one frame.

        keypoints: (V,2) or (V,3)
        returns: (V,2) or (V,3)
        """
        kp = np.asarray(keypoints, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[1] not in (2, 3):
            raise ValueError(f"Expected (V,2) or (V,3). Got {kp.shape}")

        V, C = kp.shape

        # Initialize state on first frame
        if self._prev is None:
            self._prev = kp.copy()
            if self.config.clip_to_frame is not None:
                self._prev = _clip_xy(self._prev, self.config.clip_to_frame)
            return kp

        prev = self._prev
        alpha = float(self.config.alpha)

        if C == 2:
            # No confidence: always smooth x/y
            sm = alpha * kp + (1.0 - alpha) * prev
            self._prev = sm
            if self.config.clip_to_frame is not None:
                self._prev = _clip_xy(self._prev, self.config.clip_to_frame)
            return self._prev

        # C == 3 : (x,y,conf)
        conf = kp[:, 2]
        valid = conf >= float(self.config.conf_thr)

        sm = prev.copy()

        # Smooth x/y only for valid keypoints
        # For invalid, apply missing policy
        if valid.any():
            sm_xy = alpha * kp[valid, 0:2] + (1.0 - alpha) * prev[valid, 0:2]
            sm[valid, 0:2] = sm_xy

        if not valid.all():
            if self.config.missing_policy == "hold":
                # keep previous x/y as already in sm
                pass
            elif self.config.missing_policy == "zero":
                sm[~valid, 0:2] = 0.0
            else:
                raise ValueError(f"Unknown missing_policy: {self.config.missing_policy}")

        # Confidence handling
        if self.config.smooth_conf:
            sm_conf = alpha * kp[:, 2] + (1.0 - alpha) * prev[:, 2]
            sm[:, 2] = np.clip(sm_conf, 0.0, 1.0)
        else:
            sm[:, 2] = kp[:, 2]  # keep raw confidence

        # If missing_policy == zero, also zero confidence for missing points
        if self.config.missing_policy == "zero":
            sm[~valid, 2] = 0.0

        if self.config.clip_to_frame is not None:
            sm = _clip_xy(sm, self.config.clip_to_frame)

        self._prev = sm
        return sm

    def smooth_sequence(self, poses: np.ndarray) -> np.ndarray:
        """
        Smooth an entire pose sequence.

        poses: (T,V,2) or (T,V,3)
        returns: same shape
        """
        arr = np.asarray(poses, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] not in (2, 3):
            raise ValueError(f"Expected (T,V,2) or (T,V,3). Got {arr.shape}")

        T = arr.shape[0]
        out = np.empty_like(arr)

        self.reset()
        for t in range(T):
            out[t] = self.smooth_frame(arr[t])

        return out


def _clip_xy(frame: np.ndarray, clip_to_frame: Tuple[int, int]) -> np.ndarray:
    """
    Clip x/y channels of a (V,2) or (V,3) array to frame bounds.
    """
    w, h = clip_to_frame
    out = frame.copy()
    out[:, 0] = np.clip(out[:, 0], 0.0, max(0.0, float(w - 1)))
    out[:, 1] = np.clip(out[:, 1], 0.0, max(0.0, float(h - 1)))
    return out
