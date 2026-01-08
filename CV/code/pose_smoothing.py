import numpy as np

class EMAPoseSmoother:
    """
    Exponential Moving Average smoother for pose keypoints.
    Smooths per keypoint over time.
    """

    def __init__(self, alpha: float = 0.7):
        """
        alpha:
            0.0 → infinite smoothing (very laggy)
            1.0 → no smoothing
        Recommended: 0.6–0.8
        """
        self.alpha = alpha
        self.prev_keypoints = None

    def reset(self):
        """Reset state (call when a new video starts)."""
        self.prev_keypoints = None

    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """
        keypoints shape: (num_keypoints, 2) or (num_keypoints, 3)
        Returns smoothed keypoints with same shape.
        """
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            return keypoints

        smoothed = (
            self.alpha * keypoints +
            (1.0 - self.alpha) * self.prev_keypoints
        )

        self.prev_keypoints = smoothed
        return smoothed
