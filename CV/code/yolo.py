# yolo.py
from __future__ import annotations

import os
import cv2
import numpy as np
from ultralytics import YOLO


VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv",
    ".MP4", ".MOV", ".M4V", ".AVI", ".MKV"
}


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def iter_videos(root: str):
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1] in VIDEO_EXTS:
                yield os.path.join(r, f)


def load_model(model_path: str) -> YOLO:
    return YOLO(abspath(model_path))


def pick_single_person_keypoints(result0, num_kpts: int = 17) -> np.ndarray:
    """
    Returns exactly one skeleton as (V,3) [x,y,score] in pixel coordinates.
    If no person detected, returns zeros (V,3).
    Chooses the person with the highest mean keypoint confidence.
    """
    if result0.keypoints is None:
        return np.zeros((num_kpts, 3), dtype=np.float32)

    kpts = result0.keypoints
    if kpts.xy is None or len(kpts) == 0:
        return np.zeros((num_kpts, 3), dtype=np.float32)

    xy = kpts.xy.cpu().numpy()  # (M, V, 2)
    conf = None
    if getattr(kpts, "conf", None) is not None:
        conf = kpts.conf.cpu().numpy()  # (M, V)

    V = xy.shape[1]
    if V != num_kpts:
        num_kpts = V

    if conf is None:
        chosen = 0
        out = np.zeros((num_kpts, 3), dtype=np.float32)
        out[:, 0:2] = xy[chosen]
        out[:, 2] = 1.0
        return out

    mean_conf = conf.mean(axis=1)
    chosen = int(np.argmax(mean_conf))

    out = np.zeros((num_kpts, 3), dtype=np.float32)
    out[:, 0] = xy[chosen, :, 0]
    out[:, 1] = xy[chosen, :, 1]
    out[:, 2] = conf[chosen, :]
    return out


def extract_pose_from_video(
    video_path: str,
    model: YOLO,
    *,
    num_kpts: int = 17,
    device: int | str | None = None,
    batch_size: int = 8,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Returns:
      poses: (T, V, 3) [x,y,score] pixel coords
      meta: dict with fps, width, height, num_frames, video_path, backend
    """
    video_abs = abspath(video_path)

    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_abs}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses_list: list[np.ndarray] = []
    frame_idx = 0
    batch_frames: list[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        batch_frames.append(frame)
        frame_idx += 1

        if len(batch_frames) >= batch_size:
            results = model(batch_frames, device=device, verbose=verbose)
            for r0 in results:
                poses_list.append(pick_single_person_keypoints(r0, num_kpts=num_kpts))
            batch_frames.clear()

    if batch_frames:
        results = model(batch_frames, device=device, verbose=verbose)
        for r0 in results:
            poses_list.append(pick_single_person_keypoints(r0, num_kpts=num_kpts))

    cap.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames read from video: {video_abs}")

    poses = np.stack(poses_list, axis=0).astype(np.float32, copy=False)

    meta = {
        "backend": "yolo",
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "num_frames": int(frame_idx),
        "video_path": video_abs,
        "num_kpts": int(poses.shape[1]),
    }
    return poses, meta
