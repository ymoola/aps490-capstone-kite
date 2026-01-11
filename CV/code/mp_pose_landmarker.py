# mp_pose_landmarker.py
from __future__ import annotations

import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def load_model(
    model_path: str,
    *,
    num_poses: int = 1,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    base_options = python.BaseOptions(model_asset_path=abspath(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)

def visualize_direct_outputs(
    video_path: str,
    landmarker,
    *,
    out_path: str,
    draw_on_black: bool = False,
    conf_thr: float = 0.05,
    max_frames: int | None = None,
    preview: bool = True,
) -> None:
    """
    Visualize raw MediaPipe Pose Landmarker outputs directly (no saving to npz, no interp/smooth).
    Writes an MP4 to out_path.

    draw_on_black=True -> black background
    draw_on_black=False -> overlay on original video frames

    conf_thr filters joints/bones by landmark.visibility (if present).
    """
    video_abs = abspath(video_path)
    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_abs}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")

    # MediaPipe 33 edges (same as in your visualizer)
    edges = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (20, 22),
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 11), (0, 12),
    ]

    frame_idx = 0
    last_ts = 0  # we'll keep monotonic locally for this viz run

    window = "MP Direct Output"  # only used if preview=True

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        ts = int(round((frame_idx - 1) * 1000.0 / fps))
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        result = landmarker.detect_for_video(mp_image, ts)

        if draw_on_black:
            canvas = np.zeros_like(frame_bgr)
        else:
            canvas = frame_bgr.copy()

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm_list = result.pose_landmarks[0]

            # convert to pixel coords
            xy = np.zeros((33, 2), dtype=np.float32)
            conf = np.ones((33,), dtype=np.float32)

            K = min(33, len(lm_list))
            for i in range(K):
                lm = lm_list[i]
                xy[i, 0] = float(lm.x) * width
                xy[i, 1] = float(lm.y) * height
                conf[i] = float(getattr(lm, "visibility", 1.0))

            # bones
            for a, b in edges:
                if conf[a] < conf_thr or conf[b] < conf_thr:
                    continue
                ax, ay = xy[a]
                bx, by = xy[b]
                if not np.isfinite([ax, ay, bx, by]).all():
                    continue
                cv2.line(canvas, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 0), 2, cv2.LINE_AA)

            # joints
            for i in range(33):
                if conf[i] < conf_thr:
                    continue
                x, y = xy[i]
                if not np.isfinite([x, y]).all():
                    continue
                cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(canvas, f"{frame_idx}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        writer.write(canvas)

        if preview:
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    writer.release()
    if preview:
        cv2.destroyWindow(window)



def extract_pose_from_video(
    video_path: str,
    landmarker,
    *,
    num_kpts: int = 33,
    timestamp_base_ms: int = 0,   # <<< NEW
) -> tuple[np.ndarray, dict, int]:
    """
    Returns:
      poses: (T, 33, 3) [x_px, y_px, score]
      meta: dict
      next_timestamp_ms: int  (timestamp to use as base for the next video)
    """
    video_abs = abspath(video_path)
    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_abs}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses_list: list[np.ndarray] = []
    frame_idx = 0

    # IMPORTANT: this must be monotonic across the *entire landmarker lifetime*
    last_ts = timestamp_base_ms - 1

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # derive ms from frame index, then offset by timestamp_base_ms
        ts = timestamp_base_ms + int(round((frame_idx - 1) * 1000.0 / fps))

        # extra safety: force strict monotonicity even if rounding duplicates
        if ts <= last_ts:
            ts = last_ts + 1
        last_ts = ts

        result = landmarker.detect_for_video(mp_image, ts)

        out = np.zeros((num_kpts, 3), dtype=np.float32)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lm_list = result.pose_landmarks[0]
            K = min(num_kpts, len(lm_list))
            for i in range(K):
                lm = lm_list[i]
                out[i, 0] = float(lm.x) * width
                out[i, 1] = float(lm.y) * height
                out[i, 2] = float(getattr(lm, "visibility", 1.0))

        poses_list.append(out)

    cap.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames read from video: {video_abs}")

    poses = np.stack(poses_list, axis=0).astype(np.float32, copy=False)

    meta = {
        "backend": "mediapipe_pose_landmarker",
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "num_frames": int(frame_idx),
        "video_path": video_abs,
        "num_kpts": int(poses.shape[1]),
        "landmarks": "pose_landmarker_33",
        "timestamp_base_ms": int(timestamp_base_ms),
        "timestamp_last_ms": int(last_ts),
    }

    # Ensure next video starts strictly after this video ended
    next_timestamp_ms = int(last_ts + 1)
    return poses, meta, next_timestamp_ms
