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

# COCO 17 skeleton edges (Ultralytics pose models are usually COCO-17)
# You can tweak if your model uses a different kpt order
COCO17_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


import re as _re
_SUB_DIR_RE = _re.compile(r'^sub\d+$', _re.IGNORECASE)


def iter_videos(root: str):
    for r, _, files in os.walk(root):
        if not _SUB_DIR_RE.match(os.path.basename(r)):
            continue
        for f in files:
            if os.path.splitext(f)[1] in VIDEO_EXTS:
                yield os.path.join(r, f)


def load_model(model_path: str) -> YOLO:
    return YOLO(abspath(model_path))


def _skeleton_center(xy: np.ndarray, conf: np.ndarray | None, conf_thr: float) -> np.ndarray:
    """
    xy: (V,2)
    conf: (V,) or None
    returns (2,) center in pixel coords
    """
    if conf is None:
        return np.nanmean(xy, axis=0)

    m = conf >= conf_thr
    if m.sum() < 2:
        return np.nanmean(xy, axis=0)
    return xy[m].mean(axis=0)


def _skeleton_area(xy: np.ndarray, conf: np.ndarray | None, conf_thr: float) -> float:
    """
    Approximates "closeness" via area of bounding box around confident joints.
    """
    if conf is not None:
        m = conf >= conf_thr
        if m.sum() < 2:
            return 0.0
        pts = xy[m]
    else:
        pts = xy

    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
    if not np.isfinite([x1, y1, x2, y2]).all():
        return 0.0
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def pick_single_person_keypoints(
    result0,
    *,
    num_kpts: int = 17,
    prev_center: np.ndarray | None = None,
    conf_thr: float = 0.05,
    width: int | None = None,
    height: int | None = None,
    w_conf: float = 0.25,
    w_size: float = 0.55,
    w_track: float = 0.20,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Returns:
      out_kpts: (V,3) [x,y,score] pixel coords (one person)
      new_center: (2,) center of chosen skeleton (for continuity)

    Goal:
      - Prefer the closest person (largest skeleton area).
      - Prefer stable tracking: stay near previous center.
      - Use mean confidence as a mild tie-breaker.

    score = w_size * size_norm + w_track * track_score + w_conf * mean_conf_norm
      track_score is higher when candidate is closer to prev_center.
    """
    if result0.keypoints is None:
        return np.zeros((num_kpts, 3), dtype=np.float32), prev_center

    kpts = result0.keypoints
    if kpts.xy is None or len(kpts) == 0:
        return np.zeros((num_kpts, 3), dtype=np.float32), prev_center

    xy = kpts.xy.cpu().numpy()  # (M, V, 2)
    conf = None
    if getattr(kpts, "conf", None) is not None:
        conf = kpts.conf.cpu().numpy()  # (M, V)

    M, V = xy.shape[0], xy.shape[1]
    if V != num_kpts:
        num_kpts = V

    # Fallback if width/height not provided
    if width is None or height is None:
        # estimate from points (best-effort)
        finite = np.isfinite(xy)
        if finite.any():
            max_x = np.nanmax(xy[..., 0])
            max_y = np.nanmax(xy[..., 1])
            width = int(max(1, max_x))
            height = int(max(1, max_y))
        else:
            width, height = 1920, 1080

    # If no conf available, fall back to largest skeleton area
    if conf is None:
        areas = np.array([_skeleton_area(xy[i], None, conf_thr) for i in range(M)], dtype=np.float32)
        chosen = int(np.argmax(areas)) if M > 0 else 0

        out = np.zeros((num_kpts, 3), dtype=np.float32)
        out[:, 0:2] = xy[chosen]
        out[:, 2] = 1.0
        center = _skeleton_center(xy[chosen], None, conf_thr)
        return out, center

    # Mean confidence per person
    mean_conf = conf.mean(axis=1).astype(np.float32)  # (M,)
    # Normalize mean_conf to [0,1] within the frame to avoid scale issues
    if mean_conf.max() > mean_conf.min():
        mean_conf_norm = (mean_conf - mean_conf.min()) / (mean_conf.max() - mean_conf.min() + 1e-6)
    else:
        mean_conf_norm = np.zeros_like(mean_conf)

    centers = np.stack([_skeleton_center(xy[i], conf[i], conf_thr) for i in range(M)], axis=0)  # (M,2)
    areas = np.array([_skeleton_area(xy[i], conf[i], conf_thr) for i in range(M)], dtype=np.float32)  # (M,)

    # Normalize size to [0,1] within frame
    if areas.max() > 0:
        size_norm = areas / (areas.max() + 1e-6)
    else:
        size_norm = np.zeros_like(areas)

    # Tracking continuity: higher score when closer to prev_center
    if prev_center is None or (not np.isfinite(prev_center).all()):
        track_score = np.zeros((M,), dtype=np.float32)
    else:
        frame_diag = float(np.sqrt(width * width + height * height))
        dist = np.linalg.norm(centers - prev_center[None, :], axis=1).astype(np.float32)
        dist_norm = np.clip(dist / (frame_diag + 1e-6), 0.0, 1.0)
        track_score = 1.0 - dist_norm  # closer => higher

    score = (w_size * size_norm) + (w_track * track_score) + (w_conf * mean_conf_norm)
    chosen = int(np.argmax(score))

    out = np.zeros((num_kpts, 3), dtype=np.float32)
    out[:, 0] = xy[chosen, :, 0]
    out[:, 1] = xy[chosen, :, 1]
    out[:, 2] = conf[chosen, :]

    new_center = centers[chosen]
    return out, new_center


def draw_kpts_and_skeleton(
    img: np.ndarray,
    kpt: np.ndarray,
    *,
    edges=COCO17_EDGES,
    conf_thr: float = 0.10,
) -> np.ndarray:
    """
    kpt: (V,3) [x,y,conf]
    """
    out = img

    # points
    for i in range(kpt.shape[0]):
        x, y, c = kpt[i]
        if c >= conf_thr and np.isfinite([x, y]).all():
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)

    # edges
    for a, b in edges:
        if a >= kpt.shape[0] or b >= kpt.shape[0]:
            continue
        xa, ya, ca = kpt[a]
        xb, yb, cb = kpt[b]
        if ca >= conf_thr and cb >= conf_thr and np.isfinite([xa, ya, xb, yb]).all():
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

    return out


def extract_pose_from_video(
    video_path: str,
    model: YOLO,
    *,
    num_kpts: int = 17,
    device: int | str | None = None,
    batch_size: int = 8,
    verbose: bool = False,
    conf_thr: float = 0.05,
    w_conf: float = 0.25,
    w_size: float = 0.55,
    w_track: float = 0.20,
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

    prev_center: np.ndarray | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        batch_frames.append(frame)
        frame_idx += 1

        if len(batch_frames) >= batch_size:
            results = model(batch_frames, device=device, verbose=verbose)
            for r0 in results:
                kpt, prev_center = pick_single_person_keypoints(
                    r0,
                    num_kpts=num_kpts,
                    prev_center=prev_center,
                    conf_thr=conf_thr,
                    width=width,
                    height=height,
                    w_conf=w_conf,
                    w_size=w_size,
                    w_track=w_track,
                )
                poses_list.append(kpt)
            batch_frames.clear()

    if batch_frames:
        results = model(batch_frames, device=device, verbose=verbose)
        for r0 in results:
            kpt, prev_center = pick_single_person_keypoints(
                r0,
                num_kpts=num_kpts,
                prev_center=prev_center,
                conf_thr=conf_thr,
                width=width,
                height=height,
                w_conf=w_conf,
                w_size=w_size,
                w_track=w_track,
            )
            poses_list.append(kpt)

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
        "selection_policy": {
            "type": "size_track_conf",
            "conf_thr": float(conf_thr),
            "w_conf": float(w_conf),
            "w_size": float(w_size),
            "w_track": float(w_track),
        },
    }
    return poses, meta


if __name__ == "__main__":
    # -------------------------------------------------
    # EDIT THESE VARIABLES
    # -------------------------------------------------
    VIDEO_PATH = r"D:\path\to\your_video.mp4"
    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_PATH = str(_PROJECT_ROOT / "models" / "yolo11x-pose.pt")

    DEVICE = None       # None | "cpu" | "cuda:0" | 0
    IMGSZ = 640
    CONF = 0.25

    SHOW_FPS = True
    WINDOW_NAME = "YOLO Pose (Chosen Person)"
    WAIT_MS = 1
    START_FRAME = 0
    MAX_FRAMES = None
    # -------------------------------------------------

    video_abs = abspath(VIDEO_PATH)
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_abs)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_abs}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if START_FRAME and START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_FRAME))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_i = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if START_FRAME else 0
    shown = 0
    prev_center: np.ndarray | None = None

    while True:
        if MAX_FRAMES is not None and shown >= int(MAX_FRAMES):
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            device=DEVICE,
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False,
        )
        r0 = results[0]

        kpt, prev_center = pick_single_person_keypoints(
            r0,
            prev_center=prev_center,
            conf_thr=0.05,
            width=width,
            height=height,
            w_conf=0.25,
            w_size=0.55,
            w_track=0.20,
        )

        overlay = frame.copy()
        overlay = draw_kpts_and_skeleton(overlay, kpt, conf_thr=0.10)

        if SHOW_FPS:
            cv2.putText(
                overlay,
                f"{fps:.1f} fps  |  frame {frame_i}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WINDOW_NAME, overlay)

        key = cv2.waitKey(int(WAIT_MS)) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 in (ord("q"), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit
                if k2 == ord(" "):
                    break

        frame_i += 1
        shown += 1

    cap.release()
    cv2.destroyAllWindows()
