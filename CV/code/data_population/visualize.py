# visualize.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import cv2


# =============================
# CONFIG (edit these)
# =============================
NPZ_PATH = r"D:\temp\idapt797_sub349_DP_13_12-13-01_openpose_raw.npz"

FPS = 30
CONF_THR = 0.05
PERSON_INDEX = 0

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
WINDOW_NAME = "Pose Viewer"

DRAW_ON_BLACK = True
LABEL_KPT_INDICES = False  # useful when debugging the “one point goes flying” issue

SAVE_DIR = Path(r"D:\Downloads")
SAVE_SUFFIX = "_pose_wa.mp4"
TMP_SUFFIX = "_tmp.mp4"


# -----------------------------
# Skeleton edge sets
# -----------------------------
COCO17_EDGES: List[tuple[int, int]] = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
]

# OpenPose BODY_25 (K=25)
BODY25_EDGES: List[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20), (20, 21),
    (11, 22), (22, 23), (23, 24),
]

# MediaPipe Pose Landmarker (K=33)
MEDIAPIPE33_EDGES: List[tuple[int, int]] = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (19, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (20, 22),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 11), (0, 12),
]


def pick_edges(K: int) -> List[tuple[int, int]]:
    if K == 17:
        return COCO17_EDGES
    if K == 25:
        return BODY25_EDGES
    if K == 33:
        return MEDIAPIPE33_EDGES
    return []


def guess_format(K: int) -> str:
    return {17: "coco17/yolo", 25: "openpose_body25", 33: "mediapipe33"}.get(K, f"unknown(K={K})")


def reencode_for_whatsapp(src: str, dst: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", src,
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            dst,
        ],
        check=True,
    )


def load_npz(npz_path: Path) -> Tuple[np.ndarray, dict]:
    data = np.load(npz_path, allow_pickle=True)
    if "poses" not in data:
        raise ValueError(f"{npz_path} missing 'poses' array. Keys: {list(data.keys())}")
    poses = data["poses"]

    meta = {}
    if "meta_json" in data:
        try:
            meta = json.loads(str(data["meta_json"]))
        except Exception:
            meta = {}

    return poses, meta


def to_TNKC(poses: np.ndarray) -> np.ndarray:
    arr = np.asarray(poses)
    if arr.ndim == 3:
        return arr[:, None, :, :]  # (T,1,K,C)
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Unsupported pose shape: {arr.shape}")


def normalize_if_needed(xy: np.ndarray, W: int, H: int) -> np.ndarray:
    mx = float(np.nanmax(xy[..., 0]))
    my = float(np.nanmax(xy[..., 1]))
    # If coords look normalized, scale.
    if mx <= 2.0 and my <= 2.0:
        xy = xy.copy()
        xy[..., 0] *= W
        xy[..., 1] *= H
    return xy


def draw_frame(
    t: int,
    poses: np.ndarray,
    W: int,
    H: int,
    conf_thr: float,
    person_index: int,
    edges: List[tuple[int, int]],
    draw_on_black: bool,
    label_indices: bool,
) -> np.ndarray:
    bg = 0 if draw_on_black else 255
    img = np.full((H, W, 3), bg, dtype=np.uint8)

    T, N, K, C = poses.shape
    if t < 0 or t >= T:
        return img
    if person_index < 0 or person_index >= N:
        raise ValueError(f"PERSON_INDEX={person_index} out of range for N={N}")

    pts = poses[t, person_index]
    xy = normalize_if_needed(pts[:, :2].astype(np.float32), W, H)
    conf = pts[:, 2].astype(np.float32) if C >= 3 else None

    # bones
    for a, b in edges:
        if a >= K or b >= K:
            continue
        if conf is not None and (conf[a] < conf_thr or conf[b] < conf_thr):
            continue

        ax, ay = xy[a]
        bx, by = xy[b]
        if not np.isfinite([ax, ay, bx, by]).all():
            continue

        cv2.line(img, (int(ax), int(ay)), (int(bx), int(by)), (255, 255, 255), 2, cv2.LINE_AA)

    # joints (+ optional indices)
    for k in range(K):
        if conf is not None and conf[k] < conf_thr:
            continue
        x, y = xy[k]
        if not np.isfinite([x, y]).all():
            continue

        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
        if label_indices:
            cv2.putText(
                img,
                str(k),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255) if draw_on_black else (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        img,
        f"{t+1}/{T}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255) if draw_on_black else (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return img


def render_full_video_to_tmp(poses: np.ndarray, tmp_path: Path, edges: List[tuple[int, int]]) -> None:
    T, _, _, _ = poses.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, FPS, (CANVAS_WIDTH, CANVAS_HEIGHT), isColor=True)
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open. Try output .avi or ensure codecs installed.")

    for t in range(T):
        frame = draw_frame(
            t,
            poses,
            CANVAS_WIDTH,
            CANVAS_HEIGHT,
            CONF_THR,
            PERSON_INDEX,
            edges,
            DRAW_ON_BLACK,
            LABEL_KPT_INDICES,
        )
        writer.write(frame)

    writer.release()


def play_video_file(video_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path} for playback.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)

    delay_ms = max(1, int(1000 / FPS))
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()


def main():
    npz_path = Path(NPZ_PATH)
    poses_raw, meta = load_npz(npz_path)
    poses = to_TNKC(poses_raw)

    T, N, K, C = poses.shape
    fmt = guess_format(K)
    edges = pick_edges(K)

    print(f"[info] poses shape: T={T}, N={N}, K={K}, C={C} format={fmt}")
    if meta:
        backend = meta.get("backend", "unknown")
        stage = (meta.get("poses_saved_stage") or meta.get("pipeline", {}))
        print(f"[info] meta backend={backend} stage={stage}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    final_path = SAVE_DIR / (npz_path.stem + SAVE_SUFFIX)
    tmp_path = SAVE_DIR / (npz_path.stem + TMP_SUFFIX)

    print(f"[info] rendering to temp: {tmp_path}")
    render_full_video_to_tmp(poses, tmp_path, edges)

    print(f"[info] re-encoding to WhatsApp-safe: {final_path}")
    reencode_for_whatsapp(str(tmp_path), str(final_path))

    try:
        tmp_path.unlink()
    except Exception:
        pass

    print(f"[done] wrote: {final_path}")
    print("[info] now playing the converted file")
    play_video_file(final_path)


if __name__ == "__main__":
    main()
