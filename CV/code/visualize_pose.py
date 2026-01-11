# visualize_pose.py
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2


# =============================
# CONFIG
# =============================
NPZ_PATH = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\out\2025-02-06\sub354\idapt7800_sub354_DP_5_11-26-23.npz"

FPS = 120.0
CONF_THR = 0.05
PERSON_INDEX = 0

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
WINDOW_NAME = "Pose Viewer"

SAVE_DIR = Path(r"D:\Downloads")
SAVE_SUFFIX = "_pose_wa.mp4"   # final output only
TMP_SUFFIX = "_tmp.mp4"        # intermediate (deleted)


# -----------------------------
# Skeleton edges
# -----------------------------

# COCO / YOLO 17-keypoint skeleton (Ultralytics pose default)
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

# MediaPipe Pose Landmarker (33 landmarks) indices (common map):
# 0 nose
# 1-4 eyes/ears
# 11/12 shoulders, 13/14 elbows, 15/16 wrists
# 23/24 hips, 25/26 knees, 27/28 ankles
# 29/30 heels, 31/32 foot index (toes)
MEDIAPIPE33_EDGES: List[tuple[int, int]] = [
    # torso
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),

    # left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19), (19, 21),

    # right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20), (20, 22),

    # left leg + foot
    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (27, 31),

    # right leg + foot
    (24, 26), (26, 28),
    (28, 30), (30, 32),
    (28, 32),

    # head-ish
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 11), (0, 12),
]


def pick_edges_for_K(K: int) -> List[tuple[int, int]]:
    if K == 33:
        return MEDIAPIPE33_EDGES
    if K == 17:
        return COCO17_EDGES
    # fallback: no bones, only points
    return []


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
        return arr  # (T,N,K,C)
    raise ValueError(f"Unsupported pose shape: {arr.shape}")


def normalize_if_needed(xy: np.ndarray, W: int, H: int) -> np.ndarray:
    mx = float(np.nanmax(xy[..., 0]))
    my = float(np.nanmax(xy[..., 1]))
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
) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)

    T, N, K, C = poses.shape
    if t < 0 or t >= T:
        return img
    if person_index < 0 or person_index >= N:
        raise ValueError(f"PERSON_INDEX={person_index} out of range for N={N}")

    edges = pick_edges_for_K(K)

    pts = poses[t, person_index]  # (K,C)
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

    # joints
    for k in range(K):
        if conf is not None and conf[k] < conf_thr:
            continue
        x, y = xy[k]
        if not np.isfinite([x, y]).all():
            continue
        cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.putText(
        img,
        f"{t+1}/{T}  K={K}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def render_full_video_to_tmp(poses: np.ndarray, tmp_path: Path) -> None:
    T, _, _, _ = poses.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(tmp_path),
        fourcc,
        FPS,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        isColor=True,
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open. Try .avi or install codecs.")

    for t in range(T):
        frame = draw_frame(t, poses, CANVAS_WIDTH, CANVAS_HEIGHT, CONF_THR, PERSON_INDEX)
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
    print(f"[info] poses shape: T={T}, N={N}, K={K}, C={C}")
    if meta:
        print(f"[info] backend: {meta.get('backend', 'unknown')}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    final_path = SAVE_DIR / (npz_path.stem + SAVE_SUFFIX)
    tmp_path = SAVE_DIR / (npz_path.stem + TMP_SUFFIX)

    print(f"[info] rendering full sequence to temp: {tmp_path}")
    render_full_video_to_tmp(poses, tmp_path)

    print(f"[info] converting to WhatsApp-safe MP4: {final_path}")
    reencode_for_whatsapp(str(tmp_path), str(final_path))

    try:
        tmp_path.unlink()
    except Exception:
        pass

    print(f"[done] wrote: {final_path}")
    print("[info] now playing the converted file (this is what you'll upload)")
    play_video_file(final_path)


if __name__ == "__main__":
    main()
