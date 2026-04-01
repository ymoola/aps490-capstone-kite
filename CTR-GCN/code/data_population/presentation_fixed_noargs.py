# presentation_fixed_noargs.py
from __future__ import annotations

"""
A .py version of CV/code/data_population/presentation.ipynb (no CLI args).

What changed vs the notebook:
- FIX: when pipeline label is "raw", we overlay ALL backends (YOLO / MediaPipe / OpenPose)
  onto the original video frame (previously only YOLO used the raw-video background).
- For non-raw labels (interp/smooth/etc), we keep drawing on black (unless DRAW_ON_BLACK=False).
"""

import sys
import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from dataclasses import asdict

from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = THIS_FILE.parents[2]  # .../CTR-GCN
CODE_ROOT = THIS_FILE.parents[1]      # .../CTR-GCN/code

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

print("Added to PYTHONPATH:", CODE_ROOT)
print("sys.path[0]:", sys.path[0])

# =============================
# EDIT THESE (copied from notebook constants)
# =============================
VIDEO_PATH = r"C:\Users\brad\OneDrive - UHN\Li, Yue (Sophia)'s files - WinterLab videos\raw videos to rename the gopro files\videos_renamed\2025-02-19\sub349\idapt797_sub349_DP_13_12-13-01.MP4"  # absolute path to raw mp4

# Processing toggles (define BOTH pipeline AND label tag)
DO_INTERP = False
DO_SMOOTH = False

# Render options
FPS = 120 if DO_INTERP else 30
CONF_THR = 0.05
PERSON_INDEX = 0

PANEL_W = 1280
PANEL_H = 720
CANVAS_W = PANEL_W * 3
CANVAS_H = PANEL_H

DRAW_ON_BLACK = True
LABEL_KPT_INDICES = False

# Save everything here
SAVE_DIR = Path(r"D:\temp")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUT_BASENAME = "pose_compare_3panel"
TMP_MP4 = SAVE_DIR / f"{OUT_BASENAME}_tmp.mp4"
FINAL_MP4 = SAVE_DIR / f"{OUT_BASENAME}.mp4"

# =============================
# Backend model/exe paths (these existed in the notebook cell; fill them in if needed)
# =============================
# ---- These should match the config in your main.py ----
# OpenPose
OPENPOSE_EXE = str(_PROJECT_ROOT / "frameworks" / "openpose" / "bin" / "OpenPoseDemo.exe")
OPENPOSE_MODEL_FOLDER = str(_PROJECT_ROOT / "frameworks" / "openpose" / "models")
OPENPOSE_MODEL_POSE = "BODY_25"
OPENPOSE_NUMBER_PEOPLE_MAX = 1

# YOLO
YOLO_MODEL_PATH = str(_PROJECT_ROOT / "models" / "yolo26x-pose.pt")
YOLO_DEVICE = "cuda:0"           # change if you want
YOLO_BATCH_SIZE = 8
YOLO_VERBOSE = False

# MediaPipe
MP_MODEL_PATH = str(_PROJECT_ROOT / "models" / "pose_landmarker_heavy.task")
MP_MIN_DET = 0.5
MP_MIN_PRES = 0.5
MP_MIN_TRACK = 0.5

# =============================
# Skeleton edges (same as visualize.py)
# =============================
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


def stage_tag(do_interp: bool, do_smooth: bool) -> str:
    if do_interp and do_smooth:
        return "raw_interp_smooth"
    if do_interp and not do_smooth:
        return "raw_interp"
    if (not do_interp) and do_smooth:
        return "raw_smooth"
    return "raw"


def reencode_h264(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(dst),
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
            meta = json.loads(data["meta_json"].item())
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


def normalize_if_needed_xy(xy: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    mx = float(np.nanmax(xy[:, 0]))
    my = float(np.nanmax(xy[:, 1]))
    if mx <= 2.0 and my <= 2.0:
        xy = xy.copy()
        xy[:, 0] *= float(src_w)
        xy[:, 1] *= float(src_h)
    return xy


def draw_pose_on_img(
    img_bgr: np.ndarray,
    pts_kc: np.ndarray,            # (K,C)
    edges: List[tuple[int, int]],
    conf_thr: float,
    src_w: Optional[int],
    src_h: Optional[int],
    label_indices: bool = False,
) -> np.ndarray:
    """
    Draws pose onto img_bgr in-place and returns it.
    Assumes pose coordinates are in the source video coordinate space (or normalized).
    """
    H, W = img_bgr.shape[:2]
    K, C = pts_kc.shape

    xy = pts_kc[:, :2].astype(np.float32)
    conf = pts_kc[:, 2].astype(np.float32) if C >= 3 else None

    if src_w is not None and src_h is not None and src_w > 0 and src_h > 0:
        xy = normalize_if_needed_xy(xy, int(src_w), int(src_h))
        sx = W / float(src_w)
        sy = H / float(src_h)
        xy = xy.copy()
        xy[:, 0] *= sx
        xy[:, 1] *= sy
    else:
        # fallback: if normalized in panel space
        mx = float(np.nanmax(xy[:, 0]))
        my = float(np.nanmax(xy[:, 1]))
        if mx <= 2.0 and my <= 2.0:
            xy = xy.copy()
            xy[:, 0] *= float(W)
            xy[:, 1] *= float(H)

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
        cv2.line(img_bgr, (int(ax), int(ay)), (int(bx), int(by)), (0, 255, 0), 2, cv2.LINE_AA)

    # joints
    for k in range(K):
        if conf is not None and conf[k] < conf_thr:
            continue
        x, y = xy[k]
        if not np.isfinite([x, y]).all():
            continue
        cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
        if label_indices:
            cv2.putText(
                img_bgr,
                str(k),
                (int(x) + 4, int(y) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return img_bgr


# =============================
# Pipeline imports (same as notebook)
# =============================
from data_population.pose import PoseExtractor, PoseBackendConfig
from preprocessing.pose_interpolation import InterpolationConfig, interpolate_pose_sequence
from preprocessing.pose_smoothing import SmoothingConfig, EMAPoseSmoother


def build_pose_extractor(backend: str) -> PoseExtractor:
    name = backend.lower()

    if name == "openpose":
        cfg = PoseBackendConfig(
            name="openpose",
            op_exe_path=OPENPOSE_EXE,
            op_model_folder=OPENPOSE_MODEL_FOLDER,
            op_model_pose=OPENPOSE_MODEL_POSE,
            op_number_people_max=OPENPOSE_NUMBER_PEOPLE_MAX,
        )
        return PoseExtractor(cfg)

    if name == "yolo":
        cfg = PoseBackendConfig(
            name="yolo",
            yolo_model_path=YOLO_MODEL_PATH,
            yolo_device=YOLO_DEVICE,
            yolo_batch_size=YOLO_BATCH_SIZE,
            yolo_verbose=YOLO_VERBOSE,
            yolo_num_kpts=17,
        )
        return PoseExtractor(cfg)

    if name == "mediapipe":
        cfg = PoseBackendConfig(
            name="mediapipe",
            mp_model_path=MP_MODEL_PATH,
            mp_num_poses=1,
            mp_min_det_conf=MP_MIN_DET,
            mp_min_presence_conf=MP_MIN_PRES,
            mp_min_track_conf=MP_MIN_TRACK,
        )
        return PoseExtractor(cfg)

    raise ValueError(f"Unknown backend={backend!r}")


def run_pipeline_and_save_npz(
    *,
    backend: str,
    video_abs: str,
    do_interp: bool,
    do_smooth: bool,
    out_dir: Path,
    conf_thr: float,
) -> Path:
    extractor = build_pose_extractor(backend)

    # match main.py defaults used in notebook
    interp_cfg = InterpolationConfig(
        scale_factor=4,      # FPS_SCALE from main.py
        mode="linear",       # INTERP_MODE from main.py
        conf_thr=conf_thr,
        frame_min_kpts=10,
        frame_min_frac=0.0,
        clip_to_frame=None,
    )

    smoother = EMAPoseSmoother(SmoothingConfig(
        alpha=0.35,          # EMA_ALPHA from main.py (as in notebook)
        conf_thr=conf_thr,
        clip_to_frame=None,
    ))

    tag = stage_tag(do_interp, do_smooth)
    stem = Path(video_abs).stem
    out_path = out_dir / f"{stem}_{backend}_{tag}.npz"

    print(f"[gen] extracting backend={backend} -> {out_path.name}")
    poses_raw, meta = extractor.extract_pose_from_video(video_abs, conf_thr=conf_thr)

    # run stages
    poses_stage = poses_raw
    poses_interp = None
    poses_smooth = None

    if do_interp:
        poses_interp = interpolate_pose_sequence(poses_stage, interp_cfg)
        poses_stage = poses_interp

    if do_smooth:
        poses_smooth = smoother.smooth_sequence(poses_stage)
        poses_stage = poses_smooth

    payload_meta = {
        "backend": backend,
        "video_path": video_abs,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
        "raw_meta": meta,
        "interp_config": asdict(interp_cfg) if do_interp else None,
        "smooth_config": asdict(smoother.config) if do_smooth else None,
        "poses_raw_shape": list(np.asarray(poses_raw).shape),
        "poses_interp_shape": list(np.asarray(poses_interp).shape) if poses_interp is not None else None,
        "poses_smooth_shape": list(np.asarray(poses_smooth).shape) if poses_smooth is not None else None,
        "poses_saved_stage": tag,
    }

    np.savez_compressed(
        out_path,
        poses=np.asarray(poses_stage, dtype=np.float32),
        meta_json=json.dumps(payload_meta),
    )

    return out_path


def main() -> None:
    video_abs = os.path.abspath(os.path.expanduser(VIDEO_PATH))
    if not os.path.isfile(video_abs):
        raise FileNotFoundError(video_abs)

    tag = stage_tag(DO_INTERP, DO_SMOOTH)
    print("Pipeline tag:", tag)

    # Generate NPZs every run (same as notebook)
    npz_yolo = run_pipeline_and_save_npz(
        backend="yolo",
        video_abs=video_abs,
        do_interp=DO_INTERP,
        do_smooth=DO_SMOOTH,
        out_dir=SAVE_DIR,
        conf_thr=CONF_THR,
    )

    npz_mp = run_pipeline_and_save_npz(
        backend="mediapipe",
        video_abs=video_abs,
        do_interp=DO_INTERP,
        do_smooth=DO_SMOOTH,
        out_dir=SAVE_DIR,
        conf_thr=CONF_THR,
    )

    npz_op = run_pipeline_and_save_npz(
        backend="openpose",
        video_abs=video_abs,
        do_interp=DO_INTERP,
        do_smooth=DO_SMOOTH,
        out_dir=SAVE_DIR,
        conf_thr=CONF_THR,
    )

    print("YOLO:", npz_yolo)
    print("MP  :", npz_mp)
    print("OP  :", npz_op)

    # Load poses for rendering
    poses_yolo_raw, meta_yolo = load_npz(Path(npz_yolo))
    poses_mp_raw, meta_mp = load_npz(Path(npz_mp))
    poses_op_raw, meta_op = load_npz(Path(npz_op))

    poses_yolo = to_TNKC(poses_yolo_raw)
    poses_mp = to_TNKC(poses_mp_raw)
    poses_op = to_TNKC(poses_op_raw)

    edges_yolo = pick_edges(poses_yolo.shape[2])
    edges_mp = pick_edges(poses_mp.shape[2])
    edges_op = pick_edges(poses_op.shape[2])

    # build human-readable label (same as notebook)
    if DO_INTERP and DO_SMOOTH:
        pipeline_label = "interp + smooth"
    elif DO_INTERP and not DO_SMOOTH:
        pipeline_label = "interp"
    elif (not DO_INTERP) and DO_SMOOTH:
        pipeline_label = "smooth"
    else:
        pipeline_label = "raw"

    # IMPORTANT FIX:
    # - If pipeline_label == "raw", overlay onto the original video for ALL backends.
    # - Otherwise respect DRAW_ON_BLACK.
    effective_draw_on_black = (DRAW_ON_BLACK and pipeline_label != "raw")

    # Open source video for background frames
    cap = cv2.VideoCapture(str(video_abs))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_abs}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # (Notebook seeks to mid-video for demo sometimes; here we start at 0 for full render.)
    black_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)

    # meta sizes used for coordinate mapping
    src_w_yolo = meta_yolo.get("width")
    src_h_yolo = meta_yolo.get("height")

    src_w_mp = meta_mp.get("width")
    src_h_mp = meta_mp.get("height")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    src_w_op = meta_op.get("width") or video_w
    src_h_op = meta_op.get("height") or video_h

    print("src sizes:",
          "yolo", (src_w_yolo, src_h_yolo),
          "mp", (src_w_mp, src_h_mp),
          "op", (src_w_op, src_h_op))

    # ---- KEY FIX FOR 120 FPS OUTPUT ----
    # This MUST match InterpolationConfig.scale_factor used in run_pipeline_and_save_npz().
    FPS_SCALE = 4

    # Drive the render loop by pose length (interpolated length), not by cap.read() length.
    T_out = min(poses_yolo.shape[0], poses_mp.shape[0], poses_op.shape[0])

    print(f"Source video: fps={src_fps:.3f}, frames={src_frame_count}")
    print(f"Pose frames:  yolo={poses_yolo.shape[0]}, mp={poses_mp.shape[0]}, op={poses_op.shape[0]}")
    print(f"Rendering:   T_out={T_out} at FPS={FPS}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(TMP_MP4), fourcc, FPS, (CANVAS_W, CANVAS_H), isColor=True)
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open. Try output .avi or ensure codecs installed.")

    # Helper: map output index (120fps timeline) -> source frame index (30fps)
    def t_to_src_frame(t: int) -> int:
        if DO_INTERP:
            return t // FPS_SCALE
        return t

    # Render
    for t in range(T_out):
        if effective_draw_on_black:
            base_bg = black_panel
        else:
            t_src = t_to_src_frame(t)

            # Clamp to last valid frame so we don't stop early due to seek/read edge cases
            if src_frame_count > 0:
                t_src = min(t_src, src_frame_count - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, t_src)
            ok, frame_bgr = cap.read()
            if not ok:
                break

            raw_panel = cv2.resize(frame_bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
            base_bg = raw_panel

        # Panel 1: YOLO
        p1 = base_bg.copy()
        p1 = draw_pose_on_img(
            p1,
            poses_yolo[t, PERSON_INDEX],
            edges_yolo,
            CONF_THR,
            src_w_yolo,
            src_h_yolo,
            label_indices=LABEL_KPT_INDICES,
        )
        cv2.rectangle(p1, (0, 0), (PANEL_W, 40), (0, 0, 0), -1)
        cv2.putText(p1, f"YOLO | {pipeline_label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 2: MediaPipe
        p2 = base_bg.copy()
        p2 = draw_pose_on_img(
            p2,
            poses_mp[t, PERSON_INDEX],
            edges_mp,
            CONF_THR,
            src_w_mp,
            src_h_mp,
            label_indices=LABEL_KPT_INDICES,
        )
        cv2.rectangle(p2, (0, 0), (PANEL_W, 40), (0, 0, 0), -1)
        cv2.putText(p2, f"MediaPipe | {pipeline_label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Panel 3: OpenPose
        p3 = base_bg.copy()
        p3 = draw_pose_on_img(
            p3,
            poses_op[t, PERSON_INDEX],
            edges_op,
            CONF_THR,
            src_w_op,
            src_h_op,
            label_indices=LABEL_KPT_INDICES,
        )
        cv2.rectangle(p3, (0, 0), (PANEL_W, 40), (0, 0, 0), -1)
        cv2.putText(p3, f"OpenPose | {pipeline_label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:, 0:PANEL_W] = p1
        canvas[:, PANEL_W:2*PANEL_W] = p2
        canvas[:, 2*PANEL_W:3*PANEL_W] = p3

        writer.write(canvas)

    writer.release()
    cap.release()

    print("Wrote tmp:", TMP_MP4)
    print("Re-encoding ->", FINAL_MP4)
    reencode_h264(TMP_MP4, FINAL_MP4)

    try:
        TMP_MP4.unlink()
    except Exception:
        pass

    print("DONE:", FINAL_MP4)


if __name__ == "__main__":
    main()
