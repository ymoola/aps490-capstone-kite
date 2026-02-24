# main.py
from __future__ import annotations

import os
import json
from dataclasses import asdict
from multiprocessing import get_context
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from pose import iter_videos, PoseExtractor, PoseBackendConfig
from preprocessing.pose_interpolation import InterpolationConfig, interpolate_pose_sequence
from preprocessing.pose_smoothing import SmoothingConfig, EMAPoseSmoother

POSE_BACKEND = "yolo"  # "openpose" | "yolo" | "mediapipe"

# -----------------------------
# PATHS
# -----------------------------
DATA_ROOT = r"C:\Users\brad\OneDrive - UHN\Li, Yue (Sophia)'s files - WinterLab videos\raw videos to rename the gopro files\videos_renamed"
POSE_OUT_ROOT = rf"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\outputs\out_{POSE_BACKEND.lower()}"

# -----------------------------
# BACKEND CONFIG
# -----------------------------

# OpenPose
OPENPOSE_EXE = r"CV\OpenPose\bin\OpenPoseDemo.exe"
OPENPOSE_MODEL_FOLDER = r"CV\OpenPose\models"
OPENPOSE_MODEL_POSE = "BODY_25"
OPENPOSE_NUMBER_PEOPLE_MAX = 1

# YOLO (if you ever swap back)
YOLO_MODEL_PATH = r"CV\models\yolo11x-pose.pt"
YOLO_DEVICE = None             # e.g., "cuda:0"
YOLO_BATCH_SIZE = 8
YOLO_VERBOSE = False

# MediaPipe (if you ever swap back)
MP_MODEL_PATH = r"CV\models\pose_landmarker_heavy.task"
MP_MIN_DET = 0.5
MP_MIN_PRES = 0.5
MP_MIN_TRACK = 0.5

# -----------------------------
# PIPELINE TOGGLES (this is what you tune)
# -----------------------------
DO_INTERP = True
DO_SMOOTH = True

# Interpolation knobs
FPS_SCALE = 4
INTERP_MODE = "linear"      # "linear" | "catmull_rom"
CONF_THR = 0.05

# Smoothing knobs
EMA_ALPHA = 0.7

# -----------------------------
# MULTIPROC
# -----------------------------
NUM_GPUS_TO_USE = 2  # used only for sharding processes; OpenPose itself can use multiple GPUs internally

OVERRIDE_EXISTING_INFERENCES = False



def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def stage_tag(do_interp: bool, do_smooth: bool) -> str:
    if do_interp and do_smooth:
        return "raw_interp_smooth"
    if do_interp and not do_smooth:
        return "raw_interp"
    if (not do_interp) and do_smooth:
        return "raw_smooth"
    return "raw"


def mirrored_output_path(video_abs: str, data_root_abs: str, out_root_abs: str, tag: str) -> str:
    """
    Mirror structure and keep filename, add tag suffix.
    in/x/y/video.mp4 -> out/x/y/video_<tag>.npz
    """
    rel_path = os.path.relpath(video_abs, data_root_abs)
    rel_no_ext, _ = os.path.splitext(rel_path)
    out_path = os.path.join(out_root_abs, rel_no_ext + f"_{tag}.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def build_pose_extractor(gpu_id: int) -> PoseExtractor:
    name = POSE_BACKEND.lower()

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
            yolo_device=YOLO_DEVICE if YOLO_DEVICE is not None else f"cuda:{gpu_id}",
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

    raise ValueError(f"Unknown POSE_BACKEND={POSE_BACKEND!r}")


def worker(
    worker_id: int,
    videos: List[str],
    data_root: str,
    out_root: str,
    do_interp: bool,
    do_smooth: bool,
    fps_scale: int,
    interp_mode: str,
    ema_alpha: float,
    conf_thr: float,
):
    data_root_abs = abspath(data_root)
    out_root_abs = abspath(out_root)

    extractor = build_pose_extractor(worker_id)

    interp_cfg = InterpolationConfig(
        scale_factor=FPS_SCALE,
        mode=INTERP_MODE,
        conf_thr=CONF_THR,
        frame_min_kpts=10,      # try 6, 8, 10
        frame_min_frac=0.0,
        clip_to_frame=None,
    )


    smoother = EMAPoseSmoother(SmoothingConfig(
        alpha=ema_alpha,
        conf_thr=conf_thr,
        smooth_conf=False,
        missing_policy="hold",
        clip_to_frame=None,
    ))

    tag = stage_tag(do_interp, do_smooth)
    pbar = tqdm(videos, desc=f"Worker {worker_id}", position=worker_id, leave=True)

    for video in pbar:
        video_abs = abspath(video)
        out_path = mirrored_output_path(video_abs, data_root_abs, out_root_abs, tag)

        if os.path.exists(out_path):
            continue

        try:
            # 1) Pose extraction
            poses_raw, meta = extractor.extract_pose_from_video(video_abs, conf_thr=CONF_THR)


            poses_stage = poses_raw
            poses_interp = None
            poses_smooth = None

            # 2) Optional interpolate
            if do_interp:
                poses_interp = interpolate_pose_sequence(poses_stage, interp_cfg)
                poses_stage = poses_interp

            # 3) Optional smooth
            if do_smooth:
                poses_smooth = smoother.smooth_sequence(poses_stage)
                poses_stage = poses_smooth

            payload_meta = {
                **meta,
                "video_relpath": os.path.relpath(video_abs, data_root_abs),
                "pipeline": {
                    "do_interp": bool(do_interp),
                    "do_smooth": bool(do_smooth),
                },
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

            pbar.set_postfix({
                "saved": os.path.basename(out_path),
                "rawT": int(np.asarray(poses_raw).shape[0]),
                "tag": tag,
            })

        except Exception as e:
            pbar.write(f"[Worker {worker_id}] FAIL {video_abs}: {e}")


def main():
    data_root_abs = abspath(DATA_ROOT)
    out_root_abs = abspath(POSE_OUT_ROOT)

    if not os.path.isdir(data_root_abs):
        raise RuntimeError(f"DATA_ROOT not found: {data_root_abs}")
    os.makedirs(out_root_abs, exist_ok=True)

    all_videos = list(iter_videos(data_root_abs))
    print(f"Found {len(all_videos)} videos under {data_root_abs}")
    if not all_videos:
        return

    # Decide which videos still need inference
    tag = stage_tag(DO_INTERP, DO_SMOOTH)  # uses your existing function
    if OVERRIDE_EXISTING_INFERENCES:
        todo_videos = all_videos
        print("[info] OVERRIDE_EXISTING_INFERENCES=True -> re-running inference for all videos.")
    else:
        todo_videos = []
        skipped = 0
        for v in all_videos:
            v_abs = abspath(v)
            out_path = mirrored_output_path(v_abs, data_root_abs, out_root_abs, tag)
            if os.path.exists(out_path):
                skipped += 1
                continue
            todo_videos.append(v)

        print(f"[info] OVERRIDE_EXISTING_INFERENCES=False -> skipping {skipped} already-computed videos.")
        print(f"[info] Remaining to process: {len(todo_videos)}")

    if not todo_videos:
        print("[info] Nothing to do.")
        return

    # Worker count:
    available = torch.cuda.device_count()
    if available == 0 or POSE_BACKEND.lower() == "openpose":
        num_workers = 1
        print("[info] Using 1 worker process (OpenPose or no CUDA visible to torch).")
    else:
        num_workers = max(1, min(NUM_GPUS_TO_USE, available))
        print(f"[info] Using {num_workers} worker process(es) (torch sees {available} CUDA device(s)).")

    # Shard only the videos that still need processing
    shards: List[List[str]] = [todo_videos[i::num_workers] for i in range(num_workers)]
    for i, shard in enumerate(shards):
        print(f"Worker {i} assigned {len(shard)} videos")

    ctx = get_context("spawn")
    procs = []
    for worker_id in range(num_workers):
        p = ctx.Process(
            target=worker,
            args=(
                worker_id,
                shards[worker_id],
                data_root_abs,
                out_root_abs,
                DO_INTERP,
                DO_SMOOTH,
                FPS_SCALE,
                INTERP_MODE,
                EMA_ALPHA,
                CONF_THR,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All workers complete.")


if __name__ == "__main__":
    main()
