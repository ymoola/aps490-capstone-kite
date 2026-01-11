# main.py

from __future__ import annotations

import os
import json
from dataclasses import asdict
from multiprocessing import get_context
from typing import List, Tuple
from pathlib import Path


import numpy as np
import torch
from tqdm import tqdm


from pose import iter_videos, extract_pose_from_video, PoseExtractor, PoseBackendConfig
from pose_interpolation import InterpolationConfig, interpolate_pose_sequence
from pose_smoothing import SmoothingConfig, EMAPoseSmoother


# -----------------------------
# FILL THESE IN
# -----------------------------
DATA_ROOT = r"C:\Users\brad\OneDrive - UHN\Li, Yue (Sophia)'s files - WinterLab videos\raw videos to rename the gopro files\videos_renamed"
POSE_OUT_ROOT = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\out"  # where compressed pose files go
MODEL_PATH = r"CV/yolo11x-pose.pt"

# Pipeline knobs
FPS_SCALE = 4
INTERP_MODE = "linear"      # or "catmull_rom"
EMA_ALPHA = 0.7
CONF_THR = 0.05

# YOLO batching
BATCH_SIZE = 8              # try 8, 16, 32 depending on VRAM
NUM_GPUS_TO_USE = 2         # as requested


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def mirrored_output_path(video_abs: str, data_root_abs: str, out_root_abs: str) -> str:
    """
    Mirror the input directory structure under out_root,
    keep the video filename, but change extension to .npz.

    Example:
      in/x/y/video.mp4
      -> out/x/y/video.npz
    """
    # path relative to DATA_ROOT, including filename
    rel_path = os.path.relpath(video_abs, data_root_abs)

    # replace extension with .npz
    rel_no_ext, _ = os.path.splitext(rel_path)
    out_path = os.path.join(out_root_abs, rel_no_ext + ".npz")

    # ensure parent directories exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path



def worker(
    gpu_id: int,
    videos: List[str],
    data_root: str,
    out_root: str,
    model_path: str,
    fps_scale: int,
    interp_mode: str,
    ema_alpha: float,
    conf_thr: float,
    batch_size: int,
):
    data_root_abs = abspath(data_root)
    out_root_abs = abspath(out_root)

    cfg = PoseBackendConfig(
        name="mediapipe",
        mp_model_path="CV/models/pose_landmarker_heavy.task",
        mp_min_det_conf=0.5,
        mp_min_presence_conf=0.5,
        mp_min_track_conf=0.5,
    )
    extractor = PoseExtractor(cfg)

    interp_cfg = InterpolationConfig(
        scale_factor=fps_scale,
        mode=interp_mode,
        conf_thr=conf_thr,
        clip_to_frame=None,
    )

    smoother = EMAPoseSmoother(SmoothingConfig(
        alpha=ema_alpha,
        conf_thr=conf_thr,
        smooth_conf=False,
        missing_policy="hold",
        clip_to_frame=None,
    ))

    pbar = tqdm(videos, desc=f"Worker {gpu_id}", position=gpu_id, leave=True)
    for video in pbar:
        video_abs = abspath(video)
        out_path = mirrored_output_path(video_abs, data_root_abs, out_root_abs)

        if os.path.exists(out_path):
            continue

        try:
            # 1) Pose extraction (backend-agnostic)
            poses_raw, meta = extract_pose_from_video(video_abs, extractor)

            # 2) Interpolate
            poses_interp = interpolate_pose_sequence(poses_raw, interp_cfg)

            # 3) Smooth
            poses_smooth = smoother.smooth_sequence(poses_interp)

            payload_meta = {
                **meta,
                "video_relpath": os.path.relpath(video_abs, data_root_abs),
                "interp_config": asdict(interp_cfg),
                "smooth_config": asdict(smoother.config),
                "poses_raw_shape": list(poses_raw.shape),
                "poses_interp_shape": list(poses_interp.shape),
                "poses_smooth_shape": list(poses_smooth.shape),
            }

            np.savez_compressed(
                out_path,
                poses=poses_smooth.astype(np.float32, copy=False),
                meta_json=json.dumps(payload_meta),
            )

            pbar.set_postfix({
                "rawT": poses_raw.shape[0],
                "interpT": poses_interp.shape[0],
                "saved": os.path.basename(out_path),
            })

        except Exception as e:
            pbar.write(f"[Worker {gpu_id}] FAIL {video_abs}: {e}")

    extractor.close()

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

    # Determine GPUs
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("No CUDA GPUs found. Set NUM_GPUS_TO_USE=0 and run on CPU if needed.")

    num_workers = min(NUM_GPUS_TO_USE, available)
    print(f"Using {num_workers} GPU worker process(es) out of {available} available GPUs.")

    # Deterministic sharding (no duplicates)
    shards: List[List[str]] = [all_videos[i::num_workers] for i in range(num_workers)]
    for i, shard in enumerate(shards):
        print(f"GPU {i} assigned {len(shard)} videos")

    ctx = get_context("spawn")  # safer with CUDA than fork
    procs = []
    for gpu_id in range(num_workers):
        p = ctx.Process(
            target=worker,
            args=(
                gpu_id,
                shards[gpu_id],
                data_root_abs,
                out_root_abs,
                MODEL_PATH,
                FPS_SCALE,
                INTERP_MODE,
                EMA_ALPHA,
                CONF_THR,
                BATCH_SIZE,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("All workers complete.")


if __name__ == "__main__":
    main()
