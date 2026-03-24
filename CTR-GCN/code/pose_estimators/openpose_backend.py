# openpose_backend.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


# OpenPose BODY_25 skeleton edges (K=25)
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

# OpenPose COCO edges (K=18) if you ever use COCO
COCO18_EDGES: List[tuple[int, int]] = [
    (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 1),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
]


def edges_for_model_pose(model_pose: str) -> List[tuple[int, int]]:
    mp = (model_pose or "").upper()
    if mp == "BODY_25":
        return BODY25_EDGES
    if mp == "COCO":
        return COCO18_EDGES
    return BODY25_EDGES


@dataclass
class OpenPoseConfig:
    openpose_exe: str
    model_folder: str
    model_pose: str = "BODY_25"
    display: int = 0
    render_pose: int = 0
    number_people_max: int = 1


def _run_openpose_video_to_json(cfg: OpenPoseConfig, video_path: str, json_out_dir: str) -> None:
    exe = abspath(cfg.openpose_exe)
    video_abs = abspath(video_path)
    model_folder = abspath(cfg.model_folder)
    json_out_dir = abspath(json_out_dir)

    os.makedirs(json_out_dir, exist_ok=True)

    cmd = [
        exe,
        "--video", video_abs,
        "--write_json", json_out_dir,
        "--model_folder", model_folder,
        "--model_pose", cfg.model_pose,
        "--display", str(cfg.display),
        "--render_pose", str(cfg.render_pose),
        "--number_people_max", str(cfg.number_people_max),
    ]

    # NOTE: If your portable build requires running from OpenPose root,
    # you can set cwd here, e.g.:
    # openpose_root = str(Path(cfg.openpose_exe).resolve().parent.parent)
    # subprocess.run(cmd, check=True, cwd=openpose_root)
    subprocess.run(cmd, check=True)


def _sorted_json_files(json_dir: str) -> List[str]:
    files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]
    files.sort()
    return [os.path.join(json_dir, f) for f in files]


def _pick_single_person_from_openpose_json(people: list, num_kpts: int) -> np.ndarray:
    """
    Returns (K,3) [x,y,conf] for the person with highest mean conf.
    If no valid person, returns zeros.
    """
    if not people:
        return np.zeros((num_kpts, 3), dtype=np.float32)

    best = None
    best_score = -1.0
    for p in people:
        arr = p.get("pose_keypoints_2d", [])
        if not arr or len(arr) < 3 * num_kpts:
            continue
        k = np.array(arr[: 3 * num_kpts], dtype=np.float32).reshape(num_kpts, 3)
        score = float(np.mean(k[:, 2]))
        if score > best_score:
            best_score = score
            best = k

    if best is None:
        return np.zeros((num_kpts, 3), dtype=np.float32)
    return best


def _should_accept_pose_by_bones(kpts: np.ndarray, edges: List[tuple[int, int]], conf_thr: float) -> bool:
    """
    Accept pose if at least 2/3 of bones are "good".
    A bone is "good" if BOTH endpoints have conf >= conf_thr.
    Reject if >= 1/3 bones are "bad".
    """
    if conf_thr <= 0:
        return True  # no gating

    if kpts.size == 0:
        return False

    conf = kpts[:, 2]
    total = 0
    bad = 0

    K = kpts.shape[0]
    for a, b in edges:
        if a >= K or b >= K:
            continue
        total += 1
        if conf[a] < conf_thr or conf[b] < conf_thr:
            bad += 1

    if total == 0:
        # if no edges defined, fall back to mean confidence
        return float(np.mean(conf)) >= conf_thr

    # reject if 1/3 or more bones are bad
    return (bad / total) < (1.0 / 3.0)


def extract_pose_from_video(
    video_path: str,
    cfg: OpenPoseConfig,
    *,
    conf_thr: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """
    Returns:
      poses: (T, K, 3)
      meta: dict

    If conf_thr > 0:
      frames where >= 2/3 bones are low-confidence are zeroed out (no pose).
    """
    video_abs = abspath(video_path)

    tmp_root = tempfile.mkdtemp(prefix="openpose_json_")
    json_dir = os.path.join(tmp_root, "json")

    try:
        _run_openpose_video_to_json(cfg, video_abs, json_dir)
        json_files = _sorted_json_files(json_dir)

        mp = cfg.model_pose.upper()
        if mp == "BODY_25":
            K = 25
        elif mp == "COCO":
            K = 18
        else:
            K = 25

        edges = edges_for_model_pose(cfg.model_pose)

        poses_list: List[np.ndarray] = []
        rejected = 0

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            people = data.get("people", [])
            one = _pick_single_person_from_openpose_json(people, num_kpts=K)

            if conf_thr > 0 and not _should_accept_pose_by_bones(one, edges, conf_thr):
                one = np.zeros((K, 3), dtype=np.float32)
                rejected += 1

            poses_list.append(one)

        if not poses_list:
            raise RuntimeError(f"OpenPose produced no JSON frames for: {video_abs}")

        poses = np.stack(poses_list, axis=0).astype(np.float32, copy=False)

        meta = {
            "backend": "openpose",
            "video_path": video_abs,
            "model_pose": cfg.model_pose,
            "num_frames": int(poses.shape[0]),
            "num_kpts": int(poses.shape[1]),
            "conf_thr_gate": float(conf_thr),
            "rejected_frames": int(rejected),
            "notes": "Parsed from OpenPose per-frame JSON output (with optional bone-confidence gating).",
        }
        return poses, meta

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)




if __name__ == "__main__":
    # -------------------------------------------------
    # EDIT THESE VARIABLES
    # -------------------------------------------------
    VIDEO_PATH = r"C:\Users\brad\OneDrive - UHN\Li, Yue (Sophia)'s files - WinterLab videos\raw videos to rename the gopro files\videos_renamed\2025-05-13\sub352\idapt802_sub352_DF_13_GP1_12-07-14.mp4"

    from pathlib import Path
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    OPENPOSE_EXE = str(_PROJECT_ROOT / "OpenPose" / "bin" / "OpenPoseDemo.exe")
    MODEL_FOLDER = str(_PROJECT_ROOT / "OpenPose" / "models")

    MODEL_POSE = "BODY_25"     # BODY_25 | COCO
    NUMBER_PEOPLE_MAX = 1

    DISPLAY = 1               # 1 = show interactive window
    RENDER_POSE = 1           # 1 = draw skeleton
    NET_RESOLUTION = "-1x368" # optional, OpenPose default
    HAND = False
    FACE = False
    # -------------------------------------------------

    exe = abspath(OPENPOSE_EXE)
    video_abs = abspath(VIDEO_PATH)
    model_folder = abspath(MODEL_FOLDER)

    cmd = [
        exe,
        "--video", video_abs,
        "--model_folder", model_folder,
        "--model_pose", MODEL_POSE,
        "--number_people_max", str(NUMBER_PEOPLE_MAX),
        "--display", str(DISPLAY),
        "--render_pose", str(RENDER_POSE),
        "--net_resolution", NET_RESOLUTION,
    ]

    if HAND:
        cmd.append("--hand")
    if FACE:
        cmd.append("--face")

    print("[OpenPose] Running command:")
    print(" ".join(cmd))

    # IMPORTANT:
    # Some Windows portable builds REQUIRE cwd to be OpenPose root
    # Uncomment if needed:
    #
    # openpose_root = str(Path(OPENPOSE_EXE).resolve().parent.parent)
    # subprocess.run(cmd, check=True, cwd=openpose_root)

    subprocess.run(cmd, check=True)
