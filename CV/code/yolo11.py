import cv2
import os
import shutil
import subprocess
from ultralytics import YOLO
import torch

from video_interpolation import RIFEVideoInterpolator


# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_ROOT = "CV/data"
OUT_ROOT = "CV/out"
MODEL_PATH = "CV/yolo11x-pose.pt"
MAX_SIZE_MB = 10
TEMP_DIR_PREFIX = "_temp_pose_frames"

VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv",
    ".MP4", ".MOV", ".M4V", ".AVI", ".MKV"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))

def ensure_ffmpeg_available():
    subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def iter_videos(root: str):
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1] in VIDEO_EXTS:
                yield os.path.join(r, f)

def compress_to_max_mb(input_path, output_path, duration_sec, max_size_mb):
    target_bits = max_size_mb * 8 * 1024 * 1024
    duration_sec = max(duration_sec, 0.1)
    target_bitrate = int(target_bits / duration_sec)
    target_bitrate = max(150_000, min(target_bitrate, 20_000_000))

    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-b:v", str(target_bitrate),
        "-maxrate", str(target_bitrate),
        "-bufsize", str(target_bitrate * 2),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ], check=True)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    data_root = abspath(DATA_ROOT)
    out_root = abspath(OUT_ROOT)

    os.makedirs(out_root, exist_ok=True)
    ensure_ffmpeg_available()

    # Load models once
    yolo = YOLO(abspath(MODEL_PATH))
    interpolator = RIFEVideoInterpolator()

    videos = list(iter_videos(data_root))
    print(f"Found {len(videos)} video(s)")

    for i, video in enumerate(videos, 1):
        video_abs = abspath(video)

        # Mirror directory structure
        rel_path = os.path.relpath(video_abs, data_root)
        rel_dir = os.path.dirname(rel_path)
        out_dir = abspath(os.path.join(out_root, rel_dir))
        os.makedirs(out_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(video_abs))[0]
        final_video = abspath(os.path.join(out_dir, f"{name}_annotated.mp4"))

        # Restorative check
        if os.path.exists(final_video):
            print(f"[{i}/{len(videos)}] SKIP: {final_video}")
            continue

        raw_video = abspath(os.path.join(out_dir, f"{name}_annotated_raw.mp4"))
        temp_dir = abspath(os.path.join(out_dir, f"{TEMP_DIR_PREFIX}_{name}"))
        interp_video = abspath(os.path.join(out_dir, f"{name}_interp_temp.mp4"))

        os.makedirs(temp_dir, exist_ok=True)

        # --------------------------------------------------
        # Step 1: Interpolate full video (RIFE)
        # --------------------------------------------------
        print(f"[{i}/{len(videos)}] Interpolating video with RIFE...")
        interpolator.interpolate_video_to_file(
            input_video=video_abs,
            output_video=interp_video,
            fps_multiplier=2
        )

        # --------------------------------------------------
        # Step 2: YOLO inference on interpolated video
        # --------------------------------------------------
        cap = cv2.VideoCapture(interp_video)
        if not cap.isOpened():
            print("  !! Failed to open interpolated video")
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(interp_video):
                os.remove(interp_video)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = yolo(frame)
            annotated = result[0].plot()

            cv2.imwrite(
                os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg"),
                annotated
            )
            frame_idx += 1

        cap.release()

        if frame_idx == 0:
            print("  !! No frames read from interpolated video")
            shutil.rmtree(temp_dir, ignore_errors=True)
            if os.path.exists(interp_video):
                os.remove(interp_video)
            continue

        duration = frame_idx / fps

        # --------------------------------------------------
        # Step 3: Build raw annotated video
        # --------------------------------------------------
        vw = cv2.VideoWriter(
            raw_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        for idx in range(frame_idx):
            img_path = os.path.join(temp_dir, f"frame_{idx:06d}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Missing frame: {img_path}")
            vw.write(img)

        vw.release()

        # --------------------------------------------------
        # Step 4: Compress final output
        # --------------------------------------------------
        compress_to_max_mb(raw_video, final_video, duration, MAX_SIZE_MB)

        # --------------------------------------------------
        # Step 5: Cleanup
        # --------------------------------------------------
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(raw_video):
            os.remove(raw_video)
        if os.path.exists(interp_video):
            os.remove(interp_video)

        size_mb = os.path.getsize(final_video) / (1024 * 1024)
        print(f"[{i}/{len(videos)}] Done ({size_mb:.2f} MB)")

    print("All videos processed.")

if __name__ == "__main__":
    main()
