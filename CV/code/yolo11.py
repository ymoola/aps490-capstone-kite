import cv2
import os
import shutil
import subprocess
from ultralytics import YOLO

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_ROOT = "CV/data"
OUT_ROOT = "CV/out"
MODEL_PATH = "CV/yolo11x-pose.pt"
MAX_SIZE_MB = 10
TEMP_DIR_PREFIX = "_temp_pose_frames"

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv",
              ".MP4", ".MOV", ".M4V", ".AVI", ".MKV"}

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

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-b:v", str(target_bitrate),
        "-maxrate", str(target_bitrate),
        "-bufsize", str(target_bitrate * 2),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    data_root = abspath(DATA_ROOT)
    out_root = abspath(OUT_ROOT)
    model_path = abspath(MODEL_PATH)

    if not os.path.isdir(data_root):
        raise RuntimeError(f"DATA_ROOT not found: {data_root}")

    os.makedirs(out_root, exist_ok=True)
    ensure_ffmpeg_available()

    model = YOLO(model_path)

    videos = list(iter_videos(data_root))
    print(f"Found {len(videos)} video(s)")

    for i, video in enumerate(videos, 1):
        video_abs = abspath(video)

        rel_path = os.path.relpath(video_abs, data_root)
        rel_dir = os.path.dirname(rel_path)

        out_dir = abspath(os.path.join(out_root, rel_dir))
        os.makedirs(out_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(video_abs))[0]
        final_video = abspath(os.path.join(out_dir, f"{name}_annotated.mp4"))

        # --------------------------------------------------
        # RESTORATIVE CHECK (key addition)
        # --------------------------------------------------
        if os.path.exists(final_video):
            print(f"[{i}/{len(videos)}] SKIP (exists): {final_video}")
            continue

        raw_video = abspath(os.path.join(out_dir, f"{name}_annotated_raw.mp4"))
        temp_dir = abspath(os.path.join(out_dir, f"{TEMP_DIR_PREFIX}_{name}"))

        print(f"[{i}/{len(videos)}] Processing: {video_abs}")
        os.makedirs(temp_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_abs)
        if not cap.isOpened():
            print("  !! Failed to open video, skipping")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = frames_total / fps if frames_total else 0.0

        # --------------------------------------------------
        # Pose inference → temp frames
        # --------------------------------------------------
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(frame)
            annotated = result[0].plot()

            frame_path = abspath(
                os.path.join(temp_dir, f"frame_{frame_idx:06d}.jpg")
            )
            cv2.imwrite(frame_path, annotated)
            frame_idx += 1

        cap.release()

        if frame_idx == 0:
            print("  !! No frames read")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        if duration <= 0:
            duration = frame_idx / fps

        # --------------------------------------------------
        # Build raw annotated video
        # --------------------------------------------------
        vw = cv2.VideoWriter(
            raw_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        for fidx in range(frame_idx):
            frame_path = abspath(
                os.path.join(temp_dir, f"frame_{fidx:06d}.jpg")
            )
            img = cv2.imread(frame_path)
            if img is None:
                raise RuntimeError(f"Missing frame: {frame_path}")
            vw.write(img)

        vw.release()

        # --------------------------------------------------
        # Compress
        # --------------------------------------------------
        compress_to_max_mb(raw_video, final_video, duration, MAX_SIZE_MB)

        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(raw_video):
            os.remove(raw_video)

        size_mb = os.path.getsize(final_video) / (1024 * 1024)
        print(f"  ✓ Done ({size_mb:.2f} MB)")

    print("\nAll videos processed (or skipped).")

if __name__ == "__main__":
    main()
