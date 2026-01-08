import cv2
import os
import json
import csv
from ultralytics import YOLO
from tqdm import tqdm

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SET_ROOT = "C:/Users/user/Desktop/Github/APS490/aps490-capstone-kite/CV/sets"
DATA_ROOT = "C:/Users/user/Desktop/Github/APS490/aps490-capstone-kite/CV/data"
OUT_ROOT = "C:/Users/user/Desktop/Github/APS490/aps490-capstone-kite/CV/out"
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

def iter_videos(root: str):
    for r, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1] in VIDEO_EXTS:
                yield os.path.join(r, f)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    set_root = abspath(SET_ROOT)
    data_root = abspath(DATA_ROOT)
    out_root = abspath(OUT_ROOT)
    model_path = abspath(MODEL_PATH)

    if not os.path.isdir(set_root):
        raise RuntimeError(f"SET_ROOT not found: {set_root}")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"DATA_ROOT not found: {data_root}")

    # Read all CSV files from sets directory
    all_videos = []
    for setname in os.listdir(set_root):
        set_path = os.path.join(set_root, setname)
        if not set_path.endswith('.csv'):
            continue

        print(f"Reading set: {setname}")
        with open(set_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = os.path.join(data_root, row['path'])
                label = int(row['label'])
                all_videos.append({
                    'path': video_path,
                    'label': label,
                    'set': os.path.splitext(setname)[0]  # Remove .csv extension
                })

    os.makedirs(out_root, exist_ok=True)

    model = YOLO(model_path)

    print(f"Found {len(all_videos)} video(s) from CSV files")

    for video_info in tqdm(all_videos, desc="Processing videos"):
        video_abs = abspath(video_info['path'])
        label = video_info['label']
        set_name = video_info['set']

        rel_path = os.path.relpath(video_abs, data_root)
        rel_dir = os.path.dirname(rel_path)

        out_dir = abspath(os.path.join(out_root, rel_dir))
        os.makedirs(out_dir, exist_ok=True)

        name = os.path.splitext(os.path.basename(video_abs))[0]
        skeleton_json = abspath(os.path.join(out_dir, f"{set_name}_{name}_skeleton.json"))

        # --------------------------------------------------
        # RESTORATIVE CHECK (key addition)
        # --------------------------------------------------
        if os.path.exists(skeleton_json):
            tqdm.write(f"SKIP (exists): {os.path.basename(skeleton_json)}")
            continue

        tqdm.write(f"Processing: {os.path.basename(video_abs)} (set: {set_name}, label: {label})")

        cap = cv2.VideoCapture(video_abs)
        if not cap.isOpened():
            tqdm.write(f"  !! Failed to open video, skipping")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = frames_total / fps if frames_total else 0.0

        # --------------------------------------------------
        # Pose inference → temp frames + skeleton data
        # --------------------------------------------------
        frame_idx = 0
        skeleton_data = {
            "data": []
        }

        # Create progress bar for frames
        frame_pbar = tqdm(total=frames_total if frames_total > 0 else None,
                         desc="Frames", leave=False, unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(frame)
            frame_pbar.update(1)

            # Extract skeleton keypoints
            frame_skeletons = []
            if result[0].keypoints is not None:
                keypoints = result[0].keypoints
                for person_idx in range(len(keypoints)):
                    # Get keypoints for this person (shape: [17, 2] or [17, 3] with confidence)
                    kpts = keypoints.xy[person_idx].cpu().numpy()
                    conf = keypoints.conf[person_idx].cpu().numpy() if keypoints.conf is not None else None

                    # Initialize pose and score arrays with zeros (for 18 keypoints)
                    # YOLO has 17 keypoints, but we'll use 18 to match the format
                    pose = []
                    score = []

                    # COCO 17 keypoints - convert to flat array with normalized coordinates
                    for kpt_idx in range(17):
                        x, y = kpts[kpt_idx]
                        c = conf[kpt_idx] if conf is not None else 0.0

                        # Normalize coordinates (divide by width/height)
                        x_norm = float(x) / width if x > 0 else 0.0
                        y_norm = float(y) / height if y > 0 else 0.0

                        # Only add if confidence is above threshold, otherwise use 0.0
                        if c < 0.05:  # Low confidence threshold
                            pose.extend([0.0, 0.0])
                            score.append(0.0)
                        else:
                            pose.extend([round(x_norm, 3), round(y_norm, 3)])
                            score.append(round(float(c), 3))

                    # Add 18th keypoint as zeros to match format
                    pose.extend([0.0, 0.0])
                    score.append(0.0)

                    person_skeleton = {
                        "pose": pose,
                        "score": score
                    }
                    frame_skeletons.append(person_skeleton)

            skeleton_data["data"].append({
                "frame_index": frame_idx + 1,  # 1-indexed
                "skeleton": frame_skeletons
            })

            frame_idx += 1

        frame_pbar.close()
        cap.release()

        if frame_idx == 0:
            tqdm.write("  !! No frames read")
            continue

        # --------------------------------------------------
        # Save skeleton JSON
        # --------------------------------------------------
        with open(skeleton_json, 'w') as f:
            json.dump(skeleton_data, f, indent=2)

        tqdm.write(f"  ✓ Done - {frame_idx} frames processed")

    print("\nAll videos processed (or skipped).")

if __name__ == "__main__":
    main()
