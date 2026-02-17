import os
import subprocess
import multiprocessing
from pathlib import Path
from tqdm import tqdm  # pip install tqdm (Optional, for a nice progress bar)

# ==========================================
# 1. Config
# ==========================================
INPUT_DIR = Path("/Users/yusufmoola/Library/CloudStorage/OneDrive-UHN/Li, Yue (Sophia)'s files - new video and tipper files/Videos")       
OUTPUT_DIR = Path("/Users/yusufmoola/Desktop/videos_new_360p") 

# How many CPU cores to use
NUM_WORKERS = os.cpu_count() - 1 

# ==========================================
# 2. Worker Function
# ==========================================
def resize_video(video_path_str):
    input_path = Path(video_path_str)
    
    # Reconstruct output path to maintain folder structure
    # We use try/except in case the file is not inside INPUT_DIR
    try:
        rel_path = input_path.relative_to(INPUT_DIR)
    except ValueError:
        # Fallback if path manipulation fails
        rel_path = input_path.name
        
    output_path = OUTPUT_DIR / rel_path
    
    # Skip if already exists
    if output_path.exists():
        return None # Return None to indicate skipped
    
    # Ensure parent dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command: Resize short side to 360px
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "scale='if(gt(iw,ih),-2,360)':'if(gt(iw,ih),360,-2)'", 
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an", # Remove audio
        str(output_path)
    ]
    
    try:
        # Check if ffmpeg is actually installed
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"✅ Resized {rel_path}"
    except FileNotFoundError:
        return "Error: FFmpeg not found. Please install it (brew install ffmpeg)"
    except Exception as e:
        return f"❌ Failed {rel_path}: {e}"

# ==========================================
# 3. Helper: List Files
# ==========================================
def list_files(directory):
    files = []
    # Add any other extensions you have
    extensions = ["**/*.mp4", "**/*.avi", "**/*.mov", "**/*.MP4", "**/*.MTS", "**/*.MP4"]
    
    for ext in extensions:
        found = list(directory.glob(ext))
        files.extend(found)
        
    return [str(p) for p in files]

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    if not INPUT_DIR.exists():
        print(f"❌ Error: Input directory does not exist: {INPUT_DIR}")
        exit(1)

    print(f"Scanning files in: {INPUT_DIR}")
    all_videos = list_files(INPUT_DIR)
    print(f"Found {len(all_videos)} videos.")
    
    if len(all_videos) == 0:
        print("Nothing to do.")
        exit()

    print(f"Starting resize on {NUM_WORKERS} local cores...")
    print(f"Saving to: {OUTPUT_DIR}")

    # Use multiprocessing Pool to parallelize locally
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        # Use tqdm for a progress bar if available, otherwise just iterate
        try:
            results = list(tqdm(pool.imap_unordered(resize_video, all_videos), total=len(all_videos)))
        except ImportError:
            print("Tip: 'pip install tqdm' for a progress bar.")
            results = pool.map(resize_video, all_videos)

    # Print summary
    success = sum(1 for r in results if r and "✅" in r)
    skipped = sum(1 for r in results if r is None)
    failed = sum(1 for r in results if r and "❌" in r)

    print("-" * 30)
    print(f"Done! Processed: {success} | Skipped: {skipped} | Failed: {failed}")