from pathlib import Path
import mp_pose_landmarker as mp

VIDEO = r"C:\Users\brad\OneDrive - UHN\Li, Yue (Sophia)'s files - WinterLab videos\raw videos to rename the gopro files\videos_renamed\2025-02-06\sub354\idapt798_sub354_DF_14_11-16-00.mp4"
MODEL = r"CV\models\pose_landmarker_heavy.task"
OUT = r"D:\Downloads\raw_mediapipe_overlay.mp4"

landmarker = mp.load_model(MODEL)
mp.visualize_direct_outputs(
    VIDEO,
    landmarker,
    out_path=OUT,
    draw_on_black=False,   # True if you want black
    conf_thr=0.05,
    preview=True,
    max_frames=None,
)
landmarker.close()
print("wrote:", OUT)
