from ultralytics import YOLO
import numpy as np

# Load trained model
model_path = r"C:\Capstone\Yolo\YoloPoseTesting\runs\classify\train2\weights\best.pt"
model = YOLO(model_path)

#class index for "two_foot"
two_foot_idx = 1

# Path to video
video_path = r"C:\Capstone\Yolo\YoloPoseTesting\dataset\test\two_foot\GX011493.mp4"

frame_probs = []

# Run inference efficiently
for result in model(video_path, stream=True):
    if len(result.probs.data) > 0:  # check there are predictions
        prob_slip = float(result.probs.data[two_foot_idx].cpu().numpy())
        frame_probs.append(prob_slip)

# Average probability across all frames
if frame_probs:
    avg_prob_slip = np.mean(frame_probs)
    print(f"Average two-foot slip probability for video: {avg_prob_slip:.2f}")
    final_label = "slip" if avg_prob_slip >= 0.5 else "no slip"
    print(f"Predicted video label: {final_label}")
else:
    print("No valid frames were processed.")
