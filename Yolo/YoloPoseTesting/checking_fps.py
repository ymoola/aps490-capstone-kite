import cv2

video_path = r"C:\Capstone\Yolo\YoloPoseTesting\dataset\train\two_foot\GX011322-2.mp4"
vid = cv2.VideoCapture(video_path)

fps = vid.get(cv2.CAP_PROP_FPS)  # get frames per second
frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
duration = frame_count / fps  # duration in seconds

print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Duration (s): {duration}")

vid.release()