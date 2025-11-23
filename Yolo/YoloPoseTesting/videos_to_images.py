import cv2
import os
import shutil

# Backup folders where your videos currently are
backup_folders = [
    r"C:\Capstone\Yolo\YoloPoseTesting\video_backups\two_foot",
    r"C:\Capstone\Yolo\YoloPoseTesting\video_backups\no_two_foot",
]

# Corresponding train folders where frames will go
train_folders = [
    r"C:\Capstone\Yolo\YoloPoseTesting\dataset\train\two_foot",
    r"C:\Capstone\Yolo\YoloPoseTesting\dataset\train\no_two_foot",
]

# Extract frames from backup videos into train folders
for i in range(len(backup_folders)):
    backup_folder = backup_folders[i]
    train_folder = train_folders[i]

    # List all video files in backup
    videos = []
    for f in os.listdir(backup_folder):
        if f.lower().endswith((".mp4", ".avi", ".mov")):
            videos.append(f)

    for vid_file in videos:
        video_path = os.path.join(backup_folder, vid_file)
        vid = cv2.VideoCapture(video_path)
        count = 0

        while True:
            success, frame = vid.read()
            if not success:
                break
            # Save frames directly in the train folder
            frame_file = os.path.join(train_folder, os.path.splitext(vid_file)[0] + "_" + str(count).zfill(5) + ".jpg")
            cv2.imwrite(frame_file, frame)
            count += 1

        vid.release()
        print("Extracted frames from", vid_file)

print("All videos processed successfully!")