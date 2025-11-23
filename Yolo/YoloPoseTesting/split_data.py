import os
import shutil
import random

original_data = "C:\\Capstone\\Yolo\\YoloPoseTesting\\slip_dataset\\slip_dataset"
output_data   = "C:\\Capstone\\Yolo\\YoloPoseTesting\\dataset"
train_ratio   = 0.95


def process_two_foot_slip():
    # FIRST CLASS: two_foot_slip
    original_class = "two_foot_slip"
    new_class  = "two_foot"

    class_path = os.path.join(original_data, original_class)
    videos = []  # create an empty list to store video filenames
    for files in os.listdir(class_path):  # loop over all files/folders in the folder
        if files.endswith(('.mp4', '.avi', '.mov')):  # check if the file is a video
            videos.append(files)  # add it to the list

    random.shuffle(videos)  # shuffle the list of videos randomly

    # Split into train/test
    train_count = int(len(videos) * train_ratio)
    train_videos = []
    test_videos  = []

    for i in range(train_count):
        train_videos.append(videos[i])
    for i in range(train_count, len(videos)):
        test_videos.append(videos[i])

    # Copy train videos
    for vid in train_videos:
        src = os.path.join(class_path, vid)
        dst = os.path.join(output_data, "train", new_class, vid)
        shutil.copy(src, dst)

    # Copy test videos
    for vid in test_videos:
        src = os.path.join(class_path, vid)
        dst = os.path.join(output_data, "test", new_class, vid)
        shutil.copy(src, dst)

def process_no_two_foot_slip():
    # SECOND CLASS: no_two_foot_slip
    original_class = "no_two_foot_slip"
    new_class  = "no_two_foot"

    class_path = os.path.join(original_data, original_class)

    # List only video files
    videos = []
    for files in os.listdir(class_path):
        if files.endswith(".mp4") or files.endswith(".avi") or files.endswith(".mov"):
            videos.append(files)

    # Shuffle videos randomly
    random.shuffle(videos)

    # Split into train/test
    train_count = int(len(videos) * train_ratio)
    train_videos = []
    test_videos  = []

    for i in range(train_count):
        train_videos.append(videos[i])
    for i in range(train_count, len(videos)):
        test_videos.append(videos[i])

    # Copy train videos
    for vid in train_videos:
        src = os.path.join(class_path, vid)
        dst = os.path.join(output_data, "train", new_class, vid)
        shutil.copy(src, dst)

    # Copy test videos
    for vid in test_videos:
        src = os.path.join(class_path, vid)
        dst = os.path.join(output_data, "test", new_class, vid)
        shutil.copy(src, dst)


process_two_foot_slip()
process_no_two_foot_slip()
print("Train/test splitting and copying completed!")

