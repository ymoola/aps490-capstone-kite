

import os

folders = {
    "two_foot": "C:\\Capstone\\aps490-capstone-kite\\Yolo\\YoloPoseTesting\\dataset\\train\\two_foot",
    "no_two_foot": "C:\\Capstone\\aps490-capstone-kite\\Yolo\\YoloPoseTesting\\dataset\\train\\no_two_foot"
}

for label, folder in folders.items():
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        print(f"Number of images in '{label}' class: {count}")
    else:
        print(f"Folder not found: {folder}")