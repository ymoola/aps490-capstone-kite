import os
import numpy as np

#Parsing files within my OneDrive Folder
DATA_ROOT = r"C:\Users\archa\UHN\Li, Yue (Sophia) - raw videos to rename the gopro files\videos_renamed"
video_paths = []
labels = []

for root, _, files in os.walk(DATA_ROOT):
    for file in files:
        if not file.lower().endswith(".mp4"):
            continue

        full_path = os.path.join(root, file)

        # Label parsing from filename only
        filename = file.upper()

        if "DP" in filename:
            label = 1   # P class
        elif "DF" in filename:
            label = 0   # F class
        else:
            continue  # skip unlabeled files

        video_paths.append(full_path)
        labels.append(label)

print(f"Found {len(video_paths)} labeled videos")        

#creating our dataset

import pandas as pd

df = pd.DataFrame({
    "video_path": video_paths,
    "label": labels
})

# Set random seed for reproducibility
randomSeed = np.random.default_rng(seed=42)

train_idx, val_idx, test_idx = [], [], []

# Loop through each class to preserve class ratios
for label in df["label"].unique():
    cls_idx = df[df["label"] == label].index.to_numpy()
    randomSeed.shuffle(cls_idx)

    n = len(cls_idx)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val  # remaining

    train_idx.extend(cls_idx[:n_train])
    val_idx.extend(cls_idx[n_train:n_train + n_val])
    test_idx.extend(cls_idx[n_train + n_val:])

# Create the splits
train_df = df.loc[train_idx].reset_index(drop=True)
val_df   = df.loc[val_idx].reset_index(drop=True)
test_df  = df.loc[test_idx].reset_index(drop=True)

print("Train class distribution:")
print(train_df["label"].value_counts(normalize=True))
print("Val class distribution:")
print(val_df["label"].value_counts(normalize=True))
print("Test class distribution:")
print(test_df["label"].value_counts(normalize=True))


output_dir = r"C:\Users\archa\UHN\Li, Yue (Sophia) - raw videos to rename the gopro files\BaselineDataset"
os.makedirs(output_dir, exist_ok=True)

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)