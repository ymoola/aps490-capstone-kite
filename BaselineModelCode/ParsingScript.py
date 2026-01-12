import os
from pathlib import Path
import numpy as np
import pandas as pd

# 1) Get dataset root from environment variable (portable across computers)
DATA_ROOT = Path(os.environ["WINTERLAB_VIDEO_ROOT"])

# 2) Crawl for videos and parse labels from filename
video_rel_paths = []
labels = []

for full_path in DATA_ROOT.rglob("*.mp4"):
    filename = full_path.name.upper()

    # Label parsing from filename 
    if "DP" in filename:
        label = 1
    elif "DF" in filename:
        label = 0
    else:
        continue  # skip unlabeled files

    # Store RELATIVE path (portable)
    rel_path = full_path.relative_to(DATA_ROOT).as_posix()
    video_rel_paths.append(rel_path)
    labels.append(label)

print(f"Found {len(video_rel_paths)} labeled videos")

# 3) Build dataframe
df = pd.DataFrame({
    "path": video_rel_paths,   # relative to DATA_ROOT
    "label": labels
})

# 4) Stratified split 
rng = np.random.default_rng(seed=42)

train_idx, val_idx, test_idx = [], [], []

for label in sorted(df["label"].unique()):
    cls_idx = df.index[df["label"] == label].to_numpy()
    rng.shuffle(cls_idx)

    n = len(cls_idx)
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)

    train_idx.extend(cls_idx[:n_train])
    val_idx.extend(cls_idx[n_train:n_train + n_val])
    test_idx.extend(cls_idx[n_train + n_val:])

train_df = df.loc[train_idx].reset_index(drop=True)
val_df   = df.loc[val_idx].reset_index(drop=True)
test_df  = df.loc[test_idx].reset_index(drop=True)

print("\nTrain class distribution:")
print(train_df["label"].value_counts(normalize=True))
print("\nVal class distribution:")
print(val_df["label"].value_counts(normalize=True))
print("\nTest class distribution:")
print(test_df["label"].value_counts(normalize=True))

# 5) Output folder (portable): save alongside the script, or inside your repo
output_dir = Path("BaselineDataset")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print(f"\nSaved splits to: {output_dir.resolve()}")