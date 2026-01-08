# =========================
# PyTorchVideo + X3D (Dataset + Dataloaders)
# OneDrive setup: videos + split CSVs live in OneDrive, paths are portable
# Requires env vars:
#   WINTERLAB_VIDEO_ROOT -> .../videos_renamed (contains date folders)
#   WINTERLAB_SPLIT_ROOT -> .../BaselineDataset (contains train.csv/val.csv/test.csv)
# =========================

import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import torch.utils.data
import pytorchvideo.data
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import torch.optim as optim
import pytorch_lightning as L  
from torch.utils.data import DataLoader
import pytorch_lightning
import pytorchvideo


# =========================
# Apply a transform to a dict key (like ApplyTransformToKey)
# =========================

class ApplyToKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform
    def __call__(self, x):
        x[self.key] = self.transform(x[self.key])
        return x

# Temporal subsample on (C, T, H, W)
class UniformTemporalSubsample:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        C, T, H, W = video.shape
        if T <= self.num_samples:
            # If clip is short, repeat last frame to reach num_samples (safer than failing)
            idx = torch.arange(T)
            if T < self.num_samples:
                pad = idx.new_full((self.num_samples - T,), T - 1)
                idx = torch.cat([idx, pad], dim=0)
            return video[:, idx, :, :]
        idx = torch.linspace(0, T - 1, self.num_samples).long()
        return video[:, idx, :, :]

# Normalize (C, T, H, W) using TF.normalize frame-by-frame
class VideoNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        frames = [TF.normalize(video[:, t], self.mean, self.std) for t in range(video.shape[1])]
        return torch.stack(frames, dim=1)

# Resize each frame so short side = s (random range for train, fixed for val)
class RandomShortSideScale:
    def __init__(self, min_size=256, max_size=320):
        self.min_size = min_size
        self.max_size = max_size
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        s = int(torch.randint(self.min_size, self.max_size + 1, (1,)).item())
        return torch.stack([TF.resize(video[:, t], s) for t in range(video.shape[1])], dim=1)

class ShortSideScale:
    def __init__(self, size=256):
        self.size = size
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return torch.stack([TF.resize(video[:, t], self.size) for t in range(video.shape[1])], dim=1)

class RandomCrop:
    def __init__(self, size=224):
        self.size = size
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, H, W = video.shape
        th, tw = self.size, self.size
        if H < th or W < tw:
            # If small, pad (rare, but avoids crashing)
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            video = torch.nn.functional.pad(video, (0, pad_w, 0, pad_h))
            _, _, H, W = video.shape
        i = int(torch.randint(0, H - th + 1, (1,)).item())
        j = int(torch.randint(0, W - tw + 1, (1,)).item())
        return video[:, :, i:i+th, j:j+tw]

class CenterCrop:
    def __init__(self, size=224):
        self.size = size
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, H, W = video.shape
        th, tw = self.size, self.size
        i = max(0, (H - th) // 2)
        j = max(0, (W - tw) // 2)
        return video[:, :, i:i+th, j:j+tw]

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(video, dims=[3])  # flip width
        return video

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# -------------------------
# 0) Data Loading Roots (OneDrive approach)
# -------------------------
VIDEO_ROOT = Path(os.environ["WINTERLAB_VIDEO_ROOT"])
SPLIT_ROOT = Path(os.environ["WINTERLAB_SPLIT_ROOT"])

assert VIDEO_ROOT.exists(), f"WINTERLAB_VIDEO_ROOT does not exist: {VIDEO_ROOT}"
assert (SPLIT_ROOT / "train.csv").exists(), f"train.csv not found in WINTERLAB_SPLIT_ROOT: {SPLIT_ROOT}"
assert (SPLIT_ROOT / "val.csv").exists(), f"val.csv not found in WINTERLAB_SPLIT_ROOT: {SPLIT_ROOT}"
assert (SPLIT_ROOT / "test.csv").exists(), f"test.csv not found in WINTERLAB_SPLIT_ROOT: {SPLIT_ROOT}"


# -------------------------
# 1) Load labeled paths from split CSV
#    CSV format expected: columns ["path", "label"]
#    where "path" is RELATIVE to VIDEO_ROOT
# -------------------------
def load_labeled_video_paths(split_csv_path: Path):
    df = pd.read_csv(split_csv_path)

    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{split_csv_path} must contain columns: 'path' and 'label'")

    labeled = []
    for _, row in df.iterrows():
        rel_path = str(row["path"])
        label = int(row["label"])

        full_path = (VIDEO_ROOT / rel_path).as_posix()

        # IMPORTANT: label must be a mapping/dict for this PyTorchVideo version
        labeled.append((full_path, {"label": label}))

    return labeled

# -------------------------
# 2) Transforms (Tune later)
# -------------------------
NUM_FRAMES = 16
CROP_SIZE = 224
CLIP_DURATION = 2.0

train_transform = Compose([
    ApplyToKey("video", Compose([
        UniformTemporalSubsample(NUM_FRAMES),
        lambda v: v / 255.0,
        VideoNormalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        RandomShortSideScale(256, 320),
        RandomCrop(CROP_SIZE),
        RandomHorizontalFlip(0.5),
    ]))
])

val_transform = Compose([
    ApplyToKey("video", Compose([
        UniformTemporalSubsample(NUM_FRAMES),
        lambda v: v / 255.0,
        VideoNormalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ShortSideScale(256),
        CenterCrop(CROP_SIZE),
    ]))
])


# -------------------------
# 3) Make PyTorchVideo dataset 
#    Train uses random clip sampler, val/test use uniform sampler.
# -------------------------
def make_dataset(labeled_video_paths, split: str):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    clip_sampler = pytorchvideo.data.make_clip_sampler(
        "random" if split == "train" else "uniform",
        CLIP_DURATION
    )

    transform = train_transform if split == "train" else val_transform

    return pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths=labeled_video_paths,  # list[(video_path_str, label_int)]
        clip_sampler=clip_sampler,
        decode_audio=False,
        transform=transform,
    )


# -------------------------
# 4) Build datasets + dataloaders
# -------------------------
BATCH_SIZE = 4
NUM_WORKERS = 0  ##changed from 4 to 0 in order for me to run this

train_labeled = load_labeled_video_paths(SPLIT_ROOT / "train.csv")
val_labeled   = load_labeled_video_paths(SPLIT_ROOT / "val.csv")
test_labeled  = load_labeled_video_paths(SPLIT_ROOT / "test.csv")

train_ds = make_dataset(train_labeled, split="train")
val_ds   = make_dataset(val_labeled, split="val")
test_ds  = make_dataset(test_labeled, split="test")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# -------------------------
# 5) Sanity check one batch
# -------------------------

batch = next(iter(train_loader))

print("Batch keys:", batch.keys())
print("video shape:", batch["video"].shape)   # (B, C, T, H, W)
print("label shape:", batch["label"].shape)   # (B,)
print("labels:", batch["label"])
print("video_name (first 2):", batch["video_name"][:2])

# =========================
# Done Data Pipeline - confirm code runs sucessfully above this line 
# =========================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load pretrained X3D-S ----
# This uses torch.hub to fetch the model definition + weights.
# If this fails due to network restrictions, set pretrained=False.
model = torch.hub.load(
    "facebookresearch/pytorchvideo",
    "x3d_s",
    pretrained=True
)

num_classes = 2 #two-foot slip vs no slip
in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, num_classes) #overwrite previous classifier

# =========================
# Sanity Check forward pass to confirm that the model, data, and GPU setup work before training.
# =========================

model = model.to(device)
model.train()  # training mode

batch = next(iter(train_loader))
video = batch["video"].to(device)   # (B, C, T, H, W)
label = batch["label"].to(device)   # (B,)

with torch.no_grad():
    logits = model(video)

print("logits shape:", logits.shape)   # expect: (B, 2)
print("labels shape:", label.shape)    # expect: (B,)


# =========================
# Loss + Optimizer
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()  # mixed precision on GPU

# Training Loop - Accuracy is Clip-level here

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) #reset gradients before computing new ones

        # Mixed precision speeds up on GPU
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")): #forward pass-run clips through X3D 
            logits = model(video)
            loss = criterion(logits, label)

        if device.type == "cuda": #GPU
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: #CPU
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * video.size(0) #Update Metrics
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    return total_loss / total, correct / total

# =========================
# Validation Loop
# =========================
@torch.no_grad() #no gradients are stores
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(video)
            loss = criterion(logits, label)

        total_loss += loss.item() * video.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    return total_loss / total, correct / total

#Aggregate predictions per video, e.g.: max probability of slip across clips (good for “slip might occur briefly”) or average

@torch.no_grad()
def eval_video_level_max(model, loader, device):
    model.eval()

    # store max slip prob per video_index
    max_prob = defaultdict(lambda: 0.0)
    true_label = {}

    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].cpu()
        vid_idx = batch["video_index"].cpu().tolist()

        logits = model(video)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu()  # prob of class=1 (slip)

        for i, v in enumerate(vid_idx):
            max_prob[v] = max(max_prob[v], float(probs[i]))
            true_label[v] = int(labels[i])

    # compute accuracy at video-level using threshold 0.5
    correct = 0
    total = 0
    for v, p in max_prob.items():
        pred = 1 if p >= 0.5 else 0
        correct += (pred == true_label[v])
        total += 1

    return correct / total if total > 0 else 0.0


# =========================
# Test Loop
# =========================
@torch.no_grad()
def test_video_level_max(model, loader, device, threshold=0.5):
    model.eval()

    max_prob = defaultdict(float)
    true_label = {}

    for batch in loader:
        video = batch["video"].to(device)
        labels = batch["label"].cpu()
        vid_idx = batch["video_index"].cpu().tolist()

        logits = model(video)
        probs = F.softmax(logits, dim=1)[:, 1].cpu()  # slip prob

        for i, v in enumerate(vid_idx):
            max_prob[v] = max(max_prob[v], float(probs[i]))
            true_label[v] = int(labels[i])

    correct = 0
    total = 0

    for v, p in max_prob.items():
        pred = 1 if p >= threshold else 0
        correct += (pred == true_label[v])
        total += 1

    return correct / total if total > 0 else 0.0

# =========================
# Test Loop at a Video Level
# =========================
@torch.no_grad()
def test_clip_level(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        video = batch["video"].to(device)
        label = batch["label"].to(device)

        logits = model(video)
        loss = criterion(logits, label)

        total_loss += loss.item() * video.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    return total_loss / total, correct / total


# Run training for N epochs 

EPOCHS = 10

for epoch in range(1, EPOCHS + 1):

    # ---- Train ----
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    # ---- Clip-level validation ----
    val_loss, val_acc = eval_one_epoch(
        model, val_loader, criterion, device
    )

    # ---- Video-level validation ----
    video_val_acc = eval_video_level_max(
        model, val_loader, device
    )

    # ---- Print metrics ----
    print(
        f"Epoch {epoch:02d} | "
        f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
        f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
        f"val VIDEO acc {video_val_acc:.4f}"
    )

# ---- Final test (RUN ONCE) ----
test_loss, test_clip_acc = test_clip_level(model, test_loader, criterion, device)
test_video_acc = test_video_level_max(model, test_loader, device)

print("FINAL TEST RESULTS")
print(f"Clip-level accuracy:  {test_clip_acc:.4f}")
print(f"Video-level accuracy: {test_video_acc:.4f}")




