# =========================
# PyTorchVideo + X3D
# Video-level classification with multi-instance learning (MIL)
# Requires env vars:
# WINTERLAB_VIDEO_ROOT -> .../videos_renamed (contains date folders)
# WINTERLAB_SPLIT_ROOT -> .../BaselineDataset (contains train.csv/val.csv)
# =========================

import os
import random
import time
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms.functional as TF
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.encoded_video_pyav import EncodedVideoPyAV

# Load .env file
load_dotenv()


# =========================
# Transforms
# =========================

class ApplyToKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, x):
        x[self.key] = self.transform(x[self.key])
        return x


class UniformTemporalSubsample:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, T, _, _ = video.shape
        if T <= self.num_samples:
            idx = torch.arange(T)
            if T < self.num_samples:
                pad = idx.new_full((self.num_samples - T,), T - 1)
                idx = torch.cat([idx, pad], dim=0)
            return video[:, idx, :, :]
        idx = torch.linspace(0, T - 1, self.num_samples).long()
        return video[:, idx, :, :]


class VideoNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        frames = [TF.normalize(video[:, t], self.mean, self.std) for t in range(video.shape[1])]
        return torch.stack(frames, dim=1)


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
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)
            video = torch.nn.functional.pad(video, (0, pad_w, 0, pad_h))
            _, _, H, W = video.shape
        i = int(torch.randint(0, H - th + 1, (1,)).item())
        j = int(torch.randint(0, W - tw + 1, (1,)).item())
        return video[:, :, i:i + th, j:j + tw]


class CenterCrop:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, H, W = video.shape
        th, tw = self.size, self.size
        i = max(0, (H - th) // 2)
        j = max(0, (W - tw) // 2)
        return video[:, :, i:i + th, j:j + tw]


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(video, dims=[3])
        return video


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# =========================
# Config
# =========================

@dataclass
class TrainConfig:
    num_frames: int = 16
    crop_size: int = 224
    clip_duration: float = 2.0
    clips_per_video: int = 8
    batch_size: int = 4
    num_workers: int = 2
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_classes: int = 2
    pretrained: bool = True
    mil_pool: str = "logsumexp" # max | mean | logsumexp
    decoder: str = "pyav"
    seed: int = 1337
    log_dir: str = "runs/x3d_slip"
    save_dir: str = "."
    run_sanity_check: bool = False


# =========================
# Data
# =========================

class Div255:
    """
    Scales the video tensor from [0, 255] to [0, 1].
    """
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video / 255.0

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labeled_video_paths(split_csv_path: Path, video_root: Path):
    df = pd.read_csv(split_csv_path)
    if "path" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{split_csv_path} must contain columns: 'path' and 'label'")

    labeled = []
    for _, row in df.iterrows():
        rel_path = str(row["path"])
        label = int(row["label"])
        full_path = (video_root / rel_path).as_posix()
        labeled.append((full_path, {"label": label}))
    return labeled


def build_transforms(cfg: TrainConfig, split: str):
    if split == "train":
        return Compose([
            ApplyToKey("video", Compose([
                UniformTemporalSubsample(cfg.num_frames),
                Div255(),
                VideoNormalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(256, 320),
                RandomCrop(cfg.crop_size),
                RandomHorizontalFlip(0.5),
            ]))
        ])
    return Compose([
        ApplyToKey("video", Compose([
            UniformTemporalSubsample(cfg.num_frames),
            Div255(),
            VideoNormalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ShortSideScale(256),
            CenterCrop(cfg.crop_size),
        ]))
    ])


def open_encoded_video(path: str, decode_audio: bool, decoder: str):
    if decoder == "pyav":
        try:
            return EncodedVideoPyAV(path, Path(path).name, decode_audio=decode_audio)
        except TimeoutError as exc:
            raise RuntimeError(
                "Timed out reading video. If your videos are in OneDrive, "
                "ensure they are fully downloaded or copy them to a local folder."
            ) from exc
        except Exception:
            return EncodedVideo.from_path(path, decode_audio=decode_audio, decoder=decoder)
    return EncodedVideo.from_path(path, decode_audio=decode_audio, decoder=decoder)


class VideoBagDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_video_paths, clip_duration, num_clips, transform, mode: str, decoder: str):
        self.labeled_video_paths = labeled_video_paths
        self.clip_duration = clip_duration
        self.num_clips = num_clips
        self.transform = transform
        self.mode = mode
        self.decoder = decoder

    def __len__(self):
        return len(self.labeled_video_paths)

    def _sample_starts(self, duration: float):
        if duration <= self.clip_duration:
            return [0.0] * self.num_clips

        max_start = duration - self.clip_duration
        if self.mode == "train":
            return [random.uniform(0.0, max_start) for _ in range(self.num_clips)]

        if self.num_clips == 1:
            return [max_start / 2.0]

        step = max_start / (self.num_clips - 1)
        return [i * step for i in range(self.num_clips)]

    def __getitem__(self, index):
        path, label_dict = self.labeled_video_paths[index]
        label = int(label_dict["label"])

        video = open_encoded_video(path, decode_audio=False, decoder=self.decoder)
        duration = float(video.duration)
        starts = self._sample_starts(duration)

        clips = []
        for start in starts:
            clip = video.get_clip(start, start + self.clip_duration)
            if clip is None or clip.get("video") is None:
                continue
            clip_video = clip["video"]
            if self.transform is not None:
                clip_video = self.transform({"video": clip_video})["video"]
            clips.append(clip_video)

        if not clips:
            raise RuntimeError(f"Failed to decode clips for {path}")

        while len(clips) < self.num_clips:
            clips.append(clips[-1])

        if len(clips) > self.num_clips:
            clips = clips[:self.num_clips]

        return {
            "video": torch.stack(clips, dim=0),
            "label": torch.tensor(label, dtype=torch.long),
            "video_index": torch.tensor(index, dtype=torch.long),
            "video_name": Path(path).name,
        }


# =========================
# Model
# =========================

def build_model(cfg: TrainConfig):
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "x3d_s",
        pretrained=cfg.pretrained,
    )

    # Freeze entire backbone
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier head
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, cfg.num_classes)

    # Unfreeze classifier head
    for p in model.blocks[-1].proj.parameters():
        p.requires_grad = True

    # Remove activation if present
    if hasattr(model.blocks[-1], "activation"):
        model.blocks[-1].activation = nn.Identity()

    return model




# =========================
# MIL + Training
# =========================

def mil_pool(clip_logits: torch.Tensor, method: str) -> torch.Tensor:
    if method == "max":
        return clip_logits.max(dim=1).values
    if method == "mean":
        return clip_logits.mean(dim=1)
    if method == "logsumexp":
        return torch.logsumexp(clip_logits, dim=1) - math.log(clip_logits.size(1))
    raise ValueError(f"Unsupported MIL pool: {method}")


def forward_batch(model, batch, device):
    non_blocking = device.type == "cuda"
    video = batch["video"].to(device, non_blocking=non_blocking)
    labels = batch["label"].to(device, non_blocking=non_blocking)
    b, k, c, t, h, w = video.shape
    clip_logits = model(video.flatten(0, 1))
    return clip_logits.view(b, k, -1), labels


def train_one_epoch(model, loader, optimizer, criterion, device, cfg: TrainConfig, scaler, amp_device, amp_enabled):
    model.train()
    loss_sum = 0.0
    video_correct = 0
    video_total = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
            clip_logits, labels = forward_batch(model, batch, device)
            video_logits = mil_pool(clip_logits, cfg.mil_pool)
            loss = criterion(video_logits, labels)

        if device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        preds = video_logits.argmax(dim=1)
        video_correct += (preds == labels).sum().item()
        video_total += labels.size(0)

    avg_loss = loss_sum / video_total if video_total else 0.0
    video_acc = video_correct / video_total if video_total else 0.0
    return avg_loss, video_acc


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, cfg: TrainConfig, amp_device, amp_enabled):
    model.eval()
    loss_sum = 0.0
    video_correct = 0
    video_total = 0

    for batch in loader:
        with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
            clip_logits, labels = forward_batch(model, batch, device)
            video_logits = mil_pool(clip_logits, cfg.mil_pool)
            loss = criterion(video_logits, labels)

        loss_sum += loss.item() * labels.size(0)
        preds = video_logits.argmax(dim=1)
        video_correct += (preds == labels).sum().item()
        video_total += labels.size(0)

    avg_loss = loss_sum / video_total if video_total else 0.0
    video_acc = video_correct / video_total if video_total else 0.0
    return avg_loss, video_acc


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, cfg: TrainConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg.__dict__,
        },
        path,
    )


# =========================
# Main
# =========================

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    video_root = Path(os.environ["WINTERLAB_VIDEO_ROOT"])
    split_root = Path(os.environ["WINTERLAB_SPLIT_ROOT"])

    assert video_root.exists(), f"WINTERLAB_VIDEO_ROOT does not exist: {video_root}"
    assert (split_root / "train.csv").exists(), f"train.csv not found in WINTERLAB_SPLIT_ROOT: {split_root}"
    assert (split_root / "val.csv").exists(), f"val.csv not found in WINTERLAB_SPLIT_ROOT: {split_root}"

    train_labeled = load_labeled_video_paths(split_root / "train.csv", video_root)
    val_labeled = load_labeled_video_paths(split_root / "val.csv", video_root)

    print("Collected data")
    train_ds = VideoBagDataset(
        train_labeled,
        clip_duration=cfg.clip_duration,
        num_clips=cfg.clips_per_video,
        transform=build_transforms(cfg, "train"),
        mode="train",
        decoder=cfg.decoder,
    )
    val_ds = VideoBagDataset(
        val_labeled,
        clip_duration=cfg.clip_duration,
        num_clips=cfg.clips_per_video,
        transform=build_transforms(cfg, "val"),
        mode="val",
        decoder=cfg.decoder,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device found. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found. Using GPU.")
    else:
        device = torch.device("cpu")
        print("No GPU found. Using CPU.")

    pin_memory = device.type == "cuda"
    generator = torch.Generator().manual_seed(cfg.seed)

    print("Performing data loader...")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    if cfg.run_sanity_check:
        batch = next(iter(train_loader))
        print("Batch video shape:", batch["video"].shape)
        print("Batch label shape:", batch["label"].shape)
        print("Video name (first):", batch["video_name"][0])

    print("building model...")
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if amp_enabled else "cpu"

    writer = SummaryWriter(log_dir=cfg.log_dir)

    best_val_acc = -1.0
    start_time = time.perf_counter()

    print("starting training...")

    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

        train_start = time.perf_counter()
        train_loss, train_video_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg, scaler, amp_device, amp_enabled
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        train_time = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        val_loss, val_video_acc = eval_one_epoch(
            model, val_loader, criterion, device, cfg, amp_device, amp_enabled
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        eval_time = time.perf_counter() - eval_start
        epoch_time = train_time + eval_time

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train_video", train_video_acc, epoch)
        writer.add_scalar("acc/val_video", val_video_acc, epoch)
        writer.add_scalar("time/train_sec", train_time, epoch)
        writer.add_scalar("time/val_sec", eval_time, epoch)
        writer.add_scalar("time/epoch_sec", epoch_time, epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f} VIDEO acc {train_video_acc:.4f} | "
            f"Val loss {val_loss:.4f} VIDEO acc {val_video_acc:.4f} | "
            f"Time train {train_time:.1f}s eval {eval_time:.1f}s total {epoch_time:.1f}s"
        )

        metrics = {
            "train_loss": train_loss,
            "train_video_acc": train_video_acc,
            "val_loss": val_loss,
            "val_video_acc": val_video_acc,
        }

        last_path = Path(cfg.save_dir) / "x3d_last.pth"
        save_checkpoint(last_path, model, optimizer, epoch, metrics, cfg)

        if val_video_acc > best_val_acc:
            best_val_acc = val_video_acc
            best_path = Path(cfg.save_dir) / "x3d_best.pth"
            save_checkpoint(best_path, model, optimizer, epoch, metrics, cfg)

    total_time = time.perf_counter() - start_time
    writer.add_scalar("time/total_sec", total_time, cfg.epochs)
    writer.close()

    print(f"Training complete in {total_time:.1f}s")


if __name__ == "__main__":
    main()