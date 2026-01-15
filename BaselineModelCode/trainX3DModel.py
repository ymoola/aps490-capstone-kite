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
import numpy as np
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
    log_plots: bool = True
    log_plots_every: int = 1


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


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    labels = labels.astype(np.int64)
    probs = probs.astype(np.float64)
    preds = (probs >= threshold).astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    brier = float(np.mean((probs - labels) ** 2)) if labels.size > 0 else 0.0

    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        order = np.argsort(-probs)
        labels_sorted = labels[order]
        tps = np.cumsum(labels_sorted)
        fps = np.cumsum(1 - labels_sorted)
        tpr = tps / pos
        fpr = fps / neg
        tpr = np.concatenate([[0.0], tpr, [1.0]])
        fpr = np.concatenate([[0.0], fpr, [1.0]])
        roc_auc = float(np.trapz(tpr, fpr))

        precision_curve = tps / (tps + fps + 1e-12)
        recall_curve = tps / pos
        recall_curve = np.concatenate([[0.0], recall_curve, [1.0]])
        precision_curve = np.concatenate([[1.0], precision_curve, [0.0]])
        pr_auc = float(np.trapz(precision_curve, recall_curve))

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "mcc": mcc,
        "brier": brier,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_class_weights(labeled_video_paths, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, label_dict in labeled_video_paths:
        label = int(label_dict["label"])
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label {label} out of range [0, {num_classes - 1}]")
        counts[label] += 1
    counts = counts.clamp(min=1)
    total = counts.sum().float()
    weights = total / (counts.float() * num_classes)
    return weights


def log_eval_plots(writer, labels: np.ndarray, probs: np.ndarray, epoch: int):
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc   
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if labels.size == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Validation Metrics")

    # ROC curve
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[0].set_title("ROC")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[0].legend(loc="lower right")
    else:
        axes[0].set_title("ROC (n/a)")
        axes[0].text(0.5, 0.5, "Single class", ha="center", va="center")
        axes[0].set_axis_off()

    # PR curve
    if len(np.unique(labels)) > 1:
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, label=f"AUC={pr_auc:.3f}")
        axes[1].set_title("PR")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="lower left")
    else:
        axes[1].set_title("PR (n/a)")
        axes[1].text(0.5, 0.5, "Single class", ha="center", va="center")
        axes[1].set_axis_off()

    # Confusion matrix
    preds = (probs >= 0.5).astype(np.int64)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    im = axes[2].imshow(cm, cmap="Blues")
    axes[2].set_title("Confusion")
    axes[2].set_xlabel("Pred")
    axes[2].set_ylabel("True")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    for (i, j), val in np.ndenumerate(cm):
        axes[2].text(j, i, str(val), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    writer.add_figure("val/curves", fig, epoch)
    plt.close(fig)


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
    all_labels = []
    all_probs = []

    for batch in loader:
        with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
            clip_logits, labels = forward_batch(model, batch, device)
            video_logits = mil_pool(clip_logits, cfg.mil_pool)
            loss = criterion(video_logits, labels)

        loss_sum += loss.item() * labels.size(0)
        preds = video_logits.argmax(dim=1)
        video_correct += (preds == labels).sum().item()
        video_total += labels.size(0)
        probs = torch.softmax(video_logits, dim=1)[:, 1]
        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

    avg_loss = loss_sum / video_total if video_total else 0.0
    video_acc = video_correct / video_total if video_total else 0.0
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    probs_np = torch.cat(all_probs).numpy() if all_probs else np.array([])
    metrics = compute_binary_metrics(labels_np, probs_np)
    return avg_loss, video_acc, metrics, labels_np, probs_np


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
    class_weights = compute_class_weights(train_labeled, cfg.num_classes).to(device)
    print("Class weights:", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        val_loss, val_video_acc, val_metrics, val_labels, val_probs = eval_one_epoch(
            model, val_loader, criterion, device, cfg, amp_device, amp_enabled
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
        eval_time = time.perf_counter() - eval_start
        epoch_time = train_time + eval_time

        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc/video", {"train": train_video_acc, "val": val_video_acc}, epoch)
        writer.add_scalar("time/train_sec", train_time, epoch)
        writer.add_scalar("time/val_sec", eval_time, epoch)
        writer.add_scalar("time/epoch_sec", epoch_time, epoch)
        writer.add_scalar("val/precision", val_metrics["precision"], epoch)
        writer.add_scalar("val/recall", val_metrics["recall"], epoch)
        writer.add_scalar("val/specificity", val_metrics["specificity"], epoch)
        writer.add_scalar("val/f1", val_metrics["f1"], epoch)
        writer.add_scalar("val/balanced_acc", val_metrics["balanced_acc"], epoch)
        writer.add_scalar("val/mcc", val_metrics["mcc"], epoch)
        writer.add_scalar("val/brier", val_metrics["brier"], epoch)
        if not np.isnan(val_metrics["roc_auc"]):
            writer.add_scalar("val/roc_auc", val_metrics["roc_auc"], epoch)
        if not np.isnan(val_metrics["pr_auc"]):
            writer.add_scalar("val/pr_auc", val_metrics["pr_auc"], epoch)
        writer.add_scalar("val/confusion/tp", val_metrics["tp"], epoch)
        writer.add_scalar("val/confusion/tn", val_metrics["tn"], epoch)
        writer.add_scalar("val/confusion/fp", val_metrics["fp"], epoch)
        writer.add_scalar("val/confusion/fn", val_metrics["fn"], epoch)
        if cfg.log_plots and (epoch % cfg.log_plots_every == 0):
            log_eval_plots(writer, val_labels, val_probs, epoch)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_loss:.4f} VIDEO acc {train_video_acc:.4f} | "
            f"Val loss {val_loss:.4f} VIDEO acc {val_video_acc:.4f} F1 {val_metrics['f1']:.4f} | "
            f"Time train {train_time:.1f}s eval {eval_time:.1f}s total {epoch_time:.1f}s"
        )

        metrics = {
            "train_loss": train_loss,
            "train_video_acc": train_video_acc,
            "val_loss": val_loss,
            "val_video_acc": val_video_acc,
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_mcc": val_metrics["mcc"],
            "val_brier": val_metrics["brier"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_tp": val_metrics["tp"],
            "val_tn": val_metrics["tn"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
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
