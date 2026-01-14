# =========================
# X3D test-only evaluation (video-level MIL)
# Requires env vars:
#   WINTERLAB_VIDEO_ROOT -> .../videos_renamed (contains date folders)
#   WINTERLAB_SPLIT_ROOT -> .../BaselineDataset (contains test.csv)
# =========================

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


class Div255:
    """
    Scales the video tensor from [0, 255] to [0, 1].
    """
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return video / 255.0


class VideoNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        frames = [TF.normalize(video[:, t], self.mean, self.std) for t in range(video.shape[1])]
        return torch.stack(frames, dim=1)


class ShortSideScale:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return torch.stack([TF.resize(video[:, t], self.size) for t in range(video.shape[1])], dim=1)


class CenterCrop:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        _, _, H, W = video.shape
        th, tw = self.size, self.size
        i = max(0, (H - th) // 2)
        j = max(0, (W - tw) // 2)
        return video[:, :, i:i + th, j:j + tw]


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
class EvalConfig:
    num_frames: int = 16
    crop_size: int = 224
    clip_duration: float = 2.0
    clips_per_video: int = 8
    batch_size: int = 4
    num_workers: int = 2
    num_classes: int = 2
    mil_pool: str = "logsumexp"
    decoder: str = "pyav"
    seed: int = 1337


# =========================
# Data
# =========================

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


def build_transforms(cfg: EvalConfig):
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
    def __init__(self, labeled_video_paths, clip_duration, num_clips, transform, decoder: str):
        self.labeled_video_paths = labeled_video_paths
        self.clip_duration = clip_duration
        self.num_clips = num_clips
        self.transform = transform
        self.decoder = decoder

    def __len__(self):
        return len(self.labeled_video_paths)

    def _sample_starts(self, duration: float):
        if duration <= self.clip_duration:
            return [0.0] * self.num_clips

        max_start = duration - self.clip_duration
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
            "video_name": Path(path).name,
        }


# =========================
# Model + Metrics
# =========================

def build_model(cfg: EvalConfig):
    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "x3d_s",
        pretrained=False,
    )
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, cfg.num_classes)
    if hasattr(model.blocks[-1], "activation"):
        model.blocks[-1].activation = nn.Identity()
    return model


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


def log_eval_plots(writer, labels: np.ndarray, probs: np.ndarray, step: int):
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
    fig.suptitle("Test Metrics")

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
    writer.add_figure("test/curves", fig, step)
    plt.close(fig)


def forward_batch(model, batch, device):
    non_blocking = device.type == "cuda"
    video = batch["video"].to(device, non_blocking=non_blocking)
    labels = batch["label"].to(device, non_blocking=non_blocking)
    b, k, c, t, h, w = video.shape
    clip_logits = model(video.flatten(0, 1))
    return clip_logits.view(b, k, -1), labels


@torch.no_grad()
def eval_test(model, loader, criterion, device, mil_method: str, amp_device, amp_enabled):
    model.eval()
    loss_sum = 0.0
    total = 0
    all_labels = []
    all_probs = []

    for batch in loader:
        with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
            clip_logits, labels = forward_batch(model, batch, device)
            video_logits = mil_pool(clip_logits, mil_method)
            loss = criterion(video_logits, labels)

        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
        probs = torch.softmax(video_logits, dim=1)[:, 1]
        all_labels.append(labels.detach().cpu())
        all_probs.append(probs.detach().cpu())

    avg_loss = loss_sum / total if total else 0.0
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    probs_np = torch.cat(all_probs).numpy() if all_probs else np.array([])
    metrics = compute_binary_metrics(labels_np, probs_np)
    return avg_loss, metrics, labels_np, probs_np


# =========================
# Main
# =========================

def load_config_from_checkpoint(path: Path) -> EvalConfig:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    cfg = EvalConfig()
    for key, value in cfg_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Evaluate X3D on test split.")
    parser.add_argument("--checkpoint", required=True, help="Path to best checkpoint (.pth)")
    parser.add_argument("--log-dir", default="runs/x3d_test", help="TensorBoard log directory")
    args = parser.parse_args()

    video_root = Path(os.environ["WINTERLAB_VIDEO_ROOT"])
    split_root = Path(os.environ["WINTERLAB_SPLIT_ROOT"])

    assert video_root.exists(), f"WINTERLAB_VIDEO_ROOT does not exist: {video_root}"
    assert (split_root / "test.csv").exists(), f"test.csv not found in WINTERLAB_SPLIT_ROOT: {split_root}"

    cfg = load_config_from_checkpoint(Path(args.checkpoint))
    set_seed(cfg.seed)

    test_labeled = load_labeled_video_paths(split_root / "test.csv", video_root)
    test_ds = VideoBagDataset(
        test_labeled,
        clip_duration=cfg.clip_duration,
        num_clips=cfg.clips_per_video,
        transform=build_transforms(cfg),
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
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if amp_enabled else "cpu"

    test_loss, metrics, labels_np, probs_np = eval_test(
        model, test_loader, criterion, device, cfg.mil_pool, amp_device, amp_enabled
    )

    writer = SummaryWriter(log_dir=args.log_dir)
    writer.add_scalar("test/loss", test_loss, 0)
    writer.add_scalar("test/acc", metrics["acc"], 0)
    writer.add_scalar("test/precision", metrics["precision"], 0)
    writer.add_scalar("test/recall", metrics["recall"], 0)
    writer.add_scalar("test/specificity", metrics["specificity"], 0)
    writer.add_scalar("test/f1", metrics["f1"], 0)
    writer.add_scalar("test/balanced_acc", metrics["balanced_acc"], 0)
    writer.add_scalar("test/mcc", metrics["mcc"], 0)
    writer.add_scalar("test/brier", metrics["brier"], 0)
    if not np.isnan(metrics["roc_auc"]):
        writer.add_scalar("test/roc_auc", metrics["roc_auc"], 0)
    if not np.isnan(metrics["pr_auc"]):
        writer.add_scalar("test/pr_auc", metrics["pr_auc"], 0)
    writer.add_scalar("test/confusion/tp", metrics["tp"], 0)
    writer.add_scalar("test/confusion/tn", metrics["tn"], 0)
    writer.add_scalar("test/confusion/fp", metrics["fp"], 0)
    writer.add_scalar("test/confusion/fn", metrics["fn"], 0)
    log_eval_plots(writer, labels_np, probs_np, 0)
    writer.close()

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc: {metrics['acc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Balanced acc: {metrics['balanced_acc']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"Brier: {metrics['brier']:.4f}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    if not np.isnan(metrics["pr_auc"]):
        print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"Confusion (TP, TN, FP, FN): {metrics['tp']}, {metrics['tn']}, {metrics['fp']}, {metrics['fn']}")


if __name__ == "__main__":
    main()
