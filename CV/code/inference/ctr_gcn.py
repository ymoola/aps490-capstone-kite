# inference/ctr_gcn.py
from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# -----------------------------
# Dataset
# -----------------------------
class PoseNPZDataset(Dataset):
    """
    Expects NPZ with:
      - data:   (N, C, T, V, M) float32
      - labels: (N,) int64  (0=pass, 1=fail) by your builder
      - meta:   (N,) object (optional)
    """
    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"Dataset not found: {npz_path}")

        z = np.load(npz_path, allow_pickle=True)
        self.data = z["data"].astype(np.float32, copy=False)
        self.labels = z["labels"].astype(np.int64, copy=False)

        self.meta = None
        if "meta" in z.files:
            self.meta = z["meta"]  # object array

        # Basic sanity
        if self.data.ndim != 5:
            raise RuntimeError(f"Expected data ndim=5 (N,C,T,V,M), got {self.data.shape} in {npz_path}")
        if self.labels.ndim != 1:
            raise RuntimeError(f"Expected labels ndim=1, got {self.labels.shape} in {npz_path}")
        if self.data.shape[0] != self.labels.shape[0]:
            raise RuntimeError(f"N mismatch: data N={self.data.shape[0]} labels N={self.labels.shape[0]}")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.data[idx])           # (C,T,V,M)
        y = torch.tensor(int(self.labels[idx]))        # scalar
        return x, y

    def class_counts(self) -> Dict[int, int]:
        vals, counts = np.unique(self.labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    def get_meta(self, idx: int) -> Optional[dict]:
        if self.meta is None:
            return None
        m = self.meta[idx]
        # builder saved meta_list as python dict objects
        if isinstance(m, dict):
            return m
        # sometimes becomes 0-d object array item
        try:
            if hasattr(m, "item"):
                mm = m.item()
                if isinstance(mm, dict):
                    return mm
        except Exception:
            pass
        return None


# -----------------------------
# CTR-GCN model loader (robust)
# -----------------------------
def _try_import_ctr_gcn_model(ctr_repo_root: str):
    """
    Tries a few common module paths used by CTR-GCN forks.

    Returns: ModelClass
    Raises: ImportError with helpful message if not found.
    """
    ctr_repo_root = os.path.abspath(os.path.expanduser(ctr_repo_root))
    if not os.path.isdir(ctr_repo_root):
        raise FileNotFoundError(f"CTR-GCN repo root not found: {ctr_repo_root}")

    # Ensure repo is importable
    if ctr_repo_root not in sys.path:
        sys.path.insert(0, ctr_repo_root)

    tried: List[str] = []

    # Common paths across forks:
    candidates = [
        # official-ish
        ("model.ctrgcn", "Model"),
        ("model.ctr_gcn", "Model"),
        ("model.ctr_gcn", "CTRGCN"),
        ("model.ctrgcn", "CTRGCN"),
        # other layouts
        ("net.ctrgcn", "Model"),
        ("net.ctr_gcn", "Model"),
        ("models.ctrgcn", "Model"),
        ("models.ctr_gcn", "Model"),
    ]

    for mod_name, cls_name in candidates:
        tried.append(f"{mod_name}:{cls_name}")
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            if hasattr(mod, cls_name):
                return getattr(mod, cls_name)
        except Exception:
            continue

    raise ImportError(
        "Could not import CTR-GCN Model class.\n"
        f"Repo root: {ctr_repo_root}\n"
        "Tried:\n  - " + "\n  - ".join(tried) + "\n\n"
        "Fix options:\n"
        "  1) Open CV/frameworks/CTR-GCN and find the model definition file.\n"
        "  2) Tell me the file path + class name, and we’ll set it explicitly.\n"
    )


def build_ctr_gcn_model(
    ctr_repo_root: str,
    num_class: int,
    num_point: int,
    num_person: int,
    in_channels: int,
    graph: str = "graph.ntu_rgb_d.Graph",
    graph_args: Optional[dict] = None,
    drop_out: float = 0.0,
) -> nn.Module:
    """
    Builds the CTR-GCN Model using the repo’s Model class.

    Important:
    - Many CTR-GCN repos default graph to NTU (25 joints). You have COCO 17.
    - We will likely need a COCO graph later. For now, we allow overriding `graph`.
    """
    ModelClass = _try_import_ctr_gcn_model(ctr_repo_root)

    kwargs = dict(
        num_class=num_class,
        num_point=num_point,
        num_person=num_person,
        in_channels=in_channels,
        drop_out=drop_out,
        graph=graph,
        graph_args=graph_args or {},
    )

    try:
        model = ModelClass(**kwargs)
    except TypeError as e:
        # Some forks use different arg names; fail with a useful error.
        raise TypeError(
            "CTR-GCN Model init signature mismatch.\n"
            f"Tried kwargs={kwargs}\n"
            f"Error: {e}\n\n"
            "Next step: open the CTR-GCN Model class __init__ signature and we’ll map args.\n"
        )

    return model


# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    epochs: int = 50
    batch_size: int = 16
    num_workers: int = 4

    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30

    # imbalance handling
    use_weighted_sampler: bool = True
    use_class_weighted_loss: bool = True

    # logging/checkpointing
    out_dir: str = "runs/ctr_gcn"
    save_best: bool = True
    best_metric: str = "val_balanced_acc"  # or "val_acc"
    


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def compute_binary_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    logits: (N,2) or (N,) if binary-logit (we assume (N,2) here)
    y: (N,)
    """
    if logits.ndim == 1:
        # treat >0 as class 1
        pred = (logits > 0).long()
    else:
        pred = torch.argmax(logits, dim=1)

    y = y.long()
    correct = (pred == y).sum().item()
    acc = correct / max(1, y.numel())

    # confusion
    # 0=pass, 1=fail
    tp = ((pred == 1) & (y == 1)).sum().item()
    tn = ((pred == 0) & (y == 0)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()

    # avoid div0
    tpr = tp / max(1, (tp + fn))  # recall for fail
    tnr = tn / max(1, (tn + fp))  # recall for pass
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "acc": float(acc),
        "balanced_acc": float(bal_acc),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tpr_fail": float(tpr),
        "tnr_pass": float(tnr),
    }


def _make_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    returns weight tensor for CrossEntropyLoss of shape (2,)
    weight[c] inversely proportional to count of class c.
    """
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    # inverse frequency
    weights = 1.0 / np.clip(counts, 1.0, None)
    # normalize to mean=1 (optional)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def make_train_loader(
    ds: PoseNPZDataset,
    cfg: TrainConfig,
) -> DataLoader:
    if cfg.use_weighted_sampler:
        labels = ds.labels
        counts = np.bincount(labels, minlength=2).astype(np.float64)
        # sample weight per item: inverse of class frequency
        w_per_class = 1.0 / np.clip(counts, 1.0, None)
        sample_w = w_per_class[labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_w).double(),
            num_samples=len(labels),
            replacement=True,
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_eval_loader(ds: PoseNPZDataset, cfg: TrainConfig) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


# -----------------------------
# Train / Eval loops
# -----------------------------
def run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)  # (B,C,T,V,M)
        y = y.to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # expect (B, num_class)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    mets = compute_binary_metrics_from_logits(logits_cat, y_cat)

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        **mets,
    }


@torch.no_grad()
def run_one_epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    mets = compute_binary_metrics_from_logits(logits_cat, y_cat)

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        **mets,
    }


# -----------------------------
# High-level API: train/validate/test
# -----------------------------
def train_validate_test(
    *,
    ctr_repo_root: str,
    train_npz: str,
    val_npz: str,
    test_npz: str,
    model_kwargs: dict,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # datasets
    ds_train = PoseNPZDataset(train_npz)
    ds_val = PoseNPZDataset(val_npz)
    ds_test = PoseNPZDataset(test_npz)

    train_counts = ds_train.class_counts()
    val_counts = ds_val.class_counts()
    test_counts = ds_test.class_counts()

    # loaders (balance emphasis)
    train_loader = make_train_loader(ds_train, cfg)
    val_loader = make_eval_loader(ds_val, cfg)
    test_loader = make_eval_loader(ds_test, cfg)

    # model
    model = build_ctr_gcn_model(ctr_repo_root=ctr_repo_root, **model_kwargs)
    model = model.to(device)
    print(f"[ctr_gcn] Model device: {next(model.parameters()).device}")

    # loss
    if cfg.use_class_weighted_loss:
        w = _make_class_weights(ds_train.labels).to(device)  # (2,)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # training
    best_score = -1e9
    best_path = os.path.join(cfg.out_dir, "best.pt")
    history_path = os.path.join(cfg.out_dir, "history.json")

    history: List[dict] = []
    t0 = time.time()
    epochs_no_improve = 0
    for epoch in range(1, cfg.epochs + 1):
        tr = run_one_epoch_train(model, train_loader, optimizer, criterion, device)
        va = run_one_epoch_eval(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "train_balanced_acc": tr["balanced_acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "val_balanced_acc": va["balanced_acc"],
        }
        row.update({
            "val_tp": int(va.get("tp", -1)),
            "val_tn": int(va.get("tn", -1)),
            "val_fp": int(va.get("fp", -1)),
            "val_fn": int(va.get("fn", -1)),
        })
        history.append(row)

        # NEW: persist history each epoch so you can plot while training
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_kwargs": model_kwargs,
                    "train_config": cfg.__dict__,
                    "history": history,
                },
                f,
                indent=2,
            )

        # choose best
        score = row.get(cfg.best_metric, row["val_balanced_acc"])
        if cfg.save_best and score > best_score:
            best_score = score
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "train_config": cfg.__dict__,
                    "epoch": epoch,
                    "best_score": best_score,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            
        print(
            f"[ctr_gcn] epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.3f} bal {tr['balanced_acc']:.3f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.3f} bal {va['balanced_acc']:.3f}"
        )

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break



    # load best and test
    if cfg.save_best and os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    te = run_one_epoch_eval(model, test_loader, criterion, device)

    # write summary
    summary = {
        "paths": {
            "train_npz": train_npz,
            "val_npz": val_npz,
            "test_npz": test_npz,
            "ctr_repo_root": os.path.abspath(os.path.expanduser(ctr_repo_root)),
            "best_ckpt": best_path if os.path.isfile(best_path) else None,
        },
        "class_counts": {
            "train": train_counts,
            "val": val_counts,
            "test": test_counts,
        },
        "final_test": te,
        "history": history,
        "elapsed_sec": time.time() - t0,
        "notes": {
            "imbalance_handling": {
                "use_weighted_sampler": cfg.use_weighted_sampler,
                "use_class_weighted_loss": cfg.use_class_weighted_loss,
            }
        },
    }

    with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[ctr_gcn] TEST | loss {te['loss']:.4f} acc {te['acc']:.3f} bal {te['balanced_acc']:.3f} "
        f"(tp={int(te['tp'])} tn={int(te['tn'])} fp={int(te['fp'])} fn={int(te['fn'])})"
    )
    print(f"[ctr_gcn] Wrote summary: {os.path.join(cfg.out_dir, 'summary.json')}")

    return summary
