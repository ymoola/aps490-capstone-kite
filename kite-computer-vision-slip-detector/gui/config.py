from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


PROJECT_FILE_NAME = ".slopesense_project.json"
RECENT_PROJECTS_DIR = Path.home() / ".slopesense"
RECENT_PROJECTS_FILE = RECENT_PROJECTS_DIR / "recent.json"
MAX_RECENT = 10


@dataclass
class ProjectConfig:
    # --- Paths ---
    video_root: str = ""
    pose_output_root: str = ""
    yolo_model_path: str = ""
    ctr_gcn_repo_path: str = ""
    runs_root: str = ""
    production_output_dir: str = ""

    # --- Pose extraction ---
    pose_backend: str = "yolo"
    device: str = "cuda"
    num_gpus: int = 1
    do_interp: bool = True
    do_smooth: bool = True
    fps_scale: int = 4
    interp_mode: str = "linear"
    ema_alpha: float = 0.7
    conf_thr: float = 0.05

    # --- Dataset building ---
    fixed_t: int = 100
    k_folds: int = 5
    cv_seed: int = 12345
    val_strategy: str = "next_fold"

    # --- Training (HPO grid) ---
    epochs: int = 250
    patience: int = 25
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    learning_rates: List[float] = field(default_factory=lambda: [1e-3, 1e-2, 1e-4, 1e-5])
    weight_decays: List[float] = field(default_factory=lambda: [1e-4, 1e-5])
    num_workers: int = 4
    dropout: float = 0.4
    use_weighted_sampler: bool = True
    use_class_weighted_loss: bool = True
    best_metric: str = "val_balanced_acc"

    # --- Production ---
    production_val_ratio: float = 0.2
    production_split_seed: int = 42
    production_patience: int = 40

    # --- Model shape (fixed for COCO-17 + CTR-GCN) ---
    num_class: int = 2
    num_point: int = 17
    num_person: int = 1
    in_channels: int = 3
    graph: str = "graph.coco17.Graph"

    def project_file_path(self) -> str:
        return os.path.join(self.pose_output_root, PROJECT_FILE_NAME)

    def slopesense_dir(self) -> str:
        return os.path.join(self.pose_output_root, ".slopesense")

    def fill_defaults_from_project_root(self, project_root: str) -> None:
        """Set path fields relative to a project root if they are empty."""
        root = Path(project_root)
        if not self.pose_output_root:
            self.pose_output_root = str(root / "outputs" / "out_yolo")
        if not self.yolo_model_path:
            self.yolo_model_path = str(root / "models" / "yolo26x-pose.pt")
        if not self.ctr_gcn_repo_path:
            self.ctr_gcn_repo_path = str(root / "frameworks" / "CTR-GCN")
        if not self.runs_root:
            self.runs_root = str(root / "runs" / "ctr_gcn_kfold_hpo")
        if not self.production_output_dir:
            self.production_output_dir = str(root / "production")

    def validate(self) -> List[str]:
        """Return list of validation errors (empty = OK)."""
        errors = []
        if not self.video_root or not os.path.isdir(self.video_root):
            errors.append(f"Video root not found: {self.video_root!r}")
        if not self.yolo_model_path or not os.path.isfile(self.yolo_model_path):
            errors.append(f"YOLO model not found: {self.yolo_model_path!r}")
        if not self.ctr_gcn_repo_path or not os.path.isdir(self.ctr_gcn_repo_path):
            errors.append(f"CTR-GCN repo not found: {self.ctr_gcn_repo_path!r}")
        if self.fixed_t < 1:
            errors.append("fixed_t must be >= 1")
        if self.k_folds < 2:
            errors.append("k_folds must be >= 2")
        if not self.batch_sizes:
            errors.append("At least one batch size required")
        if not self.learning_rates:
            errors.append("At least one learning rate required")
        if not self.weight_decays:
            errors.append("At least one weight decay required")
        return errors

    # --- Serialization ---
    def save(self, path: Optional[str] = None) -> str:
        save_path = path or self.project_file_path()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        _add_recent_project(save_path)
        return save_path

    @classmethod
    def load(cls, path: str) -> "ProjectConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        _add_recent_project(path)
        return cfg


def _add_recent_project(path: str) -> None:
    RECENT_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    recents: List[str] = []
    if RECENT_PROJECTS_FILE.is_file():
        try:
            recents = json.loads(RECENT_PROJECTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            recents = []

    abs_path = os.path.abspath(path)
    recents = [p for p in recents if os.path.abspath(p) != abs_path]
    recents.insert(0, abs_path)
    recents = recents[:MAX_RECENT]

    RECENT_PROJECTS_FILE.write_text(json.dumps(recents, indent=2), encoding="utf-8")


def get_recent_projects() -> List[str]:
    if not RECENT_PROJECTS_FILE.is_file():
        return []
    try:
        return json.loads(RECENT_PROJECTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
