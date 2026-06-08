"""
Checkpoint / restore system.

Persists pipeline state to hidden .slopesense/ directory so the app
can resume after interruption and detect new/changed inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


STAGE_NAMES = [
    "pose_extraction",
    "dataset_building",
    "hpo_training",
    "production_training",
]

STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


@dataclass
class VideoStatus:
    video_path: str
    status: str = STATUS_NOT_STARTED  # not_started | completed | failed
    output_npz: str = ""
    error: str = ""
    config_hash: str = ""


@dataclass
class StageState:
    pose_extraction: str = STATUS_NOT_STARTED
    dataset_building: str = STATUS_NOT_STARTED
    hpo_training: str = STATUS_NOT_STARTED
    production_training: str = STATUS_NOT_STARTED


@dataclass
class HPORunStatus:
    run_name: str
    fold: int
    batch_size: int
    lr: float
    weight_decay: float
    status: str = "pending"  # pending | completed | failed


class CheckpointManager:
    """Manages pipeline state in <pose_output_root>/.slopesense/"""

    def __init__(self, pose_output_root: str):
        self._root = os.path.join(pose_output_root, ".slopesense")
        self._stage_state = StageState()
        self._pose_manifest: Dict[str, VideoStatus] = {}
        self._hpo_grid: Dict[str, HPORunStatus] = {}

    @property
    def slopesense_dir(self) -> str:
        return self._root

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def ensure_dirs(self) -> None:
        os.makedirs(self._root, exist_ok=True)

    def load(self) -> None:
        """Load all checkpoint files if they exist."""
        self.ensure_dirs()
        self._load_stage_state()
        self._load_pose_manifest()
        self._load_hpo_grid()

    # ------------------------------------------------------------------
    # Stage state
    # ------------------------------------------------------------------
    @property
    def stage_state(self) -> StageState:
        return self._stage_state

    def set_stage_status(self, stage: str, status: str) -> None:
        if hasattr(self._stage_state, stage):
            setattr(self._stage_state, stage, status)
            self._save_stage_state()

    def _load_stage_state(self) -> None:
        path = os.path.join(self._root, "stage_state.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._stage_state = StageState(**{
                k: v for k, v in data.items() if k in StageState.__dataclass_fields__
            })

    def _save_stage_state(self) -> None:
        self.ensure_dirs()
        path = os.path.join(self._root, "stage_state.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._stage_state), f, indent=2)

    # ------------------------------------------------------------------
    # Pose extraction manifest
    # ------------------------------------------------------------------
    @property
    def pose_manifest(self) -> Dict[str, VideoStatus]:
        return self._pose_manifest

    def get_video_status(self, video_path: str) -> Optional[VideoStatus]:
        key = os.path.normpath(video_path)
        return self._pose_manifest.get(key)

    def set_video_status(
        self,
        video_path: str,
        status: str,
        output_npz: str = "",
        error: str = "",
        config_hash: str = "",
    ) -> None:
        key = os.path.normpath(video_path)
        self._pose_manifest[key] = VideoStatus(
            video_path=video_path,
            status=status,
            output_npz=output_npz,
            error=error,
            config_hash=config_hash,
        )

    def save_pose_manifest(self) -> None:
        self.ensure_dirs()
        path = os.path.join(self._root, "pose_manifest.json")
        data = {k: asdict(v) for k, v in self._pose_manifest.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_pose_manifest(self) -> None:
        path = os.path.join(self._root, "pose_manifest.json")
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._pose_manifest = {}
        for key, val in data.items():
            self._pose_manifest[key] = VideoStatus(**{
                k: v for k, v in val.items() if k in VideoStatus.__dataclass_fields__
            })

    def detect_new_videos(self, all_video_paths: List[str]) -> List[str]:
        """Return video paths not in the manifest (new files)."""
        return [
            p for p in all_video_paths
            if os.path.normpath(p) not in self._pose_manifest
        ]

    def detect_stale_videos(self, current_config_hash: str) -> List[str]:
        """Return video paths whose config_hash doesn't match current config."""
        return [
            vs.video_path for vs in self._pose_manifest.values()
            if vs.status == STATUS_COMPLETED and vs.config_hash != current_config_hash
        ]

    def summarize_video_sync(
        self,
        all_video_paths: List[str],
        current_config_hash: str,
    ) -> Dict[str, object]:
        """Compare current videos against the saved manifest."""
        normalized_paths = [os.path.normpath(os.path.abspath(p)) for p in all_video_paths]
        current_set = set(normalized_paths)

        new_videos: List[str] = []
        stale_videos: List[str] = []
        completed_videos: List[str] = []
        failed_videos: List[str] = []
        pending_videos: List[str] = []

        for path in normalized_paths:
            vs = self._pose_manifest.get(path)
            if vs is None:
                new_videos.append(path)
                continue

            if vs.status == STATUS_FAILED:
                failed_videos.append(path)
                continue

            if vs.status == STATUS_COMPLETED:
                output_missing = not vs.output_npz or not os.path.isfile(vs.output_npz)
                config_stale = vs.config_hash != current_config_hash
                if output_missing or config_stale:
                    stale_videos.append(path)
                else:
                    completed_videos.append(path)
                continue

            pending_videos.append(path)

        orphaned_manifest = sorted(
            vs.video_path
            for key, vs in self._pose_manifest.items()
            if key not in current_set
        )

        return {
            "total_videos": len(normalized_paths),
            "new_videos": sorted(new_videos),
            "stale_videos": sorted(stale_videos),
            "completed_videos": sorted(completed_videos),
            "failed_videos": sorted(failed_videos),
            "pending_videos": sorted(pending_videos),
            "orphaned_manifest": orphaned_manifest,
        }

    def refresh_stage_state(self, config, all_video_paths: Optional[List[str]] = None) -> StageState:
        """
        Recompute stage progress from the filesystem and current manifest state.
        """
        if all_video_paths is None:
            all_video_paths = self._scan_video_root(getattr(config, "video_root", ""))

        self._reconcile_hpo_grid_with_filesystem(getattr(config, "runs_root", ""))

        current_hash = self.pipeline_config_hash(
            config.pose_backend,
            config.do_interp,
            config.do_smooth,
            config.fps_scale,
            config.interp_mode,
            config.ema_alpha,
            config.conf_thr,
        )
        summary = self.summarize_video_sync(all_video_paths, current_hash)

        new_state = StageState(
            pose_extraction=self._infer_pose_stage(summary),
            dataset_building=self._infer_dataset_stage(config),
            hpo_training=self._infer_hpo_stage(config),
            production_training=self._infer_production_stage(config),
        )

        if asdict(new_state) != asdict(self._stage_state):
            self._stage_state = new_state
            self._save_stage_state()

        return self._stage_state

    # ------------------------------------------------------------------
    # HPO grid state
    # ------------------------------------------------------------------
    @property
    def hpo_grid(self) -> Dict[str, HPORunStatus]:
        return self._hpo_grid

    def set_hpo_run_status(self, run_name: str, status: str, **kwargs) -> None:
        if run_name in self._hpo_grid:
            self._hpo_grid[run_name].status = status
        else:
            self._hpo_grid[run_name] = HPORunStatus(run_name=run_name, status=status, **kwargs)
        self._save_hpo_grid()

    def get_completed_hpo_runs(self) -> List[str]:
        return [name for name, rs in self._hpo_grid.items() if rs.status == STATUS_COMPLETED]

    def _load_hpo_grid(self) -> None:
        path = os.path.join(self._root, "hpo_grid_state.json")
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._hpo_grid = {}
        for key, val in data.items():
            self._hpo_grid[key] = HPORunStatus(**{
                k: v for k, v in val.items() if k in HPORunStatus.__dataclass_fields__
            })

    def _save_hpo_grid(self) -> None:
        self.ensure_dirs()
        path = os.path.join(self._root, "hpo_grid_state.json")
        data = {k: asdict(v) for k, v in self._hpo_grid.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _reconcile_hpo_grid_with_filesystem(self, runs_root: str) -> None:
        """Downgrade cached completed runs when their artifacts were deleted."""
        if not self._hpo_grid:
            return

        changed = False
        for run_name, run_state in self._hpo_grid.items():
            run_dir = os.path.join(runs_root, run_name) if runs_root else ""
            summary_exists = bool(run_dir) and os.path.isfile(os.path.join(run_dir, "summary.json"))
            history_exists = bool(run_dir) and os.path.isfile(os.path.join(run_dir, "history.json"))

            if summary_exists:
                desired = STATUS_COMPLETED
            elif history_exists:
                desired = "pending"
            else:
                desired = "pending"

            if run_state.status != desired:
                run_state.status = desired
                changed = True

        if changed:
            self._save_hpo_grid()

    # ------------------------------------------------------------------
    # Config hashing (for stale detection)
    # ------------------------------------------------------------------
    @staticmethod
    def pipeline_config_hash(
        pose_backend: str,
        do_interp: bool,
        do_smooth: bool,
        fps_scale: int,
        interp_mode: str,
        ema_alpha: float,
        conf_thr: float,
    ) -> str:
        """Hash of pipeline parameters that affect pose output."""
        h = hashlib.sha256()
        h.update(json.dumps({
            "backend": pose_backend,
            "interp": do_interp,
            "smooth": do_smooth,
            "fps_scale": fps_scale,
            "interp_mode": interp_mode,
            "ema_alpha": ema_alpha,
            "conf_thr": conf_thr,
        }, sort_keys=True).encode())
        return h.hexdigest()[:12]

    def _scan_video_root(self, video_root: str) -> List[str]:
        if not video_root or not os.path.isdir(video_root):
            return []

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        out: List[str] = []
        for root, _, files in os.walk(video_root):
            for name in files:
                if os.path.splitext(name)[1].lower() in video_exts:
                    out.append(os.path.normpath(os.path.abspath(os.path.join(root, name))))
        out.sort()
        return out

    def _infer_pose_stage(self, summary: Dict[str, object]) -> str:
        total = int(summary["total_videos"])
        new_count = len(summary["new_videos"])
        stale_count = len(summary["stale_videos"])
        failed_count = len(summary["failed_videos"])
        pending_count = len(summary["pending_videos"])
        completed_count = len(summary["completed_videos"])

        if total == 0 and not self._pose_manifest:
            return STATUS_NOT_STARTED
        if total > 0 and completed_count == total and not (new_count or stale_count or failed_count or pending_count):
            return STATUS_COMPLETED
        if completed_count or new_count or stale_count or failed_count or pending_count:
            return STATUS_IN_PROGRESS
        return STATUS_NOT_STARTED

    def _infer_dataset_stage(self, config) -> str:
        pose_root = getattr(config, "pose_output_root", "")
        if not pose_root:
            return STATUS_NOT_STARTED

        base_dir = os.path.dirname(os.path.abspath(pose_root))
        cv_split_dir = os.path.join(base_dir, "data", "cv_splits")
        dataset_dir = os.path.join(base_dir, "data", "dataset_ctr_gcn")
        expected_files = [
            os.path.join(dataset_dir, f"fold_{fold}_{split}.npz")
            for fold in range(getattr(config, "k_folds", 0))
            for split in ("train", "val", "test")
        ]
        split_meta = os.path.join(cv_split_dir, "cv_splits.json")
        fingerprint = os.path.join(dataset_dir, "fingerprint.json")

        if expected_files and all(os.path.isfile(p) for p in expected_files) and os.path.isfile(split_meta):
            return STATUS_COMPLETED

        partial = os.path.isfile(split_meta) or os.path.isfile(fingerprint) or any(os.path.isfile(p) for p in expected_files)
        return STATUS_IN_PROGRESS if partial else STATUS_NOT_STARTED

    def _infer_hpo_stage(self, config) -> str:
        runs_root = getattr(config, "runs_root", "")
        if not runs_root:
            return STATUS_NOT_STARTED

        summary_path = os.path.join(runs_root, "summary_by_hparams.json")
        if os.path.isfile(summary_path):
            return STATUS_COMPLETED

        if os.path.isdir(runs_root):
            for entry in os.listdir(runs_root):
                entry_path = os.path.join(runs_root, entry)
                if os.path.isdir(entry_path) and os.path.isfile(os.path.join(entry_path, "history.json")):
                    return STATUS_IN_PROGRESS
        return STATUS_NOT_STARTED

    def _infer_production_stage(self, config) -> str:
        out_dir = getattr(config, "production_output_dir", "")
        if not out_dir:
            return STATUS_NOT_STARTED

        best_model = os.path.join(out_dir, "best_model.pt")
        history = os.path.join(out_dir, "run", "history.json")
        split_info = os.path.join(out_dir, "split_info.json")

        if os.path.isfile(best_model) and os.path.isfile(history):
            return STATUS_COMPLETED

        partial = os.path.isfile(history) or os.path.isfile(split_info)
        return STATUS_IN_PROGRESS if partial else STATUS_NOT_STARTED
