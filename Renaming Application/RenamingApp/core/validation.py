"""
Integrated CV model validation for the SlopeSense renaming application.

After videos are renamed, this module classifies each video as Pass/Fail
using the CTR-GCN skeleton-based action recognition model.

Pipeline per video:
  MP4 -> YOLO pose extraction -> interpolation -> smoothing ->
  temporal resampling (T=100) -> normalization -> CTR-GCN inference -> Pass/Fail

The CTR-GCN model architecture is embedded directly so no external repo
is required at runtime -- only a trained checkpoint (.pt) is needed.
"""

from __future__ import annotations

import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# torch / ultralytics are imported lazily inside functions that need them,
# so the rest of the renaming app works even when they are not installed.


# ===================================================================
# Configuration
# ===================================================================
@dataclass
class ValidationConfig:
    """All knobs needed to run the validation pipeline."""

    yolo_model_path: str
    ctr_gcn_checkpoint_path: str

    # Device
    device: str = "cpu"

    # Pose extraction
    yolo_batch_size: int = 8
    conf_thr: float = 0.05

    # Preprocessing
    do_interp: bool = True
    do_smooth: bool = True
    fps_scale: int = 4
    ema_alpha: float = 0.7

    # CTR-GCN input shape
    fixed_t: int = 100
    num_class: int = 2
    num_point: int = 17
    num_person: int = 1
    in_channels: int = 3


@dataclass
class ValidationResult:
    """Result of validating a single renamed video."""

    original_video: str
    renamed_video: str
    renamed_video_path: str
    tipper_label: str       # from tipper filename: Pass / Fail / Undecided
    predicted_label: str    # from model: Pass / Fail
    predicted_prob: float   # softmax confidence
    labels_match: bool
    error: Optional[str] = None


# ===================================================================
# Embedded COCO-17 Graph
# ===================================================================
class COCO17Graph:
    """
    COCO 17-keypoint skeleton graph for CTR-GCN.

    Keypoints:
      0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
      5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
      9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
      13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """

    def __init__(self, labeling_mode: str = "spatial", **kwargs):
        num_node = 17
        self_link = [(i, i) for i in range(num_node)]

        # Inward edges: (child, parent) directed toward root (node 0 = nose)
        inward = [
            (1, 0), (2, 0),       # eyes -> nose
            (3, 1), (4, 2),       # ears -> eyes
            (5, 0), (6, 0),       # shoulders -> nose
            (7, 5), (8, 6),       # elbows -> shoulders
            (9, 7), (10, 8),      # wrists -> elbows
            (11, 5), (12, 6),     # hips -> shoulders
            (13, 11), (14, 12),   # knees -> hips
            (15, 13), (16, 14),   # ankles -> knees
        ]
        outward = [(j, i) for (i, j) in inward]

        if labeling_mode == "spatial":
            self.A = self._spatial_adjacency(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Unknown labeling mode: {labeling_mode}")

        self.num_node = num_node

    @staticmethod
    def _edge2mat(edges: list, num_node: int) -> np.ndarray:
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i, j in edges:
            A[j, i] = 1.0
        return A

    @staticmethod
    def _normalize_digraph(A: np.ndarray) -> np.ndarray:
        Dl = A.sum(axis=0)
        Dn = np.zeros_like(A)
        for i in range(A.shape[0]):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return A @ Dn

    @classmethod
    def _spatial_adjacency(cls, num_node, self_link, inward, outward):
        I = cls._edge2mat(self_link, num_node)
        In = cls._normalize_digraph(cls._edge2mat(inward, num_node))
        Out = cls._normalize_digraph(cls._edge2mat(outward, num_node))
        return np.stack((I, In, Out))  # (3, V, V)


# ===================================================================
# Embedded CTR-GCN Model Architecture
# (faithful reproduction of the official CTR-GCN ICCV 2021 paper)
# ===================================================================

def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _conv_init(conv):
    import torch.nn as nn
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def _bn_init(bn, scale):
    import torch.nn as nn
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def _weights_init(m):
    import torch.nn as nn
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv:
    """Lazy-constructed to avoid top-level torch import."""
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is not None:
            return cls._cls
        torch, nn = _import_torch()

        class _TemporalConv(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
                super().__init__()
                pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
                self.conv = nn.Conv2d(
                    in_channels, out_channels,
                    (kernel_size, 1), (stride, 1), (pad, 0), (dilation, 1),
                )
                self.bn = nn.BatchNorm2d(out_channels)

            def forward(self, x):
                return self.bn(self.conv(x))

        cls._cls = _TemporalConv
        return _TemporalConv


class MultiScaleTemporalConv:
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is not None:
            return cls._cls
        torch, nn = _import_torch()
        TC = TemporalConv.get_class()

        class _MultiScaleTemporalConv(nn.Module):
            def __init__(
                self,
                in_channels,
                out_channels,
                kernel_size=5,
                stride=1,
                dilations=None,
                residual=True,
                residual_kernel_size=1,
            ):
                super().__init__()
                if dilations is None:
                    dilations = [1, 2]

                assert out_channels % (len(dilations) + 2) == 0, (
                    f"out_channels={out_channels} must be divisible by {len(dilations) + 2}"
                )
                self.num_branches = len(dilations) + 2
                branch_channels = out_channels // self.num_branches

                if isinstance(kernel_size, list):
                    assert len(kernel_size) == len(dilations)
                else:
                    kernel_size = [kernel_size] * len(dilations)

                self.branches = nn.ModuleList()
                for ks, dilation in zip(kernel_size, dilations):
                    self.branches.append(nn.Sequential(
                        nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                        nn.BatchNorm2d(branch_channels),
                        nn.ReLU(inplace=True),
                        TC(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
                    ))

                # MaxPool branch
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
                    nn.BatchNorm2d(branch_channels),
                ))

                # Stride-only branch
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=(stride, 1), padding=0),
                    nn.BatchNorm2d(branch_channels),
                ))

                # Residual connection
                if not residual:
                    self.residual = lambda x: 0
                elif (in_channels == out_channels) and (stride == 1):
                    self.residual = lambda x: x
                else:
                    self.residual = TC(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

                self.apply(_weights_init)

            def forward(self, x):
                branch_outs = [branch(x) for branch in self.branches]
                out = torch.cat(branch_outs, dim=1)
                out += self.residual(x)
                return out

        cls._cls = _MultiScaleTemporalConv
        return _MultiScaleTemporalConv


class CTRGC:
    """Channel-wise Topology Refinement Graph Convolution."""
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is not None:
            return cls._cls
        torch, nn = _import_torch()

        class _CTRGC(nn.Module):
            def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                if in_channels == 3 or in_channels == 9:
                    self.rel_channels = 8
                    self.mid_channels = 16
                else:
                    self.rel_channels = in_channels // rel_reduction
                    self.mid_channels = in_channels // mid_reduction
                self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
                self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
                self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
                self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
                self.tanh = nn.Tanh()
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        _conv_init(m)
                    elif isinstance(m, nn.BatchNorm2d):
                        _bn_init(m, 1)

            def forward(self, x, A=None, alpha=1):
                x1 = self.conv1(x).mean(-2)          # (N, rel_c, V)
                x2 = self.conv2(x).mean(-2)          # (N, rel_c, V)
                x3 = self.conv3(x)                    # (N, out_c, T, V)
                x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))  # (N, rel_c, V, V)
                x1 = self.conv4(x1) * alpha + (       # (N, out_c, V, V)
                    A.unsqueeze(0).unsqueeze(0) if A is not None else 0
                )
                x1 = torch.einsum("ncuv,nctv->nctu", x1, x3)
                return x1

        cls._cls = _CTRGC
        return _CTRGC


class UnitGCN:
    """Spatial graph convolution unit using CTRGC."""
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is not None:
            return cls._cls
        torch, nn = _import_torch()
        _CTRGC = CTRGC.get_class()

        class _UnitGCN(nn.Module):
            def __init__(self, in_channels, out_channels, A, adaptive=True):
                super().__init__()
                self.out_c = out_channels
                self.in_c = in_channels
                self.num_subset = A.shape[0]
                self.adaptive = adaptive

                if adaptive:
                    self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
                else:
                    self.A = torch.autograd.Variable(
                        torch.from_numpy(A.astype(np.float32)), requires_grad=False,
                    )

                self.convs = nn.ModuleList()
                for _ in range(self.num_subset):
                    self.convs.append(_CTRGC(in_channels, out_channels))
                self.alpha = nn.Parameter(torch.zeros(1))

                if in_channels != out_channels:
                    self.down = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1),
                        nn.BatchNorm2d(out_channels),
                    )
                else:
                    self.down = lambda x: x

                self.bn = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        _conv_init(m)
                    elif isinstance(m, nn.BatchNorm2d):
                        _bn_init(m, 1)
                _bn_init(self.bn, 1e-6)

            def forward(self, x):
                if self.adaptive:
                    A = self.PA
                else:
                    A = self.A.cuda(x.get_device()) if x.is_cuda else self.A

                y = None
                for i in range(self.num_subset):
                    z = self.convs[i](x, A[i], alpha=self.alpha)
                    y = z + y if y is not None else z
                y = self.bn(y)
                y += self.down(x)
                y = self.relu(y)
                return y

        cls._cls = _UnitGCN
        return _UnitGCN


class TCNGCNUnit:
    """Combined temporal + spatial convolution block."""
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is not None:
            return cls._cls
        torch, nn = _import_torch()
        _UnitGCN = UnitGCN.get_class()
        _MSTCN = MultiScaleTemporalConv.get_class()

        _TC = TemporalConv.get_class()

        class _TCNGCNUnit(nn.Module):
            def __init__(
                self,
                in_channels,
                out_channels,
                A,
                stride=1,
                residual=True,
                adaptive=True,
                kernel_size=5,
                dilations=None,
            ):
                super().__init__()
                if dilations is None:
                    dilations = [1, 2]
                self.gcn1 = _UnitGCN(in_channels, out_channels, A, adaptive=adaptive)
                # MSTCN never owns the residual — the unit-level self.residual handles it
                self.tcn1 = _MSTCN(
                    out_channels, out_channels,
                    stride=stride, kernel_size=kernel_size, dilations=dilations,
                    residual=False,
                )
                self.relu = nn.ReLU(inplace=True)

                # Unit-level residual only for channel mismatch — stored as plain TC
                # (no residual for stride-only, that lives inside tcn1)
                if not residual:
                    self.residual = lambda x: 0
                elif in_channels == out_channels:
                    self.residual = lambda x: x
                else:
                    self.residual = _TC(in_channels, out_channels, kernel_size=1, stride=stride)

            def forward(self, x):
                y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
                return y

        cls._cls = _TCNGCNUnit
        return _TCNGCNUnit


class CTRGCNModel:
    """Full CTR-GCN classifier."""
    _cls = None

    @classmethod
    def get_class(cls):
        if cls._cls is not None:
            return cls._cls
        torch, nn = _import_torch()
        _TCNGCNUnit = TCNGCNUnit.get_class()

        class _CTRGCNModel(nn.Module):
            def __init__(
                self,
                num_class=2,
                num_point=17,
                num_person=1,
                in_channels=3,
                graph=None,
                graph_args=None,
                drop_out=0,
                adaptive=True,
            ):
                super().__init__()

                # Build graph adjacency
                if graph_args is None:
                    graph_args = {}
                if isinstance(graph, np.ndarray):
                    A = graph
                else:
                    # Use embedded COCO-17 graph regardless of the string value
                    A = COCO17Graph(**graph_args).A

                self.num_point = num_point
                self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

                base_channel = 64
                self.l1 = _TCNGCNUnit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
                self.l2 = _TCNGCNUnit(base_channel, base_channel, A, adaptive=adaptive)
                self.l3 = _TCNGCNUnit(base_channel, base_channel, A, adaptive=adaptive)
                self.l4 = _TCNGCNUnit(base_channel, base_channel, A, adaptive=adaptive)
                self.l5 = _TCNGCNUnit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
                self.l6 = _TCNGCNUnit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
                self.l7 = _TCNGCNUnit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
                self.l8 = _TCNGCNUnit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
                self.l9 = _TCNGCNUnit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
                self.l10 = _TCNGCNUnit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

                self.fc = nn.Linear(base_channel * 4, num_class)
                nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
                _bn_init(self.data_bn, 1)

                if drop_out:
                    self.drop_out = nn.Dropout(drop_out)
                else:
                    self.drop_out = lambda x: x

            def forward(self, x):
                # x: (N, C, T, V, M)
                if x.size(-1) > 1:
                    raise NotImplementedError("Multi-person not supported in this build")

                N, C, T, V, M = x.size()
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
                x = self.data_bn(x)
                x = (
                    x.view(N, M, V, C, T)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                    .view(N * M, C, T, V)
                )

                x = self.l1(x)
                x = self.l2(x)
                x = self.l3(x)
                x = self.l4(x)
                x = self.l5(x)
                x = self.l6(x)
                x = self.l7(x)
                x = self.l8(x)
                x = self.l9(x)
                x = self.l10(x)

                # N*M, C, T, V
                c_new = x.size(1)
                x = x.view(N, M, c_new, -1)
                x = x.mean(3).mean(1)
                x = self.drop_out(x)
                return self.fc(x)

        cls._cls = _CTRGCNModel
        return _CTRGCNModel


# ===================================================================
# YOLO Pose Extraction
# (adapted from CV/code/pose_estimators/yolo.py)
# ===================================================================
def _skeleton_center(
    xy: np.ndarray, conf: Optional[np.ndarray], conf_thr: float,
) -> np.ndarray:
    if conf is None:
        return np.nanmean(xy, axis=0)
    m = conf >= conf_thr
    if m.sum() < 2:
        return np.nanmean(xy, axis=0)
    return xy[m].mean(axis=0)


def _skeleton_area(
    xy: np.ndarray, conf: Optional[np.ndarray], conf_thr: float,
) -> float:
    pts = xy if conf is None else xy[conf >= conf_thr]
    if len(pts) < 2:
        return 0.0
    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
    if not np.isfinite([x1, y1, x2, y2]).all():
        return 0.0
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _pick_single_person(
    result0,
    num_kpts: int = 17,
    prev_center: Optional[np.ndarray] = None,
    conf_thr: float = 0.05,
    width: int = 1920,
    height: int = 1080,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Select the most likely single person from a YOLO pose result."""

    if result0.keypoints is None:
        return np.zeros((num_kpts, 3), dtype=np.float32), prev_center

    kpts = result0.keypoints
    if kpts.xy is None or len(kpts) == 0:
        return np.zeros((num_kpts, 3), dtype=np.float32), prev_center

    xy = kpts.xy.cpu().numpy()  # (M, V, 2)
    conf = (
        kpts.conf.cpu().numpy()
        if getattr(kpts, "conf", None) is not None
        else None
    )
    M, V = xy.shape[0], xy.shape[1]

    # Single person fast path
    if M == 1:
        out = np.zeros((V, 3), dtype=np.float32)
        out[:, :2] = xy[0]
        out[:, 2] = conf[0] if conf is not None else 1.0
        center = _skeleton_center(
            xy[0], conf[0] if conf is not None else None, conf_thr,
        )
        return out, center

    # No confidence info -> pick largest skeleton
    if conf is None:
        areas = [_skeleton_area(xy[i], None, conf_thr) for i in range(M)]
        chosen = int(np.argmax(areas))
        out = np.zeros((V, 3), dtype=np.float32)
        out[:, :2] = xy[chosen]
        out[:, 2] = 1.0
        return out, _skeleton_center(xy[chosen], None, conf_thr)

    # Composite scoring: size + tracking + confidence
    centers = np.stack(
        [_skeleton_center(xy[i], conf[i], conf_thr) for i in range(M)],
    )
    areas = np.array(
        [_skeleton_area(xy[i], conf[i], conf_thr) for i in range(M)],
        dtype=np.float32,
    )
    mean_conf = conf.mean(axis=1).astype(np.float32)

    size_norm = areas / (areas.max() + 1e-6) if areas.max() > 0 else np.zeros(M)
    mc_range = mean_conf.max() - mean_conf.min()
    conf_norm = (
        (mean_conf - mean_conf.min()) / (mc_range + 1e-6)
        if mc_range > 0
        else np.zeros(M)
    )

    if prev_center is not None and np.isfinite(prev_center).all():
        diag = float(np.sqrt(width ** 2 + height ** 2))
        dist = np.linalg.norm(centers - prev_center[None, :], axis=1)
        track_score = 1.0 - np.clip(dist / (diag + 1e-6), 0, 1)
    else:
        track_score = np.zeros(M, dtype=np.float32)

    score = 0.55 * size_norm + 0.20 * track_score + 0.25 * conf_norm
    chosen = int(np.argmax(score))

    out = np.zeros((V, 3), dtype=np.float32)
    out[:, 0] = xy[chosen, :, 0]
    out[:, 1] = xy[chosen, :, 1]
    out[:, 2] = conf[chosen]
    return out, centers[chosen]


def extract_poses(
    video_path: str,
    yolo_model,
    batch_size: int = 8,
    conf_thr: float = 0.05,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Extract per-frame poses from a video using a YOLO pose model.

    Returns
    -------
    poses : ndarray, shape (T, V, 3)  --  [x, y, confidence] in pixel coords
    meta  : dict with fps, width, height, num_frames
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses_list: list[np.ndarray] = []
    prev_center: Optional[np.ndarray] = None
    batch: list[np.ndarray] = []
    infer_kwargs = {"verbose": False}
    if device:
        infer_kwargs["device"] = device

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(frame)

        if len(batch) >= batch_size:
            results = yolo_model(batch, **infer_kwargs)
            for r in results:
                kpt, prev_center = _pick_single_person(
                    r,
                    prev_center=prev_center,
                    conf_thr=conf_thr,
                    width=width,
                    height=height,
                )
                poses_list.append(kpt)
            batch.clear()

    if batch:
        results = yolo_model(batch, **infer_kwargs)
        for r in results:
            kpt, prev_center = _pick_single_person(
                r,
                prev_center=prev_center,
                conf_thr=conf_thr,
                width=width,
                height=height,
            )
            poses_list.append(kpt)

    cap.release()

    if not poses_list:
        raise RuntimeError(f"No frames in video: {video_path}")

    meta = {
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "num_frames": len(poses_list),
    }
    return np.stack(poses_list).astype(np.float32), meta


# ===================================================================
# Preprocessing
# (adapted from CV/code/preprocessing/)
# ===================================================================
def interpolate_poses(
    poses: np.ndarray,
    scale_factor: int = 4,
    conf_thr: float = 0.05,
    min_kpts: int = 8,
) -> np.ndarray:
    """Temporal upsampling via linear interpolation.  Input (T, V, 3)."""
    T, V, C = poses.shape
    if T <= 1 or scale_factor <= 1:
        return poses.copy()

    T_out = (T - 1) * scale_factor + 1
    out = np.zeros((T_out, V, C), dtype=np.float32)

    # Frame validity: needs enough confident keypoints
    frame_ok = (poses[:, :, 2] >= conf_thr).sum(axis=1) >= min_kpts

    for t in range(T - 1):
        base = t * scale_factor
        out[base] = poses[t] if frame_ok[t] else 0.0

        if not (frame_ok[t] and frame_ok[t + 1]):
            continue  # leave in-betweens as zeros

        a, b = poses[t], poses[t + 1]
        kp_ok = (a[:, 2] >= conf_thr) & (b[:, 2] >= conf_thr)

        for k in range(1, scale_factor):
            u = k / float(scale_factor)
            idx = base + k
            if kp_ok.any():
                out[idx, kp_ok, 0] = (1 - u) * a[kp_ok, 0] + u * b[kp_ok, 0]
                out[idx, kp_ok, 1] = (1 - u) * a[kp_ok, 1] + u * b[kp_ok, 1]
                out[idx, kp_ok, 2] = np.clip(
                    (1 - u) * a[kp_ok, 2] + u * b[kp_ok, 2], 0, 1,
                )

    # Last frame
    out[-1] = poses[-1] if frame_ok[-1] else 0.0
    return out


def smooth_poses(
    poses: np.ndarray,
    alpha: float = 0.7,
    conf_thr: float = 0.05,
) -> np.ndarray:
    """EMA smoothing.  Input (T, V, 3)."""
    T = poses.shape[0]
    out = np.empty_like(poses)
    prev = poses[0].copy()
    out[0] = prev

    for t in range(1, T):
        kp = poses[t]
        valid = kp[:, 2] >= conf_thr
        sm = prev.copy()

        if valid.any():
            sm[valid, :2] = alpha * kp[valid, :2] + (1 - alpha) * prev[valid, :2]

        sm[~valid, :2] = 0.0
        sm[~valid, 2] = 0.0
        sm[:, 2] = kp[:, 2]
        sm[~valid, 2] = 0.0

        prev = sm
        out[t] = sm

    return out


def _uniform_sample(T_orig: int, T: int) -> np.ndarray:
    if T_orig <= 0:
        return np.zeros(T, dtype=np.int64)
    return np.linspace(0, T_orig - 1, T).astype(np.int64)


def prepare_for_model(
    poses: np.ndarray,
    meta: dict,
    fixed_t: int = 100,
) -> "torch.Tensor":
    """
    Convert (T, V, 3) poses to a batched CTR-GCN tensor (1, C=3, T, V, M=1).
    """
    import torch

    T_orig, V, C = poses.shape
    width, height = meta["width"], meta["height"]

    poses = poses.astype(np.float32, copy=True)
    poses[..., 0] /= float(width)
    poses[..., 1] /= float(height)

    # Temporal resampling / padding
    if T_orig >= fixed_t:
        idx = _uniform_sample(T_orig, fixed_t)
        poses_t = poses[idx]
    else:
        poses_t = np.zeros((fixed_t, V, C), dtype=np.float32)
        poses_t[:T_orig] = poses

    # (T, V, C) -> (C, T, V) -> (C, T, V, 1) -> (1, C, T, V, 1)
    data = poses_t.transpose(2, 0, 1)[..., np.newaxis]
    return torch.from_numpy(data[np.newaxis]).float()


# ===================================================================
# Model Loading
# ===================================================================
def _load_legacy_dir_checkpoint(dir_path: str, map_location=None):
    """
    Load a PyTorch checkpoint stored in the legacy directory format:
      dir_path/
        data.pkl          -- pickled object with persistent storage refs
        data/0, data/1 .. -- raw binary tensor storage files
    """
    import pickle
    import torch

    data_dir = os.path.join(dir_path, "data")
    cached: dict = {}

    # Map storage class name -> torch dtype
    _type_to_dtype = {
        "FloatStorage":    torch.float32,
        "DoubleStorage":   torch.float64,
        "HalfStorage":     torch.float16,
        "BFloat16Storage": torch.bfloat16,
        "LongStorage":     torch.int64,
        "IntStorage":      torch.int32,
        "ShortStorage":    torch.int16,
        "ByteStorage":     torch.uint8,
        "CharStorage":     torch.int8,
        "BoolStorage":     torch.bool,
    }

    def _get_storage(storage_type, key, numel):
        if key in cached:
            return cached[key]

        storage_path = os.path.join(data_dir, str(key))
        numel = int(numel)

        # Resolve dtype from the storage class name
        type_name = getattr(storage_type, "__name__", str(storage_type))
        torch_dtype = _type_to_dtype.get(type_name, torch.float32)
        elem_size = torch.tensor([], dtype=torch_dtype).element_size()

        with open(storage_path, "rb") as f:
            data = f.read(numel * elem_size)

        # Build UntypedStorage then wrap in TypedStorage so _rebuild_tensor_v2
        # can read the .dtype attribute it requires.
        untyped = torch.UntypedStorage.from_buffer(data, byte_order="little", dtype=torch.uint8)
        st = torch.storage.TypedStorage(wrap_storage=untyped, dtype=torch_dtype)

        cached[key] = st
        return st

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            # Normalize bytes → str for very old formats
            if isinstance(pid[0], bytes):
                pid = tuple(p.decode() if isinstance(p, bytes) else p for p in pid)
            if pid[0] != "storage":
                return pid
            _, storage_type, key, _location, numel = pid
            return _get_storage(storage_type, key, numel)

    pkl_path = os.path.join(dir_path, "data.pkl")
    with open(pkl_path, "rb") as f:
        return _Unpickler(f).load()


def load_ctr_gcn_model(config: ValidationConfig):
    """Build the CTR-GCN model and load trained weights from checkpoint."""
    import torch

    ckpt_path = os.path.abspath(os.path.expanduser(config.ctr_gcn_checkpoint_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(config.device)

    if os.path.isdir(ckpt_path):
        ckpt = _load_legacy_dir_checkpoint(ckpt_path, map_location=device)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    _Model = CTRGCNModel.get_class()

    default_model_kwargs = {
        "num_class": config.num_class,
        "num_point": config.num_point,
        "num_person": config.num_person,
        "in_channels": config.in_channels,
        "graph": "graph.coco17.Graph",
        "graph_args": {},
        "drop_out": 0.0,
    }

    def _try_load(model, state_dict):
        """Try strict then non-strict load; raise with diagnostics on failure."""
        try:
            model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError:
            pass
        # Try stripping common prefixes (e.g. "module." from DataParallel)
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(stripped, strict=True)
            return
        except RuntimeError:
            pass
        # Non-strict as last resort — report what's missing/unexpected
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        if missing or unexpected:
            import warnings
            warnings.warn(
                f"Checkpoint loaded non-strictly. "
                f"Missing keys ({len(missing)}): {missing[:5]}... "
                f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}..."
            )

    if isinstance(ckpt, torch.nn.Module):
        # Full model was saved directly
        model = ckpt
    elif isinstance(ckpt, dict):
        # Determine which key holds the state dict
        # Common keys: "model_state", "model", "state_dict", or the dict IS the state dict
        state_dict = None
        for key in ("model_state", "model", "state_dict", "net", "weights"):
            if key in ckpt:
                state_dict = ckpt[key]
                break

        # If none of the known keys matched, assume the dict itself is the state dict
        # (i.e. torch.save(model.state_dict(), path))
        if state_dict is None:
            # Validate: does it look like a state dict (values are tensors)?
            first_val = next(iter(ckpt.values()), None)
            if isinstance(first_val, torch.Tensor):
                state_dict = ckpt
            else:
                top_keys = list(ckpt.keys())
                raise RuntimeError(
                    f"Unrecognized checkpoint format. Top-level keys: {top_keys}"
                )

        model_kwargs = (ckpt.get("model_kwargs") or {}) if isinstance(ckpt, dict) else {}
        if not model_kwargs:
            model_kwargs = default_model_kwargs
        model = _Model(**model_kwargs)
        _try_load(model, state_dict)
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint format at {ckpt_path} "
            f"(type={type(ckpt).__name__}). Expected dict or nn.Module."
        )

    model = model.to(device)
    model.eval()
    return model


def load_yolo_model(model_path: str):
    """Load YOLO pose estimation model."""
    from ultralytics import YOLO

    path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"YOLO model not found: {path}")
    return YOLO(path)


# ===================================================================
# Label extraction from tipper / renamed filenames
# ===================================================================
def extract_tipper_label(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract direction and result from a tipper-style filename.

    Example: ``idapt811_sub001_DP_5_14-01-48.mp4`` -> ``('D', 'P')``
    """
    stem = Path(filename).stem
    match = re.search(r"_([DU][PFU])_", stem)
    if match:
        code = match.group(1)
        return code[0], code[1]
    return None, None


_RESULT_READABLE = {"P": "Pass", "F": "Fail", "U": "Undecided"}


# ===================================================================
# Single-video classification
# ===================================================================
def classify_video(
    video_path: str,
    yolo_model,
    ctr_gcn_model,
    config: ValidationConfig,
) -> Tuple[str, float]:
    """
    Run the full pipeline on one video.

    Returns ``(predicted_label, confidence)`` where label is 'Pass' or 'Fail'.
    """
    import torch

    # 1. Pose extraction
    poses, meta = extract_poses(
        video_path,
        yolo_model,
        batch_size=config.yolo_batch_size,
        conf_thr=config.conf_thr,
        device=config.device,
    )

    # 2. Interpolation
    if config.do_interp:
        poses = interpolate_poses(
            poses, scale_factor=config.fps_scale, conf_thr=config.conf_thr,
        )

    # 3. Smoothing
    if config.do_smooth:
        poses = smooth_poses(
            poses, alpha=config.ema_alpha, conf_thr=config.conf_thr,
        )

    # 4. Prepare tensor
    data = prepare_for_model(poses, meta, fixed_t=config.fixed_t)
    data = data.to(config.device)

    # 5. Inference
    with torch.no_grad():
        logits = ctr_gcn_model(data)          # (1, num_class)
        probs = torch.softmax(logits, dim=1)  # (1, num_class)
        pred = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred].item())

    label = "Pass" if pred == 0 else "Fail"
    return label, confidence


# ===================================================================
# Batch validation
# ===================================================================
def validate_videos(
    video_entries: List[Tuple[str, str, str]],
    config: ValidationConfig,
    log: Callable[[str], None] = print,
    progress: Callable[[int, int], None] = lambda _c, _t: None,
    stop_requested: Callable[[], bool] = lambda: False,
) -> List[ValidationResult]:
    """
    Classify every renamed video.

    Parameters
    ----------
    video_entries
        List of ``(original_filename, renamed_filename, video_file_path)`` tuples.
        ``video_file_path`` must point to the actual file to process.
    config
        Full validation configuration.
    log / progress / stop_requested
        Callbacks for UI integration.

    Returns
    -------
    List[ValidationResult]
    """
    log("[Validate] Loading YOLO model...")
    yolo_model = load_yolo_model(config.yolo_model_path)

    log("[Validate] Loading CTR-GCN model...")
    ctr_gcn_model = load_ctr_gcn_model(config)

    results: List[ValidationResult] = []
    total = len(video_entries)
    log(f"[Validate] {total} videos to classify.")

    for i, (orig, renamed, path) in enumerate(video_entries):
        if stop_requested():
            log("[Validate] Cancelled by user.")
            break

        log(f"[Validate] ({i + 1}/{total}) {renamed}")
        progress(i + 1, total)

        _dir, result_code = extract_tipper_label(renamed)
        tipper_readable = _RESULT_READABLE.get(result_code, "Unknown")

        try:
            pred_label, pred_prob = classify_video(
                path, yolo_model, ctr_gcn_model, config,
            )

            if tipper_readable in ("Pass", "Fail"):
                match = tipper_readable == pred_label
            else:
                match = False

            results.append(
                ValidationResult(
                    original_video=orig,
                    renamed_video=renamed,
                    renamed_video_path=str(path),
                    tipper_label=tipper_readable,
                    predicted_label=pred_label,
                    predicted_prob=pred_prob,
                    labels_match=match,
                )
            )
        except Exception as exc:
            log(f"  [WARN] Failed: {exc}")
            results.append(
                ValidationResult(
                    original_video=orig,
                    renamed_video=renamed,
                    renamed_video_path=str(path),
                    tipper_label=tipper_readable,
                    predicted_label="Error",
                    predicted_prob=0.0,
                    labels_match=False,
                    error=str(exc),
                )
            )

    return results


# ===================================================================
# Spreadsheet output
# ===================================================================
def write_validation_report(
    results: List[ValidationResult],
    output_path: Path,
) -> None:
    """Write validation results to an Excel spreadsheet."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Validation Results"

    headers = [
        "Original Video",
        "Renamed Video",
        "Tipper Classification",
        "Model Classification",
        "Model Confidence",
        "Match",
        "Error",
    ]

    header_fill = PatternFill(fill_type="solid", fgColor="18366F")
    header_font = Font(bold=True, color="FFFFFF")

    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    pass_fill = PatternFill(fill_type="solid", fgColor="D4EDDA")
    fail_fill = PatternFill(fill_type="solid", fgColor="F8D7DA")
    match_fill = PatternFill(fill_type="solid", fgColor="D4EDDA")
    mismatch_fill = PatternFill(fill_type="solid", fgColor="FFF3CD")
    error_fill = PatternFill(fill_type="solid", fgColor="F8D7DA")

    for row_idx, r in enumerate(results, 2):
        ws.cell(row=row_idx, column=1, value=r.original_video)
        ws.cell(row=row_idx, column=2, value=r.renamed_video)

        tipper_cell = ws.cell(row=row_idx, column=3, value=r.tipper_label)
        if r.tipper_label == "Pass":
            tipper_cell.fill = pass_fill
        elif r.tipper_label == "Fail":
            tipper_cell.fill = fail_fill

        pred_cell = ws.cell(row=row_idx, column=4, value=r.predicted_label)
        if r.predicted_label == "Pass":
            pred_cell.fill = pass_fill
        elif r.predicted_label == "Fail":
            pred_cell.fill = fail_fill
        elif r.predicted_label == "Error":
            pred_cell.fill = error_fill

        ws.cell(row=row_idx, column=5, value=round(r.predicted_prob, 4))

        match_cell = ws.cell(
            row=row_idx, column=6, value="Yes" if r.labels_match else "No",
        )
        match_cell.fill = match_fill if r.labels_match else mismatch_fill

        if r.error:
            err_cell = ws.cell(row=row_idx, column=7, value=r.error)
            err_cell.fill = error_fill

    # Summary section
    total = len(results)
    errors = sum(1 for r in results if r.error)
    comparable = total - errors
    matches = sum(1 for r in results if r.labels_match)

    summary_row = total + 3
    ws.cell(row=summary_row, column=1, value="Summary").font = Font(bold=True)
    ws.cell(row=summary_row + 1, column=1, value=f"Total videos: {total}")
    ws.cell(
        row=summary_row + 2,
        column=1,
        value=f"Matches: {matches}/{comparable}",
    )
    if comparable > 0:
        ws.cell(
            row=summary_row + 3,
            column=1,
            value=f"Agreement rate: {matches / comparable:.1%}",
        )
    ws.cell(row=summary_row + 4, column=1, value=f"Errors: {errors}")

    # Auto-fit columns
    for col_cells in ws.columns:
        col_letter = get_column_letter(col_cells[0].column)
        max_len = max((len(str(c.value or "")) for c in col_cells), default=0)
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 12), 45)

    ws.freeze_panes = "A2"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))
