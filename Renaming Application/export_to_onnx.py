"""
One-time export script: converts .pt models to ONNX for use with ONNX Runtime.
Run from the "Renaming Application/" directory with torch + ultralytics installed:
    python export_to_onnx.py
Outputs:
    ../models/classifier.onnx
    ../models/yolo26x-pose.onnx
"""
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODELS_DIR = Path(__file__).parent.parent / "models"


# ===================================================================
# COCO-17 Graph
# ===================================================================
class COCO17Graph:
    def __init__(self, labeling_mode="spatial", **kwargs):
        num_node = 17
        self_link = [(i, i) for i in range(num_node)]
        inward = [
            (1, 0), (2, 0), (3, 1), (4, 2),
            (5, 0), (6, 0), (7, 5), (8, 6),
            (9, 7), (10, 8), (11, 5), (12, 6),
            (13, 11), (14, 12), (15, 13), (16, 14),
        ]
        outward = [(j, i) for (i, j) in inward]
        I   = self._edge2mat(self_link, num_node)
        In  = self._normalize_digraph(self._edge2mat(inward, num_node))
        Out = self._normalize_digraph(self._edge2mat(outward, num_node))
        self.A = np.stack((I, In, Out))

    @staticmethod
    def _edge2mat(edges, num_node):
        A = np.zeros((num_node, num_node), dtype=np.float32)
        for i, j in edges:
            A[j, i] = 1.0
        return A

    @staticmethod
    def _normalize_digraph(A):
        Dl = A.sum(axis=0)
        Dn = np.zeros_like(A)
        for i in range(A.shape[0]):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return A @ Dn


# ===================================================================
# CTR-GCN Architecture
# ===================================================================
def _conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def _bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def _weights_init(m):
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


class TemporalConv(nn.Module):
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


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 dilations=None, residual=True, residual_kernel_size=1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2]
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
                TemporalConv(branch_channels, branch_channels, kernel_size=ks,
                             stride=stride, dilation=dilation),
            ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1,
                      stride=(stride, 1), padding=0),
            nn.BatchNorm2d(branch_channels),
        ))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=residual_kernel_size, stride=stride)
        self.apply(_weights_init)

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out += self.residual(x)
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels in (3, 9):
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
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (
            A.unsqueeze(0).unsqueeze(0) if A is not None else 0
        )
        return torch.einsum("ncuv,nctv->nctu", x1, x3)


class UnitGCN(nn.Module):
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
        self.convs = nn.ModuleList([CTRGC(in_channels, out_channels)
                                    for _ in range(self.num_subset)])
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
        A = self.PA if self.adaptive else (
            self.A.cuda(x.get_device()) if x.is_cuda else self.A
        )
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], alpha=self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCNGCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 adaptive=True, kernel_size=5, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2]
        self.gcn1 = UnitGCN(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScaleTemporalConv(
            out_channels, out_channels, stride=stride, kernel_size=kernel_size,
            dilations=dilations, residual=False,
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels,
                                         kernel_size=1, stride=stride)

    def forward(self, x):
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class CTRGCNModel(nn.Module):
    def __init__(self, num_class=2, num_point=17, num_person=1, in_channels=3,
                 graph=None, graph_args=None, drop_out=0, adaptive=True):
        super().__init__()
        if graph_args is None:
            graph_args = {}
        A = COCO17Graph(**graph_args).A
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        c = 64
        self.l1  = TCNGCNUnit(in_channels, c,     A, residual=False, adaptive=adaptive)
        self.l2  = TCNGCNUnit(c,           c,     A, adaptive=adaptive)
        self.l3  = TCNGCNUnit(c,           c,     A, adaptive=adaptive)
        self.l4  = TCNGCNUnit(c,           c,     A, adaptive=adaptive)
        self.l5  = TCNGCNUnit(c,           c * 2, A, stride=2, adaptive=adaptive)
        self.l6  = TCNGCNUnit(c * 2,       c * 2, A, adaptive=adaptive)
        self.l7  = TCNGCNUnit(c * 2,       c * 2, A, adaptive=adaptive)
        self.l8  = TCNGCNUnit(c * 2,       c * 4, A, stride=2, adaptive=adaptive)
        self.l9  = TCNGCNUnit(c * 4,       c * 4, A, adaptive=adaptive)
        self.l10 = TCNGCNUnit(c * 4,       c * 4, A, adaptive=adaptive)
        self.fc = nn.Linear(c * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        _bn_init(self.data_bn, 1)
        self.drop_out = nn.Dropout(drop_out) if drop_out else lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x);  x = self.l2(x);  x = self.l3(x);  x = self.l4(x)
        x = self.l5(x);  x = self.l6(x);  x = self.l7(x)
        x = self.l8(x);  x = self.l9(x);  x = self.l10(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1).mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x)


# ===================================================================
# Legacy directory checkpoint loader
# ===================================================================
def _load_legacy_dir_checkpoint(dir_path, map_location=None):
    data_dir = os.path.join(dir_path, "data")
    cached = {}
    _type_to_dtype = {
        "FloatStorage": torch.float32, "DoubleStorage": torch.float64,
        "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16,
        "LongStorage": torch.int64, "IntStorage": torch.int32,
        "ShortStorage": torch.int16, "ByteStorage": torch.uint8,
        "CharStorage": torch.int8, "BoolStorage": torch.bool,
    }

    def _get_storage(storage_type, key, numel):
        if key in cached:
            return cached[key]
        storage_path = os.path.join(data_dir, str(key))
        type_name = getattr(storage_type, "__name__", str(storage_type))
        torch_dtype = _type_to_dtype.get(type_name, torch.float32)
        elem_size = torch.tensor([], dtype=torch_dtype).element_size()
        with open(storage_path, "rb") as f:
            data = f.read(int(numel) * elem_size)
        untyped = torch.UntypedStorage.from_buffer(data, byte_order="little", dtype=torch.uint8)
        st = torch.storage.TypedStorage(wrap_storage=untyped, dtype=torch_dtype)
        cached[key] = st
        return st

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            if isinstance(pid[0], bytes):
                pid = tuple(p.decode() if isinstance(p, bytes) else p for p in pid)
            if pid[0] != "storage":
                return pid
            _, storage_type, key, _location, numel = pid
            return _get_storage(storage_type, key, numel)

    with open(os.path.join(dir_path, "data.pkl"), "rb") as f:
        return _Unpickler(f).load()


# ===================================================================
# Export functions
# ===================================================================
def export_ctrgcn():
    out = MODELS_DIR / "classifier.onnx"
    ckpt_path = str(MODELS_DIR / "classifier")
    print("Loading CTR-GCN checkpoint...")

    ckpt = _load_legacy_dir_checkpoint(ckpt_path)

    model = CTRGCNModel(num_class=2, num_point=17, num_person=1, in_channels=3)
    model.eval()

    # Try loading state dict with common key patterns
    if isinstance(ckpt, dict):
        state_dict = None
        for key in ("model_state", "model", "state_dict", "net", "weights"):
            if key in ckpt:
                state_dict = ckpt[key]
                break
        if state_dict is None:
            state_dict = ckpt
    else:
        # ckpt is the model or state dict directly
        if hasattr(ckpt, "state_dict"):
            model = ckpt
            model.eval()
            state_dict = None
        else:
            state_dict = ckpt

    if state_dict is not None:
        # Strip "module." prefix if present
        stripped = {k.replace("module.", ""): v for k, v in state_dict.items()}
        try:
            model.load_state_dict(stripped, strict=True)
        except RuntimeError:
            model.load_state_dict(stripped, strict=False)

    dummy = torch.zeros(1, 3, 100, 17, 1)
    print(f"Exporting CTR-GCN -> {out}")
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )
    print("  CTR-GCN export done.")


def export_yolo():
    from ultralytics import YOLO
    pt = MODELS_DIR / "yolo26x-pose.pt"
    out = MODELS_DIR / "yolo26x-pose.onnx"
    print(f"Loading YOLO model {pt.name}...")
    model = YOLO(str(pt))
    print(f"Exporting YOLO -> {out}")
    model.export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
    auto = pt.with_suffix(".onnx")
    if auto.exists() and auto != out:
        shutil.move(str(auto), str(out))
    print("  YOLO export done.")


if __name__ == "__main__":
    export_ctrgcn()
    export_yolo()
    print(f"\nAll models exported to {MODELS_DIR}")
    print("  classifier.onnx")
    print("  yolo26x-pose.onnx")
