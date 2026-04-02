#!/usr/bin/env python3
"""
Compare inference backends on a single video.

This script runs the exact app ONNX path and the original CTR-GCN PyTorch path
side by side, then reports:
  - final prediction / probabilities
  - stage timings
  - pose / model-input / logit deltas against a reference backend

Examples:
  python debug_backend_compare.py \
    --video /Users/yusufmoola/Desktop/renamed-07/2025-07-07/sub295/idapt804_sub295_DP_5_GP1_13-04-54.mp4

  python debug_backend_compare.py \
    --video /path/to/video.mp4 \
    --backends pt_cpu onnx_cpu onnx_coreml \
    --json-out backend_debug.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
APP_VALIDATION_PATH = Path(__file__).resolve().parent / "RenamingApp" / "core" / "validation.py"
CTR_ROOT = REPO_ROOT / "CTR-GCN"


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _label_from_probs(probs: np.ndarray) -> str:
    return "Pass" if int(np.argmax(probs[0])) == 0 else "Fail"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _load_app_validation_module():
    spec = importlib.util.spec_from_file_location("slopesense_validation", APP_VALIDATION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_ctr_import_paths() -> None:
    ctr_root = str(CTR_ROOT)
    framework_root = str(CTR_ROOT / "frameworks" / "CTR-GCN")
    if ctr_root not in sys.path:
        sys.path.insert(0, ctr_root)
    if framework_root not in sys.path:
        sys.path.insert(0, framework_root)


def _import_ctr_attr(module_name: str, attr_name: str):
    _ensure_ctr_import_paths()
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_name}' does not export '{attr_name}'") from exc


def _available_pt_backends() -> List[str]:
    import torch

    backends = ["pt_cpu"]
    if torch.cuda.is_available():
        backends.append("pt_cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        backends.append("pt_mps")
    return backends


def _available_onnx_backends() -> List[str]:
    import onnxruntime as ort

    providers = set(ort.get_available_providers())
    backends = ["onnx_cpu"]
    if "CUDAExecutionProvider" in providers:
        backends.append("onnx_cuda")
    if "CoreMLExecutionProvider" in providers:
        backends.append("onnx_coreml")
    return backends


def _default_ultra_coreml_model() -> Optional[Path]:
    candidates = [
        MODELS_DIR / "yolo26x-pose.mlpackage",
        MODELS_DIR / "yolo26x-pose.mlmodel",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in ("*pose*.mlpackage", "*pose*.mlmodel"):
        matches = sorted(MODELS_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def _available_hybrid_backends() -> List[str]:
    if _default_ultra_coreml_model() is not None:
        return ["ultra_coreml_yolo", "direct_coreml_yolo"]
    return []


def _normalize_onnx_backend_name(name: str) -> str:
    if name.startswith("onnx_"):
        return name
    return f"onnx_{name}"


def _coreml_provider_options(args: argparse.Namespace) -> Dict[str, str]:
    options: Dict[str, str] = {}
    if args.coreml_model_format:
        options["ModelFormat"] = args.coreml_model_format
    if args.coreml_compute_units:
        options["MLComputeUnits"] = args.coreml_compute_units
    if args.coreml_require_static_input_shapes is not None:
        options["RequireStaticInputShapes"] = str(int(args.coreml_require_static_input_shapes))
    if args.coreml_enable_on_subgraphs is not None:
        options["EnableOnSubgraphs"] = str(int(args.coreml_enable_on_subgraphs))
    if args.coreml_profile_compute_plan is not None:
        options["ProfileComputePlan"] = str(int(args.coreml_profile_compute_plan))
    if args.coreml_allow_low_precision_gpu_accumulation is not None:
        options["AllowLowPrecisionAccumulationOnGPU"] = str(
            int(args.coreml_allow_low_precision_gpu_accumulation)
        )
    if args.coreml_cache_dir:
        options["ModelCacheDirectory"] = str(Path(args.coreml_cache_dir).expanduser().resolve())
    return options


def _onnx_provider_spec(
    backend: str,
    model_path: Path,
    args: argparse.Namespace,
) -> Tuple[Path, List[object]]:
    if backend == "onnx_cpu":
        return model_path, ["CPUExecutionProvider"]
    if backend == "onnx_cuda":
        return model_path, ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if backend != "onnx_coreml":
        raise ValueError(f"Unsupported ONNX backend: {backend}")

    os.environ.setdefault("TMPDIR", str(REPO_ROOT / ".codex_tmp" / "ort"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    staged_path = _stage_coreml_model(model_path)
    options = _coreml_provider_options(args)
    if options:
        providers: List[object] = [("CoreMLExecutionProvider", options), "CPUExecutionProvider"]
    else:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return staged_path, providers


def _stage_coreml_model(model_path: Path) -> Path:
    """
    CoreML session creation is sensitive to some local path setups.
    Stage the model under a simple workspace path if needed.
    """
    stage_dir = REPO_ROOT / ".codex_tmp" / "coreml_models"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_dir / model_path.name
    shutil.copy2(model_path, staged)
    sidecar = model_path.with_suffix(model_path.suffix + ".data")
    if sidecar.exists():
        shutil.copy2(sidecar, stage_dir / sidecar.name)
    return staged


def _diff_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {
            "same_shape": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    diff = np.abs(a - b)
    return {
        "same_shape": True,
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def _prepare_and_classify_with_onnx(
    poses_raw: np.ndarray,
    meta: Dict[str, Any],
    classifier_backend: str,
    args: argparse.Namespace,
    valmod,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    import onnxruntime as ort

    classifier_path = MODELS_DIR / "classifier.onnx"
    classifier_path, classifier_providers = _onnx_provider_spec(classifier_backend, classifier_path, args)
    clf_session = ort.InferenceSession(str(classifier_path), providers=classifier_providers)

    poses_interp = valmod.interpolate_poses(poses_raw, scale_factor=4, conf_thr=0.05)
    poses_smooth = valmod.smooth_poses(poses_interp, alpha=0.7, conf_thr=0.05)
    model_input = valmod.prepare_for_model(poses_smooth, meta, fixed_t=100)
    logits = clf_session.run(None, {clf_session.get_inputs()[0].name: model_input})[0]
    return poses_interp, poses_smooth, model_input, logits, clf_session


def _run_ultra_coreml_yolo_backend(
    video_path: Path,
    backend: str,
    valmod,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import cv2
    from ultralytics import YOLO

    if backend != "ultra_coreml_yolo":
        raise ValueError(f"Unsupported hybrid backend: {backend}")

    model_path = Path(args.ultra_yolo_model).expanduser().resolve() if args.ultra_yolo_model else _default_ultra_coreml_model()
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            "No CoreML YOLO model found. Export a pose model to .mlpackage/.mlmodel "
            "and pass it with --ultra-yolo-model."
        )

    pick_single_person_keypoints = _import_ctr_attr(
        "code.pose_estimators.yolo", "pick_single_person_keypoints"
    )

    classifier_backend = _normalize_onnx_backend_name(args.onnx_classifier_backend or "onnx_cpu")

    t0 = time.time()
    yolo_model = YOLO(str(model_path), task="pose", verbose=False)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses_list: List[np.ndarray] = []
    prev_center = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame, imgsz=640, conf=0.05, verbose=False)
        kpt, prev_center = pick_single_person_keypoints(
            results[0],
            prev_center=prev_center,
            conf_thr=0.05,
            width=width,
            height=height,
            w_conf=0.25,
            w_size=0.55,
            w_track=0.20,
        )
        poses_list.append(kpt.astype(np.float32, copy=False))
        frame_count += 1
    cap.release()

    if not poses_list:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    poses_raw = np.stack(poses_list, axis=0).astype(np.float32, copy=False)
    meta = {
        "backend": "ultralytics_coreml",
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "num_frames": int(frame_count),
        "video_path": str(video_path),
        "num_kpts": int(poses_raw.shape[1]),
    }


def _run_direct_coreml_yolo_backend(
    video_path: Path,
    backend: str,
    valmod,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import cv2
    import coremltools as ct
    from PIL import Image

    if backend != "direct_coreml_yolo":
        raise ValueError(f"Unsupported hybrid backend: {backend}")

    model_path = Path(args.ultra_yolo_model).expanduser().resolve() if args.ultra_yolo_model else _default_ultra_coreml_model()
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(
            "No CoreML YOLO model found. Export a pose model to .mlpackage/.mlmodel "
            "and pass it with --ultra-yolo-model."
        )

    classifier_backend = _normalize_onnx_backend_name(args.onnx_classifier_backend or "onnx_cpu")

    t0 = time.time()
    yolo_model = ct.models.MLModel(str(model_path))
    input_name = yolo_model.get_spec().description.input[0].name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses_list: List[np.ndarray] = []
    prev_center = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img, ratio, pad_top, pad_left = valmod._letterbox(frame, 640)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        output = yolo_model.predict({input_name: pil})
        raw = np.asarray(next(iter(output.values())), dtype=np.float32)
        detections = valmod._yolo_postprocess(raw, ratio, pad_top, pad_left, conf_thr=0.05)
        kpt, prev_center = valmod._pick_single_person(
            detections,
            num_kpts=17,
            prev_center=prev_center,
            conf_thr=0.05,
            width=width,
            height=height,
        )
        poses_list.append(kpt.astype(np.float32, copy=False))
        frame_count += 1
    cap.release()

    if not poses_list:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    poses_raw = np.stack(poses_list, axis=0).astype(np.float32, copy=False)
    meta = {
        "backend": "direct_coreml",
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "num_frames": int(frame_count),
        "video_path": str(video_path),
        "num_kpts": int(poses_raw.shape[1]),
    }
    t1 = time.time()

    poses_interp, poses_smooth, model_input, logits, clf_session = _prepare_and_classify_with_onnx(
        poses_raw, meta, classifier_backend, args, valmod
    )
    probs = _softmax(logits)
    t2 = time.time()

    return {
        "backend": backend,
        "kind": "hybrid",
        "providers": {
            "yolo": ["CoreMLDirect"],
            "classifier": clf_session.get_providers(),
        },
        "requested_runtime": {
            "yolo": "direct_coreml",
            "classifier": classifier_backend,
        },
        "ultra_yolo_model": str(model_path),
        "prediction": _label_from_probs(probs),
        "confidence": float(probs[0, int(np.argmax(probs[0]))]),
        "probabilities": {
            "pass": float(probs[0, 0]),
            "fail": float(probs[0, 1]),
        },
        "timings_sec": {
            "pose_extraction": round(t1 - t0, 3),
            "postprocess_and_classify": round(t2 - t1, 3),
            "total": round(t2 - t0, 3),
        },
        "meta": meta,
        "nonzero_keypoints_raw": int((poses_raw[:, :, 2] >= 0.05).sum()),
        "arrays": {
            "poses_raw": poses_raw,
            "poses_smooth": poses_smooth,
            "model_input": model_input,
            "logits": logits,
        },
    }
    t1 = time.time()

    poses_interp, poses_smooth, model_input, logits, clf_session = _prepare_and_classify_with_onnx(
        poses_raw, meta, classifier_backend, args, valmod
    )
    probs = _softmax(logits)
    t2 = time.time()

    return {
        "backend": backend,
        "kind": "hybrid",
        "providers": {
            "yolo": [f"Ultralytics({model_path.suffix})"],
            "classifier": clf_session.get_providers(),
        },
        "requested_runtime": {
            "yolo": "ultra_coreml",
            "classifier": classifier_backend,
        },
        "ultra_yolo_model": str(model_path),
        "prediction": _label_from_probs(probs),
        "confidence": float(probs[0, int(np.argmax(probs[0]))]),
        "probabilities": {
            "pass": float(probs[0, 0]),
            "fail": float(probs[0, 1]),
        },
        "timings_sec": {
            "pose_extraction": round(t1 - t0, 3),
            "postprocess_and_classify": round(t2 - t1, 3),
            "total": round(t2 - t0, 3),
        },
        "meta": meta,
        "nonzero_keypoints_raw": int((poses_raw[:, :, 2] >= 0.05).sum()),
        "arrays": {
            "poses_raw": poses_raw,
            "poses_smooth": poses_smooth,
            "model_input": model_input,
            "logits": logits,
        },
    }


def _run_onnx_backend(
    video_path: Path,
    backend: str,
    valmod,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    import onnxruntime as ort

    if backend not in {"onnx_cpu", "onnx_cuda", "onnx_coreml"}:
        raise ValueError(f"Unsupported ONNX backend: {backend}")

    yolo_path = MODELS_DIR / "yolo26x-pose.onnx"
    classifier_path = MODELS_DIR / "classifier.onnx"
    yolo_backend = _normalize_onnx_backend_name(args.onnx_yolo_backend or backend)
    classifier_backend = _normalize_onnx_backend_name(args.onnx_classifier_backend or backend)
    yolo_path, yolo_providers = _onnx_provider_spec(yolo_backend, yolo_path, args)
    classifier_path, classifier_providers = _onnx_provider_spec(classifier_backend, classifier_path, args)

    t0 = time.time()
    yolo_session = ort.InferenceSession(str(yolo_path), providers=yolo_providers)
    clf_session = ort.InferenceSession(str(classifier_path), providers=classifier_providers)
    yolo_model = valmod.YoloPoseONNX(str(yolo_path), session=yolo_session)

    t1 = time.time()
    poses_raw, meta = valmod.extract_poses(str(video_path), yolo_model, conf_thr=0.05)
    t2 = time.time()
    poses_interp = valmod.interpolate_poses(poses_raw, scale_factor=4, conf_thr=0.05)
    poses_smooth = valmod.smooth_poses(poses_interp, alpha=0.7, conf_thr=0.05)
    model_input = valmod.prepare_for_model(poses_smooth, meta, fixed_t=100)
    logits = clf_session.run(None, {clf_session.get_inputs()[0].name: model_input})[0]
    probs = _softmax(logits)
    t3 = time.time()

    return {
        "backend": backend,
        "kind": "onnx",
        "providers": {
            "yolo": yolo_session.get_providers(),
            "classifier": clf_session.get_providers(),
        },
        "requested_runtime": {
            "yolo": yolo_backend,
            "classifier": classifier_backend,
        },
        "coreml_options": _coreml_provider_options(args) if (
            yolo_backend == "onnx_coreml" or classifier_backend == "onnx_coreml"
        ) else {},
        "prediction": _label_from_probs(probs),
        "confidence": float(probs[0, int(np.argmax(probs[0]))]),
        "probabilities": {
            "pass": float(probs[0, 0]),
            "fail": float(probs[0, 1]),
        },
        "timings_sec": {
            "session_init": round(t1 - t0, 3),
            "pose_extraction": round(t2 - t1, 3),
            "postprocess_and_classify": round(t3 - t2, 3),
            "total": round(t3 - t0, 3),
        },
        "meta": meta,
        "nonzero_keypoints_raw": int((poses_raw[:, :, 2] >= 0.05).sum()),
        "arrays": {
            "poses_raw": poses_raw,
            "poses_smooth": poses_smooth,
            "model_input": model_input,
            "logits": logits,
        },
    }


def _run_pt_backend(video_path: Path, backend: str) -> Dict[str, Any]:
    import torch

    if backend not in {"pt_cpu", "pt_cuda", "pt_mps"}:
        raise ValueError(f"Unsupported PyTorch backend: {backend}")

    load_model = _import_ctr_attr("code.pose_estimators.yolo", "load_model")
    extract_pose_from_video = _import_ctr_attr("code.pose_estimators.yolo", "extract_pose_from_video")
    InterpolationConfig = _import_ctr_attr("code.preprocessing.pose_interpolation", "InterpolationConfig")
    interpolate_pose_sequence = _import_ctr_attr("code.preprocessing.pose_interpolation", "interpolate_pose_sequence")
    SmoothingConfig = _import_ctr_attr("code.preprocessing.pose_smoothing", "SmoothingConfig")
    EMAPoseSmoother = _import_ctr_attr("code.preprocessing.pose_smoothing", "EMAPoseSmoother")
    uniform_sample_indices = _import_ctr_attr("code.inference.dataset_builder", "uniform_sample_indices")
    Model = _import_ctr_attr("model.ctrgcn", "Model")

    device = backend.split("_", 1)[1]

    t0 = time.time()
    pt_yolo = load_model(str(MODELS_DIR / "yolo26x-pose.pt"))

    poses_raw, meta = extract_pose_from_video(
        str(video_path),
        pt_yolo,
        device=device,
        batch_size=8,
        verbose=False,
        conf_thr=0.05,
    )
    t1 = time.time()

    interp_cfg = InterpolationConfig(
        scale_factor=4,
        mode="linear",
        conf_thr=0.05,
        frame_min_kpts=8,
    )
    poses_interp = interpolate_pose_sequence(poses_raw, interp_cfg)
    smoother = EMAPoseSmoother(
        SmoothingConfig(
            alpha=0.7,
            conf_thr=0.05,
            smooth_conf=False,
            missing_policy="hold",
        )
    )
    poses_smooth = smoother.smooth_sequence(poses_interp)

    poses_pp = poses_smooth.astype(np.float32, copy=True)
    poses_pp[..., 0] /= float(meta["width"])
    poses_pp[..., 1] /= float(meta["height"])
    if poses_pp.shape[0] >= 100:
        idx = uniform_sample_indices(poses_pp.shape[0], 100)
        poses_pp = poses_pp[idx]
    else:
        padded = np.zeros((100, poses_pp.shape[1], poses_pp.shape[2]), dtype=np.float32)
        padded[: poses_pp.shape[0]] = poses_pp
        poses_pp = padded
    model_input = poses_pp.transpose(2, 0, 1)[..., None][None].astype(np.float32)

    checkpoint = torch.load(CTR_ROOT / "production" / "best_model.pt", map_location=device)
    model = Model(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(model_input).to(device)).detach().cpu().numpy()
    probs = _softmax(logits)
    t2 = time.time()

    return {
        "backend": backend,
        "kind": "pytorch",
        "providers": [device],
        "prediction": _label_from_probs(probs),
        "confidence": float(probs[0, int(np.argmax(probs[0]))]),
        "probabilities": {
            "pass": float(probs[0, 0]),
            "fail": float(probs[0, 1]),
        },
        "timings_sec": {
            "pose_extraction": round(t1 - t0, 3),
            "postprocess_and_classify": round(t2 - t1, 3),
            "total": round(t2 - t0, 3),
        },
        "meta": meta,
        "nonzero_keypoints_raw": int((poses_raw[:, :, 2] >= 0.05).sum()),
        "arrays": {
            "poses_raw": poses_raw,
            "poses_smooth": poses_smooth,
            "model_input": model_input,
            "logits": logits,
        },
    }


def _run_backend(video_path: Path, backend: str, valmod, args: argparse.Namespace) -> Dict[str, Any]:
    try:
        if backend.startswith("onnx_"):
            result = _run_onnx_backend(video_path, backend, valmod, args)
        elif backend.startswith("pt_"):
            result = _run_pt_backend(video_path, backend)
        elif backend == "ultra_coreml_yolo":
            result = _run_ultra_coreml_yolo_backend(video_path, backend, valmod, args)
        elif backend == "direct_coreml_yolo":
            result = _run_direct_coreml_yolo_backend(video_path, backend, valmod, args)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        result["status"] = "ok"
        return result
    except Exception as exc:
        return {
            "backend": backend,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _compare_to_reference(reference: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "poses_raw": _diff_stats(reference["arrays"]["poses_raw"], current["arrays"]["poses_raw"]),
        "poses_smooth": _diff_stats(reference["arrays"]["poses_smooth"], current["arrays"]["poses_smooth"]),
        "model_input": _diff_stats(reference["arrays"]["model_input"], current["arrays"]["model_input"]),
        "logits": _diff_stats(reference["arrays"]["logits"], current["arrays"]["logits"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX inference backends on one video.")
    parser.add_argument("--video", required=True, help="Absolute or relative path to the video file.")
    parser.add_argument(
        "--backends",
        nargs="*",
        default=None,
        help=(
            "Backends to run. Default: all detected. "
            "Choices include pt_cpu pt_cuda pt_mps onnx_cpu onnx_cuda onnx_coreml "
            "ultra_coreml_yolo direct_coreml_yolo."
        ),
    )
    parser.add_argument(
        "--reference-backend",
        default=None,
        help="Backend to use as the diff reference. Default: pt_cpu if available, else first successful backend.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON output path for the full report.",
    )
    parser.add_argument(
        "--onnx-yolo-backend",
        choices=["cpu", "cuda", "coreml", "onnx_cpu", "onnx_cuda", "onnx_coreml"],
        default=None,
        help="Override ONNX runtime backend for YOLO only.",
    )
    parser.add_argument(
        "--onnx-classifier-backend",
        choices=["cpu", "cuda", "coreml", "onnx_cpu", "onnx_cuda", "onnx_coreml"],
        default=None,
        help="Override ONNX runtime backend for the classifier only.",
    )
    parser.add_argument(
        "--coreml-model-format",
        choices=["MLProgram", "NeuralNetwork"],
        default=None,
        help="Optional CoreMLExecutionProvider ModelFormat override.",
    )
    parser.add_argument(
        "--coreml-compute-units",
        choices=["CPUOnly", "CPUAndNeuralEngine", "CPUAndGPU", "ALL"],
        default=None,
        help="Optional CoreMLExecutionProvider MLComputeUnits override.",
    )
    parser.add_argument(
        "--coreml-require-static-input-shapes",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional CoreMLExecutionProvider RequireStaticInputShapes override.",
    )
    parser.add_argument(
        "--coreml-enable-on-subgraphs",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional CoreMLExecutionProvider EnableOnSubgraphs override.",
    )
    parser.add_argument(
        "--coreml-profile-compute-plan",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional CoreMLExecutionProvider ProfileComputePlan override.",
    )
    parser.add_argument(
        "--coreml-allow-low-precision-gpu-accumulation",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional CoreMLExecutionProvider AllowLowPrecisionAccumulationOnGPU override.",
    )
    parser.add_argument(
        "--coreml-cache-dir",
        default=None,
        help="Optional CoreMLExecutionProvider ModelCacheDirectory override.",
    )
    parser.add_argument(
        "--ultra-yolo-model",
        default=None,
        help="Path to a direct Ultralytics CoreML YOLO pose export (.mlpackage or .mlmodel).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return 1

    valmod = _load_app_validation_module()

    detected_backends = _available_pt_backends() + _available_onnx_backends() + _available_hybrid_backends()
    backends = args.backends or detected_backends
    invalid = [
        b for b in backends
        if b not in detected_backends
        and b not in {
            "pt_cpu", "pt_cuda", "pt_mps",
            "onnx_cpu", "onnx_cuda", "onnx_coreml",
            "ultra_coreml_yolo", "direct_coreml_yolo",
        }
    ]
    if invalid:
        print(f"[ERROR] Unknown backend(s): {', '.join(invalid)}")
        return 1

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Detected backends: {', '.join(detected_backends)}")
    print(f"[INFO] Requested backends: {', '.join(backends)}")

    results: List[Dict[str, Any]] = []
    for backend in backends:
        print(f"\n=== Running {backend} ===")
        res = _run_backend(video_path, backend, valmod, args)
        results.append(res)
        if res["status"] == "ok":
            print(
                f"[OK] {backend}: pred={res['prediction']} "
                f"pass={res['probabilities']['pass']:.6f} "
                f"fail={res['probabilities']['fail']:.6f} "
                f"time={res['timings_sec']['total']:.3f}s"
            )
        else:
            print(f"[ERR] {backend}: {res['error']}")

    successful = [r for r in results if r["status"] == "ok"]
    if not successful:
        print("\n[ERROR] No backend completed successfully.")
        return 1

    reference_name = args.reference_backend or ("pt_cpu" if any(r["backend"] == "pt_cpu" for r in successful) else successful[0]["backend"])
    reference = next((r for r in successful if r["backend"] == reference_name), None)
    if reference is None:
        print(f"[WARN] Reference backend {reference_name} not available; using {successful[0]['backend']} instead.")
        reference = successful[0]
        reference_name = reference["backend"]

    print(f"\n[INFO] Reference backend: {reference_name}")
    for res in successful:
        if res["backend"] == reference_name:
            continue
        diffs = _compare_to_reference(reference, res)
        res["diff_vs_reference"] = diffs
        print(f"\n--- {res['backend']} vs {reference_name} ---")
        for key in ("poses_raw", "poses_smooth", "model_input", "logits"):
            stats = diffs[key]
            if stats["same_shape"]:
                print(
                    f"{key}: max_abs_diff={stats['max_abs_diff']:.6f}, "
                    f"mean_abs_diff={stats['mean_abs_diff']:.6f}"
                )
            else:
                print(f"{key}: shape mismatch {stats['shape_a']} vs {stats['shape_b']}")

    report = {
        "video": str(video_path),
        "reference_backend": reference_name,
        "results": [],
    }
    for res in results:
        clean = {k: v for k, v in res.items() if k != "arrays"}
        report["results"].append(_to_jsonable(clean))

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
    else:
        out_path = Path.cwd() / f"backend_compare_{video_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[INFO] Wrote report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
