#!/usr/bin/env python3
"""
Batch-run the direct CoreML YOLO + ONNX CTR-GCN validation path on a folder.

This is a debug-only runner intended to evaluate the Mac-accelerated path
without touching the desktop app:

  YOLO pose extraction: direct Ultralytics CoreML export (.mlpackage/.mlmodel)
  CTR-GCN classifier: ONNX Runtime (CoreML or CPU)

It writes an Excel report in the same format as the app validation report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parent.parent
APP_VALIDATION_PATH = Path(__file__).resolve().parent / "RenamingApp" / "core" / "validation.py"
DEBUG_COMPARE_PATH = Path(__file__).resolve().parent / "debug_backend_compare.py"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _iter_videos(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def _default_report_path(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"ultra_coreml_batch_{root.name}_{stamp}.xlsx"


def _default_json_path(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"ultra_coreml_batch_{root.name}_{stamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct CoreML YOLO + ONNX CTR-GCN on every video in a folder."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Folder containing the videos to evaluate.",
    )
    parser.add_argument(
        "--ultra-yolo-model",
        required=True,
        help="Path to the direct Ultralytics CoreML YOLO pose export (.mlpackage or .mlmodel).",
    )
    parser.add_argument(
        "--onnx-classifier-backend",
        default="coreml",
        choices=["cpu", "coreml", "onnx_cpu", "onnx_coreml"],
        help="Backend for the ONNX CTR-GCN classifier. Default: coreml.",
    )
    parser.add_argument(
        "--xlsx-out",
        default=None,
        help="Optional output path for the Excel report.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path for the JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] Folder not found: {root}")
        return 1

    validation = _load_module("slopesense_validation_batch", APP_VALIDATION_PATH)
    debug_compare = _load_module("slopesense_debug_compare_batch", DEBUG_COMPARE_PATH)

    videos = list(_iter_videos(root))
    if not videos:
        print(f"[ERROR] No videos found under: {root}")
        return 1

    run_args = SimpleNamespace(
        onnx_classifier_backend=args.onnx_classifier_backend,
        ultra_yolo_model=str(Path(args.ultra_yolo_model).expanduser().resolve()),
        onnx_yolo_backend=None,
        coreml_model_format=None,
        coreml_compute_units=None,
        coreml_require_static_input_shapes=None,
        coreml_enable_on_subgraphs=None,
        coreml_profile_compute_plan=None,
        coreml_allow_low_precision_gpu_accumulation=None,
        coreml_cache_dir=None,
    )

    results: List[Any] = []
    errors: List[Dict[str, Any]] = []

    print(f"[INFO] Running ultra_coreml_yolo batch on {len(videos)} videos from {root}")
    print(
        "[INFO] Backends: "
        f"YOLO=Ultralytics CoreML ({run_args.ultra_yolo_model}), "
        f"CTR-GCN={args.onnx_classifier_backend}"
    )

    for idx, video_path in enumerate(videos, start=1):
        print(f"[INFO] ({idx}/{len(videos)}) {video_path.name}")
        outcome = debug_compare._run_backend(
            video_path=video_path,
            backend="ultra_coreml_yolo",
            valmod=validation,
            args=run_args,
        )

        direction, result_code = validation.extract_tipper_label(video_path.name)
        tipper_readable = validation._RESULT_READABLE.get(result_code, "Unknown")

        if outcome["status"] == "ok":
            pred_label = outcome["prediction"]
            pred_prob = outcome["confidence"]
            labels_match = tipper_readable in ("Pass", "Fail") and tipper_readable == pred_label
            print(
                f"  [OK] pred={pred_label} "
                f"pass={outcome['probabilities']['pass']:.6f} "
                f"fail={outcome['probabilities']['fail']:.6f} "
                f"time={outcome['timings_sec']['total']:.3f}s"
            )
            results.append(
                validation.ValidationResult(
                    original_video=video_path.name,
                    renamed_video=video_path.name,
                    renamed_video_path=str(video_path),
                    tipper_label=tipper_readable,
                    predicted_label=pred_label,
                    predicted_prob=pred_prob,
                    labels_match=labels_match,
                )
            )
        else:
            print(f"  [ERR] {outcome['error']}")
            results.append(
                validation.ValidationResult(
                    original_video=video_path.name,
                    renamed_video=video_path.name,
                    renamed_video_path=str(video_path),
                    tipper_label=tipper_readable,
                    predicted_label="Error",
                    predicted_prob=0.0,
                    labels_match=False,
                    error=outcome["error"],
                )
            )
            errors.append(
                {
                    "video": str(video_path),
                    "error": outcome["error"],
                }
            )

    xlsx_out = Path(args.xlsx_out).expanduser().resolve() if args.xlsx_out else _default_report_path(root)
    xlsx_out.parent.mkdir(parents=True, exist_ok=True)
    validation.write_validation_report(results, xlsx_out)

    comparable = [r for r in results if r.tipper_label in ("Pass", "Fail") and r.predicted_label in ("Pass", "Fail")]
    matches = sum(1 for r in comparable if r.labels_match)
    summary = {
        "root": str(root),
        "ultra_yolo_model": run_args.ultra_yolo_model,
        "onnx_classifier_backend": args.onnx_classifier_backend,
        "num_videos": len(results),
        "num_comparable": len(comparable),
        "num_matches": matches,
        "accuracy": (matches / len(comparable)) if comparable else None,
        "num_errors": len(errors),
        "report_xlsx": str(xlsx_out),
        "errors": errors,
    }

    json_out = Path(args.json_out).expanduser().resolve() if args.json_out else _default_json_path(root)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[INFO] Comparable accuracy: {matches}/{len(comparable)}"
        if comparable else "[INFO] No comparable labels found."
    )
    print(f"[INFO] Excel report: {xlsx_out}")
    print(f"[INFO] JSON summary: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
