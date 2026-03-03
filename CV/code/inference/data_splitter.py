from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple


FILENAME_RE = re.compile(
    r"^(?P<footwear>[^_]+)_"
    r"(?P<participant>sub\d+)_"
    r"(?P<label>[DU][PFU])_"
    r"(?P<rest>.+?)_"
    r"(?P<tag>raw_interp_smooth)"
    r"\.npz$",
    re.IGNORECASE,
)

# --- NEW: participant normalization ---
SUB_RE = re.compile(r"(sub\d+)", re.IGNORECASE)


def normalize_participant_key(s: str) -> str:
    """
    Convert any string containing 'sub###' into normalized participant key 'sub###' (lowercase).
    If it doesn't contain sub###, return original lowercased string.
    """
    m = SUB_RE.search(s)
    if not m:
        return s.strip().lower()
    return m.group(1).lower()


@dataclass(frozen=True)
class DatumMeta:
    npz_path: str
    rel_path: str

    participant_dir: str
    participant_id: str

    footwear_id: str

    label_code: str
    label_binary: str
    slope_dir: str

    angle_deg: int
    time_str: str

    stage_tag: str


# ---- OLD SplitConfig/SplitResult can remain if you still want the single split mode ----

# --- NEW: K-fold config/result ---
@dataclass(frozen=True)
class KFoldConfig:
    seed: int = 12345
    k: int = 5
    val_strategy: str = "next_fold"  # currently only "next_fold"


@dataclass(frozen=True)
class FoldSummary:
    fold_index: int
    participants: Dict[str, List[str]]        # train/val/test participant keys
    counts: Dict[str, Dict[str, int]]         # participants/samples per split


@dataclass(frozen=True)
class KFoldResult:
    config: KFoldConfig
    out_root: str
    folds: List[FoldSummary]
    warnings: List[str]


def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def iter_npz_files(out_root: str) -> Iterable[str]:
    out_root_abs = abspath(out_root)
    for r, _, files in os.walk(out_root_abs):
        for f in files:
            if f.lower().endswith(".npz"):
                yield os.path.join(r, f)


def participant_dir_from_npz(npz_abs: str) -> str:
    parent = os.path.basename(os.path.dirname(npz_abs))
    if not parent:
        raise RuntimeError(f"Could not infer participant_dir from path: {npz_abs}")
    return parent


def extract_angle_from_rest(rest: str) -> Optional[float]:
    for token in rest.split("_"):
        if re.fullmatch(r"\d+(\.\d+)?", token):
            return float(token)
    return None


def parse_npz_filename(npz_abs: str) -> DatumMeta:
    base = os.path.basename(npz_abs)
    m = FILENAME_RE.match(base)
    if not m:
        raise ValueError(
            f"NPZ filename did not match expected pattern:\n"
            f"  {base}\n"
            f"Expected like:idaptXXX_subYYY_DF_<anything>_raw_interp_smooth.npz"
        )

    footwear_id = m.group("footwear")
    participant_id = m.group("participant")
    label_code = m.group("label").upper()
    rest = m.group("rest")

    angle_val = extract_angle_from_rest(rest)
    angle_deg = int(round(angle_val)) if angle_val is not None else -1

    time_match = re.search(r"\d{2}-\d{2}-\d{2}", rest)
    time_str = time_match.group(0) if time_match else "unknown"
    stage_tag = m.group("tag")

    slope_dir = "downhill" if label_code[0] == "D" else "uphill"

    if label_code[1] == "F":
        label_binary = "fail"
    elif label_code[1] == "P":
        label_binary = "pass"
    else:
        label_binary = "fail"
        print(f"[warn] label is unknown and overrode as fail (kept): {label_code} → {npz_abs}")

    participant_dir = participant_dir_from_npz(npz_abs)

    if participant_dir.lower() != participant_id.lower():
        print(
            f"[warn] participant_dir '{participant_dir}' does not match "
            f"participant_id '{participant_id}' in filename: {npz_abs}"
        )

    return DatumMeta(
        npz_path=npz_abs,
        rel_path="",  # filled later
        participant_dir=participant_dir,
        participant_id=participant_id,
        footwear_id=footwear_id,
        label_code=label_code,
        label_binary=label_binary,
        slope_dir=slope_dir,
        angle_deg=angle_deg,
        time_str=time_str,
        stage_tag=stage_tag,
    )


def build_index(out_root: str) -> Tuple[List[DatumMeta], List[str]]:
    out_root_abs = abspath(out_root)
    items: List[DatumMeta] = []
    warnings: List[str] = []

    for npz_abs in iter_npz_files(out_root_abs):
        try:
            meta = parse_npz_filename(npz_abs)
            meta = DatumMeta(**{**asdict(meta), "rel_path": os.path.relpath(npz_abs, out_root_abs)})
            items.append(meta)

            if meta.participant_dir.lower() != meta.participant_id.lower():
                warnings.append(
                    f"[warn] participant mismatch: dir='{meta.participant_dir}' vs filename='{meta.participant_id}' "
                    f"for {meta.rel_path.replace(os.sep, '/')}"
                )
        except Exception as e:
            warnings.append(f"[warn] skipping npz (parse failed): {npz_abs} reason={e}")

    if not items:
        raise RuntimeError(f"No valid .npz pose files found under: {out_root_abs}")

    return items, warnings


# -----------------------------
# NEW: 5-fold participant CV
# -----------------------------
def _chunk_into_k(parts: List[str], k: int) -> List[List[str]]:
    """
    Split list into k chunks as evenly as possible: sizes differ by at most 1.
    """
    n = len(parts)
    base = n // k
    rem = n % k
    chunks: List[List[str]] = []
    idx = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        chunks.append(parts[idx:idx + size])
        idx += size
    return chunks


def make_kfold_splits(
    out_root: str,
    cfg: KFoldConfig,
) -> Tuple[KFoldResult, Dict[int, Dict[str, List[DatumMeta]]]]:
    """
    Participant-level K-fold CV.

    Returns:
      result: fold summaries + warnings
      fold_items: dict[fold_index] -> {"train":[...], "val":[...], "test":[...]} expanded per npz
    """
    out_root_abs = abspath(out_root)
    items, warnings = build_index(out_root_abs)

    # group by normalized participant key (sub###)
    by_p: Dict[str, List[DatumMeta]] = {}
    for it in items:
        key = normalize_participant_key(it.participant_dir)
        by_p.setdefault(key, []).append(it)

    participant_keys = sorted(by_p.keys())
    if cfg.k < 2:
        raise ValueError("k must be >= 2")
    if len(participant_keys) < cfg.k:
        raise ValueError(f"Not enough participants ({len(participant_keys)}) for k={cfg.k}")

    rng = random.Random(cfg.seed)
    shuffled = participant_keys[:]
    rng.shuffle(shuffled)

    folds_p = _chunk_into_k(shuffled, cfg.k)  # list of participant lists

    def expand(pids: List[str]) -> List[DatumMeta]:
        out: List[DatumMeta] = []
        for pid in pids:
            out.extend(sorted(by_p[pid], key=lambda x: x.rel_path))
        return out

    fold_items: Dict[int, Dict[str, List[DatumMeta]]] = {}
    fold_summaries: List[FoldSummary] = []

    for i in range(cfg.k):
        test_p = folds_p[i]

        if cfg.val_strategy != "next_fold":
            raise ValueError(f"Unsupported val_strategy={cfg.val_strategy!r}")

        val_p = folds_p[(i + 1) % cfg.k]
        train_p: List[str] = []
        for j in range(cfg.k):
            if j == i or j == (i + 1) % cfg.k:
                continue
            train_p.extend(folds_p[j])

        split_items = {
            "train": expand(train_p),
            "val": expand(val_p),
            "test": expand(test_p),
        }
        fold_items[i] = split_items

        counts = {
            "participants": {
                "train": len(train_p),
                "val": len(val_p),
                "test": len(test_p),
                "total": len(participant_keys),
            },
            "samples": {
                "train": len(split_items["train"]),
                "val": len(split_items["val"]),
                "test": len(split_items["test"]),
                "total": len(items),
            },
        }

        fold_summaries.append(
            FoldSummary(
                fold_index=i,
                participants={"train": train_p, "val": val_p, "test": test_p},
                counts=counts,
            )
        )

    result = KFoldResult(
        config=cfg,
        out_root=out_root_abs,
        folds=fold_summaries,
        warnings=warnings,
    )
    return result, fold_items


def write_kfold_artifacts(
    out_dir: str,
    kfold_result: KFoldResult,
    fold_items: Dict[int, Dict[str, List[DatumMeta]]],
) -> None:
    """
    Writes:
      - cv_splits.json
      - fold_{i}_train.jsonl / fold_{i}_val.jsonl / fold_{i}_test.jsonl
    """
    out_dir_abs = abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    summary_path = os.path.join(out_dir_abs, "cv_splits.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "out_root": kfold_result.out_root,
                "config": asdict(kfold_result.config),
                "folds": [asdict(fs) for fs in kfold_result.folds],
                "warnings": kfold_result.warnings,
                "notes": {
                    "split_key": "participant (normalized sub### extracted from folder name one level above each npz)",
                    "val_strategy": "val is next fold after test fold (cyclic)",
                    "label_binary": "fail if label_code endswith 'F', else pass",
                    "label_code": "DF/DP/UF/UP",
                },
            },
            f,
            indent=2,
        )

    for fold_idx, splits in fold_items.items():
        for split_name, items in splits.items():
            path = os.path.join(out_dir_abs, f"fold_{fold_idx}_{split_name}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for it in items:
                    d = asdict(it)
                    d["npz_path"] = d["npz_path"].replace("\\", "/")
                    d["rel_path"] = d["rel_path"].replace("\\", "/")
                    # also add a stable participant key (helpful downstream)
                    d["participant_key"] = normalize_participant_key(it.participant_dir)
                    f.write(json.dumps(d) + "\n")


# -----------------------------
# Optional self-test hook
# -----------------------------
if __name__ == "__main__":
    OUT_ROOT = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\outputs\out_yolo"
    SPLIT_OUT_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\data\cv_splits"

    CFG = KFoldConfig(seed=12345, k=5, val_strategy="next_fold")

    res, fold_items = make_kfold_splits(OUT_ROOT, CFG)
    write_kfold_artifacts(SPLIT_OUT_DIR, res, fold_items)

    print("[data_splitter] Done.")
    for fs in res.folds:
        c = fs.counts
        print(
            f"fold {fs.fold_index}: "
            f"p(train/val/test)=({c['participants']['train']}/{c['participants']['val']}/{c['participants']['test']}), "
            f"s(train/val/test)=({c['samples']['train']}/{c['samples']['val']}/{c['samples']['test']})"
        )
    if res.warnings:
        print(f"[data_splitter] Warnings: {len(res.warnings)} (see cv_splits.json)")