# data_splitter.py
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


@dataclass(frozen=True)
class DatumMeta:
    # paths
    npz_path: str            # absolute path to npz
    rel_path: str            # relative to out_root

    # participant identity
    participant_dir: str     # folder one level above npz (split key)
    participant_id: str      # parsed from filename (may or may not match dir)

    # sample identity
    footwear_id: str

    # label
    label_code: str          # DF/DP/UF/UP (uppercased)
    label_binary: str        # "pass" or "fail"
    slope_dir: str           # "downhill" or "uphill"

    # capture info
    angle_deg: int
    time_str: str            # "HH-MM-SS" from filename

    # processing tag
    stage_tag: str           # e.g., raw_interp_smooth


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 12345
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


@dataclass(frozen=True)
class SplitResult:
    config: SplitConfig
    out_root: str
    counts: Dict[str, Dict[str, int]]  # participants/videos per split
    participants: Dict[str, List[str]] # split -> list of participant_dir
    warnings: List[str]                # any filename/dir mismatches etc.


# -----------------------------
# Public API
# -----------------------------
def abspath(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def iter_npz_files(out_root: str) -> Iterable[str]:
    """Recursively yield absolute .npz paths."""
    out_root_abs = abspath(out_root)
    for r, _, files in os.walk(out_root_abs):
        for f in files:
            if f.lower().endswith(".npz"):
                yield os.path.join(r, f)


def participant_dir_from_npz(npz_abs: str) -> str:
    """
    Participant is defined as the folder one level above the npz.
    Example: out/2025-0205/sub323/datum.npz -> sub323
    """
    parent = os.path.basename(os.path.dirname(npz_abs))
    if not parent:
        raise RuntimeError(f"Could not infer participant_dir from path: {npz_abs}")
    return parent


def extract_angle_from_rest(rest: str) -> Optional[float]:
    """
    Extract first numeric token from rest (int or float).
    Returns None if not found.
    Examples:
      "11_15-12-08"        -> 11.0
      "12.1_GP0_13-24-00" -> 12.1
      "0_GP1_14-15-14"    -> 0.0
    """
    for token in rest.split("_"):
        if re.fullmatch(r"\d+(\.\d+)?", token):
            return float(token)
    return None


def parse_npz_filename(npz_abs: str) -> DatumMeta:
    """
    Parse metadata from the filename, plus participant_dir from folder.
    Raises ValueError if the filename doesn't match the expected pattern.
    """
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

    # best-effort time extraction (optional, non-fatal)
    time_match = re.search(r"\d{2}-\d{2}-\d{2}", rest)
    time_str = time_match.group(0) if time_match else "unknown"
    stage_tag = m.group("tag")

    # Label decoding
    # DF/DP/UF/UP:
    #   first char: D/U  => downhill / uphill
    #   second char: F/P => fail / pass
    slope_dir = "downhill" if label_code[0] == "D" else "uphill"

    if label_code[1] == "F":
        label_binary = "fail"
    elif label_code[1] == "P":
        label_binary = "pass"
    else:
        label_binary = "fail"
        print(f"[warn] label is unknown and overrode as fail (kept): {label_code} → {npz_abs}")



    participant_dir = participant_dir_from_npz(npz_abs)

    if participant_dir != participant_id:
        # not fatal, but suspicious
        print(
            f"[warn] participant_dir '{participant_dir}' does not match "
            f"participant_id '{participant_id}' in filename: {npz_abs}"
        )

    # relative path will be filled by caller when out_root known
    return DatumMeta(
        npz_path=npz_abs,
        rel_path="",  # placeholder; caller fills
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
    """
    Walks out_root, parses all .npz files, returns:
      - list of DatumMeta
      - warnings
    """
    out_root_abs = abspath(out_root)
    items: List[DatumMeta] = []
    warnings: List[str] = []

    for npz_abs in iter_npz_files(out_root_abs):
        try:
            meta = parse_npz_filename(npz_abs)
            meta = DatumMeta(**{**asdict(meta), "rel_path": os.path.relpath(npz_abs, out_root_abs)})
            items.append(meta)

            # warn if participant_dir doesn't match participant_id (sub323 vs sub356)
            if meta.participant_dir.lower() != meta.participant_id.lower():
                warnings.append(
                    f"[warn] participant mismatch: dir='{meta.participant_dir}' vs filename='{meta.participant_id}' "
                    f"for {meta.rel_path.replace(os.sep, '/')}"
                )

        except Exception as e:
            warnings.append(f"[warn] skipping npz (parse failed): {npz_abs} السبب={e}")

    if not items:
        raise RuntimeError(f"No valid .npz pose files found under: {out_root_abs}")

    return items, warnings


def split_participants(
    participant_dirs: List[str],
    cfg: SplitConfig,
) -> Tuple[List[str], List[str], List[str]]:
    s = cfg.train_frac + cfg.val_frac + cfg.test_frac
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {s}")

    rng = random.Random(cfg.seed)
    parts = participant_dirs[:]
    rng.shuffle(parts)

    n = len(parts)
    n_train = int(round(n * cfg.train_frac))
    n_val = int(round(n * cfg.val_frac))

    # clamp then ensure total coverage
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    # remainder goes to test
    train_p = parts[:n_train]
    val_p = parts[n_train:n_train + n_val]
    test_p = parts[n_train + n_val:]

    # disjoint sanity
    assert len(set(train_p) & set(val_p)) == 0
    assert len(set(train_p) & set(test_p)) == 0
    assert len(set(val_p) & set(test_p)) == 0
    assert len(train_p) + len(val_p) + len(test_p) == n

    return train_p, val_p, test_p


def make_splits(
    out_root: str,
    cfg: SplitConfig,
) -> Tuple[SplitResult, Dict[str, List[DatumMeta]]]:
    """
    Returns:
      split_result: summary info
      split_items: dict: {"train": [...], "val": [...], "test": [...]}

    Splitting key is participant_dir (folder one level above npz).
    """
    out_root_abs = abspath(out_root)
    items, warnings = build_index(out_root_abs)

    # group by participant_dir (this is the split key)
    by_p: Dict[str, List[DatumMeta]] = {}
    for it in items:
        by_p.setdefault(it.participant_dir, []).append(it)

    participant_dirs = sorted(by_p.keys())
    train_p, val_p, test_p = split_participants(participant_dirs, cfg)

    def expand(pids: List[str]) -> List[DatumMeta]:
        out: List[DatumMeta] = []
        for pid in pids:
            # stable order per participant
            out.extend(sorted(by_p[pid], key=lambda x: x.rel_path))
        return out

    split_items = {
        "train": expand(train_p),
        "val": expand(val_p),
        "test": expand(test_p),
    }

    counts = {
        "participants": {
            "train": len(train_p),
            "val": len(val_p),
            "test": len(test_p),
            "total": len(participant_dirs),
        },
        "samples": {
            "train": len(split_items["train"]),
            "val": len(split_items["val"]),
            "test": len(split_items["test"]),
            "total": len(items),
        },
    }

    result = SplitResult(
        config=cfg,
        out_root=out_root_abs,
        counts=counts,
        participants={"train": train_p, "val": val_p, "test": test_p},
        warnings=warnings,
    )
    return result, split_items


def write_split_artifacts(
    out_dir: str,
    split_result: SplitResult,
    split_items: Dict[str, List[DatumMeta]],
) -> None:
    """
    Writes:
      - splits.json (summary)
      - train.jsonl / val.jsonl / test.jsonl (one DatumMeta dict per line)
    """
    out_dir_abs = abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    # summary json
    summary_path = os.path.join(out_dir_abs, "splits.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "out_root": split_result.out_root,
                "config": asdict(split_result.config),
                "counts": split_result.counts,
                "participants": split_result.participants,
                "warnings": split_result.warnings,
                "notes": {
                    "split_key": "participant_dir (folder one level above each npz)",
                    "label_binary": "fail if label_code endswith 'F', else pass",
                    "label_code": "DF/DP/UF/UP",
                },
            },
            f,
            indent=2,
        )

    # jsonl manifests
    for split_name, items in split_items.items():
        path = os.path.join(out_dir_abs, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                d = asdict(it)
                # normalize paths for cross-tooling
                d["npz_path"] = d["npz_path"].replace("\\", "/")
                d["rel_path"] = d["rel_path"].replace("\\", "/")
                f.write(json.dumps(d) + "\n")


# -----------------------------
# Optional: quick self-test hook
# (You can delete this; it doesn't add CLI args.)
# -----------------------------
if __name__ == "__main__":
    # EDIT THESE IF YOU RUN THIS MODULE DIRECTLY
    OUT_ROOT = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\out"
    SPLIT_OUT_DIR = r"D:\Brad\School\UofT\Year4\CSC494_eng\aps490-capstone-kite\CV\splits"
    CFG = SplitConfig(seed=12345, train_frac=0.70, val_frac=0.15, test_frac=0.15)

    res, split_items = make_splits(OUT_ROOT, CFG)
    write_split_artifacts(SPLIT_OUT_DIR, res, split_items)

    print("[data_splitter] Done.")
    print("[data_splitter] Counts:", res.counts)
    if res.warnings:
        print(f"[data_splitter] Warnings: {len(res.warnings)} (see splits.json)")
