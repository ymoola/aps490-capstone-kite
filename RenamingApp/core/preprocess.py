from __future__ import annotations

from typing import List, Optional

from .models import HitlCallbacks, LogFn, ProcessingCancelled, StopCheck, TipperInfo
from .reporting import ReportCollector
from .tipper import update_tipper_result

_ALLOWED_ANGLE_DECISIONS = {"U", "P", "F"}


def _normalize_angle_decision(decision: Optional[str], fallback: str, log: LogFn) -> Optional[str]:
    if decision is None:
        return None
    normalized = decision.strip().upper()
    if normalized in _ALLOWED_ANGLE_DECISIONS:
        return normalized
    log(f"[WARN] Ignoring unsupported angle decision '{decision}'. Keeping original tipper result.")
    return fallback


def preprocess_tippers(
    tippers: List[TipperInfo],
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
    date: str,
    sub: str,
    dry_run: bool = False,
    reporter: Optional[ReportCollector] = None,
) -> List[TipperInfo]:
    """Review angle-0/undecided tippers with HITL before matching."""
    processed: List[TipperInfo] = []
    for tipper in tippers:
        if stop_requested():
            raise ProcessingCancelled("Cancelled while preprocessing tippers.")
        if tipper.result == "U" and tipper.angle is not None and abs(tipper.angle) < 1e-9:
            try:
                decision = callbacks.decide_angle_zero(tipper)
            except ProcessingCancelled:
                raise
            except Exception as exc:
                log(f"[WARN] Failed to collect angle decision for {tipper.path.name}: {exc}")
                decision = tipper.result

            normalized_decision = _normalize_angle_decision(decision, tipper.result, log)
            if normalized_decision is None:
                log(f"  - Deleting/Skipping tipper {tipper.path.name}")
                continue
            if normalized_decision != tipper.result:
                tipper = update_tipper_result(
                    tipper,
                    normalized_decision,
                    log,
                    dry_run=dry_run,
                    reporter=reporter,
                    date=date,
                    sub=sub,
                )
        processed.append(tipper)
    processed.sort(key=lambda t: (t.time_tuple, t.path.name))
    return processed
