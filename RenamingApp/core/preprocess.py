from __future__ import annotations

from typing import List

from .models import HitlCallbacks, LogFn, ProcessingCancelled, StopCheck, TipperInfo
from .tipper import update_tipper_result


def preprocess_tippers(
    tippers: List[TipperInfo],
    callbacks: HitlCallbacks,
    log: LogFn,
    stop_requested: StopCheck,
) -> List[TipperInfo]:
    """Review angle-0/undecided tippers with HITL before matching."""
    processed: List[TipperInfo] = []
    for tipper in tippers:
        if stop_requested():
            raise ProcessingCancelled("Cancelled while preprocessing tippers.")
        if tipper.result == "U" and tipper.angle is not None and abs(tipper.angle) < 1e-9:
            decision = callbacks.decide_angle_zero(tipper)
            if decision is None:
                log(f"  - Deleting/Skipping tipper {tipper.path.name}")
                continue
            if decision != tipper.result:
                tipper = update_tipper_result(tipper, decision, log)
        processed.append(tipper)
    processed.sort(key=lambda t: (t.time_tuple, t.path.name))
    return processed

