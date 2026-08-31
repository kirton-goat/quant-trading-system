"""Point-in-time execution policy for isolated V3 research.

This module deliberately separates observable trading constraints from factor
selection.  It never removes a security from a historical universe because a
later price gap is known.  It is used only after a portfolio has been formed.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


POLICY_ID = "stale_mark_and_rebalance_deferral_v1"


@dataclass(frozen=True)
class CloseMark:
    price: float | None
    observed_on: str | None
    is_stale: bool


def direct_close(history: pd.DataFrame | None, date: str) -> float | None:
    """Return the official close only when it was available on ``date``."""
    if history is None or history.empty:
        return None
    rows = history[history["date"].astype(str).eq(str(date))]
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[-1].get("close"), errors="coerce")
    return float(value) if pd.notna(value) and value > 0 else None


def close_mark_as_of(history: pd.DataFrame | None, date: str) -> CloseMark:
    """Mark a suspended position at its last *previously observable* close.

    A stale mark carries the same close forward and therefore produces a zero
    return for the affected security.  It does not drop the security from an
    equal-weight return denominator, which would silently reallocate capital.
    """
    if history is None or history.empty:
        return CloseMark(None, None, True)
    frame = history.copy()
    frame["_date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame[frame["_date"] <= pd.Timestamp(date)].dropna(subset=["_date"])
    if frame.empty:
        return CloseMark(None, None, True)
    row = frame.sort_values("_date").iloc[-1]
    value = pd.to_numeric(row.get("close"), errors="coerce")
    if pd.isna(value) or value <= 0:
        return CloseMark(None, None, True)
    observed_on = row["_date"].date().isoformat()
    return CloseMark(float(value), observed_on, observed_on != str(date))


def rebalance_execution_decision(
    histories: dict[str, pd.DataFrame], current_codes: list[str], execution_date: str
) -> tuple[bool, list[str]]:
    """Return whether an entire planned rebalance may execute.

    If an existing position is suspended at the execution close, V3 defers the
    *whole* rebalance and continues the current portfolio.  This conservative
    rule avoids assuming a sale price or financing new purchases from a locked
    holding.  It only uses information observable at ``execution_date``.
    """
    locked = [code for code in current_codes if direct_close(histories.get(code), execution_date) is None]
    return (not locked, locked)

