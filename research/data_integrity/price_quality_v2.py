"""V2 price-integrity audit: tradeable life, not arbitrary backtest end.

End-of-trading and listing-start boundaries are valid facts.  Only missing
prices inside a security's expected tradeable life count as TRUE_PRICE_GAP.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.universe.index_members import DEFAULT_CACHE_DIR
from research.universe.stock_filter import read_keyed_csv
from research.v3.price_quality import DETAIL_PATH, membership_windows
from research.v3.preflight import OUTPUT_DIR, V3_STRATEGY_PRICE_CACHE, default_v3_config
from research.v4.config import V4_OUTPUT_DIR


DETAIL_V2 = OUTPUT_DIR / "price_quality_audit_v2.csv"
SUMMARY_V2 = OUTPUT_DIR / "price_quality_summary_v2.json"
SPECIAL_EXIT = OUTPUT_DIR / "special_security_exit_audit.csv"


def _iso(value: Any) -> str:
    value = pd.to_datetime(value, errors="coerce")
    return value.date().isoformat() if pd.notna(value) else ""


def _metadata(code: str) -> dict[str, str]:
    return read_keyed_csv(DEFAULT_CACHE_DIR / "security_basics.csv").get(code, {})


def _boundary_status(row: dict[str, Any], expected_start: str, expected_end: str, list_date: str, delist_date: str) -> tuple[str, str]:
    actual_start, actual_end = str(row.get("first_date") or ""), str(row.get("last_date") or "")
    start_ok = actual_start and actual_start <= (pd.Timestamp(expected_start) + pd.Timedelta(days=7)).date().isoformat()
    end_ok = actual_end and actual_end >= (pd.Timestamp(expected_end) - pd.Timedelta(days=7)).date().isoformat()
    raw_reason = str(row.get("reason") or "")
    structural_issue = any(str(row.get(key, "0")) not in {"0", "0.0", "False", "false", ""} for key in ("duplicate_dates", "non_monotonic_dates", "invalid_price_rows", "non_qfq_rows"))
    if structural_issue:
        return "TRUE_PRICE_GAP", "price_structure_or_adjustment_issue"
    # Preserve the economic reason for an old coverage flag.  After expected
    # dates are capped to a security's life these rows would otherwise look
    # like generic passing data, which hides the delisting/listing boundary.
    if "missing_start_coverage" in raw_reason and list_date and actual_start and actual_start <= (pd.Timestamp(list_date) + pd.Timedelta(days=7)).date().isoformat():
        return "VALID_LISTING_START", "history_begins_at_actual_listing"
    if "missing_end_coverage" in raw_reason and delist_date and actual_end and actual_end >= (pd.Timestamp(delist_date) - pd.Timedelta(days=7)).date().isoformat():
        return "VALID_END_OF_TRADING", "history_ends_at_delisting_or_termination"
    if not start_ok and list_date and actual_start and actual_start <= (pd.Timestamp(list_date) + pd.Timedelta(days=7)).date().isoformat():
        return "VALID_LISTING_START", "history_begins_at_actual_listing"
    if not end_ok and delist_date and actual_end and actual_end >= (pd.Timestamp(delist_date) - pd.Timedelta(days=7)).date().isoformat():
        return "VALID_END_OF_TRADING", "history_ends_at_delisting_or_termination"
    if start_ok and end_ok:
        return "VALID_TRADING_DATA", "expected_tradeable_life_covered"
    if "missing_" in raw_reason:
        return "TRUE_PRICE_GAP", raw_reason
    return "UNRESOLVED", raw_reason or "unclassified"


def _load_last_price(code: str, cutoff: str) -> tuple[str, float | None]:
    path = V3_STRATEGY_PRICE_CACHE / f"{code}.csv"
    if not path.exists():
        return "", None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    frame = frame[frame["date"].notna() & (frame["date"] <= pd.Timestamp(cutoff))]
    if frame.empty:
        return "", None
    last = frame.sort_values("date").iloc[-1]
    return _iso(last["date"]), float(pd.to_numeric(last.get("close"), errors="coerce"))


def _strategy_holding_dates(code: str, cutoff: str) -> tuple[bool, str]:
    """Read the frozen V4 baseline only; lab experiments are intentionally ignored."""
    path = V4_OUTPUT_DIR / "runs" / "model_a_no_gate" / "daily_equity.csv"
    if not path.exists():
        return False, "baseline_daily_equity_not_found"
    daily = pd.read_csv(path)
    daily["date"] = pd.to_datetime(daily.get("date"), errors="coerce")
    prior = daily[daily["date"].notna() & (daily["date"] <= pd.Timestamp(cutoff))]
    held = prior[prior.get("holding_codes", pd.Series("", index=prior.index)).fillna("").astype(str).str.contains(code, regex=False)]
    if held.empty:
        return False, "not_held_by_v4_baseline_before_termination"
    return True, _iso(held.iloc[-1]["date"])


def _write_special_exit_audit(detail: pd.DataFrame) -> dict[str, int]:
    execution_path = OUTPUT_DIR / "execution_eligibility_audit.csv"
    execution = pd.read_csv(execution_path, dtype={"stock_code": str}) if execution_path.exists() else pd.DataFrame()
    issues = execution[execution.get("issue", pd.Series("ok", index=execution.index)).ne("ok")].copy()
    rows: list[dict[str, Any]] = []
    for item in issues.to_dict("records"):
        code = str(item.get("stock_code") or "").zfill(6)
        audit = detail[detail["stock_code"].eq(code)]
        metadata = audit.iloc[0].to_dict() if not audit.empty else {}
        termination = str(metadata.get("delisting_date") or "")
        last_date, last_price = _load_last_price(code, termination or str(item.get("next_execution_date") or ""))
        held, held_until = _strategy_holding_dates(code, termination or str(item.get("next_execution_date") or ""))
        classification = str(metadata.get("classification") or "UNRESOLVED")
        rows.append({
            "stock_code": code,
            "stock_name": metadata.get("stock_name", ""),
            "signal_date": item.get("signal_date", ""),
            "execution_date": item.get("execution_date", ""),
            "next_execution_date": item.get("next_execution_date", ""),
            "last_trade_date": last_date,
            "last_available_price": last_price,
            "delisting_or_termination_date": termination,
            "classification": classification,
            "strategy_holding_before_termination": held,
            "last_v4_baseline_holding_date": held_until,
            "strategy_attempted_exit": False,
            "exit_executable": bool(last_date),
            "special_event_exit_policy": "not_triggered_not_held" if not held else "manual_review_required_no_synthetic_exit",
            "conclusion": "no_v4_baseline_strategy_impact" if not held else "requires_special_event_exit_handling",
        })
    result = pd.DataFrame(rows)
    result.to_csv(SPECIAL_EXIT, index=False, encoding="utf-8-sig")
    held_count = int(result.get("strategy_holding_before_termination", pd.Series(dtype=bool)).astype(bool).sum())
    return {"special_exit_records": len(result), "special_exit_holding_records": held_count}


def run() -> dict[str, Any]:
    base = pd.read_csv(DETAIL_PATH, dtype={"code": str})
    windows = membership_windows(); config = default_v3_config(); basics = read_keyed_csv(DEFAULT_CACHE_DIR / "security_basics.csv")
    rows: list[dict[str, Any]] = []
    for source in base.to_dict("records"):
        code = str(source["code"]).zfill(6); info = basics.get(code, {})
        first_signal, last_signal = windows.get(code, ("", ""))
        raw_start = (pd.Timestamp(first_signal) - pd.Timedelta(days=120)).date().isoformat() if first_signal else config["sample_period"]["start"]
        raw_end = min(pd.Timestamp(config["sample_period"]["end"]), pd.Timestamp(last_signal) + pd.Timedelta(days=45)).date().isoformat() if last_signal else config["sample_period"]["end"]
        list_date, delist_date = _iso(info.get("list_date")), _iso(info.get("delist_date"))
        expected_start = max(raw_start, list_date) if list_date else raw_start
        expected_end = min(raw_end, delist_date) if delist_date else raw_end
        classification, reason = _boundary_status(source, expected_start, expected_end, list_date, delist_date)
        rows.append({**source, "stock_code": code, "stock_name": str(info.get("name") or ""), "listing_date": list_date, "delisting_date": delist_date, "expected_start": expected_start, "expected_end": expected_end, "classification": classification, "classification_reason": reason, "true_price_gap": classification == "TRUE_PRICE_GAP", "valid_end_of_trading": classification == "VALID_END_OF_TRADING"})
    detail = pd.DataFrame(rows); DETAIL_V2.parent.mkdir(parents=True, exist_ok=True); detail.to_csv(DETAIL_V2, index=False, encoding="utf-8-sig")
    original = detail[detail["passed"].astype(str).str.lower().ne("true")]
    special_exit = _write_special_exit_audit(detail)
    summary = {"data_version": "v3_hfq_baostock_price_integrity_v2", "total_stocks": len(detail), "original_flagged_stocks": len(original), "valid_end_of_trading": int((original["classification"] == "VALID_END_OF_TRADING").sum()), "valid_listing_start": int((original["classification"] == "VALID_LISTING_START").sum()), "valid_suspension": int((original["classification"] == "VALID_SUSPENSION").sum()), "true_price_gaps": int((original["classification"] == "TRUE_PRICE_GAP").sum()), "unresolved": int((original["classification"] == "UNRESOLVED").sum()), **special_exit, "passed": not bool((original["classification"].isin(["TRUE_PRICE_GAP", "UNRESOLVED"])).any()) and special_exit["special_exit_holding_records"] == 0, "detail_file": str(DETAIL_V2), "special_exit_file": str(SPECIAL_EXIT), "rule": "expected_end is capped at delisting/termination date; no post-termination price is required or created."}
    SUMMARY_V2.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
