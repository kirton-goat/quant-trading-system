"""Audit whether V3 candidates have executable HFQ price coverage.

The earlier cache audit works from raw historical index membership.  That is a
useful provenance check, but it is deliberately stricter than the actual
strategy: many raw members would already fail the point-in-time listing-age,
liquidity, or suspension filters.  This read-only audit applies the same
historical universe filter as a formal run before judging whether a candidate
has an entry price and a complete holding interval.

It never alters a universe, a cache, or a backtest result.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_data_manager import MarketDataManager
from research.universe.index_members import get_index_snapshot
from research.universe.stock_filter import UniverseFilterConfig
from research.universe.stock_filter import load_security_metadata, normalize_optional_date
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE
from research.v3.liquidity_acquisition import LIQUIDITY_CACHE
from research.v3.preflight import OUTPUT_DIR, default_v3_config, trading_rebalance_dates


DETAIL_PATH = OUTPUT_DIR / "execution_eligibility_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "execution_eligibility_summary.json"
REPORT_PATH = OUTPUT_DIR / "execution_eligibility_report.md"


def _calendar(config: dict[str, Any]) -> list[str]:
    period = config["sample_period"]
    # The benchmark-derived calendar gives the exact same trading dates used
    # for V3 signal scheduling.  We retain every day here, not only signals.
    return trading_rebalance_dates(period["start"], period["end"], 1)


def _signal_indexes(calendar: list[str], rebalance_days: int) -> list[int]:
    # ``trading_rebalance_dates`` skips the initial 60-day warmup. Map the
    # signal dates back to the full calendar instead of assuming a fixed index.
    config = default_v3_config()
    signals = trading_rebalance_dates(
        config["sample_period"]["start"],
        config["sample_period"]["end"],
        rebalance_days,
    )
    positions = {date: index for index, date in enumerate(calendar)}
    return [positions[date] for date in signals if date in positions]


def _positive_close_dates(history: pd.DataFrame) -> set[str]:
    if history is None or history.empty:
        return set()
    data = history.copy()
    data["close"] = pd.to_numeric(data.get("close"), errors="coerce")
    return set(data.loc[data["close"].gt(0), "date"].astype(str))


def _price_issue(history: pd.DataFrame, required_dates: list[str]) -> tuple[str, int]:
    if not required_dates:
        return "ok", 0
    present = _positive_close_dates(history)
    missing = [date for date in required_dates if date not in present]
    if not missing:
        return "ok", 0
    if required_dates[0] in missing:
        return "missing_execution_price", len(missing)
    return "missing_holding_prices", len(missing)


def _history_indexes(histories: dict[str, pd.DataFrame]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pre-index history once; equivalent to the existing as-of helpers.

    ``filter_universe`` is intentionally simple and slices a DataFrame for
    every stock/date pair. That is fine in a single rebalance, but becomes
    prohibitively expensive for this 2015-2025 read-only audit. The arrays
    below implement the same count, latest-trade, and last-20-amount tests.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for code, history in histories.items():
        if history.empty:
            continue
        dates = pd.to_datetime(history["date"], errors="coerce")
        amounts = pd.to_numeric(history.get("amount", pd.Series(index=history.index, dtype=float)), errors="coerce")
        valid = dates.notna()
        result[code] = (
            dates.loc[valid].to_numpy(dtype="datetime64[ns]"),
            amounts.loc[valid].to_numpy(dtype=float),
        )
    return result


def _load_liquidity_indexes(codes: set[str], cache_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load only BaoStock dates and CNY turnover for point-in-time filtering."""
    histories: dict[str, pd.DataFrame] = {}
    for code in codes:
        path = cache_dir / f"{code}.csv"
        if not path.exists():
            continue
        try:
            data = pd.read_csv(path, usecols=["date", "amount"], encoding="utf-8-sig")
            histories[code] = data
        except (OSError, ValueError):
            continue
    return _history_indexes(histories)


def _eligible_codes(
    date: str,
    members: dict[str, dict[str, str]],
    metadata: dict[str, dict[str, str]],
    history_indexes: dict[str, tuple[np.ndarray, np.ndarray]],
    config: UniverseFilterConfig,
) -> tuple[list[str], dict[str, int]]:
    """Return the same point-in-time eligible set as the universe filters."""
    cutoff = np.datetime64(date)
    eligible: list[str] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for code, member in members.items():
        info = metadata.get(code, {})
        snapshot_date = normalize_optional_date(member.get("snapshot_date"))
        list_date = normalize_optional_date(info.get("list_date") or member.get("list_date"))
        delist_date = normalize_optional_date(info.get("delist_date") or member.get("delist_date"))
        name = str(info.get("name") or member.get("name") or "").upper().replace(" ", "")
        if snapshot_date and snapshot_date > date:
            reject("future_membership")
            continue
        if list_date and list_date > date:
            reject("future_listing")
            continue
        if delist_date and delist_date < date:
            reject("delisted")
            continue
        if name.startswith(("ST", "*ST", "S*ST", "SST")):
            reject("st")
            continue
        if info.get("trade_status") == "0":
            reject("suspended")
            continue
        indexed = history_indexes.get(code)
        if indexed is None:
            reject("missing_history")
            continue
        dates, amounts = indexed
        count = int(np.searchsorted(dates, cutoff, side="right"))
        if count < config.min_listing_trading_days:
            reject("insufficient_listing")
            continue
        last_trade = dates[count - 1]
        if int((cutoff - last_trade) / np.timedelta64(1, "D")) > config.max_calendar_days_without_trade:
            reject("suspended")
            continue
        recent = amounts[max(0, count - config.liquidity_window) : count]
        recent = recent[np.isfinite(recent)]
        if len(recent) < config.liquidity_window or float(recent.mean()) < config.min_average_amount:
            reject("illiquid")
            continue
        eligible.append(code)
    return eligible, rejected


def run(price_cache_dir: Path = HFQ_BAOSTOCK_CACHE, liquidity_cache_dir: Path = LIQUIDITY_CACHE) -> dict[str, Any]:
    config = default_v3_config()
    calendar = _calendar(config)
    signal_indexes = _signal_indexes(calendar, int(config["rebalance_days"]))
    if not calendar or not signal_indexes:
        raise RuntimeError("V3 benchmark calendar is unavailable; cannot audit execution eligibility.")

    # Build the raw code set from snapshots first, then load only the isolated
    # HFQ strategy-return cache. No network fallback is permitted.
    raw_codes: set[str] = set()
    signal_dates = [calendar[index] for index in signal_indexes]
    for date in signal_dates:
        for index_name in ("CSI300", "CSI500"):
            raw_codes.update(item.code for item in get_index_snapshot(date, index_name, allow_network=False).members)
    manager = MarketDataManager(cache_dir=price_cache_dir)
    price_histories = manager.load_histories(
        sorted(raw_codes),
        config["sample_period"]["start"],
        config["sample_period"]["end"],
        min_rows=1,
        allow_network=False,
    )
    filter_config = UniverseFilterConfig(require_market_history=True)
    liquidity_indexes = _load_liquidity_indexes(raw_codes, liquidity_cache_dir)
    rows: list[dict[str, Any]] = []
    total_raw_members = total_eligible = 0
    filter_reason_totals: dict[str, int] = {}

    for sequence, signal_index in enumerate(signal_indexes):
        signal_date = calendar[signal_index]
        execution_index = min(signal_index + 1, len(calendar) - 1)
        next_signal_index = signal_indexes[sequence + 1] if sequence + 1 < len(signal_indexes) else len(calendar) - 1
        next_execution_index = min(next_signal_index + 1, len(calendar) - 1)
        execution_date = calendar[execution_index]
        next_execution_date = calendar[next_execution_index]
        members: dict[str, dict[str, str]] = {}
        for index_name in ("CSI300", "CSI500"):
            snapshot = get_index_snapshot(signal_date, index_name, allow_network=False)
            for item in snapshot.members:
                members.setdefault(
                    item.code,
                    {"code": item.code, "name": item.name, "snapshot_date": item.snapshot_date},
                )
        metadata = load_security_metadata(signal_date, allow_network=False)
        eligible_codes, rejected = _eligible_codes(signal_date, members, metadata, liquidity_indexes, filter_config)
        total_raw_members += len(members)
        total_eligible += len(eligible_codes)
        for reason, count in rejected.items():
            filter_reason_totals[reason] = filter_reason_totals.get(reason, 0) + count

        # A signal is formed at T and executed at the following close. The
        # target then remains live through the next scheduled execution close.
        required_dates = calendar[execution_index : next_execution_index + 1]
        for code in eligible_codes:
            issue, missing_days = _price_issue(price_histories.get(code, pd.DataFrame()), required_dates)
            rows.append(
                {
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "next_execution_date": next_execution_date,
                    "stock_code": code,
                    "eligible": True,
                    "issue": issue,
                    "missing_price_days": missing_days,
                    "detail": "",
                }
            )

    detail = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    eligible_rows = detail[detail.get("eligible", pd.Series(dtype=bool)).eq(True)] if not detail.empty else detail
    hard_gaps = eligible_rows[eligible_rows["issue"].ne("ok")] if not eligible_rows.empty else eligible_rows
    issue_counts = eligible_rows["issue"].value_counts().to_dict() if not eligible_rows.empty else {}
    summary: dict[str, Any] = {
        "classification": "research_experiment",
        "price_semantics": "hfq_total_return",
        "liquidity_semantics": "BaoStock amount_unit=CNY, adjustflag=3",
        "scope": "point_in_time filtered historical universe, then T+1 execution through next execution",
        "signal_dates": len(signal_indexes),
        "raw_members_seen": total_raw_members,
        "eligible_candidate_rows": int(len(eligible_rows)),
        "eligible_candidates_per_signal_average": round(total_eligible / len(signal_indexes), 2),
        "filter_rejections": filter_reason_totals,
        "eligible_price_issue_counts": issue_counts,
        "hard_execution_gap_rows": int(len(hard_gaps)),
        "hard_execution_gap_codes": sorted(hard_gaps["stock_code"].dropna().astype(str).unique().tolist()) if not hard_gaps.empty else [],
        "passed": bool(len(eligible_rows)) and hard_gaps.empty,
        "detail_file": str(DETAIL_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V3 Point-in-Time Execution Eligibility Audit",
        "",
        "- Classification: `research_experiment`",
        "- Read-only: this audit does not change the V3 universe, factors, prices, or results.",
        "- Price semantics: `hfq_total_return`.",
        "- Liquidity semantics: BaoStock unadjusted daily turnover, `amount_unit=CNY`.",
        f"- Planned signal dates: {summary['signal_dates']}",
        f"- Raw historical-membership rows: {summary['raw_members_seen']}",
        f"- Candidates that pass the existing point-in-time universe filter: {summary['eligible_candidate_rows']}",
        f"- Eligible candidates with an entry/holding price gap: {summary['hard_execution_gap_rows']}",
        f"- Decision: `{'passed' if summary['passed'] else 'blocked'}`",
        "",
        "## Interpretation",
        "",
        "Raw-membership coverage failures caused by a stock that the actual strategy would reject for listing age, liquidity, suspension, ST, or missing market history are not execution gaps. A gap is blocking only when a candidate has already passed those same as-of-date filters and then lacks a positive price at T+1 or during its intended holding interval.",
        "",
        "## Filter Rejections",
        "",
        "| Reason | Candidate rows |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in sorted(filter_reason_totals.items()))
    lines += ["", "## Eligible Price Coverage", "", "| Result | Candidate rows |", "| --- | ---: |"]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in sorted(issue_counts.items()))
    if not summary["passed"]:
        lines += [
            "",
            "## Blocking Rule",
            "",
            "V3 remains blocked. The listed gaps cannot be solved by excluding those symbols from prior dates: that would use later price coverage information to alter the historical investment universe. An explicit, point-in-time execution policy or a replacement auditable price series is required.",
        ]
    else:
        lines += ["", "## Decision", "", "This specific execution-coverage check passed. It does not by itself validate PIT fundamentals, corporate-action semantics, or the V3 backtest."]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
