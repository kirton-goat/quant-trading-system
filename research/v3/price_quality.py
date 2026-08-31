"""Streaming quality audit for the isolated V3 QFQ strategy-price cache."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from research.universe.index_members import get_index_snapshot
from research.v3.data_acquisition import PRICE_COLUMNS, price_cache_dir
from research.v3.preflight import OUTPUT_DIR, V3_STRATEGY_PRICE_CACHE, default_v3_config, trading_rebalance_dates


SUMMARY_PATH = OUTPUT_DIR / "price_quality_summary.json"
DETAIL_PATH = OUTPUT_DIR / "price_quality_audit.csv"


@dataclass
class PriceAudit:
    code: str
    passed: bool
    rows: int
    first_date: str | None
    last_date: str | None
    duplicate_dates: int
    non_monotonic_dates: int
    invalid_price_rows: int
    qfq_rows: int
    non_qfq_rows: int
    missing_columns: str
    source_values: str
    reason: str


def audit_price_file(path: Path, start: str, end: str, adjustment: str) -> PriceAudit:
    code = path.stem
    required = set(PRICE_COLUMNS)
    rows = duplicate_dates = non_monotonic_dates = invalid_price_rows = qfq_rows = non_qfq_rows = 0
    first_date = last_date = previous_date = None
    sources: set[str] = set()
    seen_dates: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing_columns = required - fields
            if missing_columns:
                return PriceAudit(code, False, 0, None, None, 0, 0, 0, 0, 0, ",".join(sorted(missing_columns)), "", "missing_required_columns")
            for row in reader:
                raw_date = str(row.get("date") or "")
                try:
                    date = dt.date.fromisoformat(raw_date[:10]).isoformat()
                except ValueError:
                    invalid_price_rows += 1
                    continue
                # Only the interval in which this code can influence the V3
                # portfolio is relevant. Later vendor anomalies must not
                # invalidate an already-ended historical membership period.
                if date < start or date > end:
                    continue
                rows += 1
                first_date = first_date or date
                last_date = date
                if previous_date is not None and date < previous_date:
                    non_monotonic_dates += 1
                previous_date = date
                if date in seen_dates:
                    duplicate_dates += 1
                seen_dates.add(date)
                try:
                    close = float(row.get("close") or "nan")
                    if not math.isfinite(close) or close <= 0:
                        invalid_price_rows += 1
                except (TypeError, ValueError):
                    invalid_price_rows += 1
                if str(row.get("adjustment") or "").lower() == adjustment:
                    qfq_rows += 1
                else:
                    non_qfq_rows += 1
                source = str(row.get("source") or "").strip()
                if source:
                    sources.add(source)
    except OSError as exc:
        return PriceAudit(code, False, 0, None, None, 0, 0, 0, 0, 0, "", "", f"read_error:{exc}")

    earliest_allowed = (dt.date.fromisoformat(start) + dt.timedelta(days=7)).isoformat()
    latest_allowed = (dt.date.fromisoformat(end) - dt.timedelta(days=7)).isoformat()
    start_ok = first_date is not None and first_date <= earliest_allowed
    end_ok = last_date is not None and last_date >= latest_allowed
    passed = bool(rows and start_ok and end_ok and duplicate_dates == 0 and non_monotonic_dates == 0 and invalid_price_rows == 0 and non_qfq_rows == 0 and sources)
    reasons: list[str] = []
    for valid, reason in ((start_ok, "missing_start_coverage"), (end_ok, "missing_end_coverage"), (duplicate_dates == 0, "duplicate_dates"), (non_monotonic_dates == 0, "non_monotonic_dates"), (invalid_price_rows == 0, "invalid_prices"), (non_qfq_rows == 0, "non_qfq_rows"), (bool(sources), "missing_source")):
        if not valid:
            reasons.append(reason)
    return PriceAudit(code, passed, rows, first_date, last_date, duplicate_dates, non_monotonic_dates, invalid_price_rows, qfq_rows, non_qfq_rows, "", " | ".join(sorted(sources)), ";".join(reasons) or "ok")


def membership_windows() -> dict[str, tuple[str, str]]:
    config = default_v3_config()
    dates = trading_rebalance_dates(config["sample_period"]["start"], config["sample_period"]["end"], int(config["rebalance_days"]))
    windows: dict[str, list[str]] = {}
    for date in dates:
        for universe in ("CSI300", "CSI500"):
            for member in get_index_snapshot(date, universe, allow_network=False).members:
                windows.setdefault(member.code, []).append(date)
    return {code: (min(values), max(values)) for code, values in windows.items()}


def run_audit(cache_dir: Path = V3_STRATEGY_PRICE_CACHE, adjustment: str = "qfq") -> dict[str, object]:
    config = default_v3_config()
    windows = membership_windows()
    audits: list[PriceAudit] = []
    for code, (first_signal, last_signal) in sorted(windows.items()):
        path = cache_dir / f"{code}.csv"
        required_start = (dt.date.fromisoformat(first_signal) - dt.timedelta(days=120)).isoformat()
        required_end = min(dt.date.fromisoformat(config["sample_period"]["end"]), dt.date.fromisoformat(last_signal) + dt.timedelta(days=45)).isoformat()
        audits.append(audit_price_file(path, required_start, required_end, adjustment) if path.exists() else PriceAudit(code, False, 0, None, None, 0, 0, 0, 0, 0, "", "", "missing_file"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(audit) for audit in audits]).to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    failed = [item.code for item in audits if not item.passed]
    summary = {"classification": "research_experiment", "cache": str(cache_dir), "adjustment_required": adjustment, "coverage_rule": "120 calendar days before first historical-universe membership through 45 calendar days after last membership (or sample end)", "files_checked": len(audits), "passed_files": len(audits) - len(failed), "failed_files": failed, "passed": bool(audits) and not failed, "detail_file": str(DETAIL_PATH)}
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all isolated V3 QFQ price files.")
    parser.add_argument("--adjustment", choices=["qfq", "hfq_total_return"], default="qfq")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.cache_dir or price_cache_dir(args.adjustment), args.adjustment), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
