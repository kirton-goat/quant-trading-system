"""Quality audit for the isolated V3 BaoStock liquidity cache."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.v3.liquidity_acquisition import LIQUIDITY_CACHE
from research.v3.preflight import OUTPUT_DIR


DETAIL_PATH = OUTPUT_DIR / "liquidity_quality_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "liquidity_quality_summary.json"
REQUIRED = {"date", "stock_code", "volume", "amount", "source", "amount_unit", "fetch_time"}


def run(cache_dir: Path = LIQUIDITY_CACHE) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for path in sorted(cache_dir.glob("*.csv")):
        code = path.stem
        try:
            data = pd.read_csv(path, encoding="utf-8-sig")
            missing = sorted(REQUIRED - set(data.columns))
            dates = pd.to_datetime(data.get("date"), errors="coerce")
            amount = pd.to_numeric(data.get("amount"), errors="coerce")
            units = sorted(set(data.get("amount_unit", pd.Series(dtype=str)).dropna().astype(str)))
            sources = sorted(set(data.get("source", pd.Series(dtype=str)).dropna().astype(str)))
            duplicates = int(dates.duplicated().sum())
            monotonic = bool(dates.dropna().is_monotonic_increasing)
            invalid_amount = int((~amount.notna() | amount.lt(0)).sum())
            valid = not missing and bool(len(data)) and duplicates == 0 and monotonic and invalid_amount == 0 and units == ["CNY"] and sources == ["BaoStock query_history_k_data_plus(adjustflag=3)"]
            records.append({
                "code": code, "passed": valid, "rows": len(data),
                "first_date": dates.min().date().isoformat() if dates.notna().any() else "",
                "last_date": dates.max().date().isoformat() if dates.notna().any() else "",
                "missing_columns": ",".join(missing), "duplicate_dates": duplicates,
                "monotonic_dates": monotonic, "invalid_amount_rows": invalid_amount,
                "amount_units": " | ".join(units), "sources": " | ".join(sources),
            })
        except Exception as exc:
            records.append({"code": code, "passed": False, "rows": 0, "first_date": "", "last_date": "", "missing_columns": "read_error", "duplicate_dates": 0, "monotonic_dates": False, "invalid_amount_rows": 0, "amount_units": "", "sources": str(exc)})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(records)
    detail.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    failed = detail.loc[~detail["passed"].astype(bool), "code"].tolist() if not detail.empty else []
    summary = {
        "classification": "research_experiment", "cache": str(cache_dir), "expected_amount_unit": "CNY",
        "files_checked": len(records), "passed_files": len(records) - len(failed), "failed_files": failed,
        "passed": bool(records) and not failed, "detail_file": str(DETAIL_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
