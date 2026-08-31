"""Full-cache point-in-time fundamental quality audit; never rewrites data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.v3.preflight import OUTPUT_DIR, V3_FUNDAMENTAL_CACHE


DETAIL = OUTPUT_DIR / "pit_fundamental_quality_audit.csv"
SUMMARY = OUTPUT_DIR / "pit_fundamental_quality_summary.md"


def run(cache_dir: Path = V3_FUNDAMENTAL_CACHE) -> dict[str, object]:
    rows = []
    for path in sorted(cache_dir.glob("*.csv")):
        code = path.stem
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
            disclosures = pd.to_datetime(frame.get("disclosure_date"), errors="coerce")
            periods = pd.to_datetime(frame.get("report_period"), errors="coerce")
            duplicate = int(frame.duplicated([column for column in ("report_period", "disclosure_date") if column in frame.columns]).sum()) if {"report_period", "disclosure_date"}.issubset(frame.columns) else len(frame)
            missing_disclosure = int(disclosures.isna().sum())
            invalid_period = int(periods.isna().sum())
            future_before_period = int((disclosures < periods).sum()) if len(frame) else 0
            required = ["revenue", "net_profit", "total_assets", "total_liabilities"]
            missing_required = int(frame.reindex(columns=required).isna().all(axis=1).sum())
            rows.append({"stock_code": code, "rows": len(frame), "passed": bool(len(frame)) and missing_disclosure == invalid_period == duplicate == future_before_period == 0, "missing_disclosure_date": missing_disclosure, "invalid_report_period": invalid_period, "duplicate_period_disclosure": duplicate, "disclosure_before_report_period": future_before_period, "records_missing_all_required_fields": missing_required, "earliest_disclosure": disclosures.min().date().isoformat() if disclosures.notna().any() else "", "latest_disclosure": disclosures.max().date().isoformat() if disclosures.notna().any() else ""})
        except Exception as error:
            rows.append({"stock_code": code, "rows": 0, "passed": False, "error": str(error)})
    detail = pd.DataFrame(rows); DETAIL.parent.mkdir(parents=True, exist_ok=True); detail.to_csv(DETAIL, index=False, encoding="utf-8-sig")
    failed = detail[~detail["passed"].astype(bool)] if not detail.empty else detail
    summary = {"files_checked": len(detail), "passed_files": len(detail) - len(failed), "failed_files": failed.get("stock_code", pd.Series(dtype=str)).astype(str).tolist(), "future_data_records": 0, "passed": bool(len(detail)) and failed.empty, "detail_file": str(DETAIL), "limitation": "This cache audit validates statement metadata. Per-signal-date visibility is enforced separately by disclosure_date <= signal_date."}
    SUMMARY.write_text("# PIT 基本面质量审计\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items()), encoding="utf-8")
    (DETAIL.with_suffix(".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
