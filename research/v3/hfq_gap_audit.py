"""Explain V3 HFQ price-audit gaps without changing any backtest behavior."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.universe.index_members import get_index_snapshot
from research.v3.preflight import OUTPUT_DIR, default_v3_config, trading_rebalance_dates


DETAIL_PATH = OUTPUT_DIR / "price_quality_audit.csv"
REPORT_PATH = OUTPUT_DIR / "hfq_gap_audit.md"
JSON_PATH = OUTPUT_DIR / "hfq_gap_audit.json"


def membership_ranges(codes: set[str]) -> dict[str, tuple[str, str, int]]:
    config = default_v3_config()
    dates = trading_rebalance_dates(config["sample_period"]["start"], config["sample_period"]["end"], int(config["rebalance_days"]))
    found: dict[str, list[str]] = {code: [] for code in codes}
    for date in dates:
        members = set()
        for index in ("CSI300", "CSI500"):
            members.update(item.code for item in get_index_snapshot(date, index, allow_network=False).members)
        for code in members & codes:
            found[code].append(date)
    return {code: (min(values), max(values), len(values)) for code, values in found.items() if values}


def run() -> dict[str, object]:
    if not DETAIL_PATH.exists():
        raise FileNotFoundError("Run HFQ price-quality audit first.")
    detail = pd.read_csv(DETAIL_PATH, dtype={"code": str}, encoding="utf-8-sig")
    failed = detail[detail["passed"].astype(str).str.lower().eq("false")].copy()
    ranges = membership_ranges(set(failed["code"]))
    failed["first_historical_membership"] = failed["code"].map(lambda code: ranges.get(code, ("", "", 0))[0])
    failed["last_historical_membership"] = failed["code"].map(lambda code: ranges.get(code, ("", "", 0))[1])
    failed["historical_membership_rebalances"] = failed["code"].map(lambda code: ranges.get(code, ("", "", 0))[2])
    csv_path = OUTPUT_DIR / "hfq_gap_audit.csv"
    failed.to_csv(csv_path, index=False, encoding="utf-8-sig")
    reasons = failed.groupby("reason")["code"].count().sort_values(ascending=False).to_dict()
    payload = {"classification": "research_experiment", "gap_count": len(failed), "reason_counts": reasons, "detail_file": str(csv_path), "conclusion": "V3 remains blocked. Failed records must not be pre-emptively excluded from historical dates, because that would introduce a future-data selection bias."}
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# V3 HFQ Strategy-Price Gap Audit", "", f"- Gap records: **{len(failed)}**", "- Status: `blocked`", "- Scope: isolated V3 research cache only", "", "## Why This Blocks V3", "", "A price gap must not be solved by removing the stock from earlier historical dates. That would use future knowledge of delisting, suspension, or vendor coverage to alter the former investable universe.", "", "## Failure Types", "", "| Reason | Codes |", "| --- | ---: |"]
    lines.extend(f"| `{reason}` | {count} |" for reason, count in reasons.items())
    lines += ["", "## Required Resolution", "", "1. Obtain an auditable corporate-action-adjusted total-return series through each affected position's executable exit date; or", "2. Specify and test a point-in-time delisting/suspension execution policy before running V3.", "", "No V3 performance, factor, ablation, or dashboard result is generated from this incomplete data state."]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
