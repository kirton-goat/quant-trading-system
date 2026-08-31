"""Classify V3/V4 price-cache gaps without silently repairing prices.

Candidate downloads are written separately.  No candidate is merged into the
strategy HFQ return cache unless a later explicit source/adjustment continuity
review approves it.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.universe.stock_filter import read_keyed_csv
from research.universe.index_members import DEFAULT_CACHE_DIR
from research.v3.price_quality import DETAIL_PATH, membership_windows
from research.v3.preflight import OUTPUT_DIR, V3_STRATEGY_PRICE_CACHE, default_v3_config


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "price_gap_resolution_plan.md"
RESOLUTION_PATH = ROOT / "price_gap_resolution.csv"
IMPACT_PATH = ROOT / "price_gap_strategy_impact.csv"
PROVENANCE_PATH = ROOT / "price_source_provenance.csv"
CANDIDATE_DIR = Path(__file__).resolve().parents[2] / "data_cache" / "v3_price_repair_candidates"
EXECUTION_PATH = OUTPUT_DIR / "execution_eligibility_audit.csv"
BASICS_PATH = DEFAULT_CACHE_DIR / "security_basics.csv"


def _date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if pd.notna(parsed) else ""


def _name_and_dates(code: str) -> tuple[str, str, str]:
    basics = read_keyed_csv(BASICS_PATH)
    item = basics.get(code, {})
    return str(item.get("name") or ""), str(item.get("list_date") or ""), str(item.get("delist_date") or "")


def _reason(row: dict[str, Any], list_date: str, delist_date: str, expected_start: str, expected_end: str) -> tuple[str, str]:
    actual_start, actual_end = str(row.get("first_date") or ""), str(row.get("last_date") or "")
    if delist_date and actual_end and delist_date <= expected_end and actual_end >= delist_date:
        return "A. Delisted / Historical Symbol", "valid_no_trade_after_delisting"
    if list_date and actual_start and list_date >= expected_start and actual_start <= list_date:
        return "F. Date Alignment Issue", "listing_date_or_membership_window_requires_review"
    if actual_start and actual_start > expected_start:
        return "D. Upstream Data Missing", "missing_start_coverage"
    if actual_end and actual_end < expected_end:
        return "D. Upstream Data Missing", "missing_end_coverage"
    return "G. Other", str(row.get("reason") or "unclassified")


def build_resolution() -> dict[str, Any]:
    if not DETAIL_PATH.exists():
        raise FileNotFoundError(f"Missing source audit: {DETAIL_PATH}")
    detail = pd.read_csv(DETAIL_PATH, dtype={"code": str})
    failed = detail[detail["passed"].astype(str).str.lower().ne("true")].copy()
    windows = membership_windows()
    execution = pd.read_csv(EXECUTION_PATH, dtype={"stock_code": str}) if EXECUTION_PATH.exists() else pd.DataFrame()
    config = default_v3_config(); rows: list[dict[str, Any]] = []
    for item in failed.to_dict("records"):
        code = str(item["code"]).zfill(6)
        first_signal, last_signal = windows.get(code, ("", ""))
        expected_start = (pd.Timestamp(first_signal) - pd.Timedelta(days=120)).date().isoformat() if first_signal else ""
        expected_end = min(pd.Timestamp(config["sample_period"]["end"]), pd.Timestamp(last_signal) + pd.Timedelta(days=45)).date().isoformat() if last_signal else ""
        name, list_date, delist_date = _name_and_dates(code)
        cause, cause_detail = _reason(item, list_date, delist_date, expected_start, expected_end)
        issues = execution[execution["stock_code"].astype(str).str.zfill(6).eq(code) & execution["issue"].ne("ok")] if not execution.empty else pd.DataFrame()
        missing_days = int(pd.to_numeric(issues.get("missing_price_days", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        affected = not issues.empty
        rows.append({
            "stock_code": code, "stock_name": name, "listing_date": list_date, "delisting_date": delist_date,
            "expected_start": expected_start, "expected_end": expected_end,
            "actual_start": str(item.get("first_date") or ""), "actual_end": str(item.get("last_date") or ""),
            "missing_days": missing_days, "missing_segments": "; ".join(f"{r.execution_date}..{r.next_execution_date}" for r in issues.itertuples(index=False)),
            "historical_universe_member": True, "eligible_period_affected": affected,
            "selected_by_strategy": "not_yet_audited", "holding_period_affected": affected,
            "execution_price_affected": bool((issues.get("issue", pd.Series(dtype=str)) == "missing_execution_price").any()),
            "likely_cause": cause, "cause_detail": cause_detail,
            "repair_status": "unresolved_classified", "source_cache": "BaoStock HFQ adjustflag=1", "adjustment_mode": "hfq_total_return",
        })
    result = pd.DataFrame(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESOLUTION_PATH, index=False, encoding="utf-8-sig")
    impact = result[result["eligible_period_affected"].astype(bool)].copy()
    impact["strategy_impact"] = impact.apply(lambda row: "entry_execution" if row["execution_price_affected"] else "holding_interval", axis=1)
    impact["affected_trades"] = impact["missing_segments"].map(lambda value: 0 if not value else len(str(value).split(";")))
    impact["severity"] = impact["strategy_impact"].map({"entry_execution": "Critical", "holding_interval": "High"})
    impact[["stock_code", "stock_name", "missing_segments", "strategy_impact", "affected_trades", "severity"]].to_csv(IMPACT_PATH, index=False, encoding="utf-8-sig")
    provenance = result[["stock_code", "source_cache", "adjustment_mode", "repair_status"]].copy()
    provenance["source"] = "BaoStock query_history_k_data_plus(adjustflag=1)"
    provenance["endpoint"] = "cached historical query"
    provenance["downloaded_at"] = "existing_cache"
    provenance["data_version"] = "v3_hfq_baostock_pre_price_quality_fix"
    provenance.to_csv(PROVENANCE_PATH, index=False, encoding="utf-8-sig")
    lines = ["# 历史价格缺口修复计划", "", "## 当前事实", "", f"- 全量审计缺口股票：{len(result)}。", f"- 影响历史合格候选执行/持有区间：{len(impact)}。", "- 不会插值、前值填充或用指数收益伪造股票价格。", "- 修复候选必须先独立保存、记录来源和复权口径；未通过连续性审计前不得合并到策略收益缓存。", "", "## 分类", "", "| 代码 | 缺口原因 | 策略影响 | 修复状态 |", "|---|---|---|---|"]
    for row in result.itertuples(index=False):
        lines.append(f"| {row.stock_code} | {row.likely_cause} | {'是' if row.eligible_period_affected else '否'} | {row.repair_status} |")
    lines.extend(["", "## 下一步", "", "1. 先尝试重新取得 10 只策略影响股票的同口径 HFQ 原始序列，保存为候选来源。", "2. 将真实退市/长期停牌与可交易日缺价区分；真实停牌保留为 valid_no_trade_day。", "3. 仅在来源、复权口径和公司行动连续性一致时合并；否则保持 unresolved 并把 V4 维持为 Data Integrity Partial。", ""])
    PLAN_PATH.write_text("\n".join(lines), encoding="utf-8")
    return {"total_gaps": len(result), "strategy_impacting": len(impact), "resolution_path": str(RESOLUTION_PATH), "impact_path": str(IMPACT_PATH)}


if __name__ == "__main__":
    print(json.dumps(build_resolution(), ensure_ascii=False, indent=2))
