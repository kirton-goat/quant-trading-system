"""Read-only strategy metadata for the Dashboard Strategy Explorer.

This module is intentionally separate from strategy execution.  Values that
change by version are loaded from the frozen summary/configuration files;
formula descriptions mirror the implementation in ``factor_engine.py`` and
``fundamental_factor.py`` and are covered by contract tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.v3.preflight import default_v3_config
from research.v4.config import default_v4_config
from research.v4.factor_recipes import active_recipes


BASE_DIR = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _weights(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("factor_weights") or config.get("weights") or {}
    return {
        "market_regime": float(raw.get("market_regime", raw.get("market", 0.20))),
        "momentum": float(raw.get("momentum", 0.25)),
        "money_flow": float(raw.get("money_flow", 0.20)),
        "fundamental": float(raw.get("fundamental", 0.25)),
        "technical": float(raw.get("technical", 0.10)),
    }


def _summary_config(filename: str, fallback: dict[str, Any]) -> dict[str, Any]:
    summary = _json(BASE_DIR / filename)
    config = summary.get("config")
    return config if isinstance(config, dict) else fallback


def _factor_metadata(weights: dict[str, float], technical_variant: str) -> list[dict[str, Any]]:
    technical_is_entry = technical_variant == "entry_timing"
    technical_name = "Entry Timing" if technical_is_entry else "Legacy Technical"
    technical_formula = (
        "Base 50; MA5 deviation -3% to +3% +8, >+8% -12, <-8% -8; "
        "3D return >10% -12 or 6% to 10% -6; 5D pullback -5% to -1% +6 or <-8% -8; "
        "MACD histogram change +6/-4; RSI 45-70 +6, >82 -12, <25 -6; "
        "10D volatility >6% -8; absolute opening gap >5% -8; clipped to 0-100."
        if technical_is_entry else
        "Base 50; close > MA5 +6; close > MA20 +8; MA20 > MA60 +8; DIF > DEA +8; "
        "RSI 45-72 +6, RSI >82 -12, RSI <30 -6; clipped to 0-100."
    )
    return [
        {
            "id": "market_regime", "name": "Market", "display_name": "市场环境", "type": "time_series_common_market_score",
            "weight": weights["market_regime"], "role": "整体市场状态，不是横截面选股信号", "status": "active",
            "purpose": "判断市场趋势环境，并向所有股票提供同一市场分。",
            "data_sources": ["CSI300 historical close cache", "factor_engine.calculate_market_regime_score_from_history"],
            "inputs": ["20D return", "60D return", "MA20", "MA60", "historical close through signal date"],
            "formula": "Uses the same momentum score formula: base 50, 20D return capped +/-18, 60D return capped +/-18, close > MA20 +8, MA20 > MA60 +8; clipped to 0-100.",
            "score_range": "0-100", "components": [],
            "current_score": None,
            "score_display": "Historical score; recalculated at every signal date, so this version has no single permanent score.",
            "important_behavior": "同一信号日全部股票使用同一个 Market Score；若只做横截面排序，它本身不会改变 TopN 相对顺序。",
            "code_refs": ["factor_engine.calculate_market_regime_score_from_history", "factor_engine.calculate_momentum_score_from_standard_history"],
        },
        {
            "id": "momentum", "name": "Momentum", "display_name": "动量", "type": "cross_sectional_stock_selection", "weight": weights["momentum"], "role": "中期趋势强度", "status": "active",
            "purpose": "判断股票是否处于持续的中期趋势。",
            "data_sources": ["historical close cache", "factor_engine.calculate_momentum_score_from_standard_history"],
            "inputs": ["20D return", "60D return", "close", "MA20", "MA60"],
            "formula": "Base 50; add 20D return x 1.2 capped +/-18; add 60D return x 0.6 capped +/-18; close > MA20 +8; MA20 > MA60 +8; clipped to 0-100.",
            "score_range": "0-100", "components": [], "code_refs": ["factor_engine.calculate_momentum_score_from_standard_history"],
        },
        {
            "id": "money_flow", "name": "Money Flow", "display_name": "资金", "type": "cross_sectional_stock_selection", "weight": weights["money_flow"], "role": "量价确认", "status": "active",
            "purpose": "确认成交量和成交额是否支持当前价格变化。",
            "data_sources": ["historical volume and amount cache", "factor_engine.calculate_money_flow_score_from_standard_history"],
            "inputs": ["5D average volume", "20D average volume", "latest amount", "20D average amount", "latest close"],
            "formula": "Base 50; (5D volume / 20D volume - 1) x 22 capped +/-18; (latest amount / 20D amount - 1) x 12 capped +/-12; close up with 5D/20D volume >= 1.2 adds 8; clipped to 0-100.",
            "score_range": "0-100", "components": [], "code_refs": ["factor_engine.calculate_money_flow_score_from_standard_history"],
        },
        {
            "id": "fundamental", "name": "Fundamental", "display_name": "基本面", "type": "cross_sectional_stock_selection", "weight": weights["fundamental"], "role": "质量、成长、现金流与估值", "status": "active",
            "purpose": "用历史时点可见的财报衡量企业质量、成长、现金流和估值。",
            "data_sources": ["PIT fundamentals cache", "historical financial statements", "historical unadjusted price for PE/PB"],
            "inputs": ["ROE", "gross margin", "debt/assets", "revenue YoY", "net profit YoY", "operating cash flow", "OCF/net profit", "PE", "PB", "report_period", "disclosure_date"],
            "formula": "Only records with disclosure_date <= signal_date are visible. Quality = average of ROE(-5..25), gross margin(0..60), inverse debt/assets(20..90); Growth = revenue and net-profit growth(-30..50); Cash Flow = OCF/net profit(-0.5..1.5) and OCF sign; Valuation = inverse PE(5..60) and PB(0.5..8). Final score re-normalizes available components with Quality 30%, Growth 25%, Cash Flow 20%, Valuation 25%; missing quality or fewer than two components excludes the stock.",
            "score_range": "0-100 or missing", "components": [{"name": "Quality", "weight": 0.30}, {"name": "Growth", "weight": 0.25}, {"name": "Cash Flow", "weight": 0.20}, {"name": "Valuation", "weight": 0.25}],
            "important_behavior": "回测日只能读取 disclosure_date 不晚于信号日的财报；缺失不会伪造中性 50 分。",
            "code_refs": ["research.fundamentals.point_in_time_fundamentals.get_fundamentals", "fundamental_factor.score_point_in_time_fundamentals"],
        },
        {
            "id": "technical", "name": technical_name, "display_name": "入场位置" if technical_is_entry else "技术（旧定义）", "type": "cross_sectional_entry_timing" if technical_is_entry else "cross_sectional_stock_selection", "weight": weights["technical"], "role": "短期入场质量" if technical_is_entry else "旧技术趋势与超买超卖", "status": "active" if technical_is_entry else "legacy",
            "purpose": "评估现在是否是相对合理的进入位置。" if technical_is_entry else "冻结版本保留的技术评分；包含趋势信息，因此与 Momentum 有重叠。",
            "data_sources": ["historical OHLC cache", "factor_engine.calculate_technical_score_from_standard_history"],
            "inputs": ["close", "MA5", "RSI", "MACD", "short-term returns", "short-term volatility", "opening gap"] if technical_is_entry else ["close", "MA5", "MA20", "MA60", "DIF", "DEA", "RSI"],
            "formula": technical_formula, "score_range": "0-100", "components": [],
            "important_behavior": "已移除 Price > MA20 与 MA20 > MA60，避免重复计算中期趋势。" if technical_is_entry else "Known Research Issue: Price > MA20 和 MA20 > MA60 与 Momentum 重复。",
            "code_refs": ["factor_engine.calculate_entry_timing_scores_from_standard_history"] if technical_is_entry else ["factor_engine.calculate_technical_score_from_standard_history"],
        },
    ]


def _version(version: str, config: dict[str, Any], *, label: str, status: str, classification: str, technical_variant: str, timeline: str, limitations: list[str]) -> dict[str, Any]:
    weights = _weights(config)
    gate_enabled = bool(config.get("market_regime_gate", config.get("market_gate", True)))
    return {
        "version": version, "label": label, "status": status, "classification": classification,
        "summary": {
            "research_period": config.get("sample_period") or {"start": config.get("start_date"), "end": config.get("end_date")},
            "universe": "CSI300 + CSI500 historical constituents", "top_n": int(config.get("top_n", 20)),
            "rebalance_frequency": f"Every {int(config.get('rebalance_days', 20))} trading days", "execution_timing": "Signal at T; execute at T+1 close",
            "fee_rate": float(config.get("fee_rate", 0.0015)), "slippage_rate": float(config.get("slippage_rate", 0.0)),
            "market_regime_gate": gate_enabled, "market_min_score": float(config.get("market_min_score", 40.0)),
            "pit_universe": True, "pit_fundamentals": True, "data_version": None,
            "integrity_status": "validated" if status == "validated" else ("frozen" if status == "frozen" else "research"),
        },
        "flow": ["Historical Universe", "Eligibility / PIT filters", "Factor calculation", "Weighted score", "Cross-sectional ranking", f"Top {int(config.get('top_n', 20))}", "Risk filters", "Market Regime hard gate", "Portfolio rebalance"],
        "timeline": timeline,
        "ranking": {"formula": "Final score = weighted average of Market + Momentum + Money Flow + Fundamental + Technical/Entry Timing, plus optional bounded event adjustment.", "score_range": "0-100", "method": "Descending final score", "tie_handling": "Stable Python sort preserves input order when scores are equal."},
        "factors": _factor_metadata(weights, technical_variant),
        "market_hard_gate": {"enabled": gate_enabled, "threshold": float(config.get("market_min_score", 40.0)), "trigger": "market_score < threshold", "action": "liquidate_to_cash" if version.startswith("v2") else "no target portfolio / cash", "note": "Separate from the Market Soft Factor. The hard gate controls whether a target portfolio may be opened."},
        "risk_filters": [
            {"name": "Historical membership", "rule": "Use only index members observable on the signal date.", "effect": "exclude stock", "source": "research.universe.historical_universe"},
            {"name": "Listing age", "rule": "At least 180 trading days.", "effect": "exclude stock", "source": "research.universe.stock_filter"},
            {"name": "ST / delisting / suspension", "rule": "Exclude ST, delisted and suspended securities.", "effect": "exclude stock", "source": "research.universe.stock_filter"},
            {"name": "Liquidity", "rule": "20D average amount must meet configured 20,000,000 CNY threshold.", "effect": "exclude stock", "source": "research.universe.stock_filter"},
            {"name": "PIT fundamental availability", "rule": "Missing required historical fundamental score excludes the stock.", "effect": "exclude stock", "source": "research.fundamentals"},
            {"name": "Execution eligibility", "rule": "No executable close / locked existing position defers or blocks the rebalance under the conservative policy.", "effect": "defer or exclude", "source": "research.v3.execution_policy"},
        ],
        "known_limitations": limitations,
        "event_factor": {"active_in_base_score": False, "rule": "Event information is an optional bounded adjustment of at most 0.05 and cannot create a signal by itself.", "status": "research_only"},
        "execution": {
            "signal_date": "T: calculate only with data observable through T.",
            "execution_date": "T+1 close.",
            "portfolio_handling": "Continuous target-portfolio transition; retained stocks remain, changed positions trade at the planned execution date.",
            "transaction_cost": "Charge fee_rate on actual buy and sell turnover; slippage uses the configured rate.",
        },
        "data_integrity": {
            "historical_universe": "historical_point_in_time",
            "pit_fundamentals": "historical_point_in_time: disclosure_date <= signal_date",
            "future_data_count": 0 if status in {"frozen", "validated"} else None,
            "price_adjustment": "V1/V2 frozen local historical cache; V3/V4 research uses isolated BaoStock HFQ total-return price cache when the research run is validated.",
            "corporate_action_handling": "V3/V4 require a corporate-action price audit before performance is considered validated.",
            "data_coverage": "See the version-specific preflight/status API for current cache coverage.",
            "data_version": None,
            "validation_status": "validated" if status == "validated" else ("frozen_reference" if status == "frozen" else "research"),
        },
        "research_variants": [5, 10, 20, 50] if version == "v4_entry_timing_candidate" else [],
        "previous_version": None,
        "version_diff": [],
    }


def get_strategy_versions() -> dict[str, Any]:
    v1 = _summary_config("backtest_v1_summary.json", {})
    v2 = _summary_config("backtest_v2_summary.json", {})
    v3 = default_v3_config()
    v4 = default_v4_config()
    versions = [
        _version("v1_historical_point_in_time", v1, label="V1 Historical PIT", status="frozen", classification="historical_reference", technical_variant="legacy", timeline="Legacy serial holding/rebalance schedule. It may create a cash wait between holding periods; retained only as a frozen historical reference.", limitations=["Known schedule-gap issue; do not treat as current economic baseline."]),
        _version("v2_continuous_rebalance", v2, label="V2 Continuous Rebalance", status="validated", classification="official_baseline", technical_variant="legacy", timeline="At every signal T, form the target with information observable at T and execute the portfolio transition at T+1 close. Positions remain invested until the next execution unless a hard gate or data/execution constraint applies.", limitations=["Legacy Technical overlaps with Momentum and remains frozen for version comparability."]),
        _version("v3_long_sample_research", v3, label="V3 Long-History Research Platform", status="research", classification="research_platform", technical_variant="legacy", timeline="Continuous-rebalance research engine: signal T, execute T+1 close, then retain the portfolio through the next execution.", limitations=["Long-sample performance is research-only and should be read with data-quality and in-sample limits.", "Legacy Technical overlaps with Momentum."]),
        _version("v4_entry_timing_candidate", v4, label="V4 Single-Factor Validation", status="research", classification="historical_research_experiment", technical_variant="entry_timing", timeline="Continuous-rebalance research engine: signal T, execute T+1 close, then retain the portfolio through the next execution.", limitations=["Entry Timing removes known trend duplication, but this remains an in-sample research candidate, not an official strategy."]),
    ]
    v1, v2, v3_item, v4_item = versions
    for factor in v4_item["factors"]:
        recipe = active_recipes().get(factor["id"])
        if recipe:
            factor["active_recipe"] = recipe
            factor["components"] = recipe["components"]
    for item in (v1, v2):
        for factor in item["factors"]:
            factor["evidence_status"] = "not_validated"
    for factor in v3_item["factors"]:
        factor["evidence_status"] = "candidate_removal" if factor["id"] == "technical" else "not_validated"
    for factor in v4_item["factors"]:
        factor["evidence_status"] = "under_validation" if factor["id"] == "technical" else "not_validated"
    v1["execution"] = {
        "signal_date": "S: calculate with data observable through S.",
        "execution_date": "S+1 close, then independent holding-period exit.",
        "portfolio_handling": "Legacy serial holding/rebalance scheduling. A subsequent scheduled signal can be skipped while the previous holding period remains active.",
        "transaction_cost": "Fee applied to the whole selected portfolio at entry/exit in the legacy implementation.",
    }
    v1["version_diff"] = ["Frozen historical reference; no prior version in this catalog."]
    v2["previous_version"] = v1["version"]
    v2["version_diff"] = ["Replaced V1 serial holding/rebalance scheduling with continuous target-portfolio rebalancing.", "Removed non-permitted rebalance schedule cash gaps.", "Reports buy turnover, sell turnover and turnover-based transaction costs."]
    v3_item["previous_version"] = v2["version"]
    v3_item["version_diff"] = ["Expanded the research period to 2015-2025 and isolated research data/cache governance.", "Adds research preflight, factor snapshots and research-lab replay; it is not an official performance upgrade."]
    v4_item["previous_version"] = v3_item["version"]
    v4_item["version_diff"] = ["Replaces Legacy Technical with Entry Timing, separating medium-term trend from short-term entry quality.", "Adds Entry Timing-specific strategy-lab variants (Top5/10/20/50); results remain research-only."]
    return {"default_version": "v2_continuous_rebalance", "versions": [{key: item[key] for key in ("version", "label", "status", "classification")} for item in versions], "details": {item["version"]: item for item in versions}}


def get_strategy_version(version: str) -> dict[str, Any] | None:
    return get_strategy_versions()["details"].get(version)
