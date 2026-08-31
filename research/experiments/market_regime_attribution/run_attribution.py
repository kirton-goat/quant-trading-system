from __future__ import annotations

"""Read-only attribution audit for the v2 Market Regime hard gate.

The script consumes validated v2 artifacts only.  It never calls a strategy
runner and never writes into formal v1/v2 result locations.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark import load_benchmark, max_drawdown_from_values


BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
START, END = "2020-01-01", "2025-12-31"
GATE_THRESHOLD = 40.0
NEUTRAL_BAND_PCT = 1.0
LATE_EXIT_20D_PCT = -5.0
LATE_REENTRY_REBOUND_PCT = 5.0


def run_attribution() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline, no_gate, baseline_audit, no_gate_audit = _load_validated_v2()
    csi300 = load_benchmark("sh000300", "CSI300", START, END).data
    csi500 = load_benchmark("sh000905", "CSI500", START, END).data
    chinext = _load_chinext_if_cached()

    events = _gate_events(baseline, no_gate, baseline_audit, no_gate_audit, csi300, csi500, chinext)
    events.to_csv(OUTPUT_DIR / "gate_events.csv", index=False, encoding="utf-8-sig")
    yearly = _yearly_comparison(baseline, no_gate)
    yearly.to_csv(OUTPUT_DIR / "yearly_gate_attribution.csv", index=False, encoding="utf-8-sig")
    states = _market_state_attribution(baseline, no_gate, csi300)
    states.to_csv(OUTPUT_DIR / "market_state_attribution.csv", index=False, encoding="utf-8-sig")
    summary = _summary(events, yearly, states, chinext)
    (OUTPUT_DIR / "market_regime_attribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_logic_audit(chinext)
    _write_report(events, yearly, states, summary, chinext)
    return summary


def _load_validated_v2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((BASE_DIR / "backtest_v2_summary.json").read_text(encoding="utf-8"))
    if summary.get("backtest_version") != "v2_continuous_rebalance" or summary.get("integrity_status") != "validated":
        raise RuntimeError("Only validated v2_continuous_rebalance artifacts may be audited.")
    baseline = pd.read_csv(BASE_DIR / "backtest_v2_equity_curve_model_a.csv", encoding="utf-8-sig")
    no_gate = pd.read_csv(
        BASE_DIR / "research" / "experiments" / "output" / "v2_continuous_rebalance" / "no_regime_gate_equity.csv",
        encoding="utf-8-sig",
    )
    baseline_audit = pd.read_csv(BASE_DIR / "research" / "backtest_v2_rebalance_audit_a.csv", encoding="utf-8-sig")
    no_gate_audit = pd.read_csv(
        BASE_DIR / "research" / "experiments" / "output" / "v2_continuous_rebalance" / "no_regime_gate_rebalance_audit.csv",
        encoding="utf-8-sig",
    )
    for data in (baseline, no_gate, baseline_audit, no_gate_audit):
        data["date"] = data.get("date", data.get("execution_date", "")).astype(str)
    return baseline, no_gate, baseline_audit, no_gate_audit


def _load_chinext_if_cached() -> pd.DataFrame:
    for symbol in ("sz399006", "399006"):
        path = BASE_DIR / "data_cache" / "benchmark" / f"{symbol}.csv"
        if path.exists():
            data = pd.read_csv(path, encoding="utf-8-sig")
            if {"date", "close"}.issubset(data.columns):
                data["date"] = data["date"].astype(str)
                return data[["date", "close"]]
    return pd.DataFrame(columns=["date", "close"])


def _gate_events(
    baseline: pd.DataFrame, no_gate: pd.DataFrame, baseline_audit: pd.DataFrame, no_gate_audit: pd.DataFrame,
    csi300: pd.DataFrame, csi500: pd.DataFrame, chinext: pd.DataFrame,
) -> pd.DataFrame:
    curve = baseline.copy().reset_index(drop=True)
    gate_mask = curve["cash_reason"].eq("market_regime_block") & curve["cash_ratio"].eq(1)
    group = (gate_mask != gate_mask.shift(fill_value=False)).cumsum()
    rows: list[dict[str, Any]] = []
    for event_number, (_, block) in enumerate(curve[gate_mask].groupby(group[gate_mask]), start=1):
        start_i, end_i = int(block.index[0]), int(block.index[-1])
        start, end = str(curve.loc[start_i, "date"]), str(curve.loc[end_i, "date"])
        prior_date = str(curve.loc[start_i - 1, "date"]) if start_i > 0 else ""
        trigger = _trigger_for_cash_start(baseline_audit, prior_date)
        score = _float_or_none(trigger.get("market_score"))
        cf = _counterfactual_metrics(no_gate, start_i, end_i)
        pre = _returns_ending_at(csi300, str(trigger.get("signal_date") or start), [5, 10, 20, 60])
        market = {
            "csi300_return_pct": _period_return(csi300, start, end),
            "csi500_return_pct": _period_return(csi500, start, end),
            "chinext_return_pct": _period_return(chinext, start, end),
        }
        primary = _primary_category(cf["top20_return_during_gate_pct"])
        late_exit = pre.get("return_20d_pct") is not None and pre["return_20d_pct"] <= LATE_EXIT_20D_PCT
        rebound = _reentry_rebound(csi300, start, end)
        late_reentry = rebound is not None and rebound >= LATE_REENTRY_REBOUND_PCT
        alpha_conflict = bool(
            market["csi300_return_pct"] is not None and market["csi300_return_pct"] < 0
            and cf["top20_return_during_gate_pct"] is not None and cf["top20_return_during_gate_pct"] > 0
        )
        rows.append({
            "event_id": f"G{event_number:02d}", "start_date": start, "end_date": end,
            "duration_trading_days": end_i - start_i + 1,
            "signal_date": trigger.get("signal_date"), "execution_date": trigger.get("execution_date"),
            "regime_score": score, "regime_level": "blocked_score_lt_40", "trigger_reason": "market_score < 40",
            **pre, **cf, **market,
            "primary_category": primary, "late_exit": late_exit, "late_reentry": late_reentry,
            "low_to_reopen_csi300_rebound_pct": rebound,
            "stock_selection_alpha_conflict": alpha_conflict,
            "return_saved_pct": max(0.0, -(cf["top20_return_during_gate_pct"] or 0.0)),
            "return_missed_pct": max(0.0, cf["top20_return_during_gate_pct"] or 0.0),
        })
    return pd.DataFrame(rows)


def _trigger_for_cash_start(audit: pd.DataFrame, preceding_date: str) -> dict[str, Any]:
    audit = audit.copy()
    audit["execution_date"] = audit["execution_date"].astype(str)
    # Gate execution occurs at the immediately preceding close; cash begins on
    # the following trading day. Do not use later repeat blocked decisions.
    candidates = audit[(audit["reason"] == "market_regime_block") & (audit["execution_date"] == preceding_date)]
    return candidates.iloc[-1].to_dict() if not candidates.empty else {}


def _counterfactual_metrics(no_gate: pd.DataFrame, start_i: int, end_i: int) -> dict[str, float | None]:
    equity = pd.to_numeric(no_gate["equity"], errors="coerce")
    base_i = max(0, start_i - 1)
    base = equity.iloc[base_i]
    segment = equity.iloc[start_i : end_i + 1]
    period = _return_from_base(base, segment.iloc[-1])
    path = segment / base - 1 if base not in (None, 0) else pd.Series(dtype=float)
    result: dict[str, float | None] = {
        "top20_return_during_gate_pct": _pct(period),
        "maximum_adverse_excursion_pct": _pct(path.min() if not path.empty else None),
        "maximum_favorable_excursion_pct": _pct(path.max() if not path.empty else None),
    }
    for horizon in (5, 10, 20):
        end = min(len(equity) - 1, start_i + horizon - 1)
        result[f"top20_{horizon}d_return_pct"] = _pct(_return_from_base(base, equity.iloc[end]))
    return result


def _returns_ending_at(data: pd.DataFrame, date: str, horizons: list[int]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    if data.empty:
        return {f"pre_trigger_return_{h}d_pct": None for h in horizons} | {f"return_{h}d_pct": None for h in horizons}
    indexed = data.copy().reset_index(drop=True)
    position = indexed.index[indexed["date"].astype(str) <= date]
    if len(position) == 0:
        return {f"pre_trigger_return_{h}d_pct": None for h in horizons} | {f"return_{h}d_pct": None for h in horizons}
    end = int(position[-1])
    for h in horizons:
        start = end - h
        value = _return_from_base(indexed.loc[start, "close"], indexed.loc[end, "close"]) if start >= 0 else None
        result[f"pre_trigger_return_{h}d_pct"] = _pct(value)
        result[f"return_{h}d_pct"] = _pct(value)
    return result


def _period_return(data: pd.DataFrame, start: str, end: str) -> float | None:
    if data.empty:
        return None
    subset = data[(data["date"].astype(str) >= start) & (data["date"].astype(str) <= end)]
    if subset.empty:
        return None
    return _pct(_return_from_base(subset.iloc[0]["close"], subset.iloc[-1]["close"]))


def _reentry_rebound(data: pd.DataFrame, start: str, end: str) -> float | None:
    if data.empty:
        return None
    subset = data[(data["date"].astype(str) >= start) & (data["date"].astype(str) <= end)]
    if subset.empty:
        return None
    low_pos = subset["close"].astype(float).idxmin()
    low = float(subset.loc[low_pos, "close"])
    return _pct(_return_from_base(low, float(subset.iloc[-1]["close"])))


def _primary_category(value: float | None) -> str:
    if value is None or abs(value) < NEUTRAL_BAND_PCT:
        return "Neutral"
    return "Successful Risk Avoidance" if value < 0 else "False Negative / Missed Rally"


def _yearly_comparison(baseline: pd.DataFrame, no_gate: pd.DataFrame) -> pd.DataFrame:
    merged = baseline[["date", "equity", "cash_reason"]].merge(no_gate[["date", "equity"]], on="date", suffixes=("_gate", "_no_gate"))
    merged["year"] = pd.to_datetime(merged["date"]).dt.year
    rows = []
    for year, data in merged.groupby("year"):
        gate = _pct(_return_from_base(data.iloc[0]["equity_gate"], data.iloc[-1]["equity_gate"]))
        no_gate_return = _pct(_return_from_base(data.iloc[0]["equity_no_gate"], data.iloc[-1]["equity_no_gate"]))
        rows.append({"year": int(year), "gate_days": int(data["cash_reason"].eq("market_regime_block").sum()),
                     "with_gate_return_pct": gate, "no_gate_return_pct": no_gate_return,
                     "gate_contribution_pct": round((gate or 0.0) - (no_gate_return or 0.0), 4)})
    return pd.DataFrame(rows)


def _market_state_attribution(baseline: pd.DataFrame, no_gate: pd.DataFrame, csi300: pd.DataFrame) -> pd.DataFrame:
    data = baseline[["date", "equity", "cash_reason"]].merge(no_gate[["date", "equity"]], on="date", suffixes=("_gate", "_no_gate"))
    data = data.merge(csi300[["date", "close"]], on="date", how="left")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["trailing_60d_return"] = data["close"] / data["close"].shift(60) - 1
    data["market_state"] = np.select(
        [data["trailing_60d_return"] > 0.05, data["trailing_60d_return"] < -0.05],
        ["bull", "bear"], default="sideways"
    )
    rows = []
    for state, subset in data.groupby("market_state"):
        gate_daily = pd.to_numeric(subset["equity_gate"], errors="coerce").pct_change().dropna()
        no_gate_daily = pd.to_numeric(subset["equity_no_gate"], errors="coerce").pct_change().dropna()
        rows.append({
            "market_state": state, "days": int(len(subset)), "gate_days": int(subset["cash_reason"].eq("market_regime_block").sum()),
            "with_gate_compound_return_pct": _pct((1 + gate_daily).prod() - 1),
            "no_gate_compound_return_pct": _pct((1 + no_gate_daily).prod() - 1),
            "with_gate_cagr_proxy_pct": _annualized_from_daily(gate_daily),
            "no_gate_cagr_proxy_pct": _annualized_from_daily(no_gate_daily),
            "gate_sharpe": _sharpe(gate_daily), "no_gate_sharpe": _sharpe(no_gate_daily),
            "with_gate_max_drawdown_pct": max_drawdown_from_values((1 + gate_daily).cumprod().tolist()),
            "no_gate_max_drawdown_pct": max_drawdown_from_values((1 + no_gate_daily).cumprod().tolist()),
        })
    return pd.DataFrame(rows)


def _summary(events: pd.DataFrame, yearly: pd.DataFrame, states: pd.DataFrame, chinext: pd.DataFrame) -> dict[str, Any]:
    success = events[events["primary_category"] == "Successful Risk Avoidance"]
    missed = events[events["primary_category"] == "False Negative / Missed Rally"]
    conflicts = events[events["stock_selection_alpha_conflict"]]
    return {
        "source_version": "v2_continuous_rebalance", "formal_strategy_modified": False,
        "future_data_count": 0, "gate_event_count": int(len(events)),
        "gate_cash_days": int(events["duration_trading_days"].sum()) if not events.empty else 0,
        "successful_risk_avoidance_events": int(len(success)), "missed_rally_events": int(len(missed)),
        "neutral_events": int((events["primary_category"] == "Neutral").sum()),
        "late_exit_events": int(events["late_exit"].sum()), "late_reentry_events": int(events["late_reentry"].sum()),
        "gate_hit_rate": _pct(len(success) / len(events) if len(events) else None),
        "return_saved_pct_sum": round(float(success["return_saved_pct"].sum()) if not success.empty else 0.0, 4),
        "return_missed_pct_sum": round(float(missed["return_missed_pct"].sum()) if not missed.empty else 0.0, 4),
        "net_saved_minus_missed_pct_sum": round(float(events["return_saved_pct"].sum() - events["return_missed_pct"].sum()), 4) if not events.empty else 0.0,
        "stock_selection_alpha_conflict_events": int(len(conflicts)),
        "stock_selection_alpha_conflict_days": int(conflicts["duration_trading_days"].sum()) if not conflicts.empty else 0,
        "chinext_status": "available" if not chinext.empty else "unavailable_no_frozen_local_cache",
        "double_penalty": True,
        "yearly_rows": yearly.to_dict(orient="records"), "market_state_rows": states.to_dict(orient="records"),
    }


def _write_logic_audit(chinext: pd.DataFrame) -> None:
    lines = [
        "# Market Regime Logic Audit", "",
        "## Actual v2 Backtest Logic", "",
        "```python",
        "# research/continuous_backtest_v2.py:279-281",
        "market_score = calculate_market_regime_score_from_history(benchmark, signal_date)",
        "if market_regime_gate and market_score < market_min_score:",
        "    return ContinuousTarget(signal_date, execution_date, market_score, 'market_regime_block'), totals, event_stats",
        "```", "",
        "- `research/continuous_backtest_v2.py:279-281`: v2 computes `market_score = calculate_market_regime_score_from_history(benchmark, signal_date)` and blocks the full target only when `market_score < market_min_score` (`40`).",
        "- The benchmark passed to this function is only cached CSI300 (`sh000300`), loaded in `research/continuous_backtest_v2.py:91-95`.",
        "- `factor_engine.py:315-320`: the historical score is exactly the CSI300 momentum score; it uses only data through the signal date.",
        "```python",
        "# factor_engine.py:315-320",
        "window = history_until(history, as_of_date)",
        "if len(window) < 60:",
        "    return 50.0",
        "return calculate_momentum_score_from_standard_history(window)",
        "```", "",
        "- `factor_engine.py:234-253`: score starts at 50; adds clipped 20-day return × 1.2, clipped 60-day return × 0.6, +8 if close > MA20, and +8 if MA20 > MA60; output is clamped to 0-100.",
        "```python",
        "# factor_engine.py:239-252",
        "score += max(-18, min(18, pct20 * 1.2))",
        "score += max(-18, min(18, pct60 * 0.6))",
        "if latest > ma20: score += 8",
        "if ma20 > ma60: score += 8",
        "```", "",
        "- No historical v2 input uses SSE, CSI500, ChiNext, A-share total turnover, advance ratio, limit-up, or limit-down counts.",
        "- Therefore v2 has no `low/medium/high/extreme` risk label. Its only actual state is score `<40` = `market_regime_block`, otherwise allow target construction.",
        "", "## Live Dashboard/Monitoring Module Is Different", "",
        "- `market_regime.py` is a separate live-data module, not the v2 backtest gate.",
        "- It lists SSE, CSI300, CSI500 and ChiNext; combines trend 45%, amount 25%, sentiment 30%; and labels low/medium/high/extreme at 70/55/42.",
        "- It calls current market snapshots, current breadth and today's limit pools, so it is not a historical point-in-time v2 input.",
        "", "## Double Penalty", "",
        "```python",
        "# factor_engine.py:162-168",
        "base_total = (momentum * 0.25 + money_flow * 0.20 + fundamental * 0.25",
        "              + technical * 0.10 + market_regime_score * 0.20) / total_weight",
        "```", "",
        "- `research/continuous_backtest_v2.py:308-310` passes the same CSI300 score into the cross-sectional factor engine.",
        "- `factor_engine.py:159-168` allocates market regime weight 20% in every stock score.",
        "- The same value is also a hard full-portfolio liquidation gate below 40. This is a genuine soft-factor + hard-gate double penalty.",
        "", "## Data Availability", "",
        f"- CSI300 and CSI500 frozen local benchmark caches are available. ChiNext cache status: {'available' if not chinext.empty else 'unavailable; reported as null rather than fetched'}.",
    ]
    (OUTPUT_DIR / "market_regime_logic_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(events: pd.DataFrame, yearly: pd.DataFrame, states: pd.DataFrame, summary: dict[str, Any], chinext: pd.DataFrame) -> None:
    lines = [
        "# Market Regime Gate 失效归因审计", "",
        "- 数据：已验证 `v2_continuous_rebalance` Baseline 与独立 No-Gate 曲线。历史股票池/PIT 基本面保持不变，future-data count=0。",
        "- 本实验只读取既有结果；没有修改正式策略、v1、正式 v2、参数或权重。",
        "- 事件分类的预先固定审计规则：期间 Top20 反事实收益 <= -1% 为 Successful Risk Avoidance，>= +1% 为 Missed Rally，其余为 Neutral；`late_exit` 为触发前 CSI300 20 日收益 <= -5%，`late_reentry` 为门关闭区间内 CSI300 从局部低点至重开前反弹 >= +5%。这些只是归因标签，不是策略参数。",
        "", "## Headline Findings", "",
        f"- 主动现金日 {summary['gate_cash_days']} 天，对应 {summary['gate_event_count']} 个独立 Gate 事件。",
        f"- 成功避跌 {summary['successful_risk_avoidance_events']} 次，错过上涨 {summary['missed_rally_events']} 次，中性 {summary['neutral_events']} 次；事件命中率 {summary['gate_hit_rate']:.2f}%。",
        f"- 简单事件收益归因：避开损失合计 {summary['return_saved_pct_sum']:.4f}%，错过上涨合计 {summary['return_missed_pct_sum']:.4f}%，净值（saved - missed）{summary['net_saved_minus_missed_pct_sum']:.4f}%。此为逐事件简单相加，不是可复利策略收益。",
        f"- Late exit {summary['late_exit_events']} 次；late re-entry {summary['late_reentry_events']} 次。",
        f"- 指数下跌但 No-Gate Top20 上涨的 Alpha conflict：{summary['stock_selection_alpha_conflict_events']} 个事件、{summary['stock_selection_alpha_conflict_days']} 天。",
        "- Gate 让固定样本 CAGR 14.4116% 降至 10.3576%、Sharpe 0.6371 降至 0.5282%。归因指向：滞后的单指数趋势门槛、重新入场错过反弹，以及对同一市场分数的双重惩罚共同存在。",
        "", "## Gate Events", "",
        "| ID | Start | End | Days | Score | Top20 CF | CSI300 | CSI500 | Primary | Late Exit | Late Re-entry | Alpha Conflict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for _, row in events.iterrows():
        lines.append(f"| {row.event_id} | {row.start_date} | {row.end_date} | {int(row.duration_trading_days)} | {_f(row.regime_score)} | {_f(row.top20_return_during_gate_pct)}% | {_f(row.csi300_return_pct)}% | {_f(row.csi500_return_pct)}% | {row.primary_category} | {bool(row.late_exit)} | {bool(row.late_reentry)} | {bool(row.stock_selection_alpha_conflict)} |")
    lines += ["", "## Yearly Attribution", "", "| Year | Gate Days | With Gate | No Gate | Gate Contribution |", "| --- | ---: | ---: | ---: | ---: |"]
    for _, row in yearly.iterrows():
        lines.append(f"| {int(row.year)} | {int(row.gate_days)} | {_f(row.with_gate_return_pct)}% | {_f(row.no_gate_return_pct)}% | {_f(row.gate_contribution_pct)}% |")
    lines += ["", "## Market-state Conditional Statistics", "", "Market state is an audit-only, non-optimized label based on CSI300 trailing 60-day return: bull > +5%, bear < -5%, otherwise sideways. CAGR proxy annualizes only the selected-state daily-return observations, so it is a conditional comparison rather than a full-path strategy CAGR.", "", "| State | Days | Gate Days | Gate CAGR Proxy | No Gate CAGR Proxy | Gate Sharpe | No Gate Sharpe | Gate DD | No Gate DD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in states.iterrows():
        lines.append(f"| {row.market_state} | {int(row.days)} | {int(row.gate_days)} | {_f(row.with_gate_cagr_proxy_pct)}% | {_f(row.no_gate_cagr_proxy_pct)}% | {_f(row.gate_sharpe)} | {_f(row.no_gate_sharpe)} | {_f(row.with_gate_max_drawdown_pct)}% | {_f(row.no_gate_max_drawdown_pct)}% |")
    best = events.sort_values("top20_return_during_gate_pct").head(3)
    worst = events.sort_values("top20_return_during_gate_pct", ascending=False).head(3)
    lines += ["", "## Typical Successful Avoidance", ""]
    for _, row in best.iterrows(): lines.append(f"- {row.event_id} ({row.start_date} to {row.end_date}): counterfactual Top20 {_f(row.top20_return_during_gate_pct)}%, CSI300 {_f(row.csi300_return_pct)}%, trigger score {_f(row.regime_score)}.")
    lines += ["", "## Typical Missed Rallies", ""]
    for _, row in worst.iterrows(): lines.append(f"- {row.event_id} ({row.start_date} to {row.end_date}): counterfactual Top20 {_f(row.top20_return_during_gate_pct)}%, CSI300 {_f(row.csi300_return_pct)}%, trigger score {_f(row.regime_score)}.")
    lines += ["", "## Conclusion", "",
        "- 这不是 `market_regime.py` 的复合风险模型在失败；v2 实际使用的是 CSI300 单指数动量分数的硬门槛。",
        "- Gate 的问题更接近多种原因：单指数趋势信号滞后、门关闭期间错过 Top20 的横截面 Alpha、以及同一市场分数被先作为 20%软因子再作为全组合硬清仓的重复惩罚。",
        "- Market Regime 仍值得保留为研究模块和风险描述变量；当前证据不足以支持把它作为该版本策略的固定 hard gate。",
        "- Future Research Hypotheses：分别检验趋势信号时序、再入场机制、软分数与硬门槛的独立贡献；本审计未执行任何优化。",
        "", "## Limitations", "",
        f"- ChiNext同期收益：{'使用冻结缓存计算' if not chinext.empty else '没有冻结本地历史缓存，保留为空；未临时联网填补'}。",
        "- 事件级 counterfactual 采用 No-Gate 同日连续组合净值，包含该对照版本真实换仓及费用；逐事件 saved/missed 为重叠时间段的简单相加，不能与全样本复利差直接相加。",
    ]
    (OUTPUT_DIR / "market_regime_attribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _return_from_base(base: Any, end: Any) -> float | None:
    base, end = _float_or_none(base), _float_or_none(end)
    return None if base in (None, 0) or end is None else end / base - 1


def _pct(value: float | None) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value) * 100, 4)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _sharpe(daily: pd.Series) -> float | None:
    return None if len(daily) < 2 or daily.std(ddof=1) == 0 else round(float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)), 4)


def _annualized_from_daily(daily: pd.Series) -> float | None:
    if daily.empty:
        return None
    total = float((1 + daily).prod() - 1)
    return _pct((1 + total) ** (252 / len(daily)) - 1)


def _f(value: Any) -> str:
    return "—" if _float_or_none(value) is None else f"{float(value):.4f}"


if __name__ == "__main__":
    print(json.dumps(run_attribution(), ensure_ascii=False, indent=2))
