from __future__ import annotations

"""Isolated v2 market-regime ablation.

This experiment deliberately does not write any v1 or formal v2 output.  It
uses the continuous-rebalance engine so an exposure observation always means
the portfolio that was actually carried through that close-to-close return.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.backtest_v2 import load_config
from research.continuous_backtest_v2 import ContinuousBacktestResult, continuous_metrics, run_continuous_backtest_v2


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "v2_continuous_rebalance"
VERSION = "v2_regime_ablation"


def run_v2_regime_ablation() -> dict[str, Any]:
    """Run the no-gate treatment and preserve a copy of the formal v2 control."""
    config = load_config(BASE_DIR / "backtest_config_v2.yaml")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load_formal_baseline()
    common = {
        "universe_name": str(config["universe"]), "top_n": int(config["top_n"]),
        "holding_days": int(config["holding_days"]), "rebalance_days": int(config["rebalance_days"]),
        "start_date": str(config["start_date"]), "end_date": str(config["end_date"]),
        "initial_capital": float(config["initial_capital"]), "fee_rate": float(config["fee_rate"]),
        "market_min_score": float(config["market_min_score"]),
        "allow_fundamental_network": False, "allow_market_network": False,
        "event_boost_weight": 0.0,
    }
    no_gate_result = run_continuous_backtest_v2(
        model_label="Experiment B: No Market Regime Gate Continuous",
        market_regime_gate=False,
        **common,
    )
    _require_valid(no_gate_result)
    no_gate = _payload_from_result(no_gate_result)

    # The frozen strategy has no separate total-score hard gate. With the
    # market gate removed, B and C are economically identical while retaining
    # all universe, PIT, risk, and tradability filters. Copy the outcome into a
    # separate artifact so the experiment label cannot be mistaken for B.
    always = _clone_payload(no_gate, "Experiment C: Always Invested Top20 Continuous")
    always["equivalence_note"] = (
        "Equivalent to No Market Regime Gate under the frozen strategy: no independent total-score "
        "portfolio gate exists. It remains continuously invested after warmup whenever eligible stocks exist."
    )

    experiments = {"baseline": baseline, "no_regime_gate": no_gate, "always_invested": always}
    all_audits = []
    for key, payload in experiments.items():
        curve = payload["curve"].copy()
        curve["experiment"] = key
        audit = _cash_audit(curve, key)
        curve.to_csv(OUTPUT_DIR / f"{key}_equity.csv", index=False, encoding="utf-8-sig")
        audit.to_csv(OUTPUT_DIR / f"{key}_cash_exposure.csv", index=False, encoding="utf-8-sig")
        payload["rebalance_audit"].to_csv(OUTPUT_DIR / f"{key}_rebalance_audit.csv", index=False, encoding="utf-8-sig")
        all_audits.append(audit)
    cash_audit = pd.concat(all_audits, ignore_index=True)
    cash_audit.to_csv(OUTPUT_DIR / "cash_exposure_audit.csv", index=False, encoding="utf-8-sig")

    summary = {
        "experiment_version": VERSION,
        "legacy_v1_ablation_status": "v1_schedule_gap_contaminated",
        "formal_v1_modified": False,
        "formal_v2_modified": False,
        "config": config,
        "experiments": {key: _serializable_payload(value) for key, value in experiments.items()},
    }
    (OUTPUT_DIR / "v2_regime_ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary)
    return summary


def _load_formal_baseline() -> dict[str, Any]:
    summary = json.loads((BASE_DIR / "backtest_v2_summary.json").read_text(encoding="utf-8"))
    if summary.get("integrity_status") != "validated":
        raise RuntimeError("Formal v2 baseline is not validated.")
    return {
        "label": "Experiment A: Baseline Continuous",
        "metrics": summary["models"][0],
        "curve": pd.read_csv(BASE_DIR / "backtest_v2_equity_curve_model_a.csv", encoding="utf-8-sig"),
        "rebalance_audit": pd.read_csv(BASE_DIR / "research" / "backtest_v2_rebalance_audit_a.csv", encoding="utf-8-sig"),
        "integrity": {"status": "validated", "timeline": summary["models"][0]["timeline_validation"]},
    }


def _payload_from_result(result: ContinuousBacktestResult) -> dict[str, Any]:
    return {
        "label": result.model_label,
        "metrics": continuous_metrics(result),
        "curve": result.daily_curve,
        "rebalance_audit": pd.DataFrame(result.rebalance_audit),
        "integrity": {"status": result.integrity_status, "timeline": result.timeline_validation},
    }


def _clone_payload(payload: dict[str, Any], label: str) -> dict[str, Any]:
    copied = dict(payload)
    copied["label"] = label
    copied["metrics"] = dict(payload["metrics"])
    copied["metrics"]["model"] = label
    copied["curve"] = payload["curve"].copy()
    copied["rebalance_audit"] = payload["rebalance_audit"].copy()
    copied["integrity"] = dict(payload["integrity"])
    return copied


def _require_valid(result: ContinuousBacktestResult) -> None:
    if result.integrity_status != "validated" or result.backtest_integrity != "point_in_time_validated":
        raise RuntimeError("Ablation result failed historical point-in-time validation.")
    if int(result.fundamental_stats.get("future_records", 0)) != 0 or not result.timeline_validation.get("passed"):
        raise RuntimeError("Ablation result failed future-data or continuous-timeline validation.")


def _cash_audit(curve: pd.DataFrame, experiment: str) -> pd.DataFrame:
    columns = ["date", "portfolio_exposure", "cash_ratio", "cash_reason", "holding_count", "holding_codes", "daily_return_pct"]
    result = curve.loc[:, [column for column in columns if column in curve.columns]].copy()
    result.insert(0, "experiment", experiment)
    result.rename(columns={"cash_reason": "reason"}, inplace=True)
    return result


def _serializable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in payload.items() if key not in {"curve", "rebalance_audit"}}
    return result


def _fmt(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{float(value):.4f}{suffix}"


def _write_report(summary: dict[str, Any]) -> None:
    experiments = summary["experiments"]
    base = experiments["baseline"]["metrics"]
    no_gate = experiments["no_regime_gate"]["metrics"]
    always = experiments["always_invested"]["metrics"]
    metrics = [
        ("Total Return", "total_return_pct", "%"), ("CAGR", "annualized_return_pct", "%"),
        ("Sharpe", "sharpe_ratio", ""), ("Sortino", "sortino_ratio", ""),
        ("Calmar", "calmar_ratio", ""), ("Volatility", "annualized_volatility_pct", "%"),
        ("Max Drawdown", "max_drawdown_pct", "%"), ("Max Drawdown Duration", "max_drawdown_duration_days", " days"),
        ("Cash Days", "cash_days", ""), ("Average Cash Ratio", "average_cash_ratio", ""),
        ("Average Exposure", "average_exposure", ""), ("Turnover", "turnover_pct", "%"),
    ]
    lines = [
        "# v2 Market Regime Ablation", "",
        "- 独立研究实验；未改动或覆盖冻结的 Backtest v1.0，也未覆盖正式 Backtest v2。",
        "- 旧 v1 消融结果保留，但标记为 `v1_schedule_gap_contaminated`，不与本报告混用。",
        "- 三组均使用历史股票池、PIT 基本面、历史行情、同一因子/权重、Top20、20 日调仓、手续费与风险过滤。",
        "- Sharpe 沿用 v1/v2：每日净值收益率、样本标准差、sqrt(252)、无风险利率 0。", "",
        "## Comparison", "", "| Metric | Baseline | No Regime Gate | Always Invested Top20 |", "| --- | ---: | ---: | ---: |",
    ]
    for title, key, suffix in metrics:
        lines.append(f"| {title} | {_fmt(base.get(key), suffix)} | {_fmt(no_gate.get(key), suffix)} | {_fmt(always.get(key), suffix)} |")
    lines += ["", "## Cash Exposure Reasons", ""]
    for key in ("baseline", "no_regime_gate", "always_invested"):
        reason_days = experiments[key]["metrics"].get("cash_reason_days", {})
        lines.append(f"- {key}: {json.dumps(reason_days, ensure_ascii=False)}")
    lines += ["", "## Findings", "",
        f"- Baseline 的真实现金日为 {base['cash_days']}；其中 `{base['cash_reason_days'].get('market_regime_block', 0)}` 天来自市场环境 gate，`{base['cash_reason_days'].get('warmup_or_no_signal', 0)}` 天来自预热期。排程空档为 {base['timeline_validation']['schedule_gap_days']}。",
        f"- 移除 gate 后，CAGR {base['annualized_return_pct']:.4f}% -> {no_gate['annualized_return_pct']:.4f}%，Sharpe {base['sharpe_ratio']:.4f} -> {no_gate['sharpe_ratio']:.4f}，最大回撤 {base['max_drawdown_pct']:.4f}% -> {no_gate['max_drawdown_pct']:.4f}%。",
        "- B/C 相同是冻结策略的直接结果：没有独立的总评分硬性空仓门槛。两者在预热结束后、存在足够合格股票时保持连续市场暴露，未绕过任何历史时点或风险检查。",
        "- 是否继续研究 Market Regime 应以该对照结果为起点；本实验没有调阈值、权重、TopN 或调仓周期。", "",
        "## Integrity", "",
    ]
    for key, payload in experiments.items():
        lines.append(f"- {key}: status={payload['integrity']['status']}; timeline={json.dumps(payload['integrity']['timeline'], ensure_ascii=False)}")
    (OUTPUT_DIR / "v2_regime_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_v2_regime_ablation(), ensure_ascii=False, indent=2, default=str))
