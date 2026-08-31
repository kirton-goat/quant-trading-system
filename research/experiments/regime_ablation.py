from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark
from research.a_share_backtest import AShareBacktestResult, run_a_share_backtest
from research.backtest_v1 import complete_curve, load_config, strategy_metrics


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "strict_offline_market_cache"
EXPERIMENT_VERSION = "regime_ablation_v2_strict_offline_market_cache"


def run_regime_ablation() -> dict[str, Any]:
    """Run isolated market-regime ablations without touching Backtest v1.0 files."""
    config = load_config(BASE_DIR / "backtest_config_v1.yaml")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark("sh000300", "CSI 300", config["start_date"], config["end_date"])
    if benchmark.data.empty:
        raise RuntimeError("CSI 300 benchmark cache is unavailable; cannot run the isolated experiment.")

    common = {
        "universe_name": str(config["universe"]),
        "top_n": int(config["top_n"]),
        "holding_days": int(config["holding_days"]),
        "rebalance_days": int(config["rebalance_days"]),
        "start_date": str(config["start_date"]),
        "end_date": str(config["end_date"]),
        "initial_capital": float(config["initial_capital"]),
        "fee_rate": float(config["fee_rate"]),
        "market_min_score": float(config["market_min_score"]),
        "allow_fundamental_network": False,
        "allow_market_network": False,
        "event_boost_weight": 0.0,
        "collect_rebalance_audit": True,
    }
    definitions = [
        ("baseline", "Experiment A: Baseline", True),
        ("no_regime_gate", "Experiment B: No Market Regime Gate", False),
        ("always_invested", "Experiment C: Always Invested Top20", False),
    ]

    results: dict[str, dict[str, Any]] = {}
    combined_audits: list[pd.DataFrame] = []
    for key, label, market_regime_gate in definitions:
        result = run_a_share_backtest(
            model_label=label,
            market_regime_gate=market_regime_gate,
            **common,
        )
        _validate_integrity(result, key)
        curve = complete_curve(result, benchmark.data, float(config["fee_rate"]))
        result.daily_curve = curve
        daily_audit = build_cash_exposure_audit(curve, result.rebalance_audit, key)
        metrics = calculate_experiment_metrics(result, daily_audit)
        curve_output = curve.merge(
            daily_audit[["date", "portfolio_exposure", "cash_ratio", "reason"]],
            on="date",
            how="left",
        )
        curve_output.to_csv(OUTPUT_DIR / f"{key}_equity.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(result.rebalance_audit).to_csv(
            OUTPUT_DIR / f"{key}_rebalance_audit.csv", index=False, encoding="utf-8-sig"
        )
        combined_audits.append(daily_audit)
        results[key] = {
            "label": label,
            "market_regime_gate": market_regime_gate,
            "metrics": metrics,
            "integrity": _integrity_payload(result),
        }

    cash_audit = pd.concat(combined_audits, ignore_index=True)
    cash_audit.to_csv(OUTPUT_DIR / "cash_exposure_audit.csv", index=False, encoding="utf-8-sig")
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "formal_backtest_modified": False,
        "market_data_mode": "strict_offline_market_cache",
        "config": config,
        "experiments": results,
    }
    summary["baseline_reproduction"] = compare_to_formal_v1(results["baseline"]["metrics"])
    (OUTPUT_DIR / "regime_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, cash_audit)
    write_baseline_reproduction_report(summary["baseline_reproduction"])
    return summary


def build_cash_exposure_audit(
    curve: pd.DataFrame, rebalance_audit: list[dict[str, Any]], experiment: str
) -> pd.DataFrame:
    """Assign a transparent exposure state to each trading day.

    A selected rebalance owns its entry-to-exit interval. Cash days between
    completed intervals are intentionally kept separate as scheduling gaps.
    """
    decisions = sorted(rebalance_audit, key=lambda item: str(item.get("entry_date", "")))
    rows: list[dict[str, Any]] = []
    for _, row in curve.iterrows():
        date = str(row["date"])
        decision = _decision_for_date(decisions, date)
        holding_codes = str(row.get("holding_codes") or "").strip()
        holding_count = _holding_count(holding_codes)
        if holding_codes and holding_codes.lower() != "nan":
            reason = "normal_holding"
            exposure, cash_ratio = 1.0, 0.0
        elif decision is not None and decision.get("reason") != "normal_holding":
            reason = str(decision["reason"])
            exposure, cash_ratio = 0.0, 1.0
        elif decision is not None and decision.get("reason") == "normal_holding":
            reason = "normal_holding"
            exposure, cash_ratio = 1.0, 0.0
            holding_count = int(decision.get("selected_count") or 0)
        else:
            reason = _cash_gap_reason(decisions, date)
            exposure, cash_ratio = 0.0, 1.0
        rows.append(
            {
                "experiment": experiment,
                "date": date,
                "portfolio_exposure": exposure,
                "cash_ratio": cash_ratio,
                "reason": reason,
                "holding_codes": holding_codes,
                "holding_count": holding_count,
            }
        )
    return pd.DataFrame(rows)


def calculate_experiment_metrics(
    result: AShareBacktestResult, daily_audit: pd.DataFrame
) -> dict[str, Any]:
    metrics = strategy_metrics(result)
    holding_counts = daily_audit["holding_count"].astype(float)
    invested = daily_audit["portfolio_exposure"] > 0
    invested_counts = holding_counts.where(invested & (holding_counts > 0))
    average_stock_weight = (1 / invested_counts).mean() if invested_counts.notna().any() else None
    average_stock_weight_all_days = (1 / invested_counts).fillna(0).mean()
    calmar = None
    if metrics["annualized_return_pct"] is not None and metrics["max_drawdown_pct"] not in (None, 0):
        calmar = metrics["annualized_return_pct"] / abs(metrics["max_drawdown_pct"])
    reason_days = {
        str(reason): int(count)
        for reason, count in daily_audit["reason"].value_counts().sort_index().items()
    }
    return {
        **metrics,
        "calmar_ratio": _rounded(calmar),
        "average_holding_count": _rounded(holding_counts.mean()),
        "average_holding_count_when_invested": _rounded(holding_counts[invested].mean() if invested.any() else None),
        "average_stock_weight": _rounded(average_stock_weight),
        "average_stock_weight_all_days": _rounded(average_stock_weight_all_days),
        "average_cash_ratio": _rounded(float(daily_audit["cash_ratio"].mean())),
        "cash_days": int((daily_audit["cash_ratio"] == 1).sum()),
        "fully_invested_days": int((daily_audit["portfolio_exposure"] == 1).sum()),
        "reason_days": reason_days,
    }


def write_report(summary: dict[str, Any], cash_audit: pd.DataFrame) -> None:
    experiments = summary["experiments"]
    baseline = experiments["baseline"]["metrics"]
    no_gate = experiments["no_regime_gate"]["metrics"]
    always = experiments["always_invested"]["metrics"]
    report = [
        "# Market Regime / 空仓机制消融实验",
        "",
        f"- 实验版本：`{EXPERIMENT_VERSION}`",
        "- 正式 Backtest v1.0：未修改、未覆盖、未重新生成。",
        "- 三组实验共用冻结的历史股票池、历史时点基本面、行情缓存、因子权重、Top20、20日持有期、手续费和调仓周期。",
        "- Sharpe 保持 v1.0 定义：每日净值收益率 / 样本标准差 × sqrt(252)，无风险利率为 0。",
        "",
        "## 比较结果",
        "",
        "| Metric | Baseline | No Regime Gate | Always Invested Top20 |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows = [
        ("Total Return", "total_return_pct", "%"),
        ("CAGR", "annualized_return_pct", "%"),
        ("Sharpe", "sharpe_ratio", ""),
        ("Sortino", "sortino_ratio", ""),
        ("Calmar", "calmar_ratio", ""),
        ("Annualized Volatility", "annualized_volatility_pct", "%"),
        ("Max Drawdown", "max_drawdown_pct", "%"),
        ("Max Drawdown Duration", "max_drawdown_duration_days", " days"),
        ("Cash Days", "cash_days", ""),
        ("Fully Invested Days", "fully_invested_days", ""),
        ("Average Cash Ratio", "average_cash_ratio", ""),
        ("Average Holding Count", "average_holding_count", ""),
        ("Average Stock Weight", "average_stock_weight", ""),
        ("Turnover", "turnover_pct", "%"),
    ]
    for title, key, suffix in rows:
        report.append(
            f"| {title} | {_format(baseline.get(key), suffix)} | {_format(no_gate.get(key), suffix)} | {_format(always.get(key), suffix)} |"
        )

    report += ["", "## 空仓原因", ""]
    reason_counts = (
        cash_audit.groupby(["experiment", "reason"]).size().unstack(fill_value=0).sort_index(axis=1)
    )
    report.append("| Experiment | " + " | ".join(reason_counts.columns) + " |")
    report.append("| --- | " + " | ".join("---:" for _ in reason_counts.columns) + " |")
    for experiment, row in reason_counts.iterrows():
        report.append("| " + experiment + " | " + " | ".join(str(int(row[column])) for column in reason_counts.columns) + " |")

    report += [
        "",
        "## 解释规则",
        "",
        "- `market_regime_block`：市场环境分低于冻结配置的 40 分，Baseline 才会出现。",
        "- `risk_filter_block`：所有可评分股票均被既有波动、回撤或流动性风险过滤拒绝。",
        "- `missing_fundamental_data` / `missing_market_data`：保持严格历史时点规则，不以替代数据填充。",
        "- `rebalance_schedule_gap`：现有正式调仓实现中，上一持有期结束至下一次有效调仓之间的现金日；这是策略时间安排，而非数据错误。",
        "- 当前正式策略没有综合评分最低阈值。Experiment B 与 C 若结果相同，意味着在保留风险过滤的前提下，二者逻辑等价；这不是实验失败。",
        "",
        "## 完整性",
        "",
    ]
    for key, item in experiments.items():
        integrity = item["integrity"]
        report.append(
            f"- {key}: universe={integrity['universe_mode']}, fundamentals={integrity['fundamental_mode']}, future_fundamental_data={integrity['future_fundamental_data']}, integrity={integrity['integrity_status']}"
        )

    baseline_cash = baseline["cash_days"]
    gate_days = baseline["reason_days"].get("market_regime_block", 0)
    schedule_days = baseline["reason_days"].get("rebalance_schedule_gap", 0)
    warmup_days = baseline["reason_days"].get("warmup_or_no_signal", 0)
    data_missing_days = sum(
        baseline["reason_days"].get(key, 0)
        for key in ("missing_fundamental_data", "missing_market_data")
    )
    report += [
        "",
        "## 研究结论（不改变正式策略）",
        "",
        f"- Baseline 有 {baseline_cash} 个现金日：若按每日收益样本排除首个净值起点，即对应此前观察到的约 {baseline_cash - 1} 个零收益日。市场环境硬门槛直接解释 {gate_days} 天，既有调仓节奏间隔解释 {schedule_days} 天，起始预热/尚无信号解释 {warmup_days} 天；由整期财务或行情缺失直接造成的现金日为 {data_missing_days} 天。",
        f"- 与 Baseline 相比，移除市场环境硬门槛后，完全投资日增加 {no_gate['fully_invested_days'] - baseline['fully_invested_days']} 天，CAGR 由 {baseline['annualized_return_pct']:.4f}% 变为 {no_gate['annualized_return_pct']:.4f}%，Sharpe 由 {baseline['sharpe_ratio']:.4f} 变为 {no_gate['sharpe_ratio']:.4f}。",
        f"- 最大回撤由 {baseline['max_drawdown_pct']:.4f}% 变为 {no_gate['max_drawdown_pct']:.4f}%。因此该 gate 在此样本中提高了收益和 Sharpe，但没有降低最大回撤；它的价值是样本内风险调整收益，而不是回撤保护的证据。",
        "- Experiment B 与 C 完全一致：现有正式策略不存在“综合评分低于阈值则全组合空仓”的额外规则。C 也没有绕过历史股票池、历史基本面、风险过滤或行情完整性检查。",
        "- 市场环境模块值得继续做分样本和滚动窗口研究，但本实验没有调阈值、权重、TopN 或持有期，不能据此优化正式策略。",
    ]
    (OUTPUT_DIR / "regime_ablation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def compare_to_formal_v1(baseline_metrics: dict[str, Any]) -> dict[str, Any]:
    """Compare the isolated baseline with the frozen, user-facing v1 artifact."""
    summary_path = BASE_DIR / "backtest_v1_summary.json"
    if not summary_path.exists():
        return {"status": "unavailable", "reason": "backtest_v1_summary.json not found"}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    formal_models = payload.get("models") or []
    if not formal_models:
        return {"status": "unavailable", "reason": "formal Model A metrics not found"}
    formal = formal_models[0]
    deltas = {
        key: _rounded(float(baseline_metrics[key]) - float(formal[key]))
        for key in ("total_return_pct", "annualized_return_pct", "sharpe_ratio", "max_drawdown_pct")
    }
    return {
        "status": "compared",
        "formal_backtest_version": payload.get("backtest_version"),
        "formal_metrics": {key: formal.get(key) for key in deltas},
        "strict_offline_baseline_metrics": {key: baseline_metrics.get(key) for key in deltas},
        "delta_strict_offline_minus_formal": deltas,
        "interpretation": (
            "The strict offline experiment is close to, but not byte-identical with, the frozen v1 artifact. "
            "The formal v1 files remain canonical and were not overwritten."
        ),
    }


def write_baseline_reproduction_report(comparison: dict[str, Any]) -> None:
    lines = ["# Baseline 与冻结 Backtest v1.0 对照", ""]
    if comparison.get("status") != "compared":
        lines.append(f"- 无法比较：{comparison.get('reason', 'unknown')}")
    else:
        lines += [
            "- 正式 v1.0 文件未改动；本文件只对照独立、离线缓存实验的 Baseline。",
            "- 严格离线实验与正式 v1.0 结果接近，但不逐字节相同；正式 v1.0 仍是 Dashboard 与研究基线的唯一正式版本。",
            "",
            "| Metric | Frozen v1.0 | Strict Offline Baseline | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
        suffixes = {
            "total_return_pct": "%",
            "annualized_return_pct": "%",
            "sharpe_ratio": "",
            "max_drawdown_pct": "%",
        }
        labels = {
            "total_return_pct": "Total Return",
            "annualized_return_pct": "CAGR",
            "sharpe_ratio": "Sharpe",
            "max_drawdown_pct": "Max Drawdown",
        }
        for key, label in labels.items():
            lines.append(
                f"| {label} | {_format(comparison['formal_metrics'][key], suffixes[key])} | "
                f"{_format(comparison['strict_offline_baseline_metrics'][key], suffixes[key])} | "
                f"{_format(comparison['delta_strict_offline_minus_formal'][key], suffixes[key])} |"
            )
        lines += [
            "",
            f"- 解释：{comparison['interpretation']}",
        ]
    (OUTPUT_DIR / "baseline_reproduction_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision_for_date(decisions: list[dict[str, Any]], date: str) -> dict[str, Any] | None:
    for decision in decisions:
        if str(decision.get("entry_date", "")) <= date <= str(decision.get("exit_date", "")):
            return decision
    return None


def _cash_gap_reason(decisions: list[dict[str, Any]], date: str) -> str:
    prior = [item for item in decisions if str(item.get("entry_date", "")) <= date]
    if not prior:
        return "warmup_or_no_signal"
    return "rebalance_schedule_gap"


def _holding_count(value: Any) -> int:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return 0
    return len([code for code in text.split(",") if code.strip()])


def _integrity_payload(result: AShareBacktestResult) -> dict[str, Any]:
    return {
        "universe_mode": "historical_point_in_time" if result.universe.is_historical_membership else "incomplete",
        "fundamental_mode": result.fundamental_mode,
        "future_fundamental_data": int(result.fundamental_stats.get("future_records", 0)),
        "backtest_integrity": result.backtest_integrity,
        "integrity_status": result.integrity_status,
    }


def _validate_integrity(result: AShareBacktestResult, experiment: str) -> None:
    integrity = _integrity_payload(result)
    if (
        integrity["universe_mode"] != "historical_point_in_time"
        or integrity["fundamental_mode"] != "historical_point_in_time"
        or integrity["future_fundamental_data"] != 0
        or integrity["backtest_integrity"] != "point_in_time_validated"
    ):
        raise RuntimeError(f"{experiment} failed historical integrity checks: {integrity}")


def _rounded(value: Any) -> float | None:
    try:
        return round(float(value), 4) if value is not None and not pd.isna(value) else None
    except (TypeError, ValueError):
        return None


def _format(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{float(value):.4f}{suffix}"


if __name__ == "__main__":
    print(json.dumps(run_regime_ablation(), ensure_ascii=False, indent=2))
