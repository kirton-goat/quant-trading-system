from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark
from research.backtest_v1 import benchmark_metrics
from research.continuous_backtest_v2 import ContinuousBacktestResult, continuous_metrics, run_continuous_backtest_v2


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "backtest_config_v2.yaml"
SUMMARY_FILE = BASE_DIR / "backtest_v2_summary.json"
REPORT_FILE = BASE_DIR / "research" / "backtest_v2_report.md"
COMPARISON_FILE = BASE_DIR / "research" / "v1_v2_comparison.md"
INTEGRITY_FILE = BASE_DIR / "backtest_v2_integrity_report.md"
AUDIT_FILE = BASE_DIR / "research" / "backtest_v2_rebalance_audit.csv"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_backtest_v2(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
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
        "market_regime_gate": True,
    }
    model_a = run_continuous_backtest_v2(
        model_label="Model A: Base Multi-Factor Continuous", event_boost_weight=0.0, **common
    )
    model_b = run_continuous_backtest_v2(
        model_label="Model B: Policy/Announcement Enhanced Continuous",
        event_boost_weight=float(config["model_b_event_boost_max"]),
        **common,
    )
    metrics_a, metrics_b = continuous_metrics(model_a), continuous_metrics(model_b)
    benchmark_300 = load_benchmark("sh000300", "CSI 300", config["start_date"], config["end_date"])
    benchmark_500 = load_benchmark("sh000905", "CSI 500", config["start_date"], config["end_date"])
    comparisons = {
        "model_a": {"CSI300": benchmark_metrics(model_a.daily_curve, benchmark_300.data), "CSI500": benchmark_metrics(model_a.daily_curve, benchmark_500.data)},
        "model_b": {"CSI300": benchmark_metrics(model_b.daily_curve, benchmark_300.data), "CSI500": benchmark_metrics(model_b.daily_curve, benchmark_500.data)},
    }
    valid = all(is_valid(result) for result in (model_a, model_b))
    summary = {
        "backtest_version": config["backtest_version"],
        "integrity_status": "validated" if valid else "incomplete",
        "config": config,
        "models": [metrics_a, metrics_b],
        "benchmarks": comparisons,
        "timeline_examples": timeline_examples(model_a),
    }
    write_outputs(model_a, model_b, config, summary, valid)
    return summary


def is_valid(result: ContinuousBacktestResult) -> bool:
    return (
        result.backtest_integrity == "point_in_time_validated"
        and result.integrity_status == "validated"
        and result.fundamental_stats.get("future_records", 0) == 0
        and bool(result.timeline_validation.get("passed"))
    )


def timeline_examples(result: ContinuousBacktestResult, count: int = 3) -> list[dict[str, Any]]:
    successful = [row for row in result.rebalance_audit if int(row.get("selected_count", 0)) > 0]
    index = {str(date): pos for pos, date in enumerate(result.daily_curve["date"].astype(str))}
    examples = []
    for current, following in zip(successful, successful[1:]):
        start, end = index[current["execution_date"]], index[following["execution_date"]]
        cash_gap = int((result.daily_curve.iloc[start + 1 : end]["portfolio_exposure"] == 0).sum())
        examples.append({
            "signal_date": current["signal_date"],
            "execution_date": current["execution_date"],
            "next_signal_date": following["signal_date"],
            "next_execution_date": following["execution_date"],
            "cash_gap_days": cash_gap,
        })
        if len(examples) >= count:
            break
    return examples


def write_outputs(a: ContinuousBacktestResult, b: ContinuousBacktestResult, config: dict[str, Any], summary: dict[str, Any], valid: bool) -> None:
    version = config["backtest_version"]
    for suffix, result in (("model_a", a), ("model_b", b)):
        curve = result.daily_curve.copy()
        curve["universe_mode"] = "historical_point_in_time"
        curve["fundamental_mode"] = result.fundamental_mode
        curve["backtest_integrity"] = result.backtest_integrity
        curve["integrity_status"] = result.integrity_status
        curve["backtest_version"] = version
        curve.to_csv(BASE_DIR / f"backtest_v2_equity_curve_{suffix}.csv", index=False, encoding="utf-8-sig")
        audit = pd.DataFrame(result.rebalance_audit)
        audit["model"] = suffix
        audit["backtest_version"] = version
        audit["fundamental_data_date"] = audit.get("fundamental_data_date", pd.Series(dtype=object)).map(json.dumps)
        audit["fundamental_disclosure_date"] = audit.get("fundamental_disclosure_date", pd.Series(dtype=object)).map(json.dumps)
        mode = "a" if suffix == "model_a" else "b"
        audit.to_csv(BASE_DIR / "research" / f"backtest_v2_rebalance_audit_{mode}.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([
        pd.read_csv(BASE_DIR / "research" / "backtest_v2_rebalance_audit_a.csv", encoding="utf-8-sig"),
        pd.read_csv(BASE_DIR / "research" / "backtest_v2_rebalance_audit_b.csv", encoding="utf-8-sig"),
    ], ignore_index=True)
    combined.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_integrity_report(a, b, valid, version)
    write_report(summary)
    write_v1_v2_comparison(summary["models"][0])


def write_integrity_report(a: ContinuousBacktestResult, b: ContinuousBacktestResult, valid: bool, version: str) -> None:
    INTEGRITY_FILE.write_text(
        "\n".join([
            "# Backtest v2 Continuous Rebalance Integrity Report", "",
            f"- Version: {version}",
            "- Universe mode: historical_point_in_time",
            "- Fundamental mode: historical_point_in_time",
            f"- Future fundamental records: {a.fundamental_stats.get('future_records', 0) + b.fundamental_stats.get('future_records', 0)}",
            f"- Model A schedule-gap days: {a.timeline_validation.get('schedule_gap_days')}",
            f"- Model B schedule-gap days: {b.timeline_validation.get('schedule_gap_days')}",
            f"- Timeline validation: {'passed' if valid else 'failed'}",
            f"- Integrity status: {'validated' if valid else 'incomplete'}",
            "",
        ]), encoding="utf-8"
    )


def write_report(summary: dict[str, Any]) -> None:
    a, b = summary["models"]
    lines = [
        "# Backtest v2 Continuous Rebalance Research Report", "",
        "- v1_historical_point_in_time remains frozen and was not overwritten.",
        "- v2 uses a continuous portfolio state machine: signal on T, rebalance at T+1 close, then earn target-portfolio returns from the following close-to-close interval.",
        "- Market Regime gate keeps v1 intent: a blocked target liquidates the existing portfolio to cash at the execution date.",
        "",
        "## Metrics", "",
        "| Metric | Model A | Model B |",
        "| --- | ---: | ---: |",
    ]
    for label, key, suffix in [
        ("Total Return", "total_return_pct", "%"), ("CAGR", "annualized_return_pct", "%"),
        ("Volatility", "annualized_volatility_pct", "%"), ("Sharpe", "sharpe_ratio", ""),
        ("Sortino", "sortino_ratio", ""), ("Calmar", "calmar_ratio", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"), ("Drawdown Duration", "max_drawdown_duration_days", " days"),
        ("Turnover", "turnover_pct", "%"), ("Transaction Costs", "transaction_cost", ""),
        ("Average Exposure", "average_exposure", ""), ("Cash Days", "cash_days", ""),
        ("Average Cash Ratio", "average_cash_ratio", ""),
    ]:
        lines.append(f"| {label} | {format_metric(a.get(key), suffix)} | {format_metric(b.get(key), suffix)} |")
    lines += ["", "## Continuous Timeline Evidence", "", "| Signal | Execution | Next Signal | Next Execution | Cash Gap |", "| --- | --- | --- | --- | ---: |"]
    for item in summary["timeline_examples"]:
        lines.append(f"| {item['signal_date']} | {item['execution_date']} | {item['next_signal_date']} | {item['next_execution_date']} | {item['cash_gap_days']} |")
    lines += ["", "## Integrity", "", f"- Status: {summary['integrity_status']}", "- No v1 output was overwritten.", ""]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def write_v1_v2_comparison(v2: dict[str, Any]) -> None:
    v1_path = BASE_DIR / "backtest_v1_summary.json"
    if not v1_path.exists():
        return
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))["models"][0]
    lines = [
        "# Backtest v1 / v2 Comparison", "",
        "- v1 and v2 are not a performance optimization comparison. They represent different economic holding timelines.",
        "- v1 serializes 20-day holding and 20-day scheduling, creating real schedule cash gaps. v2 continuously exchanges the portfolio every 20 trading days.",
        "", "| Metric | v1 frozen baseline | v2 continuous Model A |", "| --- | ---: | ---: |",
    ]
    mapping = [
        ("CAGR", "annualized_return_pct", "%"), ("Sharpe", "sharpe_ratio", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"), ("Volatility", "annualized_volatility_pct", "%"),
        ("Cash Days", "cash_days", ""), ("Average Exposure", "average_exposure", ""),
        ("Turnover", "turnover_pct", "%"), ("Fees", "transaction_cost", ""),
    ]
    for label, key, suffix in mapping:
        lines.append(f"| {label} | {format_metric(v1.get(key), suffix)} | {format_metric(v2.get(key), suffix)} |")
    COMPARISON_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.4f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest v2 continuous rebalance runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run_backtest_v2(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
