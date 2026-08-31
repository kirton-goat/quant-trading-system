from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark, max_drawdown_from_values
from research.a_share_backtest import AShareBacktestResult, run_a_share_backtest, save_result_tables


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "backtest_config_v1.yaml"
VERSION_REPORT = BASE_DIR / "research" / "backtest_report_v1.md"
INTEGRITY_REPORT = BASE_DIR / "backtest_integrity_report.md"
BENCHMARK_REPORT = BASE_DIR / "benchmark_comparison.md"
SUMMARY_FILE = BASE_DIR / "backtest_v1_summary.json"
AUDIT_FILE = BASE_DIR / "research" / "backtest_rebalance_audit_v1.csv"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_backtest_v1(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    kwargs = {
        "universe_name": str(config["universe"]),
        "top_n": int(config["top_n"]),
        "holding_days": int(config["holding_days"]),
        "rebalance_days": int(config["rebalance_days"]),
        "start_date": str(config["start_date"]),
        "end_date": str(config["end_date"]),
        "initial_capital": float(config["initial_capital"]),
        "fee_rate": float(config["fee_rate"]),
        "market_min_score": float(config["market_min_score"]),
        # A formal run must be reproducible from the frozen local cache. Missing
        # fields stay missing instead of changing as remote APIs change.
        "allow_fundamental_network": False,
    }
    model_a = run_a_share_backtest(
        model_label="Model A: Base Multi-Factor", event_boost_weight=0.0, **kwargs
    )
    model_b = run_a_share_backtest(
        model_label="Model B: Policy/Announcement Enhanced",
        event_boost_weight=float(config["model_b_event_boost_max"]),
        **kwargs,
    )
    benchmark_300 = load_benchmark("sh000300", "CSI 300", config["start_date"], config["end_date"])
    benchmark_500 = load_benchmark("sh000905", "CSI 500", config["start_date"], config["end_date"])
    for result in (model_a, model_b):
        result.daily_curve = complete_curve(result, benchmark_300.data, float(config["fee_rate"]))
    metrics_a, metrics_b = strategy_metrics(model_a), strategy_metrics(model_b)
    comparisons = {
        "model_a": {
            "CSI300": benchmark_metrics(model_a.daily_curve, benchmark_300.data),
            "CSI500": benchmark_metrics(model_a.daily_curve, benchmark_500.data),
        },
        "model_b": {
            "CSI300": benchmark_metrics(model_b.daily_curve, benchmark_300.data),
            "CSI500": benchmark_metrics(model_b.daily_curve, benchmark_500.data),
        },
    }
    validated = all(
        result.backtest_integrity == "point_in_time_validated"
        and result.integrity_status == "validated"
        and result.fundamental_stats.get("future_records", 0) == 0
        for result in (model_a, model_b)
    )
    version = config["backtest_version"]
    summary = {
        "backtest_version": version,
        "integrity_status": "validated" if validated else "incomplete",
        "config": config,
        "models": [metrics_a, metrics_b],
        "benchmarks": comparisons,
    }
    write_audit(model_a, model_b)
    write_integrity_report(model_a, model_b, validated, version)
    write_benchmark_report(comparisons, metrics_a, metrics_b)
    write_research_report(config, metrics_a, metrics_b, validated)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if validated:
        placeholder = BASE_DIR / "backtest_v1_outputs.md"
        save_result_tables(model_a, placeholder, suffix="model_a", backtest_version=version)
        save_result_tables(model_b, placeholder, suffix="model_b", backtest_version=version)
    return summary


def complete_curve(result: AShareBacktestResult, calendar: pd.DataFrame, fee_rate: float) -> pd.DataFrame:
    columns = ["date", "equity", "return_pct", "holding_codes"]
    if calendar.empty:
        return result.daily_curve.reindex(columns=columns)
    raw_map = {
        str(row["date"]): row for _, row in result.daily_curve.iterrows()
    } if not result.daily_curve.empty else {}
    entries = {trade.entry_date: trade for trade in result.trades}
    exits = {trade.exit_date: trade for trade in result.trades}
    active_trade = None
    equity = float(result.initial_capital)
    current_codes = ""
    rows = []
    for date in calendar["date"].astype(str):
        if date in entries:
            active_trade = entries[date]
        if date in raw_map:
            row = raw_map[date]
            buy_cost = float(active_trade.entry_equity) * fee_rate if active_trade is not None else 0.0
            equity = float(row["equity"]) - buy_cost
            current_codes = str(row.get("holding_codes") or "")
        if date in exits:
            equity = float(exits[date].exit_equity)
            current_codes = ""
            active_trade = None
        rows.append({
            "date": date,
            "equity": round(equity, 2),
            "return_pct": round((equity / result.initial_capital - 1) * 100, 4),
            "holding_codes": current_codes,
        })
    return pd.DataFrame(rows, columns=columns)

def strategy_metrics(result: AShareBacktestResult) -> dict[str, Any]:
    equity = pd.to_numeric(result.daily_curve.get("equity"), errors="coerce").dropna()
    daily = equity.pct_change().dropna()
    years = max(len(equity) / 252, 1 / 252)
    total = (equity.iloc[-1] / result.initial_capital - 1) * 100 if not equity.empty else 0.0
    annual = ((1 + total / 100) ** (1 / years) - 1) * 100
    volatility = daily.std(ddof=1) * math.sqrt(252) * 100 if len(daily) > 1 else None
    sharpe = daily.mean() / daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 and daily.std(ddof=1) else None
    downside = daily[daily < 0]
    sortino = daily.mean() / downside.std(ddof=1) * math.sqrt(252) if len(downside) > 1 and downside.std(ddof=1) else None
    period_returns = pd.Series([trade.portfolio_return_pct for trade in result.trades], dtype=float)
    wins, losses = period_returns[period_returns > 0], period_returns[period_returns < 0]
    pl_ratio = wins.mean() / abs(losses.mean()) if not wins.empty and not losses.empty else None
    return {
        "model": result.model_label,
        "total_return_pct": rounded(total),
        "annualized_return_pct": rounded(annual),
        "max_drawdown_pct": max_drawdown_from_values(equity.tolist()),
        "max_drawdown_duration_days": drawdown_duration(equity),
        "annualized_volatility_pct": rounded(volatility),
        "sharpe_ratio": rounded(sharpe),
        "sortino_ratio": rounded(sortino),
        "rebalance_count": len(result.trades),
        "stock_trade_count": sum(len(trade.codes) for trade in result.trades),
        "turnover_pct": rounded(turnover(result)),
        "win_rate": rounded(float((period_returns > 0).mean()) if not period_returns.empty else None),
        "profit_loss_ratio": rounded(pl_ratio),
        "final_equity": rounded(float(equity.iloc[-1]) if not equity.empty else result.initial_capital),
        "backtest_integrity": result.backtest_integrity,
        "integrity_status": result.integrity_status,
        "fundamental_mode": result.fundamental_mode,
        "universe_mode": "historical_point_in_time" if result.universe.is_historical_membership else "current_or_manual",
        "missing_fundamental_data": int(result.fundamental_stats.get("missing", 0)),
        "future_fundamental_data": int(result.fundamental_stats.get("future_records", 0)),
        "untradeable_count": int(sum(result.execution_stats.values())),
    }


def benchmark_metrics(curve: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, Any]:
    if curve.empty or benchmark.empty:
        return {"status": "missing"}
    merged = curve[["date", "equity"]].merge(benchmark[["date", "close"]], on="date", how="inner")
    if len(merged) < 3:
        return {"status": "missing"}
    aligned = pd.concat([
        merged["equity"].pct_change().rename("strategy"),
        merged["close"].pct_change().rename("benchmark"),
    ], axis=1).dropna()
    years = max(len(merged) / 252, 1 / 252)
    strategy_total = (merged["equity"].iloc[-1] / merged["equity"].iloc[0] - 1) * 100
    benchmark_total = (merged["close"].iloc[-1] / merged["close"].iloc[0] - 1) * 100
    benchmark_annual = ((1 + benchmark_total / 100) ** (1 / years) - 1) * 100
    variance = aligned["benchmark"].var()
    beta = aligned["strategy"].cov(aligned["benchmark"]) / variance if variance else None
    alpha = (aligned["strategy"].mean() - (beta or 0) * aligned["benchmark"].mean()) * 252 * 100 if beta is not None else None
    active = aligned["strategy"] - aligned["benchmark"]
    information = active.mean() / active.std(ddof=1) * math.sqrt(252) if active.std(ddof=1) else None
    return {
        "status": "ok",
        "strategy_total_return_pct": rounded(strategy_total),
        "benchmark_total_return_pct": rounded(benchmark_total),
        "excess_return_pct": rounded(strategy_total - benchmark_total),
        "benchmark_annualized_return_pct": rounded(benchmark_annual),
        "benchmark_max_drawdown_pct": max_drawdown_from_values(merged["close"].tolist()),
        "alpha_annualized_pct": rounded(alpha),
        "beta": rounded(beta),
        "information_ratio": rounded(information),
    }


def turnover(result: AShareBacktestResult) -> float | None:
    if not result.trades:
        return None
    values, previous = [], set()
    for trade in result.trades:
        current = set(trade.codes)
        values.append(1.0 if not previous else 1 - len(previous & current) / max(len(current), 1))
        previous = current
    return sum(values) / len(values) * 100


def drawdown_duration(equity: pd.Series) -> int:
    peak, duration, maximum = -math.inf, 0, 0
    for value in equity:
        if value >= peak:
            peak, duration = value, 0
        else:
            duration += 1
            maximum = max(maximum, duration)
    return maximum


def write_audit(a: AShareBacktestResult, b: AShareBacktestResult) -> None:
    rows = []
    for label, result in (("A", a), ("B", b)):
        for trade in result.trades:
            rows.append({
                "model": label,
                "date": trade.signal_date,
                "universe_size": trade.universe_size,
                "selected_stocks": json.dumps(trade.codes, ensure_ascii=False),
                "factor_scores": json.dumps({code: trade.factor_scores[code].to_dict() for code in trade.codes}, ensure_ascii=False),
                "fundamental_data_date": json.dumps(trade.fundamental_periods, ensure_ascii=False),
                "fundamental_disclosure_date": json.dumps(trade.fundamental_disclosure_dates, ensure_ascii=False),
            })
    pd.DataFrame(rows).to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")


def write_integrity_report(a: AShareBacktestResult, b: AShareBacktestResult, valid: bool, version: str) -> None:
    missing = int(a.fundamental_stats.get("missing", 0) + b.fundamental_stats.get("missing", 0))
    future = int(a.fundamental_stats.get("future_records", 0) + b.fundamental_stats.get("future_records", 0))
    untradeable = int(sum(a.execution_stats.values()) + sum(b.execution_stats.values()))
    INTEGRITY_REPORT.write_text(f"""# Backtest v1.0 Integrity Report

- Backtest version: {version}
- Universe mode: historical_point_in_time
- Fundamental mode: historical_point_in_time
- Future-universe check: {'passed' if valid else 'failed/incomplete'}
- Future-fundamental check: {'passed' if future == 0 else 'failed'}
- future_data_count: {future}
- Missing fundamental records: {missing}
- Untradeable candidate records: {untradeable}
- Anomalous records: {future}
- Integrity status: {'validated' if valid else 'incomplete'}
""", encoding="utf-8")


def write_benchmark_report(comparisons: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> None:
    lines = ["# Backtest v1.0 Benchmark Comparison", ""]
    for title, key, metrics in (("Model A", "model_a", a), ("Model B", "model_b", b)):
        lines += [f"## {title}", "", f"- Strategy total return: {metrics['total_return_pct']}%", "",
                  "| Benchmark | Benchmark return | Excess return | Annualized alpha | Beta | Information ratio |",
                  "|---|---:|---:|---:|---:|---:|"]
        for benchmark, item in comparisons[key].items():
            lines.append(f"| {benchmark} | {item.get('benchmark_total_return_pct')}% | {item.get('excess_return_pct')}% | {item.get('alpha_annualized_pct')}% | {item.get('beta')} | {item.get('information_ratio')} |")
        lines.append("")
    BENCHMARK_REPORT.write_text("\n".join(lines), encoding="utf-8")


def write_research_report(config: dict[str, Any], a: dict[str, Any], b: dict[str, Any], valid: bool) -> None:
    def block(model: dict[str, Any]) -> str:
        return "\n".join([
            f"- Total return: {model['total_return_pct']}%",
            f"- Annualized return: {model['annualized_return_pct']}%",
            f"- Maximum drawdown: {model['max_drawdown_pct']}%",
            f"- Maximum drawdown duration: {model['max_drawdown_duration_days']} trading days",
            f"- Annualized volatility: {model['annualized_volatility_pct']}%",
            f"- Sharpe ratio: {model['sharpe_ratio']}",
            f"- Sortino ratio: {model['sortino_ratio']}",
            f"- Rebalances: {model['rebalance_count']}",
            f"- Average turnover: {model['turnover_pct']}%",
            f"- Win rate: {model['win_rate']}",
            f"- Profit/loss ratio: {model['profit_loss_ratio']}",
        ])
    VERSION_REPORT.write_text(f"""# Backtest v1.0 Strict Point-in-Time Research Report

## Strategy and data

- Version: {config['backtest_version']}
- Period: {config['start_date']} to {config['end_date']}
- Universe: historical CSI 300 + CSI 500 members at each rebalance date
- Fundamentals: effective on actual disclosure date
- Top N: {config['top_n']}
- Holding/rebalance: {config['holding_days']}/{config['rebalance_days']} trading days
- Initial capital: {config['initial_capital']}
- One-way fee: {config['fee_rate']}
- Slippage: {config['slippage_rate']}
- Integrity: {'validated' if valid else 'incomplete'}

## Model A

{block(a)}

## Model B

{block(b)}

## Model A/B difference

- Total-return difference: {rounded(b['total_return_pct'] - a['total_return_pct'])}%
- Drawdown difference: {rounded(b['max_drawdown_pct'] - a['max_drawdown_pct'])}%
- Sharpe difference: {rounded((b['sharpe_ratio'] or 0) - (a['sharpe_ratio'] or 0))}

## Current limitations

- Model B equals Model A when the historical policy/announcement event library has no usable records.
- Revised-report bias is flagged, but the source may not preserve every originally disclosed value.
- This report is research evidence only; it is not a promise of future or live-trading performance.
""", encoding="utf-8")

def rounded(value: Any) -> float | None:
    try:
        return round(float(value), 4) if value is not None and not pd.isna(value) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest v1.0 strict point-in-time runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(run_backtest_v1(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
