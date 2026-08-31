"""Cross-sectional factor diagnostics for isolated V3 research outputs.

This module never changes a portfolio.  It evaluates the scores that were
already available on each V3 signal date against the *subsequent* rebalance
window, keeping factor research separate from strategy construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark import load_benchmark
from market_data_manager import MarketDataManager
from research.experiments.registry import ExperimentRequest, create_experiment, update_experiment_status
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE
from research.v3.preflight import OUTPUT_DIR, default_v3_config, run_preflight


RUN_DIR = OUTPUT_DIR / "runs" / "factor_research_no_gate"
OUTPUT_DIR_FACTOR = OUTPUT_DIR / "factor_research"
FACTOR_COLUMNS = ["momentum_score", "money_flow_score", "fundamental_score", "technical_score", "total_score"]


def _direct_close(history: pd.DataFrame, date: str) -> float | None:
    rows = history[history["date"].astype(str) == str(date)]
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[-1].get("close"), errors="coerce")
    return None if pd.isna(value) or value <= 0 else float(value)


def _forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    dates = (
        panel[["signal_date", "execution_date"]].drop_duplicates()
        .sort_values("signal_date").reset_index(drop=True)
    )
    dates["forward_end_date"] = dates["execution_date"].shift(-1)
    panel = panel.merge(dates, on=["signal_date", "execution_date"], how="left")
    panel = panel[panel["forward_end_date"].notna()].copy()

    manager = MarketDataManager(cache_dir=HFQ_BAOSTOCK_CACHE)
    histories = manager.load_histories(
        sorted(panel["stock_code"].astype(str).unique()), "2014-01-01", "2025-12-31", min_rows=1, allow_network=False,
    )
    rows: list[dict] = []
    for item in panel.itertuples(index=False):
        history = histories.get(str(item.stock_code))
        if history is None:
            continue
        entry = _direct_close(history, item.execution_date)
        exit_ = _direct_close(history, item.forward_end_date)
        if entry is None or exit_ is None:
            continue
        row = item._asdict()
        row["forward_return"] = exit_ / entry - 1
        rows.append(row)
    return pd.DataFrame(rows)


def _daily_ic(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for signal_date, frame in observations.groupby("signal_date"):
        if len(frame) < 5 or frame["forward_return"].nunique() < 2:
            continue
        for factor in FACTOR_COLUMNS:
            score = pd.to_numeric(frame[factor], errors="coerce")
            future = pd.to_numeric(frame["forward_return"], errors="coerce")
            valid = score.notna() & future.notna()
            if valid.sum() < 5 or score[valid].nunique() < 2:
                continue
            rows.append({
                "signal_date": signal_date,
                "factor": factor,
                "observations": int(valid.sum()),
                "pearson_ic": score[valid].corr(future[valid], method="pearson"),
                # Spearman is Pearson correlation of ranks.  Keeping this
                # implementation dependency-free makes the research runner
                # usable in the project's lightweight Python environment.
                "spearman_ic": score[valid].rank().corr(future[valid].rank(), method="pearson"),
            })
    return pd.DataFrame(rows)


def _quantile_returns(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for signal_date, frame in observations.groupby("signal_date"):
        for factor in FACTOR_COLUMNS:
            clean = frame[[factor, "forward_return"]].dropna().copy()
            if len(clean) < 10 or clean[factor].nunique() < 5:
                continue
            clean["quantile"] = pd.qcut(clean[factor].rank(method="first"), 5, labels=False) + 1
            grouped = clean.groupby("quantile", observed=True)["forward_return"].agg(["mean", "count"])
            for quantile, result in grouped.iterrows():
                rows.append({
                    "signal_date": signal_date, "factor": factor, "quantile": int(quantile),
                    "mean_forward_return": float(result["mean"]), "observations": int(result["count"]),
                })
    return pd.DataFrame(rows)


def _factor_correlations(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for signal_date, frame in observations.groupby("signal_date"):
        data = frame[FACTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 5:
            continue
        correlation = data.rank().corr(method="pearson")
        for left in FACTOR_COLUMNS:
            for right in FACTOR_COLUMNS:
                if left < right:
                    rows.append({"signal_date": signal_date, "factor_a": left, "factor_b": right, "spearman_correlation": correlation.loc[left, right]})
    return pd.DataFrame(rows)


def _market_factor(observations: pd.DataFrame) -> pd.DataFrame:
    dates = observations[["signal_date", "execution_date", "forward_end_date", "market_score"]].drop_duplicates()
    benchmark = load_benchmark("sh000300", "CSI 300", "2015-01-01", "2025-12-31").data
    rows: list[dict] = []
    for item in dates.itertuples(index=False):
        entry = _direct_close(benchmark, item.execution_date)
        exit_ = _direct_close(benchmark, item.forward_end_date)
        if entry is not None and exit_ is not None:
            rows.append({
                "signal_date": item.signal_date, "execution_date": item.execution_date,
                "market_score": item.market_score, "csi300_forward_return": exit_ / entry - 1,
            })
    return pd.DataFrame(rows)


def _build_report(ic: pd.DataFrame, quantiles: pd.DataFrame, correlations: pd.DataFrame, market: pd.DataFrame, observations: pd.DataFrame) -> str:
    lines = [
        "# V3 因子研究报告", "", "## 研究设计", "",
        "- 数据：V3 无 Market Regime Gate 基础模型的调仓日全横截面因子快照。",
        "- 可见性：分数仅使用信号日可见的 Historical Universe、历史行情和 PIT 基本面；评价收益来自下一次调仓执行日之间。",
        "- 这是一项 in-sample 诊断，不构成因果证明、参数优化或交易建议。",
        f"- 有效股票-调仓日观测：{len(observations)}；有效调仓期：{observations['signal_date'].nunique() if not observations.empty else 0}。",
        "", "## 横截面 IC", "", "| 因子 | 平均 Pearson IC | 平均 Spearman IC | 调仓期数 |", "|---|---:|---:|---:|",
    ]
    if not ic.empty:
        summary = ic.groupby("factor", as_index=False).agg(pearson_ic=("pearson_ic", "mean"), spearman_ic=("spearman_ic", "mean"), periods=("signal_date", "nunique"))
        for item in summary.itertuples(index=False):
            lines.append(f"| {item.factor} | {item.pearson_ic:.4f} | {item.spearman_ic:.4f} | {item.periods} |")
    else:
        lines.append("| 无可用 IC 结果 | - | - | 0 |")
    lines.extend(["", "## 五分位分组收益", "", "| 因子 | Q1 平均收益 | Q5 平均收益 | Q5-Q1 |", "|---|---:|---:|---:|"])
    if not quantiles.empty:
        pivot = quantiles.groupby(["factor", "quantile"])["mean_forward_return"].mean().unstack()
        for factor, item in pivot.iterrows():
            q1, q5 = item.get(1), item.get(5)
            spread = q5 - q1 if pd.notna(q1) and pd.notna(q5) else float("nan")
            lines.append(f"| {factor} | {q1:.2%} | {q5:.2%} | {spread:.2%} |")
    else:
        lines.append("| 无可用分组结果 | - | - | - |")
    lines.extend(["", "## 因子相关性", "", "| 因子对 | 平均 Spearman 相关性 |", "|---|---:|"])
    if not correlations.empty:
        for item in correlations.groupby(["factor_a", "factor_b"])["spearman_correlation"].mean().reset_index().itertuples(index=False):
            lines.append(f"| {item.factor_a} / {item.factor_b} | {item.spearman_correlation:.4f} |")
    else:
        lines.append("| 无可用相关性结果 | - |")
    lines.extend(["", "## 市场因子时间序列诊断", ""])
    if len(market) >= 5 and market["market_score"].nunique() > 1:
        correlation = market["market_score"].rank().corr(market["csi300_forward_return"].rank(), method="pearson")
        lines.append(f"- Market Regime Score 与下一调仓期 CSI300 收益的 Spearman 相关：{correlation:.4f}。")
    else:
        lines.append("- 观测不足，未计算市场因子的时间序列相关。")
    lines.extend([
        "", "## 解释边界", "",
        "- 所有结果均是同一样本内的描述性统计，尚未进行训练/验证/测试切分或滚动样本外检验。",
        "- 基本面估值字段的历史覆盖限制仍会影响可评分股票集合，结果必须结合数据质量报告阅读。",
        "- 该报告不会改变 V1、V2 或 V3 基准策略及其已保存结果。", "",
    ])
    return "\n".join(lines)


def run_factor_research() -> dict:
    preflight = run_preflight()
    if preflight.integrity_status != "validated":
        raise RuntimeError("V3 preflight is not validated; refusing factor analysis.")
    panel_path = RUN_DIR / "factor_score_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing score panel: {panel_path}. Run the isolated factor-research baseline first.")
    config = default_v3_config()
    request = ExperimentRequest(
        hypothesis_note="Cross-sectional factor IC, quantile-return, and correlation diagnostics do not change the V3 strategy.",
        strategy_version=str(config["research_version"]), data_version="v3_hfq_baostock_pit_fundamentals_score_panel_v1",
        sample_period=dict(config["sample_period"]), factor_weights=dict(config["factor_weights"]),
        enabled_factors=sorted(config["factor_weights"]), market_regime_gate=False,
        execution_assumptions={"forward_window": "execution_date to next execution_date", "portfolio_changes": False},
        fee_assumptions={"not_applicable": True}, benchmark_setup=dict(config["benchmark_setup"]),
    )
    experiment = create_experiment(request)
    update_experiment_status(experiment.experiment_id, "running")
    try:
        panel = pd.read_csv(panel_path, dtype={"stock_code": str})
        observations = _forward_returns(panel)
        ic = _daily_ic(observations)
        quantiles = _quantile_returns(observations)
        correlations = _factor_correlations(observations)
        market = _market_factor(observations)
        OUTPUT_DIR_FACTOR.mkdir(parents=True, exist_ok=True)
        observations.to_csv(OUTPUT_DIR_FACTOR / "factor_forward_returns.csv", index=False, encoding="utf-8-sig")
        ic.to_csv(OUTPUT_DIR_FACTOR / "factor_ic_by_rebalance.csv", index=False, encoding="utf-8-sig")
        quantiles.to_csv(OUTPUT_DIR_FACTOR / "factor_quantile_returns.csv", index=False, encoding="utf-8-sig")
        correlations.to_csv(OUTPUT_DIR_FACTOR / "factor_correlations.csv", index=False, encoding="utf-8-sig")
        market.to_csv(OUTPUT_DIR_FACTOR / "market_factor_time_series.csv", index=False, encoding="utf-8-sig")
        report_path = OUTPUT_DIR_FACTOR / "factor_research_report.md"
        report_path.write_text(_build_report(ic, quantiles, correlations, market, observations), encoding="utf-8")
        summary = {"observations": len(observations), "periods": int(observations["signal_date"].nunique()) if not observations.empty else 0, "experiment_id": experiment.experiment_id, "output": str(OUTPUT_DIR_FACTOR)}
        (OUTPUT_DIR_FACTOR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        update_experiment_status(experiment.experiment_id, "completed", summary)
        return summary
    except Exception as error:
        update_experiment_status(experiment.experiment_id, "failed", {"error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V3 factor diagnostics from an isolated score panel.")
    parser.parse_args()
    print(json.dumps(run_factor_research(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
