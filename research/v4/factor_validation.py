"""Single-factor V4 snapshot validation without changing strategy defaults."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from research.v3.ablation_replay import run_replay
from research.v4.config import V4_OUTPUT_DIR, default_v4_config
from research.v4.factor_recipes import append_experiment, list_factor_experiments
from research.v4.factor_recipes import DEFAULTS
from research.v4.component_panel import build_component_score_panel
from research.v4.strategy_lab import V4_PANEL_PATH
from benchmark import load_benchmark
from market_data_manager import MarketDataManager
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE


OUTPUT = V4_OUTPUT_DIR / "factor_validation" / "runs"
STOCK_FACTORS = {"momentum", "money_flow", "fundamental", "technical"}


def _number(value: Any) -> float | None:
    result = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(result) else float(result)


def _factor_diagnostics(panel_path: Path, factor: str, start: str, end: str) -> dict[str, Any]:
    """IC / Rank IC / quantile diagnostics using next scheduled execution.

    This is intentionally descriptive in-sample research; it is saved with
    the experiment, never used to auto-select or activate a recipe.
    """
    panel = pd.read_csv(panel_path, dtype={"stock_code": str})
    dates = panel[["signal_date", "execution_date"]].drop_duplicates().sort_values("signal_date")
    dates["forward_end"] = dates["execution_date"].shift(-1)
    panel = panel.merge(dates, on=["signal_date", "execution_date"], how="left")
    panel = panel[(panel["signal_date"] >= start) & (panel["signal_date"] <= end) & panel["forward_end"].notna()].copy()
    panel["risk_allowed"] = panel["risk_allowed"].astype(str).str.lower().isin(("true", "1", "yes"))
    panel = panel[panel["risk_allowed"] & panel["entry_price_available"].astype(str).str.lower().isin(("true", "1", "yes"))]
    codes = sorted(panel["stock_code"].unique())
    histories = MarketDataManager(cache_dir=HFQ_BAOSTOCK_CACHE).load_histories(codes, "2014-01-01", "2025-12-31", min_rows=1, allow_network=False)
    forward: list[dict[str, Any]] = []
    for item in panel.itertuples(index=False):
        history = histories.get(str(item.stock_code))
        if history is None: continue
        series = history.set_index(history["date"].astype(str))["close"]
        entry, exit_ = _number(series.get(str(item.execution_date))), _number(series.get(str(item.forward_end)))
        score = _number(getattr(item, f"raw_{factor}_score"))
        if entry and exit_ and score is not None:
            forward.append({"signal_date": item.signal_date, "score": score, "forward_return": exit_ / entry - 1})
    observations = pd.DataFrame(forward)
    if observations.empty: return {"observations": 0, "mean_ic": None, "mean_rank_ic": None, "icir": None, "positive_ic_ratio": None, "quantile_returns": {}}
    daily = []
    quantiles = []
    for date, frame in observations.groupby("signal_date"):
        if len(frame) < 5 or frame["score"].nunique() < 2: continue
        ic = frame["score"].corr(frame["forward_return"])
        rank_ic = frame["score"].rank().corr(frame["forward_return"].rank())
        daily.append({"signal_date": date, "ic": ic, "rank_ic": rank_ic})
        if len(frame) >= 10:
            frame = frame.copy(); frame["quantile"] = pd.qcut(frame["score"].rank(method="first"), 5, labels=False) + 1
            quantiles.extend(frame.groupby("quantile", observed=True)["forward_return"].mean().reset_index().to_dict("records"))
    ic_data = pd.DataFrame(daily)
    grouped = pd.DataFrame(quantiles)
    mean_ic = float(ic_data["ic"].mean()) if not ic_data.empty else None
    return {"observations": int(len(observations)), "periods": int(observations["signal_date"].nunique()), "mean_ic": round(mean_ic, 6) if mean_ic is not None else None, "mean_rank_ic": round(float(ic_data["rank_ic"].mean()), 6) if not ic_data.empty else None, "icir": round(float(ic_data["ic"].mean() / ic_data["ic"].std(ddof=1)), 6) if len(ic_data) > 1 and ic_data["ic"].std(ddof=1) else None, "positive_ic_ratio": round(float((ic_data["ic"] > 0).mean()), 6) if not ic_data.empty else None, "quantile_returns": {f"Q{int(key)}": round(float(value), 6) for key, value in grouped.groupby("quantile", observed=True)["forward_return"].mean().items()} if not grouped.empty else {}}


def run_single_factor(payload: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    factor = str(payload["factor_name"])
    hypothesis = str(payload.get("hypothesis_note") or "").strip()
    if not hypothesis: raise ValueError("研究假设不能为空。")
    if factor not in STOCK_FACTORS: raise ValueError("Market 必须使用 Market 时间序列审计，不能按 TopN 单因子回放。")
    if not V4_PANEL_PATH.exists(): raise ValueError("V4 因子快照不可用。")
    top_values = payload.get("top_n_values") or [20]
    top_values = sorted({int(value) for value in top_values})
    if any(value not in {5, 10, 20} for value in top_values): raise ValueError("单因子 TopN 仅支持 5、10、20。")
    config = default_v4_config(); start = str(payload.get("start_date") or config["sample_period"]["start"]); end = str(payload.get("end_date") or config["sample_period"]["end"])
    if start > end: raise ValueError("开始日期必须早于结束日期。")
    experiment_id = f"factor_{factor}_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}_{recipe['recipe_hash'][:6]}"
    weights = {name: 1.0 if name == factor else 0.0 for name in ("market_regime", "momentum", "money_flow", "fundamental", "technical")}
    component_panel, component_metadata = build_component_score_panel(factor, recipe)
    diagnostics = _factor_diagnostics(component_panel, factor, start, end)
    rows = []
    for top_n in top_values:
        result = run_replay(name=f"{experiment_id}_top{top_n}", weights=weights, panel_path=component_panel, top_n=top_n, market_regime_gate=bool(payload.get("market_regime_gate", False)), config_overrides=config)
        curve = result.curve[(result.curve["date"] >= start) & (result.curve["date"] <= end)].copy()
        # Metrics must use the exact chosen window; replay's full-sample metrics remain in the audit output.
        if len(curve) < 3: raise ValueError("所选区间没有足够交易日。")
        from research.v3.ablation_replay import _metrics
        metrics = _metrics(curve, float(config["initial_capital"]))
        target = OUTPUT / experiment_id / f"top{top_n}"; target.mkdir(parents=True, exist_ok=True)
        curve.to_csv(target / "daily_equity.csv", index=False, encoding="utf-8-sig")
        result.audit.to_csv(target / "rebalance_audit.csv", index=False, encoding="utf-8-sig")
        rows.append({"top_n": top_n, "metrics": metrics, "output_dir": str(target)})
    record = {"experiment_id": experiment_id, "created_at": pd.Timestamp.utcnow().isoformat(), "factor_name": factor, "recipe": recipe, "hypothesis_note": hypothesis, "sample_period": {"start": start, "end": end}, "top_n_values": top_values, "market_regime_gate": bool(payload.get("market_regime_gate", False)), "data_version": "v4_entry_timing_hfq_baostock_pit_fundamentals_score_panel_v1", "benchmark": ["CSI300", "CSI500"], "fee_rate": config["fee_rate"], "results": rows, "diagnostics": diagnostics, "approved": False, "archived": False, "recipe_application": "historical_component_recompute", "component_panel": str(component_panel), "component_metadata": component_metadata}
    append_experiment(record); return record


def factor_experiment_list(factor: str | None = None) -> list[dict[str, Any]]:
    items = list_factor_experiments(factor)
    return list(reversed(items))


def market_predictive_audit() -> dict[str, Any]:
    """Time-series audit for Market; it never pretends to rank stocks."""
    if not V4_PANEL_PATH.exists():
        raise ValueError("V4 因子快照不可用。")
    panel = pd.read_csv(V4_PANEL_PATH, usecols=["signal_date", "execution_date", "market_score"])
    dates = panel.drop_duplicates(["signal_date", "execution_date"]).sort_values("signal_date").reset_index(drop=True)
    benchmark = load_benchmark("sh000300", "CSI 300", "2015-01-01", "2025-12-31").data.copy()
    benchmark["date"] = benchmark["date"].astype(str)
    close = pd.to_numeric(benchmark["close"], errors="coerce")
    by_date = dict(zip(benchmark["date"], close))
    calendar = benchmark["date"].tolist(); position = {date: index for index, date in enumerate(calendar)}
    rows: list[dict[str, Any]] = []
    for item in dates.itertuples(index=False):
        start = position.get(str(item.execution_date))
        if start is None: continue
        row = {"signal_date": str(item.signal_date), "execution_date": str(item.execution_date), "market_score": float(item.market_score)}
        for horizon in (5, 10, 20, 60):
            end = start + horizon
            if end >= len(calendar):
                row[f"forward_return_{horizon}d"] = None; row[f"forward_volatility_{horizon}d"] = None
                continue
            prices = close.iloc[start:end + 1]
            row[f"forward_return_{horizon}d"] = float(prices.iloc[-1] / prices.iloc[0] - 1)
            row[f"forward_volatility_{horizon}d"] = float(prices.pct_change().dropna().std(ddof=1)) if len(prices) > 2 else None
        rows.append(row)
    data = pd.DataFrame(rows)
    if data.empty: raise ValueError("市场审计没有可用数据。")
    data["market_bucket"] = pd.qcut(data["market_score"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    summary: dict[str, Any] = {"factor_name": "market_regime", "factor_type": "time_series", "not_stock_ranking": True, "observations": len(data), "data_visibility": "V4 historical market score and forward CSI300 returns"}
    for horizon in (5, 10, 20, 60):
        column = f"forward_return_{horizon}d"
        valid = data[["market_score", column]].dropna()
        summary[f"rank_ic_{horizon}d"] = round(float(valid["market_score"].rank().corr(valid[column].rank())), 6) if len(valid) >= 5 else None
        summary[f"bucket_return_{horizon}d"] = {str(key): round(float(value), 6) for key, value in data.groupby("market_bucket", observed=True)[column].mean().dropna().items()}
    output = V4_OUTPUT_DIR / "factor_validation" / "market_predictive_audit"
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "market_forward_outcomes.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**summary, "output_dir": str(output)}
