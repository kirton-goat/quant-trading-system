"""Fast, isolated V3 factor-ablation replayer.

The expensive factor snapshot is produced once by ``research.v3.engine``.
This module replays that immutable score panel with the same risk/execution
eligibility fields, allowing explicit factor-removal experiments without
re-reading historical financial statements for every variant.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark, max_drawdown_from_values
from market_data_manager import MarketDataManager
from research.v3.engine import _daily_return, _turnover
from research.v3.execution_policy import POLICY_ID, rebalance_execution_decision
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE
from research.v3.preflight import OUTPUT_DIR, default_v3_config


PANEL_RUN_DIR = OUTPUT_DIR / "runs" / "factor_research_no_gate_final"
OUTPUT_ROOT = OUTPUT_DIR / "factor_ablation"
STOCK_FACTORS = ("momentum", "money_flow", "fundamental", "technical", "market_regime")


@dataclass
class ReplayResult:
    name: str
    weights: dict[str, float]
    curve: pd.DataFrame
    audit: pd.DataFrame
    metrics: dict[str, Any]


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {factor: max(0.0, float(weights.get(factor, 0.0))) for factor in STOCK_FACTORS}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("At least one factor must be enabled.")
    return {factor: value / total for factor, value in cleaned.items()}


def _eligible(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("risk_allowed", "entry_price_available"):
        result[column] = result[column].astype(str).str.lower().isin(("true", "1", "yes"))
    return result[result["risk_allowed"] & result["entry_price_available"]].copy()


def _score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(
        pd.to_numeric(frame.get(f"raw_{factor}_score", frame[f"{factor}_score"]), errors="coerce").fillna(float("-inf")) * weight
        for factor, weight in weights.items()
    )


def _build_targets(
    panel: pd.DataFrame,
    weights: dict[str, float],
    top_n: int,
    selection_mode: str = "top",
    quantile: int | None = None,
) -> dict[str, list[str]]:
    """Create date-specific targets from an immutable score snapshot.

    ``top`` retains the exact legacy replay behavior.  ``quantile`` is used
    only by factor-research experiments and splits each eligible cross-section
    into deterministic fifths after applying the same eligibility filters.
    """
    if selection_mode not in {"top", "quantile"}:
        raise ValueError("selection_mode must be 'top' or 'quantile'.")
    if selection_mode == "quantile" and quantile not in {1, 2, 3, 4, 5}:
        raise ValueError("quantile selection requires a quantile from 1 to 5.")
    targets: dict[str, list[str]] = {}
    default_weights = normalize_weights(default_v3_config()["factor_weights"])
    baseline_weights = all(abs(weights[factor] - default_weights[factor]) < 1e-12 for factor in STOCK_FACTORS)
    for execution_date, frame in panel.groupby("execution_date", sort=True):
        eligible = _eligible(frame)
        if selection_mode == "top" and baseline_weights and "base_rank" in eligible:
            eligible = eligible.sort_values("base_rank", ascending=True)
        else:
            eligible["replay_total_score"] = _score(eligible, weights)
            eligible = eligible.sort_values(["replay_total_score", "stock_code"], ascending=[False, True])
        if selection_mode == "top":
            selected = eligible.head(top_n)
        else:
            # The rank resolves ties deterministically before qcut assigns a
            # fifth.  Q1 is the lowest score group; Q5 is the highest.
            ranked = eligible.copy()
            ranked["factor_quantile"] = pd.qcut(
                ranked["replay_total_score"].rank(method="first"), 5, labels=False
            ) + 1
            selected = ranked[ranked["factor_quantile"] == quantile]
        targets[str(execution_date)] = selected["stock_code"].astype(str).tolist()
    return targets


def _metrics(curve: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    daily = equity.pct_change().dropna()
    years = max(len(equity) / 252, 1 / 252)
    total = float(equity.iloc[-1] / initial_capital - 1)
    cagr = (1 + total) ** (1 / years) - 1
    volatility = float(daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 else float("nan")
    sharpe = float(daily.mean() / daily.std(ddof=1) * math.sqrt(252)) if len(daily) > 1 and daily.std(ddof=1) else float("nan")
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std(ddof=1) * math.sqrt(252)) if len(downside) > 1 and downside.std(ddof=1) else float("nan")
    max_drawdown = max_drawdown_from_values(equity.tolist())
    return {
        "total_return_pct": round(total * 100, 4), "cagr_pct": round(cagr * 100, 4),
        "annualized_volatility_pct": round(volatility * 100, 4), "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4), "max_drawdown_pct": max_drawdown,
        "calmar_ratio": round(cagr * 100 / abs(max_drawdown), 4) if max_drawdown else None,
        "cash_days": int((curve["cash_ratio"] == 1).sum()),
        "average_exposure": round(float(curve["portfolio_exposure"].mean()), 4),
        "turnover_pct": round(float((curve["buy_turnover"] + curve["sell_turnover"]).sum() * 100), 4),
        "transaction_cost": round(float(curve["transaction_cost"].sum()), 2),
        "integrity_status": "validated_replay_snapshot",
        "execution_policy": POLICY_ID,
    }


def run_replay(
    name: str,
    weights: dict[str, float],
    panel_path: Path = PANEL_RUN_DIR / "factor_score_panel.csv",
    histories: dict[str, pd.DataFrame] | None = None,
    calendar: list[str] | None = None,
    top_n: int | None = None,
    market_regime_gate: bool = False,
    market_min_score: float | None = None,
    selection_mode: str = "top",
    quantile: int | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> ReplayResult:
    config = default_v3_config() | (config_overrides or {})
    weights = normalize_weights(weights)
    panel = pd.read_csv(panel_path, dtype={"stock_code": str})
    selected_top_n = int(top_n or config["top_n"])
    targets = _build_targets(panel, weights, selected_top_n, selection_mode=selection_mode, quantile=quantile)
    if market_regime_gate:
        threshold = float(config["market_min_score"] if market_min_score is None else market_min_score)
        market_scores = panel.groupby("execution_date")["market_score"].first()
        targets = {
            date: ([] if float(market_scores.get(date, 0.0)) < threshold else codes)
            for date, codes in targets.items()
        }
    all_codes = sorted({code for codes in targets.values() for code in codes})
    if histories is None:
        manager = MarketDataManager(cache_dir=HFQ_BAOSTOCK_CACHE)
        histories = manager.load_histories(all_codes, "2014-01-01", config["sample_period"]["end"], min_rows=1, allow_network=False)
    if calendar is None:
        calendar = load_benchmark("sh000300", "CSI 300", config["sample_period"]["start"], config["sample_period"]["end"]).data["date"].astype(str).tolist()

    equity = float(config["initial_capital"])
    current: list[str] = []
    previous_date: str | None = None
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for date in calendar:
        carried = list(current)
        daily_return, stale_count = _daily_return(histories, carried, previous_date, date)
        equity *= 1 + daily_return
        buy = sell = cost = 0.0
        if date in targets:
            executable, locked = rebalance_execution_decision(histories, current, date)
            next_codes = list(current) if not executable else targets[date]
            if executable:
                buy, sell = _turnover(current, next_codes)
                cost = equity * float(config["fee_rate"]) * (buy + sell)
                equity -= cost
            audits.append({
                "execution_date": date, "target_count": len(targets[date]), "selected_count": len(next_codes),
                "previous_codes": ",".join(current), "end_codes": ",".join(next_codes),
                "locked_codes": ",".join(locked), "buy_turnover": buy, "sell_turnover": sell,
                "transaction_cost": cost,
                "reason": (
                    "market_regime_block" if market_regime_gate and not targets[date]
                    else "normal_holding" if executable else "rebalance_deferred_locked_position"
                ),
            })
            current = next_codes
        rows.append({
            "date": date, "equity": round(equity, 2), "daily_return": round(daily_return, 8),
            "holding_codes": ",".join(carried), "holding_count": len(carried),
            "portfolio_exposure": 1.0 if carried else 0.0, "cash_ratio": 0.0 if carried else 1.0,
            "stale_mark_count": stale_count, "buy_turnover": buy, "sell_turnover": sell, "transaction_cost": round(cost, 4),
        })
        previous_date = date
    curve = pd.DataFrame(rows)
    result = ReplayResult(name, weights, curve, pd.DataFrame(audits), _metrics(curve, float(config["initial_capital"])))
    result.metrics.update({
        "top_n": selected_top_n,
        "selection_mode": selection_mode,
        "quantile": quantile,
        "market_regime_gate": bool(market_regime_gate),
        "market_min_score": float(config["market_min_score"] if market_min_score is None else market_min_score),
        "input_snapshot": str(panel_path),
    })
    return result


def write_replay(result: ReplayResult, output_dir: Path = OUTPUT_ROOT) -> Path:
    target = output_dir / result.name
    target.mkdir(parents=True, exist_ok=True)
    result.curve.to_csv(target / "daily_equity.csv", index=False, encoding="utf-8-sig")
    result.audit.to_csv(target / "rebalance_audit.csv", index=False, encoding="utf-8-sig")
    (target / "summary.json").write_text(json.dumps({"weights": result.weights, **result.metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
