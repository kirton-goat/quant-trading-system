"""Isolated V3 long-sample continuous-rebalance research engine.

V3 is intentionally separate from frozen V1/V2.  It uses BaoStock HFQ prices
for return calculation, BaoStock CNY turnover for liquidity/factor inputs, and
the isolated point-in-time fundamental cache.  Results are research outputs,
not a trading recommendation or a formal strategy release.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark, max_drawdown_from_values
from event_research_data import EventResearchStats, event_scores_as_of
from factor_engine import StockFactorScore, calculate_factor_scores_for_universe, calculate_market_regime_score_from_history
from market_data_manager import MarketDataManager
from research.a_share_backtest import historical_universe_name
from research.fundamentals.fundamental_validation import FundamentalFutureDataError
from research.fundamentals.point_in_time_fundamentals import get_fundamental_scores
from research.universe.index_members import get_index_snapshot
from research.universe.stock_filter import load_security_metadata
from research.universe.stock_filter import UniverseFilterConfig
from research.v3.execution_policy import POLICY_ID, close_mark_as_of, direct_close, rebalance_execution_decision
from research.v3.execution_eligibility_audit import _eligible_codes, _load_liquidity_indexes
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE
from research.v3.liquidity_acquisition import LIQUIDITY_CACHE
from research.v3.preflight import OUTPUT_DIR, default_v3_config, run_preflight, trading_rebalance_dates
from risk_filter import evaluate_research_risk


RESULT_DIR = OUTPUT_DIR / "runs"
WARMUP_DAYS = 60


@dataclass
class V3Target:
    signal_date: str
    execution_date: str
    market_score: float
    reason: str
    scores: dict[str, StockFactorScore] = field(default_factory=dict)
    # Full scored cross-section is research-only output. Selection still uses
    # ``scores`` and is therefore unchanged for existing strategy runs.
    all_scores: list[StockFactorScore] = field(default_factory=list)
    score_eligibility: dict[str, dict[str, Any]] = field(default_factory=dict)
    universe_size: int = 0
    fundamental_periods: dict[str, str] = field(default_factory=dict)
    fundamental_disclosures: dict[str, str] = field(default_factory=dict)
    risk_rejected: int = 0
    missing_execution_price: int = 0


@dataclass
class V3Result:
    model_label: str
    version: str
    config: dict[str, Any]
    daily_curve: pd.DataFrame
    rebalance_audit: pd.DataFrame
    score_panel: pd.DataFrame
    fundamental_stats: dict[str, int]
    execution_stats: dict[str, int]
    event_stats: EventResearchStats


def _merge_liquidity(histories: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Overwrite V3 factor liquidity fields with independently audited CNY data."""
    merged: dict[str, pd.DataFrame] = {}
    for code, history in histories.items():
        path = LIQUIDITY_CACHE / f"{code}.csv"
        data = history.copy()
        if path.exists():
            try:
                liquidity = pd.read_csv(path, usecols=["date", "volume", "amount"], encoding="utf-8-sig")
                liquidity = liquidity.rename(columns={"volume": "liquidity_volume", "amount": "liquidity_amount"})
                data = data.merge(liquidity, on="date", how="left")
                for base, source in (("volume", "liquidity_volume"), ("amount", "liquidity_amount")):
                    values = pd.to_numeric(data.get(source), errors="coerce")
                    data[base] = values.where(values.notna(), pd.to_numeric(data.get(base), errors="coerce"))
                data = data.drop(columns=[column for column in ("liquidity_volume", "liquidity_amount") if column in data])
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        merged[code] = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return merged


def _select(
    scores: list[StockFactorScore], factor_histories: dict[str, pd.DataFrame], execution_histories: dict[str, pd.DataFrame],
    signal_date: str, execution_date: str, top_n: int,
) -> tuple[list[StockFactorScore], int, int]:
    selected: list[StockFactorScore] = []
    risk_rejected = missing_execution = 0
    for score in scores:
        history = factor_histories.get(score.stock_code)
        if history is None or history.empty:
            continue
        if not evaluate_research_risk(history, signal_date).allowed:
            risk_rejected += 1
            continue
        if direct_close(execution_histories.get(score.stock_code), execution_date) is None:
            missing_execution += 1
            continue
        selected.append(score)
        if len(selected) == top_n:
            break
    return selected, risk_rejected, missing_execution


def _build_target(
    signal_date: str, execution_date: str, benchmark: pd.DataFrame, histories: dict[str, pd.DataFrame],
    config: dict[str, Any], candidate_codes: list[str], code_industries: dict[str, str], event_boost_weight: float, event_stats: EventResearchStats,
) -> tuple[V3Target, dict[str, int], EventResearchStats]:
    totals = {"requested": 0, "available": 0, "missing": 0, "revised": 0, "future_records": 0}
    market_score = calculate_market_regime_score_from_history(benchmark, signal_date)
    if config.get("market_regime_gate", True) and market_score < float(config["market_min_score"]):
        return V3Target(signal_date, execution_date, market_score, "market_regime_block"), totals, event_stats
    # Candidate codes are precomputed once with the same point-in-time filter
    # used by the execution eligibility audit. This avoids re-reading 1,500
    # histories for every signal date without changing the historical universe.
    eligible = {code: histories[code] for code in candidate_codes if code in histories}
    if not eligible:
        return V3Target(signal_date, execution_date, market_score, "insufficient_universe"), totals, event_stats
    try:
        fundamentals = get_fundamental_scores(
            list(eligible), signal_date, cache_dir=Path(config["fundamental_cache"]), allow_network=False, strict=True,
        )
    except FundamentalFutureDataError:
        totals["future_records"] = 1
        raise
    totals.update({
        "requested": len(eligible), "available": len(fundamentals.scores), "missing": len(fundamentals.missing_codes),
        "revised": len(fundamentals.revised_codes), "future_records": fundamentals.future_records,
    })
    if not fundamentals.scores:
        return V3Target(signal_date, execution_date, market_score, "missing_fundamental_data", universe_size=len(eligible)), totals, event_stats
    event_scores, event_stats = event_scores_as_of(signal_date, list(eligible), code_industries=code_industries)
    weights = config["factor_weights"]
    # All current stock factors and risk filters require no more than 60
    # observations. Trimming here is performance-only; every row is still at
    # or before the signal date, while entry checks retain full histories.
    factor_histories = {
        code: history[pd.to_datetime(history["date"], errors="coerce") <= pd.Timestamp(signal_date)].tail(120).reset_index(drop=True)
        for code, history in eligible.items()
    }
    scores = calculate_factor_scores_for_universe(
        factor_histories, signal_date,
        momentum_weight=float(weights["momentum"]), money_flow_weight=float(weights["money_flow"]),
        fundamental_weight=float(weights["fundamental"]), technical_weight=float(weights["technical"]),
        market_regime_weight=float(weights["market_regime"]), market_regime_score=market_score,
        fundamental_scores=fundamentals.scores, require_fundamentals=True,
        event_scores=event_scores, event_boost_weight=event_boost_weight,
        technical_variant=str(config.get("technical_variant", "legacy")),
    )
    if not scores:
        return V3Target(signal_date, execution_date, market_score, "insufficient_factor_score", universe_size=len(eligible)), totals, event_stats
    # Persist the same per-stock eligibility facts used by the selector.  This
    # makes downstream research replays faithful to the frozen risk and
    # execution rules instead of silently reconstructing a different universe.
    score_eligibility = {}
    for score in scores:
        history = factor_histories.get(score.stock_code)
        decision = evaluate_research_risk(history, signal_date) if history is not None else None
        score_eligibility[score.stock_code] = {
            "risk_allowed": bool(decision.allowed) if decision is not None else False,
            "risk_reason": decision.reason if decision is not None else "missing_factor_history",
            "entry_price_available": direct_close(eligible.get(score.stock_code), execution_date) is not None,
        }
    selected, risk_rejected, missing_execution = _select(scores, factor_histories, eligible, signal_date, execution_date, int(config["top_n"]))
    if not selected:
        reason = "risk_filter_block" if risk_rejected >= len(scores) else "missing_market_data"
        return V3Target(signal_date, execution_date, market_score, reason, universe_size=len(eligible), risk_rejected=risk_rejected, missing_execution_price=missing_execution), totals, event_stats
    codes = [item.stock_code for item in selected]
    return V3Target(
        signal_date=signal_date,
        execution_date=execution_date,
        market_score=market_score,
        reason="normal_holding",
        scores={item.stock_code: item for item in selected},
        all_scores=scores,
        score_eligibility=score_eligibility,
        universe_size=len(eligible),
        fundamental_periods={code: fundamentals.records[code].report_period for code in codes if code in fundamentals.records},
        fundamental_disclosures={code: fundamentals.records[code].disclosure_date for code in codes if code in fundamentals.records},
        risk_rejected=risk_rejected,
        missing_execution_price=missing_execution,
    ), totals, event_stats


def _daily_return(histories: dict[str, pd.DataFrame], codes: list[str], previous_date: str | None, date: str) -> tuple[float, int]:
    if not codes or previous_date is None:
        return 0.0, 0
    values: list[float] = []
    stale_count = 0
    for code in codes:
        previous = close_mark_as_of(histories.get(code), previous_date)
        current = close_mark_as_of(histories.get(code), date)
        if previous.price in (None, 0) or current.price is None:
            # A code cannot disappear from the denominator: preserve capital
            # with a zero return until an auditable price becomes observable.
            values.append(0.0)
            stale_count += 1
            continue
        values.append(current.price / previous.price - 1)
        stale_count += int(previous.is_stale or current.is_stale)
    return sum(values) / len(values), stale_count


def _turnover(previous: list[str], target: list[str]) -> tuple[float, float]:
    old, new = set(previous), set(target)
    old_weight = 1 / len(old) if old else 0.0
    new_weight = 1 / len(new) if new else 0.0
    codes = old | new
    buy = sum(max((new_weight if code in new else 0.0) - (old_weight if code in old else 0.0), 0.0) for code in codes)
    sell = sum(max((old_weight if code in old else 0.0) - (new_weight if code in new else 0.0), 0.0) for code in codes)
    return buy, sell


def run_v3(
    model_label: str = "model_a", event_boost_weight: float = 0.0, market_regime_gate: bool | None = None,
    factor_weights: dict[str, float] | None = None, technical_variant: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> V3Result:
    preflight = run_preflight()
    if preflight.integrity_status != "validated":
        raise RuntimeError("V3 preflight is not validated; refusing to emit a performance result.")
    config = default_v3_config() | (config_overrides or {}) | {
        "fundamental_cache": str(Path(__file__).resolve().parents[2] / "data_cache" / "v3_fundamentals"),
        "price_cache": str(HFQ_BAOSTOCK_CACHE), "liquidity_cache": str(LIQUIDITY_CACHE),
        "execution_policy": POLICY_ID,
    }
    if market_regime_gate is not None:
        config["market_regime_gate"] = bool(market_regime_gate)
    technical_variant = str(technical_variant or config.get("technical_variant", "legacy"))
    if technical_variant not in {"legacy", "entry_timing"}:
        raise ValueError("technical_variant must be legacy or entry_timing")
    config["technical_variant"] = technical_variant
    if factor_weights is not None:
        if set(factor_weights) != set(config["factor_weights"]) or abs(sum(factor_weights.values()) - 1.0) > 1e-8:
            raise ValueError("V3 factor weights must include every current factor and sum to 1.0")
        config["factor_weights"] = dict(factor_weights)
    benchmark = load_benchmark("sh000300", "CSI 300", config["sample_period"]["start"], config["sample_period"]["end"])
    calendar = benchmark.data["date"].astype(str).tolist()
    if len(calendar) <= WARMUP_DAYS + 1:
        raise RuntimeError("V3 benchmark calendar is unavailable")
    signal_indices = list(range(WARMUP_DAYS, len(calendar) - 1, int(config["rebalance_days"])))
    signal_dates = [calendar[index] for index in signal_indices]
    codes: set[str] = set()
    industries: dict[str, str] = {}
    members_by_signal: dict[str, dict[str, dict[str, str]]] = {}
    for signal_date in signal_dates:
        members: dict[str, dict[str, str]] = {}
        for index_name in ("CSI300", "CSI500"):
            for item in get_index_snapshot(signal_date, index_name, allow_network=False).members:
                members.setdefault(item.code, {"code": item.code, "name": item.name, "snapshot_date": item.snapshot_date})
        members_by_signal[signal_date] = members
        codes.update(members)
    manager = MarketDataManager(cache_dir=HFQ_BAOSTOCK_CACHE)
    histories = _merge_liquidity(manager.load_histories(sorted(codes), "2012-07-15", config["sample_period"]["end"], min_rows=1, allow_network=False))
    liquidity_indexes = _load_liquidity_indexes(codes, LIQUIDITY_CACHE)
    candidates_by_signal: dict[str, list[str]] = {}
    for signal_date, members in members_by_signal.items():
        metadata = load_security_metadata(signal_date, allow_network=False)
        candidates, _ = _eligible_codes(signal_date, members, metadata, liquidity_indexes, UniverseFilterConfig(require_market_history=True))
        candidates_by_signal[signal_date] = candidates
        for code in candidates:
            industries.setdefault(code, "待补充")
    targets: dict[str, V3Target] = {}
    score_rows: list[dict[str, Any]] = []
    fundamental_totals = {"requested": 0, "available": 0, "missing": 0, "revised": 0, "future_records": 0}
    event_stats = EventResearchStats()
    for index in signal_indices:
        signal_date = calendar[index]
        target, stats, event_stats = _build_target(signal_date, calendar[index + 1], benchmark.data, histories, config, candidates_by_signal.get(signal_date, []), industries, event_boost_weight, event_stats)
        for key, value in stats.items():
            fundamental_totals[key] += int(value)
        targets[target.execution_date] = target
        for rank, score in enumerate(target.all_scores, start=1):
            score_rows.append({
                "signal_date": target.signal_date,
                "execution_date": target.execution_date,
                "market_score": target.market_score,
                "universe_size": target.universe_size,
                "signal_reason": target.reason,
                # The strategy engine sorts on its unrounded internal score.
                # Persist the resulting rank so the current-weight research
                # replay can reproduce the frozen baseline exactly.
                "base_rank": rank,
                **target.score_eligibility.get(score.stock_code, {}),
                **score.to_dict(),
                "raw_momentum_score": score.momentum_score,
                "raw_money_flow_score": score.money_flow_score,
                "raw_fundamental_score": score.fundamental_score,
                "raw_technical_score": score.technical_score,
                "technical_variant": technical_variant,
                "raw_market_regime_score": score.market_regime_score,
                "raw_total_score": score.total_score,
            })

    equity = float(config["initial_capital"])
    current: list[str] = []
    cash_reason = "warmup_or_no_signal"
    previous_date: str | None = None
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    execution_stats = {"stale_mark_observations": 0, "deferred_rebalances": 0, "locked_position_observations": 0}
    for date in calendar:
        carried = list(current)
        daily_return, stale_count = _daily_return(histories, carried, previous_date, date)
        equity *= 1 + daily_return
        execution_stats["stale_mark_observations"] += stale_count
        buy = sell = cost = 0.0
        reason = "normal_holding" if carried else cash_reason
        if date in targets:
            target = targets[date]
            executable, locked = rebalance_execution_decision(histories, current, date)
            if not executable:
                execution_stats["deferred_rebalances"] += 1
                execution_stats["locked_position_observations"] += len(locked)
                audit_reason = "rebalance_deferred_locked_position"
                next_codes = list(current)
            else:
                next_codes = list(target.scores)
                buy, sell = _turnover(current, next_codes)
                cost = equity * float(config["fee_rate"]) * (buy + sell)
                equity -= cost
                cash_reason = target.reason if not next_codes else "normal_holding"
                audit_reason = target.reason
            audits.append({
                "signal_date": target.signal_date, "execution_date": date, "reason": audit_reason,
                "market_score": round(target.market_score, 4), "universe_size": target.universe_size,
                "selected_count": len(next_codes), "target_count": len(target.scores),
                "previous_codes": ",".join(current), "target_codes": ",".join(target.scores), "end_codes": ",".join(next_codes),
                "locked_codes": ",".join(locked), "buy_turnover": round(buy, 6), "sell_turnover": round(sell, 6),
                "transaction_cost": round(cost, 4), "risk_rejected": target.risk_rejected,
                "missing_execution_price": target.missing_execution_price, "fundamental_data_date": json.dumps(target.fundamental_periods, ensure_ascii=False),
                "fundamental_disclosure_date": json.dumps(target.fundamental_disclosures, ensure_ascii=False),
            })
            current = next_codes
        rows.append({
            "date": date, "equity": round(equity, 2), "daily_return": round(daily_return, 8), "holding_codes": ",".join(carried),
            "holding_count": len(carried), "portfolio_exposure": 1.0 if carried else 0.0, "cash_ratio": 0.0 if carried else 1.0,
            "cash_reason": reason if not carried else "normal_holding", "stale_mark_count": stale_count,
            "buy_turnover": buy, "sell_turnover": sell, "transaction_cost": round(cost, 4),
        })
        previous_date = date
    return V3Result(
        model_label=model_label,
        version=str(config["research_version"]),
        config=config,
        daily_curve=pd.DataFrame(rows),
        rebalance_audit=pd.DataFrame(audits),
        score_panel=pd.DataFrame(score_rows),
        fundamental_stats=fundamental_totals,
        execution_stats=execution_stats,
        event_stats=event_stats,
    )


def metrics(result: V3Result) -> dict[str, Any]:
    curve = result.daily_curve
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    daily = equity.pct_change().dropna()
    years = max(len(equity) / 252, 1 / 252)
    total = equity.iloc[-1] / float(result.config["initial_capital"]) - 1
    cagr = (1 + total) ** (1 / years) - 1
    volatility = daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 else None
    sharpe = daily.mean() / daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 and daily.std(ddof=1) else None
    downside = daily[daily < 0]
    sortino = daily.mean() / downside.std(ddof=1) * math.sqrt(252) if len(downside) > 1 and downside.std(ddof=1) else None
    drawdown = max_drawdown_from_values(equity.tolist())
    return {
        "classification": "research_experiment", "version": result.version, "model": result.model_label,
        "integrity_status": "validated", "universe_mode": "historical_point_in_time", "fundamental_mode": "historical_point_in_time",
        "execution_policy": POLICY_ID, "future_fundamental_data": result.fundamental_stats["future_records"],
        "total_return_pct": round(total * 100, 4), "cagr_pct": round(cagr * 100, 4),
        "annualized_volatility_pct": round(volatility * 100, 4) if volatility is not None else None,
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None, "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "max_drawdown_pct": drawdown, "calmar_ratio": round(cagr * 100 / abs(drawdown), 4) if drawdown else None,
        "cash_days": int((curve["cash_ratio"] == 1).sum()), "average_exposure": round(float(curve["portfolio_exposure"].mean()), 4),
        "turnover_pct": round(float((curve["buy_turnover"] + curve["sell_turnover"]).sum() * 100), 4),
        "transaction_cost": round(float(curve["transaction_cost"].sum()), 2), "execution_stats": result.execution_stats,
    }


def write_result(result: V3Result, output_dir: Path = RESULT_DIR) -> Path:
    target = output_dir / result.model_label
    target.mkdir(parents=True, exist_ok=True)
    result.daily_curve.to_csv(target / "daily_equity.csv", index=False, encoding="utf-8-sig")
    result.rebalance_audit.to_csv(target / "rebalance_audit.csv", index=False, encoding="utf-8-sig")
    result.score_panel.to_csv(target / "factor_score_panel.csv", index=False, encoding="utf-8-sig")
    payload = metrics(result)
    payload["config"] = result.config
    payload["fundamental_stats"] = result.fundamental_stats
    (target / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated V3 long-sample research.")
    parser.add_argument("--model", choices=["model_a", "model_b"], default="model_a")
    parser.add_argument("--no-market-gate", action="store_true")
    args = parser.parse_args()
    label = f"{args.model}_no_gate" if args.no_market_gate else args.model
    result = run_v3(label, 0.0 if args.model == "model_a" else 0.05, market_regime_gate=not args.no_market_gate)
    folder = write_result(result)
    print(json.dumps({"output": str(folder), **metrics(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
