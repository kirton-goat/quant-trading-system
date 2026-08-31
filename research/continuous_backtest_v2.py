from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark, max_drawdown_from_values
from event_research_data import EventResearchStats, ensure_event_templates, event_scores_as_of
from factor_engine import StockFactorScore, calculate_factor_scores_for_universe, calculate_market_regime_score_from_history
from market_data_manager import MarketDataManager
from research.a_share_backtest import historical_universe_name, uses_historical_index_universe, warmup_start_date
from research.fundamentals.fundamental_validation import FundamentalFutureDataError, determine_backtest_integrity, public_integrity_status
from research.fundamentals.point_in_time_fundamentals import get_fundamental_scores
from research.universe.historical_universe import HistoricalUniverseError, resolve_historical_universe
from research.universe.stock_filter import UniverseFilterConfig
from risk_filter import evaluate_research_risk
from universe_manager import UniverseResult, UniverseStock, normalize_universe_name


MIN_HISTORY_DAYS = 60


@dataclass
class ContinuousTarget:
    signal_date: str
    execution_date: str
    market_score: float
    reason: str
    scores: dict[str, StockFactorScore] = field(default_factory=dict)
    universe_size: int = 0
    fundamental_periods: dict[str, str] = field(default_factory=dict)
    fundamental_disclosure_dates: dict[str, str] = field(default_factory=dict)
    risk_rejected_count: int = 0
    missing_execution_price_count: int = 0


@dataclass
class ContinuousBacktestResult:
    model_label: str
    universe: UniverseResult
    top_n: int
    holding_days: int
    rebalance_days: int
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    event_boost_weight: float
    daily_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_audit: list[dict[str, Any]] = field(default_factory=list)
    fundamental_stats: dict[str, int] = field(default_factory=dict)
    execution_stats: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    fundamental_mode: str = "incomplete"
    backtest_integrity: str = "incomplete"
    timeline_validation: dict[str, Any] = field(default_factory=dict)
    event_stats: EventResearchStats = field(default_factory=EventResearchStats)

    @property
    def integrity_status(self) -> str:
        return public_integrity_status(self.backtest_integrity)


def run_continuous_backtest_v2(
    universe_name: str,
    top_n: int,
    holding_days: int,
    rebalance_days: int,
    start_date: str,
    end_date: str,
    initial_capital: float,
    fee_rate: float,
    market_min_score: float,
    model_label: str,
    event_boost_weight: float = 0.0,
    market_regime_gate: bool = True,
    allow_fundamental_network: bool = False,
    allow_market_network: bool = False,
) -> ContinuousBacktestResult:
    """Run an end-of-day continuous portfolio backtest.

    A target is calculated with information available on signal date T. At the
    close of T+1 the old equal-weight portfolio is exchanged for the target;
    the target earns returns from the following trading day. This prevents both
    same-day double counting and the v1 serial holding/scheduling cash gap.
    """
    if not uses_historical_index_universe(universe_name):
        raise ValueError("v2 formal research requires a historical CSI300/CSI500 universe")

    ensure_event_templates()
    manager = MarketDataManager()
    benchmark = load_benchmark("sh000300", "CSI 300", start_date, end_date)
    if benchmark.data.empty:
        raise RuntimeError("CSI 300 benchmark data is unavailable")
    calendar = benchmark.data["date"].astype(str).tolist()
    if len(calendar) <= MIN_HISTORY_DAYS + 1:
        raise RuntimeError("insufficient benchmark trading calendar for v2")

    signal_indices = list(range(MIN_HISTORY_DAYS, len(calendar) - 1, max(1, rebalance_days)))
    signal_dates = [calendar[index] for index in signal_indices]
    historical_by_date: dict[str, Any] = {}
    union: dict[str, dict[str, Any]] = {}
    for signal_date in signal_dates:
        point_in_time = resolve_historical_universe(
            signal_date,
            historical_universe_name(universe_name),
            filter_config=UniverseFilterConfig(require_market_history=False),
        )
        historical_by_date[signal_date] = point_in_time
        for item in point_in_time.stocks:
            union[item["code"]] = item

    universe = UniverseResult(
        universe=normalize_universe_name(universe_name),
        stocks=[
            UniverseStock(
                stock_code=code,
                stock_name=str(item.get("name") or ""),
                industry=str(item.get("sector") or "待补充"),
                list_date=str(item.get("list_date") or ""),
                universe=str(item.get("universe") or ""),
                source=str(item.get("source") or "历史指数成分"),
            )
            for code, item in union.items()
        ],
        source="BaoStock 按调仓日历史指数成分",
        as_of_date=end_date,
        is_historical_membership=True,
        warning="严格历史股票池：每个调仓日只使用当时的沪深300/中证500成员。",
    )
    histories = manager.load_histories(
        universe.codes,
        warmup_start_date(start_date, calendar_days=420),
        end_date.replace("-", ""),
        min_rows=90,
        allow_network=allow_market_network,
    )
    if not histories:
        raise RuntimeError("no local historical market data is available for v2")

    code_industries = {item.stock_code: item.industry for item in universe.stocks}
    fundamental_totals = {"requested": 0, "available": 0, "missing": 0, "revised": 0, "future_records": 0}
    execution_stats = {"risk_rejected": 0, "missing_execution_price": 0, "held_missing_price": 0}
    event_stats = EventResearchStats()
    targets: dict[str, ContinuousTarget] = {}
    for signal_index in signal_indices:
        signal_date = calendar[signal_index]
        execution_date = calendar[signal_index + 1]
        target, stats, event_stats = build_target(
            signal_date=signal_date,
            execution_date=execution_date,
            benchmark=benchmark.data,
            histories=histories,
            universe_name=universe_name,
            code_industries=code_industries,
            top_n=top_n,
            market_min_score=market_min_score,
            market_regime_gate=market_regime_gate,
            event_boost_weight=event_boost_weight,
            allow_fundamental_network=allow_fundamental_network,
            event_stats=event_stats,
        )
        for key, value in stats.items():
            fundamental_totals[key] = fundamental_totals.get(key, 0) + int(value)
        targets[execution_date] = target

    equity = float(initial_capital)
    current_codes: list[str] = []
    cash_reason = "warmup_or_no_signal"
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    previous_date: str | None = None
    for date in calendar:
        # The return for this date belongs to the portfolio carried into it.
        # Any rebalance executes at the close and becomes next-day exposure.
        carried_codes = list(current_codes)
        carried_cash_reason = cash_reason
        daily_return, missing_prices = portfolio_close_to_close_return(histories, carried_codes, previous_date, date)
        execution_stats["held_missing_price"] += missing_prices
        equity *= 1 + daily_return
        transaction_cost = 0.0
        buy_turnover = sell_turnover = 0.0
        reason = "normal_holding" if carried_codes else carried_cash_reason
        if date in targets:
            target = targets[date]
            next_codes = list(target.scores)
            buy_turnover, sell_turnover = calculate_turnover(current_codes, next_codes)
            gross_turnover = buy_turnover + sell_turnover
            transaction_cost = equity * fee_rate * gross_turnover
            equity -= transaction_cost
            old_codes = list(current_codes)
            current_codes = next_codes
            cash_reason = target.reason if not next_codes else "normal_holding"
            audits.append({
                "signal_date": target.signal_date,
                "execution_date": date,
                "market_score": round(target.market_score, 4),
                "market_regime_gate": market_regime_gate,
                "reason": target.reason,
                "universe_size": target.universe_size,
                "selected_count": len(next_codes),
                "previous_codes": ",".join(old_codes),
                "target_codes": ",".join(next_codes),
                "buy_turnover": round(buy_turnover, 6),
                "sell_turnover": round(sell_turnover, 6),
                "gross_turnover": round(gross_turnover, 6),
                "transaction_cost": round(transaction_cost, 4),
                "risk_rejected_count": target.risk_rejected_count,
                "missing_execution_price_count": target.missing_execution_price_count,
                "fundamental_data_date": target.fundamental_periods,
                "fundamental_disclosure_date": target.fundamental_disclosure_dates,
            })
        rows.append({
            "date": date,
            "equity": round(equity, 2),
            "return_pct": round((equity / initial_capital - 1) * 100, 4),
            "daily_return_pct": round(daily_return * 100, 6),
            "holding_codes": ",".join(carried_codes),
            "holding_count": len(carried_codes),
            "portfolio_exposure": 1.0 if carried_codes else 0.0,
            "cash_ratio": 0.0 if carried_codes else 1.0,
            "cash_reason": reason if not carried_codes else "normal_holding",
            "end_holding_codes": ",".join(current_codes),
            "end_holding_count": len(current_codes),
            "end_portfolio_exposure": 1.0 if current_codes else 0.0,
            "buy_turnover": round(buy_turnover, 6),
            "sell_turnover": round(sell_turnover, 6),
            "transaction_cost": round(transaction_cost, 4),
        })
        previous_date = date

    curve = pd.DataFrame(rows)
    fundamental_mode = "historical_point_in_time" if fundamental_totals["requested"] and fundamental_totals["future_records"] == 0 else "incomplete"
    timeline_validation = validate_continuous_timeline(curve, audits)
    integrity = determine_backtest_integrity(
        "historical_point_in_time",
        fundamental_mode,
        fundamental_totals["future_records"] == 0 and timeline_validation["passed"],
    )
    notes = [
        "v2 使用连续组合状态机：信号日计算，下一交易日收盘按真实换手切换组合。",
        "Market Regime gate 沿用 v1 的“本期保持现金”语义：执行日清仓并进入现金。",
        "普通财经新闻不参与任何交易评分；政策/公告仅作为模型B辅助增强。",
    ]
    return ContinuousBacktestResult(
        model_label=model_label,
        universe=universe,
        top_n=top_n,
        holding_days=holding_days,
        rebalance_days=rebalance_days,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_equity=round(equity, 2),
        event_boost_weight=event_boost_weight,
        daily_curve=curve,
        rebalance_audit=audits,
        fundamental_stats=fundamental_totals,
        execution_stats=execution_stats,
        notes=notes,
        fundamental_mode=fundamental_mode,
        backtest_integrity=integrity,
        timeline_validation=timeline_validation,
        event_stats=event_stats,
    )


def build_target(
    *, signal_date: str, execution_date: str, benchmark: pd.DataFrame,
    histories: dict[str, pd.DataFrame], universe_name: str, code_industries: dict[str, str],
    top_n: int, market_min_score: float, market_regime_gate: bool, event_boost_weight: float,
    allow_fundamental_network: bool, event_stats: EventResearchStats,
) -> tuple[ContinuousTarget, dict[str, int], EventResearchStats]:
    totals = {"requested": 0, "available": 0, "missing": 0, "revised": 0, "future_records": 0}
    market_score = calculate_market_regime_score_from_history(benchmark, signal_date)
    if market_regime_gate and market_score < market_min_score:
        return ContinuousTarget(signal_date, execution_date, market_score, "market_regime_block"), totals, event_stats
    try:
        point_in_time = resolve_historical_universe(
            signal_date,
            historical_universe_name(universe_name),
            histories=histories,
            filter_config=UniverseFilterConfig(require_market_history=True),
        )
    except HistoricalUniverseError:
        return ContinuousTarget(signal_date, execution_date, market_score, "insufficient_universe"), totals, event_stats
    eligible = {code: histories[code] for code in point_in_time.codes if code in histories}
    if not eligible:
        return ContinuousTarget(signal_date, execution_date, market_score, "insufficient_universe"), totals, event_stats
    try:
        fundamentals = get_fundamental_scores(
            list(eligible), signal_date, price_histories=None, allow_network=allow_fundamental_network, strict=True
        )
    except FundamentalFutureDataError:
        totals["future_records"] = 1
        raise
    totals.update({
        "requested": len(eligible), "available": len(fundamentals.scores), "missing": len(fundamentals.missing_codes),
        "revised": len(fundamentals.revised_codes), "future_records": fundamentals.future_records,
    })
    if not fundamentals.scores:
        return ContinuousTarget(signal_date, execution_date, market_score, "missing_fundamental_data", universe_size=len(eligible)), totals, event_stats
    event_scores, event_stats = event_scores_as_of(signal_date, list(eligible), code_industries=code_industries)
    scores = calculate_factor_scores_for_universe(
        eligible, signal_date, market_regime_score=market_score, fundamental_scores=fundamentals.scores,
        require_fundamentals=True, event_scores=event_scores, event_boost_weight=event_boost_weight,
    )
    if not scores:
        return ContinuousTarget(signal_date, execution_date, market_score, "insufficient_factor_score", universe_size=len(eligible)), totals, event_stats
    selected, risk_rejected, missing_execution_price = select_continuous_target(scores, eligible, signal_date, execution_date, top_n)
    if not selected:
        reason = "risk_filter_block" if risk_rejected >= len(scores) else "missing_market_data"
        return ContinuousTarget(
            signal_date, execution_date, market_score, reason, universe_size=len(eligible),
            risk_rejected_count=risk_rejected, missing_execution_price_count=missing_execution_price,
        ), totals, event_stats
    records = fundamentals.records
    return ContinuousTarget(
        signal_date=signal_date,
        execution_date=execution_date,
        market_score=market_score,
        reason="normal_holding",
        scores={item.stock_code: item for item in selected},
        universe_size=len(eligible),
        fundamental_periods={code: records[code].report_period for code in [item.stock_code for item in selected] if code in records},
        fundamental_disclosure_dates={code: records[code].disclosure_date for code in [item.stock_code for item in selected] if code in records},
        risk_rejected_count=risk_rejected,
        missing_execution_price_count=missing_execution_price,
    ), totals, event_stats


def select_continuous_target(
    scores: list[StockFactorScore], histories: dict[str, pd.DataFrame], signal_date: str, execution_date: str, top_n: int
) -> tuple[list[StockFactorScore], int, int]:
    selected: list[StockFactorScore] = []
    risk_rejected = missing_execution_price = 0
    for score in scores:
        history = histories.get(score.stock_code)
        if history is None or history.empty:
            continue
        if not evaluate_research_risk(history, signal_date).allowed:
            risk_rejected += 1
            continue
        if close_on_date(history, execution_date) is None:
            missing_execution_price += 1
            continue
        selected.append(score)
        if len(selected) >= top_n:
            break
    return selected, risk_rejected, missing_execution_price


def close_on_date(history: pd.DataFrame, date: str) -> float | None:
    rows = history[history["date"].astype(str) == str(date)]
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[-1]["close"], errors="coerce")
    return float(value) if pd.notna(value) else None


def portfolio_close_to_close_return(
    histories: dict[str, pd.DataFrame], codes: list[str], previous_date: str | None, date: str
) -> tuple[float, int]:
    if not codes or previous_date is None:
        return 0.0, 0
    returns: list[float] = []
    missing = 0
    for code in codes:
        history = histories.get(code)
        previous = close_on_date(history, previous_date) if history is not None else None
        current = close_on_date(history, date) if history is not None else None
        if previous in (None, 0) or current is None:
            missing += 1
            continue
        returns.append(current / previous - 1)
    return (sum(returns) / len(returns) if returns else 0.0), missing


def calculate_turnover(previous_codes: list[str], target_codes: list[str]) -> tuple[float, float]:
    previous = set(previous_codes)
    target = set(target_codes)
    previous_weight = 1 / len(previous) if previous else 0.0
    target_weight = 1 / len(target) if target else 0.0
    all_codes = previous | target
    buy = sum(
        max((target_weight if code in target else 0.0) - (previous_weight if code in previous else 0.0), 0.0)
        for code in all_codes
    )
    sell = sum(
        max((previous_weight if code in previous else 0.0) - (target_weight if code in target else 0.0), 0.0)
        for code in all_codes
    )
    return buy, sell


def validate_continuous_timeline(curve: pd.DataFrame, audits: list[dict[str, Any]]) -> dict[str, Any]:
    schedule_gap_days = int((curve.get("cash_reason", pd.Series(dtype=str)) == "rebalance_schedule_gap").sum())
    successful = [row for row in audits if int(row.get("selected_count", 0)) > 0]
    permitted_cash_days = 0
    unexpected_cash_days = 0
    permitted_reasons = {
        "market_regime_block", "risk_filter_block", "insufficient_factor_score",
        "missing_fundamental_data", "missing_market_data", "insufficient_universe",
    }
    if len(successful) > 1:
        date_index = {str(date): index for index, date in enumerate(curve["date"].astype(str))}
        for prior, current in zip(successful, successful[1:]):
            start = date_index[str(prior["execution_date"])]
            end = date_index[str(current["execution_date"])]
            cash_rows = curve.iloc[start + 1 : end]
            cash_rows = cash_rows[cash_rows["portfolio_exposure"] == 0]
            permitted_cash_days += int(cash_rows["cash_reason"].isin(permitted_reasons).sum())
            unexpected_cash_days += int((~cash_rows["cash_reason"].isin(permitted_reasons)).sum())
    return {
        "passed": schedule_gap_days == 0 and unexpected_cash_days == 0,
        "schedule_gap_days": schedule_gap_days,
        "cash_days_between_successful_rebalances": unexpected_cash_days,
        "permitted_cash_days_between_successful_rebalances": permitted_cash_days,
        "unexpected_cash_days_between_successful_rebalances": unexpected_cash_days,
    }


def continuous_metrics(result: ContinuousBacktestResult) -> dict[str, Any]:
    curve = result.daily_curve
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    daily = equity.pct_change().dropna()
    years = max(len(equity) / 252, 1 / 252)
    total = (equity.iloc[-1] / result.initial_capital - 1) * 100
    annual = ((1 + total / 100) ** (1 / years) - 1) * 100
    volatility = daily.std(ddof=1) * math.sqrt(252) * 100 if len(daily) > 1 else None
    sharpe = daily.mean() / daily.std(ddof=1) * math.sqrt(252) if len(daily) > 1 and daily.std(ddof=1) else None
    downside = daily[daily < 0]
    sortino = daily.mean() / downside.std(ddof=1) * math.sqrt(252) if len(downside) > 1 and downside.std(ddof=1) else None
    max_drawdown = max_drawdown_from_values(equity.tolist())
    calmar = annual / abs(max_drawdown) if max_drawdown else None
    reason_days = {str(key): int(value) for key, value in curve.loc[curve["cash_ratio"] == 1, "cash_reason"].value_counts().items()}
    return {
        "model": result.model_label,
        "total_return_pct": rounded(total), "annualized_return_pct": rounded(annual),
        "annualized_volatility_pct": rounded(volatility), "sharpe_ratio": rounded(sharpe),
        "sortino_ratio": rounded(sortino), "calmar_ratio": rounded(calmar),
        "max_drawdown_pct": max_drawdown, "max_drawdown_duration_days": drawdown_duration(equity),
        "rebalance_count": len(result.rebalance_audit),
        "turnover_pct": rounded(float((curve["buy_turnover"] + curve["sell_turnover"]).sum() * 100)),
        "buy_turnover_pct": rounded(float(curve["buy_turnover"].sum() * 100)),
        "sell_turnover_pct": rounded(float(curve["sell_turnover"].sum() * 100)),
        "transaction_cost": rounded(float(curve["transaction_cost"].sum())),
        "average_exposure": rounded(float(curve["portfolio_exposure"].mean())),
        "cash_days": int((curve["cash_ratio"] == 1).sum()),
        "average_cash_ratio": rounded(float(curve["cash_ratio"].mean())),
        "cash_reason_days": reason_days,
        "future_fundamental_data": int(result.fundamental_stats.get("future_records", 0)),
        "fundamental_mode": result.fundamental_mode,
        "universe_mode": "historical_point_in_time",
        "timeline_validation": result.timeline_validation,
        "backtest_integrity": result.backtest_integrity,
        "integrity_status": result.integrity_status,
    }


def drawdown_duration(equity: pd.Series) -> int:
    peak, duration, maximum = -math.inf, 0, 0
    for value in equity:
        if value >= peak:
            peak, duration = value, 0
        else:
            duration += 1
            maximum = max(maximum, duration)
    return maximum


def rounded(value: Any) -> float | None:
    try:
        return round(float(value), 4) if value is not None and not pd.isna(value) else None
    except (TypeError, ValueError):
        return None
