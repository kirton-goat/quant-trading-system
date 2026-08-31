from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import compare_strategy_to_benchmark, load_benchmark
from event_research_data import EventResearchStats, event_scores_as_of, ensure_event_templates
from factor_engine import (
    StockFactorScore,
    calculate_factor_scores_for_universe,
    calculate_market_regime_score_from_history,
)
from market_data_manager import MarketDataManager, date_n_days_ago
from research.fundamentals.fundamental_validation import (
    FundamentalFutureDataError,
    determine_backtest_integrity,
    public_integrity_status,
)
from research.fundamentals.point_in_time_fundamentals import get_fundamental_scores
from research.universe.historical_universe import HistoricalUniverseError, resolve_historical_universe
from research.universe.stock_filter import UniverseFilterConfig
from risk_filter import evaluate_research_risk
from universe_manager import DEFAULT_UNIVERSE, UniverseResult, UniverseStock, load_universe, normalize_universe_name


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = BASE_DIR / "backtest_report.md"


@dataclass
class BacktestTrade:
    signal_date: str
    entry_date: str
    exit_date: str
    codes: list[str]
    factor_scores: dict[str, StockFactorScore]
    returns: dict[str, float]
    portfolio_return_pct: float
    entry_equity: float
    exit_equity: float
    universe_size: int = 0
    fundamental_periods: dict[str, str] = field(default_factory=dict)
    fundamental_disclosure_dates: dict[str, str] = field(default_factory=dict)


@dataclass
class AShareBacktestResult:
    universe: UniverseResult
    requested_stock_count: int
    available_stock_count: int
    top_n: int
    holding_days: int
    rebalance_days: int
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float | None
    win_rate: float | None
    max_drawdown_pct: float | None
    sharpe_ratio: float | None
    benchmark: dict[str, Any] = field(default_factory=dict)
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)
    data_stats: dict[str, Any] = field(default_factory=dict)
    model_label: str = "模型A：基础量化"
    event_boost_weight: float = 0.0
    event_stats: EventResearchStats = field(default_factory=EventResearchStats)
    fundamental_mode: str = "incomplete"
    backtest_integrity: str = "incomplete"
    fundamental_stats: dict[str, Any] = field(default_factory=dict)
    execution_stats: dict[str, int] = field(default_factory=dict)
    rebalance_audit: list[dict[str, Any]] = field(default_factory=list)
    enhanced_result: "AShareBacktestResult | None" = None

    @property
    def integrity_status(self) -> str:
        return public_integrity_status(self.backtest_integrity)


def run_a_share_backtest(
    universe_name: str = DEFAULT_UNIVERSE,
    universe_limit: int | None = None,
    top_n: int = 20,
    holding_days: int = 20,
    rebalance_days: int = 20,
    history_days: int = 720,
    start_date: str | None = None,
    end_date: str | None = None,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0015,
    model_label: str = "模型A：基础量化",
    event_boost_weight: float = 0.0,
    market_min_score: float = 40.0,
    allow_fundamental_network: bool = True,
    allow_market_network: bool = True,
    market_regime_gate: bool = True,
    collect_rebalance_audit: bool = False,
) -> AShareBacktestResult:
    end = normalize_date(end_date) or dt.datetime.now().strftime("%Y%m%d")
    start = normalize_date(start_date) or date_n_days_ago(history_days)
    manager = MarketDataManager()
    ensure_event_templates()
    benchmark = load_benchmark(start_date=start, end_date=end)
    historical_by_date: dict[str, Any] = {}
    if uses_historical_index_universe(universe_name):
        if benchmark.data.empty:
            universe = UniverseResult(
                universe=universe_name,
                stocks=[],
                source="历史股票池不可用",
                as_of_date=end,
                is_historical_membership=True,
                warning="缺少基准交易日，严格模式不允许使用当前指数成分股替代。",
            )
            return empty_result(universe, top_n, holding_days, rebalance_days, start, end, initial_capital, manager, [universe.warning])
        calendar = benchmark.data["date"].astype(str).tolist()
        signal_dates = backtest_signal_dates(calendar, holding_days, rebalance_days)
        union: dict[str, dict[str, Any]] = {}
        try:
            for signal_date in signal_dates:
                point_in_time = resolve_historical_universe(
                    signal_date,
                    historical_universe_name(universe_name),
                    filter_config=UniverseFilterConfig(require_market_history=False),
                )
                if universe_limit:
                    point_in_time.stocks = point_in_time.stocks[: max(0, universe_limit)]
                historical_by_date[signal_date] = point_in_time
                for item in point_in_time.stocks:
                    union[item["code"]] = item
        except HistoricalUniverseError as exc:
            universe = UniverseResult(
                universe=universe_name,
                stocks=[],
                source="历史股票池不可用",
                as_of_date=end,
                is_historical_membership=True,
                warning=str(exc),
            )
            return empty_result(universe, top_n, holding_days, rebalance_days, start, end, initial_capital, manager, [str(exc)])
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
            as_of_date=end,
            is_historical_membership=True,
            warning="严格历史股票池：每个调仓日只使用当时的沪深300/中证500成员。",
        )
    else:
        universe = load_universe(universe_name, limit=universe_limit)
        calendar = []
    data_start = warmup_start_date(start, calendar_days=420) if historical_by_date else start
    histories = manager.load_histories(
        universe.codes,
        data_start,
        end,
        min_rows=max(90, holding_days + 70),
        allow_network=allow_market_network,
    )
    notes = [
        "普通财经新闻、东方财富/同花顺/新浪/财联社内容不参与任何回测交易评分。",
        "选股因子只使用截至信号日收盘已经可见的历史行情，入场日为下一交易日，避免价格未来函数。",
        universe.warning,
    ]
    if not histories:
        return empty_result(universe, top_n, holding_days, rebalance_days, start, end, initial_capital, manager, notes)

    if not calendar:
        calendar = build_common_calendar(histories)
    if len(calendar) < holding_days + 70:
        notes.append("可用交易日过少，无法形成有效调仓样本。")
        return empty_result(universe, top_n, holding_days, rebalance_days, start, end, initial_capital, manager, notes)

    if benchmark.warning:
        notes.append(benchmark.warning)
    code_industries = {
        item.stock_code: item.industry
        for item in universe.stocks
    }
    event_stats = EventResearchStats()
    fundamental_totals = {"requested": 0, "available": 0, "missing": 0, "revised": 0, "future_records": 0}
    fundamental_validation_passed = True
    execution_stats = {"risk_rejected": 0, "missing_execution_price": 0}
    rebalance_audit: list[dict[str, Any]] = []
    equity = initial_capital
    trades: list[BacktestTrade] = []
    daily_rows: list[dict[str, Any]] = []
    min_history = 60
    last_exit_index = min_history
    for signal_index in range(min_history, len(calendar) - holding_days - 1, max(1, rebalance_days)):
        if signal_index < last_exit_index:
            continue
        signal_date = calendar[signal_index]
        entry_index = signal_index + 1
        exit_index = min(entry_index + holding_days, len(calendar) - 1)
        entry_date = calendar[entry_index]
        exit_date = calendar[exit_index]
        market_score = calculate_market_regime_score_from_history(benchmark.data, signal_date)
        audit_row = {
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "market_score": round(float(market_score), 4),
            "market_regime_gate": market_regime_gate,
            "eligible_universe_size": 0,
            "factor_score_count": 0,
            "selected_count": 0,
            "risk_rejected_count": 0,
            "missing_execution_price_count": 0,
            "reason": "other",
        }
        if market_regime_gate and market_score < market_min_score:
            notes.append(f"{signal_date} 市场环境分{market_score:.1f}低于{market_min_score:.1f}，本期保持现金。")
            audit_row["reason"] = "market_regime_block"
            if collect_rebalance_audit:
                rebalance_audit.append(audit_row)
            continue
        eligible_histories = histories
        if historical_by_date:
            try:
                point_in_time = resolve_historical_universe(
                    signal_date,
                    historical_universe_name(universe_name),
                    histories=histories,
                    filter_config=UniverseFilterConfig(require_market_history=True),
                )
            except HistoricalUniverseError as exc:
                notes.append(f"{signal_date} 历史股票池失败：{exc}，本期保持现金。")
                audit_row["reason"] = "insufficient_universe"
                if collect_rebalance_audit:
                    rebalance_audit.append(audit_row)
                continue
            if universe_limit:
                point_in_time.stocks = point_in_time.stocks[: max(0, universe_limit)]
            eligible_histories = {code: histories[code] for code in point_in_time.codes if code in histories}
            audit_row["eligible_universe_size"] = len(eligible_histories)
            if not eligible_histories:
                notes.append(f"{signal_date} 历史股票池经过可交易过滤后为空，本期保持现金。")
                audit_row["reason"] = "insufficient_universe"
                if collect_rebalance_audit:
                    rebalance_audit.append(audit_row)
                continue
        else:
            audit_row["eligible_universe_size"] = len(eligible_histories)
        fundamental_score_map: dict[str, float] | None = None
        fundamental_batch = None
        require_fundamentals = False
        if historical_by_date:
            try:
                fundamental_batch = get_fundamental_scores(
                    list(eligible_histories),
                    signal_date,
                    price_histories=None,
                    allow_network=allow_fundamental_network,
                    strict=True,
                )
            except FundamentalFutureDataError as exc:
                fundamental_validation_passed = False
                fundamental_totals["future_records"] += 1
                notes.append(f"{signal_date} 基本面未来数据检查失败：{exc}。正式回测已停止。")
                trades = []
                daily_rows = []
                equity = initial_capital
                break
            fundamental_totals["requested"] += len(eligible_histories)
            fundamental_totals["available"] += len(fundamental_batch.scores)
            fundamental_totals["missing"] += len(fundamental_batch.missing_codes)
            fundamental_totals["revised"] += len(fundamental_batch.revised_codes)
            fundamental_totals["future_records"] += fundamental_batch.future_records
            if not fundamental_batch.scores:
                notes.append(f"{signal_date} 无可用历史时点基本面数据，本期保持现金。")
                audit_row["reason"] = "missing_fundamental_data"
                if collect_rebalance_audit:
                    rebalance_audit.append(audit_row)
                continue
            fundamental_score_map = fundamental_batch.scores
            require_fundamentals = True
        event_scores, event_stats = event_scores_as_of(
            signal_date,
            list(eligible_histories),
            code_industries=code_industries,
        )
        scores = calculate_factor_scores_for_universe(
            eligible_histories,
            signal_date,
            market_regime_score=market_score,
            fundamental_scores=fundamental_score_map,
            require_fundamentals=require_fundamentals,
            event_scores=event_scores,
            event_boost_weight=event_boost_weight,
        )
        audit_row["factor_score_count"] = len(scores)
        if not scores:
            audit_row["reason"] = "insufficient_factor_score"
            if collect_rebalance_audit:
                rebalance_audit.append(audit_row)
            continue
        selection_stats: dict[str, int] = {"risk_rejected": 0, "missing_execution_price": 0}
        selected = select_tradeable(
            scores, eligible_histories, signal_date, entry_date, exit_date, top_n,
            stats=selection_stats,
        )
        for key, value in selection_stats.items():
            execution_stats[key] = execution_stats.get(key, 0) + value
        audit_row["risk_rejected_count"] = selection_stats["risk_rejected"]
        audit_row["missing_execution_price_count"] = selection_stats["missing_execution_price"]
        audit_row["selected_count"] = len(selected)
        if not selected:
            if selection_stats["risk_rejected"] >= len(scores):
                audit_row["reason"] = "risk_filter_block"
            elif selection_stats["missing_execution_price"] > 0:
                audit_row["reason"] = "missing_market_data"
            else:
                audit_row["reason"] = "insufficient_factor_score"
            if collect_rebalance_audit:
                rebalance_audit.append(audit_row)
            continue

        entry_equity = equity
        selected_codes = [item.stock_code for item in selected]
        entry_prices = {code: close_on_or_before(histories[code], entry_date) for code in selected_codes}
        exit_prices = {code: close_on_or_before(histories[code], exit_date) for code in selected_codes}
        returns = {
            code: round((exit_prices[code] / entry_prices[code] - 1) * 100, 4)
            for code in selected_codes
            if entry_prices.get(code) not in (None, 0) and exit_prices.get(code) is not None
        }
        if not returns:
            audit_row["reason"] = "missing_market_data"
            if collect_rebalance_audit:
                rebalance_audit.append(audit_row)
            continue
        gross_return = sum(returns.values()) / len(returns)
        net_return = gross_return - fee_rate * 2 * 100

        for date in calendar[entry_index : exit_index + 1]:
            day_return = daily_portfolio_return(histories, selected_codes, entry_prices, date)
            daily_equity = entry_equity * (1 + day_return / 100)
            daily_rows.append(
                {
                    "date": date,
                    "equity": round(daily_equity, 2),
                    "return_pct": round((daily_equity / initial_capital - 1) * 100, 4),
                    "holding_codes": ",".join(selected_codes),
                }
            )

        equity = entry_equity * (1 + net_return / 100)
        trades.append(
            BacktestTrade(
                signal_date=signal_date,
                entry_date=entry_date,
                exit_date=exit_date,
                codes=selected_codes,
                factor_scores={item.stock_code: item for item in selected},
                returns=returns,
                portfolio_return_pct=round(net_return, 4),
                entry_equity=round(entry_equity, 2),
                exit_equity=round(equity, 2),
                universe_size=len(eligible_histories),
                fundamental_periods={
                    code: fundamental_batch.records[code].report_period
                    for code in selected_codes
                    if fundamental_batch is not None and code in fundamental_batch.records
                },
                fundamental_disclosure_dates={
                    code: fundamental_batch.records[code].disclosure_date
                    for code in selected_codes
                    if fundamental_batch is not None and code in fundamental_batch.records
                },
            )
        )
        audit_row["reason"] = "normal_holding"
        if collect_rebalance_audit:
            rebalance_audit.append(audit_row)
        last_exit_index = exit_index

    if not trades:
        notes.append("没有形成有效交易，可能是股票池过小、行情缺失或历史长度不足。")
    curve = build_daily_curve(daily_rows, initial_capital)
    benchmark_stats = compare_strategy_to_benchmark(curve, benchmark.data)
    if event_boost_weight <= 0:
        notes.append("模型A不使用政策或公告增强，只测试市场、动量、资金、基本面和技术五类量化评分。")
    elif event_stats.total_rows == 0:
        notes.append("模型B事件模板尚无历史政策/公告记录，因此事件增强为中性，模型B应与模型A相同；这不是事件因子有效性的证据。")
    else:
        notes.append(f"模型B使用最近20日内的历史政策/公告记录做最多{event_boost_weight * 100:.0f}%辅助调整；事件不能单独产生选股或买入。")
    returns_list = [item.portfolio_return_pct for item in trades]
    total_return = (equity / initial_capital - 1) * 100
    fundamental_mode = (
        "historical_point_in_time"
        if historical_by_date and fundamental_validation_passed and fundamental_totals["requested"] > 0
        else "incomplete"
    )
    backtest_integrity = determine_backtest_integrity(
        "historical_point_in_time" if universe.is_historical_membership else "current_or_manual",
        fundamental_mode,
        fundamental_validation_passed and fundamental_totals["future_records"] == 0,
    )
    if not trades:
        backtest_integrity = "incomplete"
    return AShareBacktestResult(
        universe=universe,
        requested_stock_count=len(universe.codes),
        available_stock_count=len(histories),
        top_n=top_n,
        holding_days=holding_days,
        rebalance_days=rebalance_days,
        start_date=start,
        end_date=end,
        initial_capital=initial_capital,
        final_equity=round(equity, 2),
        total_return_pct=round(total_return, 4),
        annualized_return_pct=calculate_annualized(total_return, len(curve)),
        win_rate=round(sum(1 for value in returns_list if value > 0) / len(returns_list), 4) if returns_list else None,
        max_drawdown_pct=max_drawdown(curve["equity"].tolist()) if not curve.empty else None,
        sharpe_ratio=calculate_sharpe(returns_list),
        benchmark=benchmark_stats,
        trades=trades,
        daily_curve=curve,
        notes=notes,
        data_stats=manager.stats.__dict__,
        model_label=model_label,
        event_boost_weight=event_boost_weight,
        event_stats=event_stats,
        fundamental_mode=fundamental_mode,
        backtest_integrity=backtest_integrity,
        fundamental_stats=fundamental_totals,
        execution_stats=execution_stats,
        rebalance_audit=rebalance_audit,
    )


def generate_a_share_backtest_report(
    output_file: Path = DEFAULT_OUTPUT,
    **kwargs: Any,
) -> AShareBacktestResult:
    base_result = run_a_share_backtest(model_label="模型A：基础量化", event_boost_weight=0.0, **kwargs)
    enhanced_result = run_a_share_backtest(model_label="模型B：基础量化 + 政策公告增强", event_boost_weight=0.05, **kwargs)
    base_result.enhanced_result = enhanced_result
    output_file.write_text(build_report(base_result), encoding="utf-8")
    save_result_tables(base_result, output_file, suffix="model_a")
    save_result_tables(enhanced_result, output_file, suffix="model_b")
    # Preserve the former filenames as model A outputs for existing users.
    save_result_tables(base_result, output_file)
    return base_result


def save_result_tables(
    result: AShareBacktestResult,
    output_file: Path,
    suffix: str = "",
    backtest_version: str = "legacy",
) -> None:
    suffix_text = f"_{suffix}" if suffix else ""
    curve_file = output_file.with_name(f"backtest_equity_curve{suffix_text}.csv")
    trades_file = output_file.with_name(f"backtest_trades{suffix_text}.csv")
    universe_mode = "historical_point_in_time" if result.universe.is_historical_membership else "current_or_manual"
    curve = result.daily_curve.copy()
    for column in ("date", "equity", "return_pct", "holding_codes"):
        if column not in curve.columns:
            curve[column] = pd.Series(dtype=object)
    curve["universe_mode"] = universe_mode
    curve["universe_source"] = result.universe.source
    curve["fundamental_mode"] = result.fundamental_mode
    curve["backtest_integrity"] = result.backtest_integrity
    curve["integrity_status"] = result.integrity_status
    curve["backtest_version"] = backtest_version
    curve.to_csv(curve_file, index=False, encoding="utf-8-sig")
    trade_rows: list[dict[str, Any]] = []
    for trade in result.trades:
        for code in trade.codes:
            score = trade.factor_scores[code]
            trade_rows.append(
                {
                    "signal_date": trade.signal_date,
                    "entry_date": trade.entry_date,
                    "exit_date": trade.exit_date,
                    "stock_code": code,
                    "total_score": score.total_score,
                    "momentum_score": score.momentum_score,
                    "money_flow_score": score.money_flow_score,
                    "technical_score": score.technical_score,
                    "fundamental_score": score.fundamental_score,
                    "market_regime_score": score.market_regime_score,
                    "event_score": score.event_score,
                    "stock_return_pct": trade.returns.get(code),
                    "portfolio_return_pct": trade.portfolio_return_pct,
                    "entry_equity": trade.entry_equity,
                    "exit_equity": trade.exit_equity,
                    "universe_mode": universe_mode,
                    "universe_source": result.universe.source,
                    "fundamental_mode": result.fundamental_mode,
                    "backtest_integrity": result.backtest_integrity,
                    "integrity_status": result.integrity_status,
                    "backtest_version": backtest_version,
                    "universe_size": trade.universe_size,
                    "fundamental_data_date": trade.fundamental_periods.get(code, ""),
                    "fundamental_disclosure_date": trade.fundamental_disclosure_dates.get(code, ""),
                }
            )
    trade_columns = [
        "signal_date", "entry_date", "exit_date", "stock_code", "total_score",
        "momentum_score", "money_flow_score", "technical_score", "fundamental_score",
        "market_regime_score", "event_score", "stock_return_pct", "portfolio_return_pct",
        "entry_equity", "exit_equity", "universe_mode", "universe_source",
        "fundamental_mode", "backtest_integrity", "integrity_status",
        "backtest_version", "universe_size", "fundamental_data_date",
        "fundamental_disclosure_date",
    ]
    pd.DataFrame(trade_rows, columns=trade_columns).to_csv(trades_file, index=False, encoding="utf-8-sig")


def select_tradeable(
    scores: list[StockFactorScore],
    histories: dict[str, pd.DataFrame],
    signal_date: str,
    entry_date: str,
    exit_date: str,
    top_n: int,
    stats: dict[str, int] | None = None,
) -> list[StockFactorScore]:
    selected: list[StockFactorScore] = []
    for score in scores:
        history = histories.get(score.stock_code)
        if history is None or history.empty:
            continue
        risk = evaluate_research_risk(history, signal_date)
        if not risk.allowed:
            if stats is not None:
                stats["risk_rejected"] = stats.get("risk_rejected", 0) + 1
            continue
        if close_on_or_before(history, entry_date) is None or close_on_or_before(history, exit_date) is None:
            if stats is not None:
                stats["missing_execution_price"] = stats.get("missing_execution_price", 0) + 1
            continue
        selected.append(score)
        if len(selected) >= top_n:
            break
    return selected


def uses_historical_index_universe(universe_name: str) -> bool:
    return normalize_universe_name(universe_name) in {"hs300", "csi500", "hs300_csi500"}


def historical_universe_name(universe_name: str) -> str:
    normalized = normalize_universe_name(universe_name)
    return {
        "hs300": "CSI300",
        "csi500": "CSI500",
        "hs300_csi500": "CSI300_CSI500",
    }[normalized]


def backtest_signal_dates(calendar: list[str], holding_days: int, rebalance_days: int) -> list[str]:
    min_history = 60
    return [
        str(calendar[index])
        for index in range(min_history, len(calendar) - holding_days - 1, max(1, rebalance_days))
    ]


def warmup_start_date(value: str, calendar_days: int = 420) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value).replace("-", "")
    return (parsed - pd.Timedelta(days=calendar_days)).strftime("%Y%m%d")


def build_common_calendar(histories: dict[str, pd.DataFrame]) -> list[str]:
    longest = max(histories.values(), key=len)
    return [str(value) for value in longest["date"].tolist()]


def close_on_or_before(history: pd.DataFrame, date: str) -> float | None:
    data = history[history["date"] <= date]
    if data.empty:
        return None
    value = data.iloc[-1]["close"]
    try:
        return float(value)
    except Exception:
        return None


def daily_portfolio_return(histories: dict[str, pd.DataFrame], codes: list[str], entry_prices: dict[str, float | None], date: str) -> float:
    returns: list[float] = []
    for code in codes:
        entry = entry_prices.get(code)
        current = close_on_or_before(histories[code], date)
        if entry in (None, 0) or current is None:
            continue
        returns.append((current / entry - 1) * 100)
    return sum(returns) / len(returns) if returns else 0.0


def build_daily_curve(rows: list[dict[str, Any]], initial_capital: float) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date", "equity", "return_pct", "holding_codes"])
    df = pd.DataFrame(rows)
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if df.empty:
        return df
    df.loc[0, "equity"] = max(float(df.loc[0, "equity"]), initial_capital * 0.5)
    df["return_pct"] = (df["equity"].astype(float) / initial_capital - 1) * 100
    df["return_pct"] = df["return_pct"].round(4)
    return df


def build_report(result: AShareBacktestResult) -> str:
    enhanced = result.enhanced_result
    if enhanced is None:
        comparison_lines = ["- 本次仅运行模型A。", ""]
    else:
        comparison_lines = [
            "| 指标 | 模型A：基础量化 | 模型B：政策公告增强 | B-A变化 |",
            "|---|---:|---:|---:|",
            f"| 累计收益 | {format_pct(result.total_return_pct)} | {format_pct(enhanced.total_return_pct)} | {format_pct(enhanced.total_return_pct - result.total_return_pct)} |",
            f"| 年化收益 | {format_pct(result.annualized_return_pct)} | {format_pct(enhanced.annualized_return_pct)} | {format_pct(delta(enhanced.annualized_return_pct, result.annualized_return_pct))} |",
            f"| 最大回撤 | {format_pct(result.max_drawdown_pct)} | {format_pct(enhanced.max_drawdown_pct)} | {format_pct(delta(enhanced.max_drawdown_pct, result.max_drawdown_pct))} |",
            f"| 夏普比率 | {format_number(result.sharpe_ratio)} | {format_number(enhanced.sharpe_ratio)} | {format_number(delta(enhanced.sharpe_ratio, result.sharpe_ratio))} |",
            f"| 胜率 | {format_ratio(result.win_rate)} | {format_ratio(enhanced.win_rate)} | {format_ratio(delta(enhanced.win_rate, result.win_rate))} |",
            f"| 事件数据行数 | 0 | {enhanced.event_stats.total_rows} | - |",
            "",
            "模型B只有在事件数据文件中存在发布时间、股票代码或行业、评分等历史记录时才会变化。若事件数据行数为0，模型B与模型A相同是正确的防伪造行为。",
            "",
        ]
    lines = [
        "# A股多股票池多因子组合回测报告",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 这份报告在测什么",
        "",
        "本报告测试的是：从沪深300/中证500等动态股票池中批量获取股票，按调仓日横向计算市场环境、动量、资金、基本面和技术评分，再等权持有 TopN 股票。",
        "",
        "普通财经新闻仅用于研究记录。模型B只允许历史政策/公告数据做最多5%的辅助调整，不能直接触发交易。",
        "",
        "## 策略参数",
        "",
        f"- 股票池：{result.universe.universe}",
        f"- 股票池来源：{result.universe.source}",
        f"- 股票池日期：{result.universe.as_of_date}",
        f"- 请求股票数：{result.requested_stock_count}",
        f"- 可用行情股票数：{result.available_stock_count}",
        f"- TopN持仓：{result.top_n}",
        f"- 持有周期：{result.holding_days} 个交易日",
        f"- 调仓间隔：{result.rebalance_days} 个交易日",
        f"- 回测区间：{result.start_date} 至 {result.end_date}",
        "- 模型A权重：市场环境 20% + 动量 25% + 资金 20% + 基本面 25% + 技术 10%",
        "- 基本面数据：按每个调仓日选择 disclosure_date 不晚于信号日的最后一份已披露财报；缺失股票直接剔除，不使用中性50分。",
        "- 模型B：模型A + 政策/公告事件最多 5% 辅助调整",
        "- 入场规则：信号日只使用当日及以前数据计算分数，下一交易日入场。",
        "",
        "## 模型A核心结果",
        "",
        f"- 调仓次数：{len(result.trades)}",
        f"- 初始资金：{round(result.initial_capital, 2)}",
        f"- 期末权益：{result.final_equity}",
        f"- 累计收益：{format_pct(result.total_return_pct)}",
        f"- 年化收益：{format_pct(result.annualized_return_pct)}",
        f"- 胜率：{format_ratio(result.win_rate)}",
        f"- 最大回撤：{format_pct(result.max_drawdown_pct)}",
        f"- 夏普比率：{format_number(result.sharpe_ratio)}",
        "",
        "## 沪深300基准对比",
        "",
        f"- 沪深300累计收益：{format_pct(result.benchmark.get('benchmark_total_return_pct'))}",
        f"- 策略超额收益：{format_pct(result.benchmark.get('excess_return_pct'))}",
        f"- 年化收益差：{format_pct(result.benchmark.get('annualized_return_diff_pct'))}",
        f"- 沪深300最大回撤：{format_pct(result.benchmark.get('benchmark_max_drawdown_pct'))}",
        f"- 最大回撤差：{format_pct(result.benchmark.get('max_drawdown_diff_pct'))}",
        "",
        "## 模型A vs 模型B：事件增量检验",
        "",
        *comparison_lines,
        "## 数据管理统计",
        "",
        f"- 请求行情：{result.data_stats.get('requested', 0)}",
        f"- 成功加载：{result.data_stats.get('loaded', 0)}",
        f"- 缓存命中：{result.data_stats.get('cache_hits', 0)}",
        f"- 新抓取：{result.data_stats.get('fetched', 0)}",
        f"- 失败：{result.data_stats.get('failed', 0)}",
        f"- 基本面模式：{result.fundamental_mode}",
        f"- 回测完整性：{result.backtest_integrity}",
        f"- 基本面请求/可用/缺失：{result.fundamental_stats.get('requested', 0)} / {result.fundamental_stats.get('available', 0)} / {result.fundamental_stats.get('missing', 0)}",
        f"- 使用修订版记录次数：{result.fundamental_stats.get('revised', 0)}（报告中单独披露修订偏差）",
        f"- 未来基本面记录：{result.fundamental_stats.get('future_records', 0)}",
        "",
        "## 防未来函数说明",
        "",
        "- 价格因子只用信号日及以前的历史行情。",
        "- 入场发生在信号日之后的下一个交易日。",
        "- 沪深300/中证500股票池按每个调仓日获取当时的历史成分，不使用当前成分股回测过去。",
        "- 成分股快照日期、上市日期和退市日期均不得晚于信号日；检测结果写入 `future_data_check.log`。",
        "- 财报必须满足 disclosure_date <= 信号日；同比基数也必须在信号日前已披露。",
        "- PE/PB使用信号日收盘价、当时已披露利润/权益和当时财报股本计算，不读取当前PE/PB。",
        "- 基本面审计结果写入 `fundamental_future_data_check.log`。",
        "",
        "## 重要备注",
        "",
    ]
    for note in result.notes:
        if note:
            lines.append(f"- {note}")
    lines.extend(
        [
            "",
        "## 每日净值",
        "",
        "- 模型A净值：`backtest_equity_curve_model_a.csv`；模型B净值：`backtest_equity_curve_model_b.csv`。",
        "- 模型A交易：`backtest_trades_model_a.csv`；模型B交易：`backtest_trades_model_b.csv`。",
        "",
        "| 日期 | 策略权益 | 累计收益 | 持仓 |",
            "|---|---:|---:|---|",
        ]
    )
    for _, row in result.daily_curve.tail(120).iterrows():
        lines.append(f"| {row['date']} | {round(float(row['equity']), 2)} | {format_pct(float(row['return_pct']))} | {row['holding_codes']} |")
    lines.extend(
        [
            "",
            "## 交易记录",
            "",
            "| 信号日 | 入场日 | 退出日 | 股票 | 综合分 | 动量 | 资金 | 技术 | 基本面 | 市场 | 事件 | 个股收益 | 组合收益 | 退出权益 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for trade in result.trades:
        lines.append(
            "| "
            + " | ".join(
                [
                    trade.signal_date,
                    trade.entry_date,
                    trade.exit_date,
                    "<br>".join(trade.codes),
                    "<br>".join(f"{code}:{trade.factor_scores[code].total_score}" for code in trade.codes),
                    "<br>".join(f"{code}:{trade.factor_scores[code].momentum_score}" for code in trade.codes),
                    "<br>".join(f"{code}:{trade.factor_scores[code].money_flow_score}" for code in trade.codes),
                    "<br>".join(f"{code}:{trade.factor_scores[code].technical_score}" for code in trade.codes),
                    "<br>".join(f"{code}:{trade.factor_scores[code].fundamental_score}" for code in trade.codes),
                    "<br>".join(f"{code}:{trade.factor_scores[code].market_regime_score}" for code in trade.codes),
                    "<br>".join(f"{code}:{trade.factor_scores[code].event_score}" for code in trade.codes),
                    "<br>".join(f"{code}:{format_pct(trade.returns.get(code))}" for code in trade.codes),
                    format_pct(trade.portfolio_return_pct),
                    str(trade.exit_equity),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 接入历史成分股快照，消除当前成分股带来的幸存者偏差。",
            "2. 增加行业约束、单票上限、止损和最大回撤降仓。",
            "3. 增加更长历史区间和更多参数组合测试。",
            "4. 累积可审计的历史政策/公告事件数据后，再判断模型B是否有稳定增量价值。",
            "5. 回测稳定前继续不接入真实交易。",
            "",
        ]
    )
    return "\n".join(lines)


def empty_result(
    universe: UniverseResult,
    top_n: int,
    holding_days: int,
    rebalance_days: int,
    start_date: str,
    end_date: str,
    initial_capital: float,
    manager: MarketDataManager,
    notes: list[str],
) -> AShareBacktestResult:
    return AShareBacktestResult(
        universe=universe,
        requested_stock_count=len(universe.codes),
        available_stock_count=0,
        top_n=top_n,
        holding_days=holding_days,
        rebalance_days=rebalance_days,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_equity=initial_capital,
        total_return_pct=0.0,
        annualized_return_pct=None,
        win_rate=None,
        max_drawdown_pct=None,
        sharpe_ratio=None,
        notes=notes,
        data_stats=manager.stats.__dict__,
        model_label="模型A：基础量化",
    )


def calculate_annualized(total_return_pct: float, trading_days: int) -> float | None:
    if trading_days <= 0:
        return None
    years = max(1 / 252, trading_days / 252)
    return round(((1 + total_return_pct / 100) ** (1 / years) - 1) * 100, 4)


def max_drawdown(values: list[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            max_dd = min(max_dd, (value - peak) / peak * 100)
    return round(max_dd, 4)


def calculate_sharpe(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    avg = sum(returns) / len(returns)
    variance = sum((value - avg) ** 2 for value in returns) / (len(returns) - 1)
    std = variance**0.5
    if std == 0:
        return None
    return round(avg / std, 4)


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y%m%d")
    return None


def format_pct(value: float | None) -> str:
    return "-" if value is None else f"{round(float(value), 4)}%"


def format_ratio(value: float | None) -> str:
    return "-" if value is None else f"{round(float(value) * 100, 2)}%"


def format_number(value: float | None) -> str:
    return "-" if value is None else str(round(float(value), 4))


def delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return round(float(value) - float(base), 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A股多股票池组合回测")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE, help="股票池: hs300/csi500/hs300_csi500/all_a/manual")
    parser.add_argument("--universe-limit", type=int, help="限制股票数量，用于快速测试")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="报告输出路径")
    parser.add_argument("--top-n", type=int, default=20, help="每次持仓数量")
    parser.add_argument("--holding-days", type=int, default=20, help="持有周期")
    parser.add_argument("--rebalance-days", type=int, default=20, help="调仓间隔")
    parser.add_argument("--history-days", type=int, default=720, help="历史行情长度")
    parser.add_argument("--start-date", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--fee-rate", type=float, default=0.0015, help="单边费率")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = generate_a_share_backtest_report(
        output_file=Path(args.output),
        universe_name=args.universe,
        universe_limit=args.universe_limit,
        top_n=args.top_n,
        holding_days=args.holding_days,
        rebalance_days=args.rebalance_days,
        history_days=args.history_days,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
    )
    print(f"A股多股票池回测报告已生成: {args.output}")
    print(f"股票数: {result.available_stock_count}/{result.requested_stock_count}, 调仓次数: {len(result.trades)}, 总收益率: {result.total_return_pct}%")
