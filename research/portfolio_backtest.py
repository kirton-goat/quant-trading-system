from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark
from factor_engine import fetch_history, safe_float
from market_data_manager import MarketDataManager, date_n_days_ago
from research.fundamentals.fundamental_validation import (
    FundamentalFutureDataError,
    determine_backtest_integrity,
    public_integrity_status,
)
from research.fundamentals.point_in_time_fundamentals import get_fundamental_scores
from research.factor_test import calculate_momentum_score, calculate_money_flow_score, normalize_history, parse_codes
from research.universe.historical_universe import HistoricalUniverseError, resolve_historical_universe
from research.universe.stock_filter import UniverseFilterConfig
from stock_pool import DEFAULT_STOCK_POOL_FILE, StockPoolItem, load_stock_pool, load_stock_pool_items


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = BASE_DIR / "portfolio_backtest_report.md"


@dataclass
class PortfolioTrade:
    date: str
    exit_date: str
    codes: list[str]
    scores: dict[str, float]
    returns: dict[str, float]
    portfolio_return: float
    equity: float
    momentum_scores: dict[str, float] = field(default_factory=dict)
    money_flow_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioBacktestResult:
    stock_count: int
    trade_count: int
    initial_capital: float
    final_equity: float
    total_return_pct: float
    win_rate: float | None
    average_return_pct: float | None
    max_drawdown_pct: float | None
    sharpe_ratio: float | None
    trades: list[PortfolioTrade] = field(default_factory=list)
    skipped_dates: int = 0
    note: str = ""
    universe_mode: str = "manual_test_pool"
    universe_source: str = "stock_pool.csv"
    filter_summary: dict[str, int] = field(default_factory=dict)
    fundamental_mode: str = "incomplete"
    backtest_integrity: str = "incomplete"

    @property
    def integrity_status(self) -> str:
        return public_integrity_status(self.backtest_integrity)


def run_portfolio_backtest(
    stock_codes: list[str] | None = None,
    top_n: int = 5,
    holding_days: int = 20,
    rebalance_days: int = 20,
    history_days: int = 360,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0015,
    universe_type: str = "CSI300_CSI500",
    universe_limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_average_amount: float = 20_000_000.0,
) -> PortfolioBacktestResult:
    if not stock_codes:
        return run_historical_portfolio_backtest(
            universe_type=universe_type,
            universe_limit=universe_limit,
            top_n=top_n,
            holding_days=holding_days,
            rebalance_days=rebalance_days,
            history_days=history_days,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            start_date=start_date,
            end_date=end_date,
            min_average_amount=min_average_amount,
        )
    histories = load_histories(stock_codes, history_days=history_days)
    if not histories:
        return empty_result(stock_codes, initial_capital, "没有可用历史行情，可能是网络节点或股票池问题")

    dates = build_rebalance_dates(histories, min_history=60, holding_days=holding_days, rebalance_days=rebalance_days)
    equity = initial_capital
    trades: list[PortfolioTrade] = []
    skipped = 0
    for date in dates:
        candidates = score_candidates(histories, date, holding_days=holding_days)
        if not candidates:
            skipped += 1
            continue
        selected = sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_n]
        valid_returns = {item["code"]: item["forward_return"] for item in selected if item.get("forward_return") is not None}
        if not valid_returns:
            skipped += 1
            continue
        gross_return = sum(valid_returns.values()) / len(valid_returns)
        net_return = gross_return - fee_rate * 2 * 100
        equity *= 1 + net_return / 100
        exit_date = selected[0]["exit_date"]
        trades.append(
            PortfolioTrade(
                date=str(date),
                exit_date=str(exit_date),
                codes=list(valid_returns.keys()),
                scores={item["code"]: round(item["score"], 2) for item in selected},
                momentum_scores={item["code"]: round(item["momentum"], 2) for item in selected},
                money_flow_scores={item["code"]: round(item["money_flow"], 2) for item in selected},
                returns={code: round(value, 4) for code, value in valid_returns.items()},
                portfolio_return=round(net_return, 4),
                equity=round(equity, 2),
            )
        )

    if not trades:
        return empty_result(stock_codes, initial_capital, "没有形成有效调仓样本，可能历史长度不足或样本太少")

    returns = [trade.portfolio_return for trade in trades]
    total_return = (equity - initial_capital) / initial_capital * 100
    return PortfolioBacktestResult(
        stock_count=len(histories),
        trade_count=len(trades),
        initial_capital=initial_capital,
        final_equity=round(equity, 2),
        total_return_pct=round(total_return, 4),
        win_rate=round(sum(1 for value in returns if value > 0) / len(returns), 4),
        average_return_pct=round(sum(returns) / len(returns), 4),
        max_drawdown_pct=calculate_max_drawdown([trade.equity for trade in trades], initial_capital),
        sharpe_ratio=calculate_sharpe(returns),
        trades=trades,
        skipped_dates=skipped,
        note="基础组合回测：仅使用动量+资金评分，不含AI事件因子，不含真实交易执行",
    )


def run_historical_portfolio_backtest(
    universe_type: str,
    universe_limit: int | None,
    top_n: int,
    holding_days: int,
    rebalance_days: int,
    history_days: int,
    initial_capital: float,
    fee_rate: float,
    start_date: str | None,
    end_date: str | None,
    min_average_amount: float,
) -> PortfolioBacktestResult:
    end = normalize_cli_date(end_date) or dt.date.today().isoformat()
    start = normalize_cli_date(start_date) or normalize_cli_date(date_n_days_ago(history_days))
    benchmark = load_benchmark(start_date=start or "", end_date=end)
    calendar = benchmark.data["date"].astype(str).tolist() if not benchmark.data.empty else []
    dates = build_dates_from_calendar(calendar, min_history=60, holding_days=holding_days, rebalance_days=rebalance_days)
    if not dates:
        return empty_result([], initial_capital, "缺少足够的基准交易日，无法构建历史调仓日")

    preliminary: dict[str, Any] = {}
    union_codes: list[str] = []
    try:
        for signal_date in dates:
            universe = resolve_historical_universe(
                signal_date,
                universe_type,
                filter_config=UniverseFilterConfig(require_market_history=False),
            )
            if universe_limit:
                universe.stocks = universe.stocks[: max(0, universe_limit)]
            preliminary[signal_date] = universe
            for code in universe.codes:
                if code not in union_codes:
                    union_codes.append(code)
    except HistoricalUniverseError as exc:
        return empty_result([], initial_capital, f"历史股票池不可用：{exc}")

    manager = MarketDataManager()
    data_start = warmup_start_date(start, calendar_days=420)
    standard_histories = manager.load_histories(union_codes, data_start, end.replace("-", ""), min_rows=90)
    histories = {code: to_factor_history(frame) for code, frame in standard_histories.items()}
    histories = {code: frame for code, frame in histories.items() if frame is not None and not frame.empty}

    equity = initial_capital
    trades: list[PortfolioTrade] = []
    skipped = 0
    totals = {
        "future_membership": 0,
        "future_listing": 0,
        "insufficient_listing": 0,
        "st": 0,
        "delisted": 0,
        "suspended": 0,
        "illiquid": 0,
        "missing_history": 0,
        "missing_fundamental_data": 0,
        "revised_fundamental_data": 0,
        "future_fundamental_data": 0,
    }
    sources: set[str] = set()
    filter_config = UniverseFilterConfig(min_average_amount=min_average_amount, require_market_history=True)
    for signal_date in dates:
        try:
            universe = resolve_historical_universe(
                signal_date,
                universe_type,
                histories=standard_histories,
                filter_config=filter_config,
            )
        except HistoricalUniverseError:
            skipped += 1
            continue
        if universe_limit:
            universe.stocks = universe.stocks[: max(0, universe_limit)]
        sources.add(universe.source)
        for key in (
            "future_membership",
            "future_listing",
            "insufficient_listing",
            "st",
            "delisted",
            "suspended",
            "illiquid",
            "missing_history",
        ):
            totals[key] += int(getattr(universe.filter_stats, key, 0))
        eligible_histories = {code: histories[code] for code in universe.codes if code in histories}
        try:
            fundamental_batch = get_fundamental_scores(
                list(eligible_histories),
                signal_date,
                price_histories=None,
                strict=True,
            )
        except FundamentalFutureDataError:
            result = empty_result(union_codes, initial_capital, "基本面未来数据检查失败，正式回测已阻断")
            result.universe_mode = "historical_point_in_time"
            result.universe_source = " + ".join(sorted(sources)) or universe_type
            result.filter_summary = totals
            result.filter_summary["future_fundamental_data"] += 1
            return result
        totals["missing_fundamental_data"] += len(fundamental_batch.missing_codes)
        totals["revised_fundamental_data"] += len(fundamental_batch.revised_codes)
        totals["future_fundamental_data"] += fundamental_batch.future_records
        eligible_histories = {
            code: history
            for code, history in eligible_histories.items()
            if code in fundamental_batch.scores
        }
        candidates = score_candidates(eligible_histories, signal_date, holding_days=holding_days)
        if not candidates:
            skipped += 1
            continue
        selected = sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_n]
        valid_returns = {item["code"]: item["forward_return"] for item in selected if item.get("forward_return") is not None}
        if not valid_returns:
            skipped += 1
            continue
        gross_return = sum(valid_returns.values()) / len(valid_returns)
        net_return = gross_return - fee_rate * 2 * 100
        equity *= 1 + net_return / 100
        trades.append(
            PortfolioTrade(
                date=signal_date,
                exit_date=str(selected[0]["exit_date"]),
                codes=list(valid_returns),
                scores={item["code"]: round(item["score"], 2) for item in selected},
                momentum_scores={item["code"]: round(item["momentum"], 2) for item in selected},
                money_flow_scores={item["code"]: round(item["money_flow"], 2) for item in selected},
                returns={code: round(value, 4) for code, value in valid_returns.items()},
                portfolio_return=round(net_return, 4),
                equity=round(equity, 2),
            )
        )
    if not trades:
        result = empty_result(union_codes, initial_capital, "历史股票池已加载，但没有形成有效调仓样本")
        result.universe_mode = "historical_point_in_time"
        result.universe_source = " + ".join(sorted(sources)) or universe_type
        result.filter_summary = totals
        result.fundamental_mode = "historical_point_in_time"
        result.backtest_integrity = determine_backtest_integrity(
            result.universe_mode,
            result.fundamental_mode,
            totals["future_fundamental_data"] == 0,
        )
        return result
    returns = [trade.portfolio_return for trade in trades]
    return PortfolioBacktestResult(
        stock_count=len(histories),
        trade_count=len(trades),
        initial_capital=initial_capital,
        final_equity=round(equity, 2),
        total_return_pct=round((equity / initial_capital - 1) * 100, 4),
        win_rate=round(sum(1 for value in returns if value > 0) / len(returns), 4),
        average_return_pct=round(sum(returns) / len(returns), 4),
        max_drawdown_pct=calculate_max_drawdown([trade.equity for trade in trades], initial_capital),
        sharpe_ratio=calculate_sharpe(returns),
        trades=trades,
        skipped_dates=skipped,
        note="严格历史股票池组合回测：每个调仓日按当时指数成分过滤，不含AI事件因子",
        universe_mode="historical_point_in_time",
        universe_source=" + ".join(sorted(sources)) or universe_type,
        filter_summary=totals,
        fundamental_mode="historical_point_in_time",
        backtest_integrity=determine_backtest_integrity(
            "historical_point_in_time",
            "historical_point_in_time",
            totals["future_fundamental_data"] == 0,
        ),
    )


def load_histories(stock_codes: list[str], history_days: int) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for code in stock_codes:
        df = fetch_history(code, days=history_days)
        if df is None or df.empty:
            continue
        normalized = normalize_history(df)
        if normalized is not None and len(normalized) >= 90:
            histories[code] = normalized
    return histories


def build_rebalance_dates(
    histories: dict[str, pd.DataFrame],
    min_history: int,
    holding_days: int,
    rebalance_days: int,
) -> list[Any]:
    base = max(histories.values(), key=len)
    if len(base) <= min_history + holding_days:
        return []
    dates: list[Any] = []
    end_index = len(base) - holding_days - 1
    for index in range(min_history, end_index + 1, max(1, rebalance_days)):
        dates.append(base.iloc[index]["日期"])
    return dates


def build_dates_from_calendar(
    calendar: list[str],
    min_history: int,
    holding_days: int,
    rebalance_days: int,
) -> list[str]:
    if len(calendar) <= min_history + holding_days:
        return []
    end_index = len(calendar) - holding_days - 1
    return [
        str(calendar[index])
        for index in range(min_history, end_index + 1, max(1, rebalance_days))
    ]


def to_factor_history(history: pd.DataFrame) -> pd.DataFrame | None:
    renamed = history.rename(
        columns={
            "date": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量",
            "amount": "成交额",
        }
    )
    return normalize_history(renamed)


def normalize_cli_date(value: str | None) -> str | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.strftime("%Y-%m-%d") if pd.notna(parsed) else None


def warmup_start_date(value: str, calendar_days: int = 420) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value).replace("-", "")
    return (parsed - pd.Timedelta(days=calendar_days)).strftime("%Y%m%d")


def score_candidates(histories: dict[str, pd.DataFrame], date: Any, holding_days: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for code, df in histories.items():
        matches = df.index[df["日期"].astype(str) == str(date)].tolist()
        if not matches:
            continue
        index = matches[0]
        if index < 60 or index + holding_days >= len(df):
            continue
        window = df.iloc[: index + 1]
        momentum = calculate_momentum_score(window)
        money_flow = calculate_money_flow_score(window)
        score = momentum * 0.55 + money_flow * 0.45
        entry = safe_float(df.iloc[index]["收盘"])
        exit_price = safe_float(df.iloc[index + holding_days]["收盘"])
        if entry in (None, 0) or exit_price is None:
            continue
        forward_return = (exit_price - entry) / entry * 100
        items.append(
            {
                "code": code,
                "score": score,
                "momentum": momentum,
                "money_flow": money_flow,
                "forward_return": forward_return,
                "exit_date": df.iloc[index + holding_days]["日期"],
            }
        )
    return items


def calculate_max_drawdown(equity_curve: list[float], initial_capital: float) -> float | None:
    if not equity_curve:
        return None
    peak = initial_capital
    max_drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak:
            max_drawdown = min(max_drawdown, (equity - peak) / peak * 100)
    return round(max_drawdown, 4)


def calculate_sharpe(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    avg = sum(returns) / len(returns)
    variance = sum((value - avg) ** 2 for value in returns) / (len(returns) - 1)
    std = variance**0.5
    if std == 0:
        return None
    return round(avg / std, 4)


def empty_result(stock_codes: list[str], initial_capital: float, note: str) -> PortfolioBacktestResult:
    return PortfolioBacktestResult(
        stock_count=len(stock_codes),
        trade_count=0,
        initial_capital=initial_capital,
        final_equity=initial_capital,
        total_return_pct=0.0,
        win_rate=None,
        average_return_pct=None,
        max_drawdown_pct=None,
        sharpe_ratio=None,
        note=note,
    )


def generate_portfolio_backtest_report(
    stock_codes: list[str] | None = None,
    output_file: Path = DEFAULT_OUTPUT,
    pool_file: Path = DEFAULT_STOCK_POOL_FILE,
    top_n: int = 5,
    holding_days: int = 20,
    rebalance_days: int = 20,
    history_days: int = 360,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0015,
    universe_type: str = "CSI300_CSI500",
    universe_limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_average_amount: float = 20_000_000.0,
) -> PortfolioBacktestResult:
    result = run_portfolio_backtest(
        stock_codes=stock_codes,
        top_n=top_n,
        holding_days=holding_days,
        rebalance_days=rebalance_days,
        history_days=history_days,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        universe_type=universe_type,
        universe_limit=universe_limit,
        start_date=start_date,
        end_date=end_date,
        min_average_amount=min_average_amount,
    )
    output_file.write_text(
        build_report(result, stock_codes or [], pool_file, top_n, holding_days, rebalance_days, history_days, fee_rate),
        encoding="utf-8",
    )
    return result


def build_report(
    result: PortfolioBacktestResult,
    stock_codes: list[str],
    pool_file: Path,
    top_n: int,
    holding_days: int,
    rebalance_days: int,
    history_days: int,
    fee_rate: float,
) -> str:
    pool_items = load_stock_pool_items(pool_file)
    metadata = build_metadata(pool_items)
    sectors = sorted({metadata.get(code, {}).get("sector", "") for code in stock_codes if metadata.get(code, {}).get("sector")})
    historical_mode = result.universe_mode == "historical_point_in_time"
    lines = [
        "# 基础组合回测报告",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 这份报告在测什么",
        "",
        "当前测试使用每个调仓日真实存在的沪深300/中证500历史成分股，用“动量 + 资金”两个传统因子做排序，每隔固定周期买入排名靠前的股票。" if historical_mode else "这是显式开发测试模式：在手工维护的代表性股票池中，用“动量 + 资金”两个传统因子做排序。",
        "",
        "当前回测的定位是策略框架验证，用来确认行情获取、因子计算、组合调仓、收益和回撤计算是否跑通；不能直接作为实盘结论。",
        "",
        "## 数据来源",
        "",
        f"- 股票池来源：{result.universe_source}。" if historical_mode else "- 股票池来源：`stock_pool.csv` 手工维护的研究样本池。",
        "- 历史股票池按每个调仓日单独解析，严格禁止当前成分股回填过去。" if historical_mode else "- 股票名称/行业/备注：来自 `stock_pool.csv`。",
        "- 历史行情：来自 `market.py` 的多源行情链路，包括本地缓存、东方财富、腾讯和备用日线接口。",
        "- 本地行情缓存目录：`data_cache/history/`。",
        "- 回测价格：使用前复权日线收盘价，当前不使用分钟线/Tick。",
        "- 历史基本面：东方财富按报告期三大报表，经 AKShare 获取并缓存在 `data_cache/fundamentals/`；只使用披露日不晚于调仓日的记录。" if historical_mode else "- 手工开发测试模式不声明历史基本面完整性。",
        "- 当前未使用新闻、公告、政策、DeepSeek 结果作为买入依据。",
        "",
        "## 股票池说明",
        "",
        f"- 股票池模式：{'历史时点股票池' if historical_mode else '手工开发测试池'}",
        f"- 输入/联合股票数量：{result.stock_count if historical_mode else len(stock_codes)}",
        f"- 可用行情股票数：{result.stock_count}",
        f"- 覆盖行业：{', '.join(sectors) if sectors else '未识别'}",
        "- 每个调仓日只允许使用该日已经生效的指数成员和已上市证券。" if historical_mode else "- 这些股票来自当前 `stock_pool.csv` 中启用的研究样本。",
        "- 缺少历史快照时回测会停止，不会降级为当前成分股。" if historical_mode else "- 本报告属于小股票池内部排序回测，不是全市场选股能力证明。",
        "",
        "## 回测设置",
        "",
        f"- 每次持仓数量：Top {top_n}",
        f"- 持有周期：{holding_days} 个交易日",
        f"- 调仓间隔：{rebalance_days} 个交易日",
        f"- 历史行情长度：约 {history_days} 个交易日",
        f"- 单边费率假设：{round(fee_rate * 100, 4)}%",
        "- 当前因子：动量 55% + 资金 45%",
        "- 基本面不改变本基础策略的55/45权重，只作为正式研究候选资格检查；缺少可用历史财报的股票被剔除。" if historical_mode else "- 手工开发测试模式保留原行为。",
        "- 当前不含 AI 新闻事件因子、不含真实交易执行。",
        "- 调仓规则：每个调仓日对股票池全部可用股票打分，选择综合分最高的 TopN，等权持有到下一个退出日。",
        "",
        "## 核心结果",
        "",
        f"- 调仓次数：{result.trade_count}",
        f"- 初始资金：{round(result.initial_capital, 2)}",
        f"- 期末权益：{result.final_equity}",
        f"- 总收益率：{format_pct(result.total_return_pct)}",
        f"- 单期胜率：{format_ratio(result.win_rate)}",
        f"- 平均单期收益：{format_pct(result.average_return_pct)}",
        f"- 最大回撤：{format_pct(result.max_drawdown_pct)}",
        f"- 夏普近似值：{format_number(result.sharpe_ratio)}",
        f"- 跳过调仓点：{result.skipped_dates}",
        f"- 说明：{result.note}",
        f"- 历史过滤统计：{result.filter_summary}" if historical_mode else "- 历史过滤统计：开发测试池模式不适用",
        f"- 基本面模式：{result.fundamental_mode}",
        f"- 回测完整性：{result.backtest_integrity}",
        "",
        "## 结果解读",
        "",
        build_interpretation(result),
        "",
        "## 股票池清单",
        "",
        "| 股票代码 | 名称 | 行业 | 备注 |",
        "|---|---|---|---|",
    ]
    for code in stock_codes:
        info = metadata.get(code, {})
        lines.append(f"| {code} | {info.get('name', '')} | {info.get('sector', '')} | {info.get('note', '')} |")

    lines.extend(
        [
        "",
        "## 调仓明细",
        "",
        "| 日期 | 退出日 | 股票 | 综合分 | 动量 | 资金 | 个股收益 | 组合收益 | 权益 |",
        "|---|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for trade in result.trades:
        codes = format_code_list(trade.codes, metadata)
        score_text = format_metric_dict(trade.scores)
        momentum_text = format_metric_dict(trade.momentum_scores)
        money_flow_text = format_metric_dict(trade.money_flow_scores)
        return_text = format_return_dict(trade.returns)
        lines.append(
            f"| {trade.date} | {trade.exit_date} | {codes} | {score_text} | {momentum_text} | {money_flow_text} | {return_text} | {format_pct(trade.portfolio_return)} | {trade.equity} |"
        )
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 固定更大的股票池和更长历史区间，避免小样本误判。",
            "2. 增加行业约束、单票权重上限、止损和最大回撤控制。",
            "3. 对比加入 AI 事件因子前后的收益、回撤和胜率变化。",
            "4. 加入因子相关性分析，避免动量与资金因子重复计权。",
            "5. 在回测和风控稳定前，不接入真实交易。",
            "",
        ]
    )
    return "\n".join(lines)


def build_metadata(items: list[StockPoolItem]) -> dict[str, dict[str, str]]:
    return {item.code: {"name": item.name, "sector": item.sector, "note": item.note} for item in items}


def build_interpretation(result: PortfolioBacktestResult) -> str:
    parts: list[str] = []
    parts.append(f"本次回测形成 {result.trade_count} 次调仓样本。样本数量仍然偏少，主要用于观察策略框架是否正常。")
    if result.total_return_pct > 0:
        parts.append(f"期末收益为正，说明在当前小股票池和样本期内，动量+资金排序跑赢了初始资金。")
    elif result.total_return_pct < 0:
        parts.append("期末收益为负，说明当前参数和股票池在样本期内没有形成有效收益。")
    else:
        parts.append("期末收益接近 0，暂未体现明显优势。")

    if result.max_drawdown_pct is not None and result.max_drawdown_pct <= -15:
        parts.append(f"最大回撤达到 {result.max_drawdown_pct}%，回撤压力较大，后续必须加入风控。")
    elif result.max_drawdown_pct is not None:
        parts.append(f"最大回撤为 {result.max_drawdown_pct}%，仍需结合更长样本判断稳定性。")

    if result.sharpe_ratio is not None and result.sharpe_ratio < 0.5:
        parts.append(f"夏普近似值为 {result.sharpe_ratio}，风险收益质量一般，不能只看总收益率。")
    parts.append("当前没有和沪深300等基准比较，因此还不能判断是否有超额收益。")
    return "\n\n".join(parts)


def format_code_list(codes: list[str], metadata: dict[str, dict[str, str]]) -> str:
    items: list[str] = []
    for code in codes:
        name = metadata.get(code, {}).get("name", "")
        items.append(f"{code}{f'({name})' if name else ''}")
    return "<br>".join(items)


def format_metric_dict(values: dict[str, float]) -> str:
    return "<br>".join(f"{code}:{round(value, 2)}" for code, value in values.items())


def format_return_dict(values: dict[str, float]) -> str:
    return "<br>".join(f"{code}:{format_pct(value)}" for code, value in values.items())


def format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{round(value, 4)}%"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{round(value * 100, 2)}%"


def format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return str(round(value, 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基础组合回测")
    parser.add_argument("--codes", help="显式开发测试池；留空时严格使用历史指数成分")
    parser.add_argument("--pool-file", default=str(DEFAULT_STOCK_POOL_FILE), help="仅用于显式开发测试模式的股票池CSV路径")
    parser.add_argument("--universe", default="CSI300_CSI500", help="历史股票池: CSI300/CSI500/CSI300_CSI500")
    parser.add_argument("--universe-limit", type=int, help="限制每期股票数，仅用于快速测试")
    parser.add_argument("--start-date", help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--min-average-amount", type=float, default=20000000.0, help="过去20日平均成交额最低阈值")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出报告路径")
    parser.add_argument("--top-n", type=int, default=5, help="每次持仓数量")
    parser.add_argument("--holding-days", type=int, default=20, help="持有周期")
    parser.add_argument("--rebalance-days", type=int, default=20, help="调仓间隔")
    parser.add_argument("--history-days", type=int, default=360, help="历史行情长度")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="初始资金")
    parser.add_argument("--fee-rate", type=float, default=0.0015, help="单边费率")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    codes = parse_codes(args.codes) or None
    result = generate_portfolio_backtest_report(
        stock_codes=codes,
        output_file=Path(args.output),
        pool_file=Path(args.pool_file),
        top_n=args.top_n,
        holding_days=args.holding_days,
        rebalance_days=args.rebalance_days,
        history_days=args.history_days,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        universe_type=args.universe,
        universe_limit=args.universe_limit,
        start_date=args.start_date,
        end_date=args.end_date,
        min_average_amount=args.min_average_amount,
    )
    print(f"基础组合回测报告已生成: {args.output}")
    print(f"调仓次数: {result.trade_count}，总收益率: {result.total_return_pct}%")
