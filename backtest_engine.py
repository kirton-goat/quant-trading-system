from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import akshare as ak


@dataclass
class BacktestSummary:
    total_signals: int
    completed_trades: int
    win_rate: float | None
    avg_pnl_pct: float | None
    max_drawdown_pct: float | None
    sharpe_ratio: float | None
    profit_loss_ratio: float | None
    note: str


def backtest_log_file(log_file: Path, holding_days: int = 3) -> BacktestSummary:
    """Backtest pending simulated trades from a CSV log.

    This module is intentionally offline/research only. It does not send orders,
    does not change positions, and does not connect to any brokerage.
    """
    if not log_file.exists():
        return BacktestSummary(0, 0, None, None, None, None, None, "日志文件不存在")

    with log_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    completed: list[float] = []
    for row in rows:
        if row.get("sim_status") != "pending_exit":
            continue
        code = row.get("关联股票", "")
        entry_time = row.get("sim_entry_time", "")
        entry_price = safe_float(row.get("sim_entry_price"))
        direction = row.get("sim_direction", "long")
        if not (code.isdigit() and len(code) == 6 and entry_time and entry_price):
            continue

        exit_price = fetch_exit_price(code, entry_time, holding_days)
        if exit_price is None:
            continue
        pnl = calculate_pnl_pct(entry_price, exit_price, direction)
        completed.append(pnl)

    if not completed:
        return BacktestSummary(len(rows), 0, None, None, None, None, None, "没有可完成的模拟交易，可能缺少入场价或持仓周期尚未结束")

    wins = sum(1 for pnl in completed if pnl > 0)
    return BacktestSummary(
        total_signals=len(rows),
        completed_trades=len(completed),
        win_rate=round(wins / len(completed), 4),
        avg_pnl_pct=round(sum(completed) / len(completed), 4),
        max_drawdown_pct=calculate_max_drawdown(completed),
        sharpe_ratio=calculate_sharpe(completed),
        profit_loss_ratio=calculate_profit_loss_ratio(completed),
        note="回测仅基于日志模拟字段和日线收盘价，不代表真实交易结果",
    )


def backtest_factor_weights(
    log_file: Path,
    weights: dict[str, float] | None = None,
    holding_days: int = 3,
) -> BacktestSummary:
    """Research hook for testing factor weights.

    Current logs do not yet store daily factor snapshots, so this function
    delegates to the simulated-trade backtest. The signature is kept stable for
    later factor-weight grid search.
    """
    return backtest_log_file(log_file, holding_days=holding_days)


def fetch_exit_price(code: str, entry_time: str, holding_days: int) -> float | None:
    entry_date = parse_date(entry_time)
    if entry_date is None:
        return None
    start_date = entry_date.strftime("%Y%m%d")
    end_date = (entry_date + dt.timedelta(days=holding_days + 10)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df.empty or len(df) <= holding_days:
            return None
        row = df.iloc[min(holding_days, len(df) - 1)]
        return safe_float(row["收盘"])
    except Exception:
        return None


def parse_date(value: str) -> dt.datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def calculate_pnl_pct(entry_price: float, exit_price: float, direction: str) -> float:
    if direction == "short":
        return round((entry_price - exit_price) / entry_price * 100, 4)
    return round((exit_price - entry_price) / entry_price * 100, 4)


def calculate_max_drawdown(pnls: list[float]) -> float | None:
    if not pnls:
        return None
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity *= 1 + pnl / 100
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak * 100
        max_drawdown = min(max_drawdown, drawdown)
    return round(max_drawdown, 4)


def calculate_sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 2:
        return None
    avg = sum(pnls) / len(pnls)
    variance = sum((pnl - avg) ** 2 for pnl in pnls) / (len(pnls) - 1)
    std = variance ** 0.5
    if std == 0:
        return None
    return round(avg / std, 4)


def calculate_profit_loss_ratio(pnls: list[float]) -> float | None:
    gains = [pnl for pnl in pnls if pnl > 0]
    losses = [-pnl for pnl in pnls if pnl < 0]
    if not gains or not losses:
        return None
    return round((sum(gains) / len(gains)) / (sum(losses) / len(losses)), 4)


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
