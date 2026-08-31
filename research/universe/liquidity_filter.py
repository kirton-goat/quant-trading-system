from __future__ import annotations

from typing import Any

import pandas as pd


def history_as_of(history: pd.DataFrame | None, date: str) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    data = history.copy()
    date_column = "date" if "date" in data.columns else "日期" if "日期" in data.columns else None
    if date_column is None:
        return pd.DataFrame()
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    cutoff = pd.to_datetime(date)
    return data[data[date_column] <= cutoff].sort_values(date_column).reset_index(drop=True)


def average_amount(history: pd.DataFrame | None, date: str, window: int = 20) -> float | None:
    data = history_as_of(history, date)
    if data.empty:
        return None
    amount_column = "amount" if "amount" in data.columns else "成交额" if "成交额" in data.columns else None
    if amount_column is None:
        return None
    values = pd.to_numeric(data[amount_column], errors="coerce").dropna().tail(window)
    if len(values) < window:
        return None
    return float(values.mean())


def has_minimum_liquidity(
    history: pd.DataFrame | None,
    date: str,
    minimum_average_amount: float,
    window: int = 20,
) -> tuple[bool, float | None]:
    value = average_amount(history, date, window=window)
    return value is not None and value >= minimum_average_amount, value


def is_long_suspended(history: pd.DataFrame | None, date: str, max_calendar_days_without_trade: int = 20) -> bool:
    data = history_as_of(history, date)
    if data.empty:
        return True
    date_column = "date" if "date" in data.columns else "日期"
    last_trade = pd.to_datetime(data.iloc[-1][date_column], errors="coerce")
    if pd.isna(last_trade):
        return True
    return (pd.to_datetime(date) - last_trade).days > max_calendar_days_without_trade


def trading_day_count(history: pd.DataFrame | None, date: str) -> int:
    return len(history_as_of(history, date))


def numeric_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None
