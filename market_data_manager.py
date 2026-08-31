from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market import fetch_daily_history_with_fallback


BASE_DIR = Path(__file__).resolve().parent
MARKET_DATA_CACHE_DIR = BASE_DIR / "data_cache" / "market_data"
STANDARD_COLUMNS = ["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]


@dataclass
class MarketDataStats:
    requested: int = 0
    loaded: int = 0
    cache_hits: int = 0
    fetched: int = 0
    failed: int = 0


class MarketDataManager:
    def __init__(self, cache_dir: Path = MARKET_DATA_CACHE_DIR) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = MarketDataStats()

    def load_history(
        self,
        code: str,
        start_date: str,
        end_date: str,
        min_rows: int = 90,
        allow_network: bool = True,
    ) -> pd.DataFrame:
        cached = self.load_cache(code)
        if cache_covers(cached, start_date, end_date, min_rows=min_rows):
            self.stats.cache_hits += 1
            return trim(cached, start_date, end_date)

        if not allow_network:
            if not cached.empty:
                self.stats.cache_hits += 1
                return trim(cached, start_date, end_date)
            self.stats.failed += 1
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        raw = fetch_daily_history_with_fallback(code, start_date, end_date)
        normalized = normalize_market_data(raw, code)
        if not normalized.empty:
            merged = merge_history(cached, normalized, code)
            self.save_cache(code, merged)
            self.stats.fetched += 1
            return trim(merged, start_date, end_date)

        if not cached.empty:
            self.stats.cache_hits += 1
            return trim(cached, start_date, end_date)

        self.stats.failed += 1
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    def load_histories(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        min_rows: int = 90,
        limit: int | None = None,
        allow_network: bool = True,
    ) -> dict[str, pd.DataFrame]:
        self.stats = MarketDataStats(requested=len(codes[:limit] if limit else codes))
        histories: dict[str, pd.DataFrame] = {}
        for code in codes[:limit] if limit else codes:
            df = self.load_history(
                code,
                start_date,
                end_date,
                min_rows=min_rows,
                allow_network=allow_network,
            )
            if len(df) >= min_rows:
                histories[code] = df
                self.stats.loaded += 1
        return histories

    def load_cache(self, code: str) -> pd.DataFrame:
        path = self.cache_path(code)
        if not path.exists():
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        try:
            return normalize_market_data(pd.read_csv(path, encoding="utf-8-sig"), code)
        except Exception:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

    def save_cache(self, code: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        normalize_market_data(df, code).to_csv(self.cache_path(code), index=False, encoding="utf-8-sig")

    def cache_path(self, code: str) -> Path:
        return self.cache_dir / f"{code}.csv"


def normalize_market_data(df: pd.DataFrame | None, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    data = df.copy()
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vol": "volume",
        "amount": "amount",
    }
    data = data.rename(columns={column: rename_map.get(str(column), column) for column in data.columns})
    if "date" not in data.columns or "close" not in data.columns:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in data.columns:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["stock_code"] = code
    data = data.dropna(subset=["date", "close"])
    return data[STANDARD_COLUMNS].sort_values("date").drop_duplicates(["date", "stock_code"], keep="last").reset_index(drop=True)


def merge_history(old: pd.DataFrame, new: pd.DataFrame, code: str) -> pd.DataFrame:
    frames = [frame for frame in [old, new] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return normalize_market_data(pd.concat(frames, ignore_index=True), code)


def cache_covers(df: pd.DataFrame, start_date: str, end_date: str, min_rows: int = 90) -> bool:
    clipped = trim(df, start_date, end_date)
    if clipped.empty or len(clipped) < min_rows:
        return False
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    start = parse_date(start_date)
    end = parse_date(end_date)
    if dates.empty:
        return False
    if start is not None and dates.min() > pd.Timestamp(start) + pd.Timedelta(days=7):
        return False
    if end is not None and dates.max() < pd.Timestamp(end) - pd.Timedelta(days=7):
        return False
    return True


def trim(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    data = df.copy()
    start = parse_date(start_date)
    end = parse_date(end_date)
    dates = pd.to_datetime(data["date"], errors="coerce")
    if start is not None:
        data = data[dates >= start]
        dates = pd.to_datetime(data["date"], errors="coerce")
    if end is not None:
        data = data[dates <= end]
    return data.reset_index(drop=True)


def parse_date(value: str) -> dt.datetime | None:
    text = str(value)
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def date_n_days_ago(trading_days: int) -> str:
    return (dt.datetime.now() - dt.timedelta(days=max(1, trading_days) * 2)).strftime("%Y%m%d")
