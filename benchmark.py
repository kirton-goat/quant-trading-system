from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_CACHE_DIR = BASE_DIR / "data_cache" / "benchmark"


@dataclass
class BenchmarkResult:
    name: str
    symbol: str
    data: pd.DataFrame
    source: str
    warning: str = ""


def load_benchmark(symbol: str = "sh000300", name: str = "沪深300", start_date: str = "", end_date: str = "") -> BenchmarkResult:
    BENCHMARK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = load_benchmark_cache(symbol)
    trimmed = trim_benchmark(cached, start_date, end_date)
    if not trimmed.empty:
        return BenchmarkResult(name, symbol, trimmed, "本地缓存")

    fetched = fetch_benchmark(symbol)
    if not fetched.empty:
        save_benchmark_cache(symbol, fetched)
        return BenchmarkResult(name, symbol, trim_benchmark(fetched, start_date, end_date), "AkShare指数日线")

    return BenchmarkResult(name, symbol, pd.DataFrame(columns=["date", "close"]), "不可用", "沪深300基准行情获取失败，本次报告不计算基准对比。")


def fetch_benchmark(symbol: str) -> pd.DataFrame:
    try:
        import akshare as ak

        df = ak.stock_zh_index_daily(symbol=symbol)
        return normalize_benchmark(df)
    except Exception:
        return pd.DataFrame(columns=["date", "close"])


def normalize_benchmark(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    data = df.copy()
    rename_map = {"日期": "date", "收盘": "close"}
    data = data.rename(columns={column: rename_map.get(str(column), column) for column in data.columns})
    if "date" not in data.columns or "close" not in data.columns:
        return pd.DataFrame(columns=["date", "close"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "close"])
    return data[["date", "close"]].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def save_benchmark_cache(symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    normalize_benchmark(df).to_csv(cache_path(symbol), index=False, encoding="utf-8-sig")


def load_benchmark_cache(symbol: str) -> pd.DataFrame:
    path = cache_path(symbol)
    if not path.exists():
        return pd.DataFrame(columns=["date", "close"])
    try:
        return normalize_benchmark(pd.read_csv(path, encoding="utf-8-sig"))
    except Exception:
        return pd.DataFrame(columns=["date", "close"])


def cache_path(symbol: str) -> Path:
    return BENCHMARK_CACHE_DIR / f"{symbol}.csv"


def trim_benchmark(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    data = df.copy()
    dates = pd.to_datetime(data["date"], errors="coerce")
    start = parse_date(start_date)
    end = parse_date(end_date)
    if start is not None:
        data = data[dates >= start]
        dates = pd.to_datetime(data["date"], errors="coerce")
    if end is not None:
        data = data[dates <= end]
    return data.reset_index(drop=True)


def benchmark_return_on_dates(df: pd.DataFrame, dates: list[str]) -> dict[str, float]:
    if df.empty or not dates:
        return {}
    data = df[df["date"].isin(dates)].copy()
    if data.empty:
        return {}
    first = float(data.iloc[0]["close"])
    if first == 0:
        return {}
    return {str(row["date"]): round((float(row["close"]) / first - 1) * 100, 4) for _, row in data.iterrows()}


def compare_strategy_to_benchmark(strategy_curve: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict[str, Any]:
    if strategy_curve.empty or benchmark_df.empty:
        return {
            "benchmark_total_return_pct": None,
            "excess_return_pct": None,
            "annualized_return_diff_pct": None,
            "benchmark_max_drawdown_pct": None,
            "max_drawdown_diff_pct": None,
        }
    merged = strategy_curve[["date", "equity", "return_pct"]].merge(benchmark_df[["date", "close"]], on="date", how="inner")
    if len(merged) < 2:
        return {
            "benchmark_total_return_pct": None,
            "excess_return_pct": None,
            "annualized_return_diff_pct": None,
            "benchmark_max_drawdown_pct": None,
            "max_drawdown_diff_pct": None,
        }
    first_close = float(merged.iloc[0]["close"])
    last_close = float(merged.iloc[-1]["close"])
    benchmark_total = (last_close / first_close - 1) * 100 if first_close else None
    strategy_total = float(merged.iloc[-1]["return_pct"])
    trading_years = max(1 / 252, len(merged) / 252)
    strategy_annual = annualized_return(strategy_total, trading_years)
    benchmark_annual = annualized_return(benchmark_total, trading_years) if benchmark_total is not None else None
    benchmark_dd = max_drawdown_from_values(merged["close"].tolist())
    strategy_dd = max_drawdown_from_values(merged["equity"].tolist())
    return {
        "benchmark_total_return_pct": round(benchmark_total, 4) if benchmark_total is not None else None,
        "excess_return_pct": round(strategy_total - benchmark_total, 4) if benchmark_total is not None else None,
        "annualized_return_diff_pct": round(strategy_annual - benchmark_annual, 4) if benchmark_annual is not None else None,
        "benchmark_max_drawdown_pct": benchmark_dd,
        "max_drawdown_diff_pct": round(strategy_dd - benchmark_dd, 4) if strategy_dd is not None and benchmark_dd is not None else None,
    }


def annualized_return(total_return_pct: float | None, years: float) -> float:
    if total_return_pct is None:
        return 0.0
    return ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100


def max_drawdown_from_values(values: list[float]) -> float | None:
    if not values:
        return None
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            max_dd = min(max_dd, (value - peak) / peak * 100)
    return round(max_dd, 4)


def parse_date(value: str) -> pd.Timestamp | None:
    if not value:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        parsed = pd.to_datetime(value, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return None
