"""Resumable, provenance-preserving V3 historical data acquisition.

This tool writes only V3 research caches. It never rewrites the ambiguous
legacy strategy-price cache used by the frozen formal versions.
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from research.universe.index_members import HistoricalIndexDataError, get_index_snapshot
from research.v3.preflight import default_v3_config, required_history_start, trading_rebalance_dates


BASE_DIR = Path(__file__).resolve().parents[2]
V3_PRICE_CACHE = BASE_DIR / "data_cache" / "v3_strategy_prices"
V3_HFQ_PRICE_CACHE = BASE_DIR / "data_cache" / "v3_strategy_prices_hfq_total_return"
MANIFEST_PATH = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight" / "data_acquisition_manifest.csv"
PRICE_COLUMNS = ["date", "stock_code", "open", "high", "low", "close", "volume", "amount", "source", "adjustment", "fetch_time"]


def write_manifest(rows: list[dict[str, object]]) -> Path:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig") if MANIFEST_PATH.exists() else pd.DataFrame()
    frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    frame.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    return MANIFEST_PATH


def prefetch_universe(start: str, end: str, rebalance_days: int, limit: int | None = None, sleep_seconds: float = 0.15) -> dict[str, int]:
    dates = trading_rebalance_dates(start, end, rebalance_days)
    if limit is not None:
        dates = dates[:limit]
    manifest: list[dict[str, object]] = []
    completed = failed = 0
    for date in dates:
        for universe in ("CSI300", "CSI500"):
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            try:
                snapshot = get_index_snapshot(date, universe, allow_network=True)
                manifest.append({
                    "stage": "historical_universe", "key": f"{universe}:{date}", "status": "completed",
                    "source": snapshot.source, "adjustment": "not_applicable", "started_at": started,
                    "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": len(snapshot.members), "error": "",
                })
                completed += 1
            except HistoricalIndexDataError as exc:
                manifest.append({
                    "stage": "historical_universe", "key": f"{universe}:{date}", "status": "failed",
                    "source": "BaoStock", "adjustment": "not_applicable", "started_at": started,
                    "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": 0, "error": str(exc),
                })
                failed += 1
            if len(manifest) >= 25:
                write_manifest(manifest)
                manifest.clear()
            time.sleep(max(0, sleep_seconds))
    if manifest:
        write_manifest(manifest)
    return {"requested": len(dates) * 2, "completed": completed, "failed": failed}


def snapshot_codes(start: str, end: str, rebalance_days: int) -> list[str]:
    codes: set[str] = set()
    for date in trading_rebalance_dates(start, end, rebalance_days):
        for universe in ("CSI300", "CSI500"):
            snapshot = get_index_snapshot(date, universe, allow_network=False)
            codes.update(member.code for member in snapshot.members)
    return sorted(codes)


def price_cache_dir(adjustment: str) -> Path:
    return V3_HFQ_PRICE_CACHE if adjustment == "hfq_total_return" else V3_PRICE_CACHE


def normalize_adjusted_history(raw: pd.DataFrame, code: str, source: str, adjustment: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    aliases = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
    frame = raw.rename(columns={column: aliases.get(str(column), str(column)) for column in raw.columns}).copy()
    if "date" not in frame or "close" not in frame:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["stock_code"] = code
    frame["source"] = source
    frame["adjustment"] = adjustment
    frame["fetch_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return frame.dropna(subset=["date", "close"])[PRICE_COLUMNS].drop_duplicates(["date", "stock_code"], keep="last").sort_values("date")


def normalize_qfq_history(raw: pd.DataFrame, code: str, source: str) -> pd.DataFrame:
    """Compatibility wrapper retained for existing V3 governance tests."""
    return normalize_adjusted_history(raw, code, source, "qfq")


def fetch_adjusted_history(code: str, start: str, end: str, adjustment: str = "qfq") -> pd.DataFrame:
    import akshare as ak

    provider_adjustment = "hfq" if adjustment == "hfq_total_return" else "qfq"

    try:
        result = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust=provider_adjustment)
        data = normalize_adjusted_history(result, code, "AKShare stock_zh_a_hist / Eastmoney", adjustment)
        if not data.empty:
            return data
    except Exception:
        pass
    symbol = ("sh" if code.startswith(("5", "6", "9")) else "sz") + code
    try:
        result = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust=provider_adjustment, timeout=10)
        data = normalize_adjusted_history(result, code, "AKShare stock_zh_a_hist_tx / Tencent", adjustment)
        if not data.empty:
            return data
    except Exception:
        pass
    try:
        import baostock as bs

        market_code = ("sh." if code.startswith(("5", "6", "9")) else "sz.") + code
        login = bs.login()
        if login.error_code != "0":
            return pd.DataFrame(columns=PRICE_COLUMNS)
        try:
            query = bs.query_history_k_data_plus(
                market_code, "date,open,high,low,close,volume,amount", start_date=start, end_date=end,
                frequency="d", adjustflag="1" if adjustment == "hfq_total_return" else "2",
            )
            if query.error_code != "0":
                return pd.DataFrame(columns=PRICE_COLUMNS)
            rows: list[list[str]] = []
            while query.next():
                rows.append(query.get_row_data())
            return normalize_adjusted_history(pd.DataFrame(rows, columns=query.fields), code, "BaoStock query_history_k_data_plus", adjustment)
        finally:
            bs.logout()
    except Exception:
        return pd.DataFrame(columns=PRICE_COLUMNS)


def cache_path(code: str, adjustment: str = "qfq") -> Path:
    return price_cache_dir(adjustment) / f"{code}.csv"


def price_cache_covers(code: str, start: str, end: str, adjustment: str = "qfq") -> bool:
    path = cache_path(code, adjustment)
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not set(PRICE_COLUMNS).issubset(frame.columns):
            return False
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        return not dates.empty and dates.min() <= pd.Timestamp(start) + pd.Timedelta(days=7) and dates.max() >= pd.Timestamp(end) - pd.Timedelta(days=7) and set(frame["adjustment"].dropna()) == {adjustment}
    except Exception:
        return False


def prefetch_prices(codes: Iterable[str], start: str, end: str, limit: int | None = None, sleep_seconds: float = 0.25, adjustment: str = "qfq") -> dict[str, int]:
    selected = list(codes)[:limit] if limit is not None else list(codes)
    cache_dir = price_cache_dir(adjustment)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    completed = cached = failed = 0
    for code in selected:
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        if price_cache_covers(code, start, end, adjustment):
            cached += 1
            manifest.append({"stage": f"strategy_price_{adjustment}", "key": code, "status": "cached", "source": "V3 cache", "adjustment": adjustment, "started_at": started, "completed_at": started, "rows": 0, "error": ""})
        else:
            data = fetch_adjusted_history(code, start, end, adjustment)
            if data.empty:
                failed += 1
                manifest.append({"stage": f"strategy_price_{adjustment}", "key": code, "status": "failed", "source": "AKShare", "adjustment": adjustment, "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": 0, "error": "empty response"})
            else:
                data.to_csv(cache_path(code, adjustment), index=False, encoding="utf-8-sig")
                completed += 1
                manifest.append({"stage": f"strategy_price_{adjustment}", "key": code, "status": "completed", "source": data["source"].iloc[-1], "adjustment": adjustment, "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": len(data), "error": ""})
        if len(manifest) >= 25:
            write_manifest(manifest)
            manifest.clear()
        time.sleep(max(0, sleep_seconds))
    if manifest:
        write_manifest(manifest)
    return {"requested": len(selected), "completed": completed, "cached": cached, "failed": failed}


def main() -> None:
    config = default_v3_config()
    parser = argparse.ArgumentParser(description="Build resumable V3 historical research caches.")
    parser.add_argument("--stage", choices=["universe", "prices"], required=True)
    parser.add_argument("--adjustment", choices=["qfq", "hfq_total_return"], default="qfq")
    parser.add_argument("--limit", type=int, default=None, help="Limit dates (universe) or codes (prices); omit for full resumable build.")
    parser.add_argument("--sleep", type=float, default=None)
    args = parser.parse_args()
    start, end = config["sample_period"]["start"], config["sample_period"]["end"]
    if args.stage == "universe":
        result = prefetch_universe(start, end, int(config["rebalance_days"]), args.limit, args.sleep if args.sleep is not None else 0.15)
    else:
        codes = snapshot_codes(start, end, int(config["rebalance_days"]))
        result = prefetch_prices(codes, required_history_start(start), end, args.limit, args.sleep if args.sleep is not None else 0.25, args.adjustment)
    print(result)


if __name__ == "__main__":
    main()
