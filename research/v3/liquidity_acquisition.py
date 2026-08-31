"""Resumable BaoStock liquidity cache for V3 research.

Adjusted price vendors do not always publish an auditable turnover unit.  V3
therefore keeps total-return prices and liquidity data in different caches:
HFQ prices drive returns; BaoStock's unadjusted daily ``amount`` (CNY) drives
only the point-in-time liquidity eligibility filter.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd

from research.v3.data_acquisition import snapshot_codes
from research.v3.preflight import default_v3_config, required_history_start


BASE_DIR = Path(__file__).resolve().parents[2]
LIQUIDITY_CACHE = BASE_DIR / "data_cache" / "v3_liquidity_baostock"
OUTPUT_DIR = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight"
MANIFEST = OUTPUT_DIR / "liquidity_acquisition_manifest.csv"
COLUMNS = ["date", "stock_code", "volume", "amount", "source", "amount_unit", "fetch_time"]


def market_code(code: str) -> str:
    return ("sh." if str(code).startswith(("5", "6", "9")) else "sz.") + str(code)


def cache_path(code: str) -> Path:
    return LIQUIDITY_CACHE / f"{code}.csv"


def cache_covers(code: str, start: str, end: str) -> bool:
    path = cache_path(code)
    if not path.exists():
        return False
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
        dates = pd.to_datetime(data["date"], errors="coerce").dropna()
        units = set(data.get("amount_unit", pd.Series(dtype=str)).dropna().astype(str))
        return bool(
            not dates.empty
            and dates.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)
            and dates.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)
            and units == {"CNY"}
        )
    except Exception:
        return False


def _rows(query: object, code: str) -> pd.DataFrame:
    records: list[list[str]] = []
    while query.next():
        records.append(query.get_row_data())
    if not records:
        return pd.DataFrame(columns=COLUMNS)
    data = pd.DataFrame(records, columns=query.fields)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["stock_code"] = code
    data["volume"] = pd.to_numeric(data.get("volume"), errors="coerce")
    data["amount"] = pd.to_numeric(data.get("amount"), errors="coerce")
    data["source"] = "BaoStock query_history_k_data_plus(adjustflag=3)"
    data["amount_unit"] = "CNY"
    data["fetch_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return data.dropna(subset=["date", "amount"])[COLUMNS].drop_duplicates(["date", "stock_code"], keep="last").sort_values("date")


def _write_manifest(records: list[dict[str, object]]) -> None:
    if not records:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(MANIFEST, encoding="utf-8-sig") if MANIFEST.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(records)], ignore_index=True).to_csv(MANIFEST, index=False, encoding="utf-8-sig")


def prefetch(limit: int | None = None, sleep_seconds: float = 0.05) -> dict[str, int]:
    import baostock as bs

    config = default_v3_config()
    start = required_history_start(config["sample_period"]["start"])
    end = config["sample_period"]["end"]
    codes = snapshot_codes(config["sample_period"]["start"], end, int(config["rebalance_days"]))
    if limit is not None:
        codes = codes[:limit]
    LIQUIDITY_CACHE.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    completed = cached = failed = 0
    manifest: list[dict[str, object]] = []
    try:
        for position, code in enumerate(codes, start=1):
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            if cache_covers(code, start, end):
                cached += 1
                manifest.append({"code": code, "status": "cached", "rows": 0, "started_at": started, "completed_at": started, "error": ""})
            else:
                query = bs.query_history_k_data_plus(
                    market_code(code), "date,volume,amount", start_date=start, end_date=end, frequency="d", adjustflag="3"
                )
                if query.error_code != "0":
                    failed += 1
                    manifest.append({"code": code, "status": "failed", "rows": 0, "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": query.error_msg})
                else:
                    data = _rows(query, code)
                    if data.empty:
                        failed += 1
                        manifest.append({"code": code, "status": "failed", "rows": 0, "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": "empty_response"})
                    else:
                        data.to_csv(cache_path(code), index=False, encoding="utf-8-sig")
                        completed += 1
                        manifest.append({"code": code, "status": "completed", "rows": len(data), "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": ""})
            if len(manifest) >= 25:
                _write_manifest(manifest)
                manifest.clear()
            if position % 50 == 0:
                print(json.dumps({"processed": position, "total": len(codes), "completed": completed, "cached": cached, "failed": failed}, ensure_ascii=False), flush=True)
            time.sleep(max(0.0, sleep_seconds))
    finally:
        bs.logout()
    _write_manifest(manifest)
    return {"requested": len(codes), "completed": completed, "cached": cached, "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the isolated V3 BaoStock liquidity cache.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(prefetch(args.limit, args.sleep), ensure_ascii=False))
