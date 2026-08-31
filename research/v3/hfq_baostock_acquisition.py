"""Build an isolated, resumable BaoStock HFQ price cache for V3 returns.

The earlier Tencent HFQ cache has correct-looking endpoints but internal date
blocks missing for some securities.  This builder does not overwrite it.  It
uses one BaoStock session to obtain a consistent adjusted close series for the
full V3 candidate set and records provenance on every file.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd

from research.v3.data_acquisition import PRICE_COLUMNS, snapshot_codes
from research.v3.preflight import default_v3_config, required_history_start


BASE_DIR = Path(__file__).resolve().parents[2]
HFQ_BAOSTOCK_CACHE = BASE_DIR / "data_cache" / "v3_strategy_prices_hfq_baostock"
OUTPUT_DIR = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight"
MANIFEST = OUTPUT_DIR / "hfq_baostock_acquisition_manifest.csv"


def _market_code(code: str) -> str:
    return ("sh." if str(code).startswith(("5", "6", "9")) else "sz.") + str(code)


def _path(code: str) -> Path:
    return HFQ_BAOSTOCK_CACHE / f"{code}.csv"


def _covers(code: str, start: str, end: str) -> bool:
    path = _path(code)
    if not path.exists():
        return False
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
        dates = pd.to_datetime(data["date"], errors="coerce").dropna()
        return bool(
            set(PRICE_COLUMNS).issubset(data.columns)
            and not dates.empty
            and dates.min() <= pd.Timestamp(start) + pd.Timedelta(days=7)
            and dates.max() >= pd.Timestamp(end) - pd.Timedelta(days=7)
            and set(data["adjustment"].dropna()) == {"hfq_total_return"}
            and set(data["source"].dropna()) == {"BaoStock query_history_k_data_plus(adjustflag=1)"}
        )
    except Exception:
        return False


def _frame(query: object, code: str) -> pd.DataFrame:
    rows: list[list[str]] = []
    while query.next():
        rows.append(query.get_row_data())
    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    data = pd.DataFrame(rows, columns=query.fields)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in ("open", "high", "low", "close", "volume", "amount"):
        data[column] = pd.to_numeric(data.get(column), errors="coerce")
    data["stock_code"] = code
    data["source"] = "BaoStock query_history_k_data_plus(adjustflag=1)"
    data["adjustment"] = "hfq_total_return"
    data["fetch_time"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return data.dropna(subset=["date", "close"])[PRICE_COLUMNS].drop_duplicates(["date", "stock_code"], keep="last").sort_values("date")


def _append_manifest(records: list[dict[str, object]]) -> None:
    if not records:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(MANIFEST, encoding="utf-8-sig") if MANIFEST.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(records)], ignore_index=True).to_csv(MANIFEST, index=False, encoding="utf-8-sig")


def prefetch(limit: int | None = None, sleep_seconds: float = 0.03) -> dict[str, int]:
    import baostock as bs

    config = default_v3_config()
    start = required_history_start(config["sample_period"]["start"])
    end = config["sample_period"]["end"]
    codes = snapshot_codes(config["sample_period"]["start"], end, int(config["rebalance_days"]))
    if limit is not None:
        codes = codes[:limit]
    HFQ_BAOSTOCK_CACHE.mkdir(parents=True, exist_ok=True)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    completed = cached = failed = 0
    batch: list[dict[str, object]] = []
    try:
        for index, code in enumerate(codes, start=1):
            started = dt.datetime.now(dt.timezone.utc).isoformat()
            if _covers(code, start, end):
                cached += 1
                batch.append({"code": code, "status": "cached", "rows": 0, "started_at": started, "completed_at": started, "error": ""})
            else:
                query = bs.query_history_k_data_plus(
                    _market_code(code), "date,open,high,low,close,volume,amount",
                    start_date=start, end_date=end, frequency="d", adjustflag="1",
                )
                if query.error_code != "0":
                    failed += 1
                    batch.append({"code": code, "status": "failed", "rows": 0, "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": query.error_msg})
                else:
                    data = _frame(query, code)
                    if data.empty:
                        failed += 1
                        batch.append({"code": code, "status": "failed", "rows": 0, "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": "empty_response"})
                    else:
                        data.to_csv(_path(code), index=False, encoding="utf-8-sig")
                        completed += 1
                        batch.append({"code": code, "status": "completed", "rows": len(data), "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error": ""})
            if len(batch) >= 25:
                _append_manifest(batch)
                batch.clear()
            if index % 50 == 0:
                print(json.dumps({"processed": index, "total": len(codes), "completed": completed, "cached": cached, "failed": failed}, ensure_ascii=False), flush=True)
            time.sleep(max(0.0, sleep_seconds))
    finally:
        bs.logout()
    _append_manifest(batch)
    return {"requested": len(codes), "completed": completed, "cached": cached, "failed": failed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build V3 BaoStock HFQ return-price cache.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.03)
    args = parser.parse_args()
    print(json.dumps(prefetch(args.limit, args.sleep), ensure_ascii=False))
