"""Prepare and extend an isolated, date-valid PIT fundamentals cache for V3."""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd

from research.fundamentals.fundamental_cache import write_fundamental_cache
from research.fundamentals.fundamental_data_loader import fetch_fundamental_history, normalize_cached_history
from research.v3.data_acquisition import MANIFEST_PATH, snapshot_codes, write_manifest
from research.v3.preflight import default_v3_config


BASE_DIR = Path(__file__).resolve().parents[2]
SOURCE_CACHE = BASE_DIR / "data_cache" / "fundamentals"
V3_CACHE = BASE_DIR / "data_cache" / "v3_fundamentals"
QUARANTINE_PATH = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight" / "fundamental_quarantine.csv"


def valid_fundamental_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = normalize_cached_history(frame)
    if data.empty or "disclosure_date" not in data:
        return pd.DataFrame(), data
    disclosure = pd.to_datetime(data["disclosure_date"], errors="coerce")
    report = pd.to_datetime(data.get("report_period"), errors="coerce")
    valid_mask = disclosure.notna() & (disclosure >= pd.Timestamp("1991-01-01")) & report.notna()
    return data[valid_mask].copy(), data[~valid_mask].copy()


def prepare_existing_cache() -> dict[str, int]:
    V3_CACHE.mkdir(parents=True, exist_ok=True)
    quarantined: list[pd.DataFrame] = []
    source_files = list(SOURCE_CACHE.glob("*.csv"))
    copied = invalid = 0
    for path in source_files:
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        valid, rejected = valid_fundamental_rows(raw)
        if not valid.empty:
            valid.to_csv(V3_CACHE / path.name, index=False, encoding="utf-8-sig")
            copied += 1
        if not rejected.empty:
            rejected.insert(0, "source_cache_file", str(path))
            quarantined.append(rejected)
            invalid += len(rejected)
    output = pd.concat(quarantined, ignore_index=True) if quarantined else pd.DataFrame()
    QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(QUARANTINE_PATH, index=False, encoding="utf-8-sig")
    return {"source_files": len(source_files), "copied_files": copied, "quarantined_rows": invalid}


def missing_v3_codes() -> list[str]:
    config = default_v3_config()
    expected = set(snapshot_codes(config["sample_period"]["start"], config["sample_period"]["end"], int(config["rebalance_days"])))
    cached = {path.stem for path in V3_CACHE.glob("*.csv")}
    return sorted(expected - cached)


def fetch_missing_fundamentals(limit: int | None = None, sleep_seconds: float = 0.25) -> dict[str, int]:
    pending = missing_v3_codes()
    if limit is not None:
        pending = pending[:limit]
    manifest: list[dict[str, object]] = []
    completed = failed = 0
    for code in pending:
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            fetched = fetch_fundamental_history(code)
            valid, rejected = valid_fundamental_rows(fetched)
            if valid.empty:
                raise RuntimeError("no valid disclosed statement rows returned")
            valid.to_csv(V3_CACHE / f"{code}.csv", index=False, encoding="utf-8-sig")
            if not rejected.empty:
                rejected.insert(0, "source_cache_file", f"network:{code}")
                rejected.to_csv(QUARANTINE_PATH, mode="a", header=not QUARANTINE_PATH.exists(), index=False, encoding="utf-8-sig")
            manifest.append({"stage": "pit_fundamentals", "key": code, "status": "completed", "source": "Eastmoney historical statements via AKShare", "adjustment": "not_applicable", "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": len(valid), "error": ""})
            completed += 1
        except Exception as exc:
            manifest.append({"stage": "pit_fundamentals", "key": code, "status": "failed", "source": "Eastmoney historical statements via AKShare", "adjustment": "not_applicable", "started_at": started, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": 0, "error": str(exc)})
            failed += 1
        if len(manifest) >= 10:
            write_manifest(manifest)
            manifest.clear()
        time.sleep(max(0.0, sleep_seconds))
    if manifest:
        write_manifest(manifest)
    return {"requested": len(pending), "completed": completed, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated V3 point-in-time fundamentals cache.")
    parser.add_argument("--stage", choices=["prepare", "missing"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()
    if args.stage == "prepare":
        print(prepare_existing_cache())
    else:
        print(fetch_missing_fundamentals(args.limit, args.sleep))


if __name__ == "__main__":
    main()
