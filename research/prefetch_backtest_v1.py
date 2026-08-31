from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from market_data_manager import MarketDataManager
from research.a_share_backtest import warmup_start_date
from research.fundamentals.fundamental_data_loader import (
    load_fundamental_history,
    load_unadjusted_price_history,
)


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = BASE_DIR / "backtest_config_v1.yaml"
SNAPSHOT_DIR = BASE_DIR / "data_cache" / "historical_universe" / "index_members"
MANIFEST_FILE = BASE_DIR / "data_cache" / "backtest_v1_prefetch_manifest.json"
FAILURE_FILE = BASE_DIR / "data_cache" / "backtest_v1_prefetch_failures.csv"
_PRINT_LOCK = threading.Lock()


@dataclass
class StageResult:
    stage: str
    requested: int
    succeeded: int
    failed: int
    failures: list[dict[str, str]]


def load_config(path: Path = CONFIG_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_snapshot_codes(start_date: str, end_date: str) -> list[str]:
    codes: set[str] = set()
    for index_name in ("csi300", "csi500"):
        for path in (SNAPSHOT_DIR / index_name).glob("*.csv"):
            if not start_date <= path.stem <= end_date:
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
            except (OSError, pd.errors.ParserError):
                continue
            if "code" not in frame.columns:
                continue
            codes.update(
                value.zfill(6)
                for value in frame["code"].dropna().astype(str)
                if value.isdigit()
            )
    return sorted(codes)


def prefetch_market(code: str, start_date: str, end_date: str) -> tuple[bool, str]:
    manager = MarketDataManager()
    data = manager.load_history(code, start_date, end_date, min_rows=180)
    if data.empty or len(data) < 180:
        return False, f"insufficient rows: {len(data)}"
    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    if dates.empty or dates.min() > requested_start + pd.Timedelta(days=7):
        return False, "history does not cover requested start"
    if dates.max() < requested_end - pd.Timedelta(days=7):
        return False, "history does not cover requested end"
    return True, ""


def prefetch_fundamental(code: str) -> tuple[bool, str]:
    data = load_fundamental_history(code)
    if data.empty:
        return False, "no point-in-time statement history"
    required = {"report_period", "disclosure_date"}
    if not required.issubset(data.columns):
        return False, "missing report_period/disclosure_date"
    if data["disclosure_date"].astype(str).eq("").all():
        return False, "no disclosure dates"
    return True, ""


def prefetch_valuation_price(code: str) -> tuple[bool, str]:
    data = load_unadjusted_price_history(code)
    if data.empty:
        return False, "no unadjusted valuation price history"
    return True, ""


def run_stage(
    stage: str,
    codes: list[str],
    worker: Callable[[str], tuple[bool, str]],
    workers: int,
) -> StageResult:
    failures: list[dict[str, str]] = []
    succeeded = 0
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix=stage) as pool:
        futures = {pool.submit(worker, code): code for code in codes}
        for completed, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                ok, reason = future.result()
            except Exception as exc:  # A failed network request must remain visible.
                ok, reason = False, f"{type(exc).__name__}: {exc}"
            if ok:
                succeeded += 1
            else:
                failures.append({"stage": stage, "stock_code": code, "reason": reason})
            if completed % 25 == 0 or completed == len(codes):
                with _PRINT_LOCK:
                    print(
                        f"[{stage}] {completed}/{len(codes)} "
                        f"ok={succeeded} failed={len(failures)}",
                        flush=True,
                    )
    return StageResult(stage, len(codes), succeeded, len(failures), failures)


def run_prefetch(
    workers: int = 8,
    skip_market: bool = False,
    skip_fundamentals: bool = False,
    skip_prices: bool = False,
) -> dict:
    config = load_config()
    codes = load_snapshot_codes(config["start_date"], config["end_date"])
    if not codes:
        raise RuntimeError("No historical universe snapshots found for Backtest v1.0")
    market_start = warmup_start_date(config["start_date"], calendar_days=420)
    stages = []
    if not skip_market:
        stages.append(
            run_stage(
                "market",
                codes,
                lambda code: prefetch_market(code, market_start, config["end_date"]),
                workers,
            )
        )
    if not skip_fundamentals:
        stages.append(
            run_stage("fundamentals", codes, prefetch_fundamental, max(1, workers // 2))
        )
    if not skip_prices:
        stages.append(
            run_stage("valuation_prices", codes, prefetch_valuation_price, max(1, workers // 2))
        )
    manifest = {
        "backtest_version": config["backtest_version"],
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "union_stock_count": len(codes),
        "workers": workers,
        "stages": [asdict(stage) | {"failures": stage.failures[:20]} for stage in stages],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    all_failures = [item for stage in stages for item in stage.failures]
    pd.DataFrame(all_failures, columns=["stage", "stock_code", "reason"]).to_csv(
        FAILURE_FILE, index=False, encoding="utf-8-sig"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch Backtest v1.0 point-in-time datasets")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-market", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-prices", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_prefetch(
                workers=args.workers,
                skip_market=args.skip_market,
                skip_fundamentals=args.skip_fundamentals,
                skip_prices=args.skip_prices,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
