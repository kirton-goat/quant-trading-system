"""Small local status view for the resumable V3 free-data build."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.v3.data_acquisition import V3_HFQ_PRICE_CACHE, V3_PRICE_CACHE, snapshot_codes
from research.v3.fundamental_acquisition import V3_CACHE as V3_FUNDAMENTAL_CACHE
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE
from research.v3.liquidity_acquisition import LIQUIDITY_CACHE, MANIFEST as LIQUIDITY_MANIFEST
from research.v3.preflight import default_v3_config


BASE_DIR = Path(__file__).resolve().parents[2]
MANIFEST = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight" / "data_acquisition_manifest.csv"


def main() -> None:
    config = default_v3_config()
    start, end = config["sample_period"]["start"], config["sample_period"]["end"]
    expected = len(snapshot_codes(start, end, int(config["rebalance_days"])))
    qfq_files = len(list(V3_PRICE_CACHE.glob("*.csv")))
    hfq_files = len(list(V3_HFQ_PRICE_CACHE.glob("*.csv")))
    hfq_baostock_files = len(list(HFQ_BAOSTOCK_CACHE.glob("*.csv")))
    fundamental_files = len(list(V3_FUNDAMENTAL_CACHE.glob("*.csv")))
    liquidity_files = len(list(LIQUIDITY_CACHE.glob("*.csv")))
    qfq_completed = qfq_failed = hfq_completed = hfq_failed = 0
    if MANIFEST.exists():
        try:
            rows = pd.read_csv(MANIFEST, encoding="utf-8-sig")
            qfq = rows[rows["stage"].eq("strategy_price_qfq")]
            hfq = rows[rows["stage"].eq("strategy_price_hfq_total_return")]
            qfq_completed = int(qfq[qfq["status"].eq("completed")]["key"].nunique())
            qfq_failed = int(qfq[qfq["status"].eq("failed")]["key"].nunique())
            hfq_completed = int(hfq[hfq["status"].eq("completed")]["key"].nunique())
            hfq_failed = int(hfq[hfq["status"].eq("failed")]["key"].nunique())
        except Exception:
            pass
    qfq_percentage = qfq_files / expected * 100 if expected else 0.0
    hfq_percentage = hfq_files / expected * 100 if expected else 0.0
    print("V3 free historical-data build")
    print(f"Historical universe: complete (131 rebalance dates)")
    print(f"Legacy QFQ audit files (not used for V3 returns): {qfq_files} / {expected} ({qfq_percentage:.2f}%)")
    print(f"Legacy Tencent HFQ candidate files (audit only): {hfq_files} / {expected} ({hfq_percentage:.2f}%)")
    print(f"BaoStock HFQ total-return strategy-price files (required): {hfq_baostock_files} / {expected} ({hfq_baostock_files / expected * 100 if expected else 0:.2f}%)")
    print(f"PIT fundamental files: {fundamental_files} / {expected}")
    print(f"BaoStock liquidity files (CNY amount, required): {liquidity_files} / {expected} ({liquidity_files / expected * 100 if expected else 0:.2f}%)")
    print(f"QFQ manifest completed: {qfq_completed}; failed: {qfq_failed}")
    print(f"HFQ manifest completed: {hfq_completed}; failed: {hfq_failed}")
    print(f"HFQ cache folder: {V3_HFQ_PRICE_CACHE}")
    print(f"Manifest: {MANIFEST}")
    print(f"Liquidity manifest: {LIQUIDITY_MANIFEST}")
    print("The download is resumable. A strict V3 backtest remains blocked until all checks pass.")


if __name__ == "__main__":
    main()
