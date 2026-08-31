"""Read-only PIT fundamental cache audit for V3 data preparation."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data_cache" / "v3_fundamentals"
OUTPUT_PATH = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight" / "fundamental_invalid_disclosure_dates.csv"


def audit_invalid_disclosure_dates(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    findings: list[pd.DataFrame] = []
    for path in cache_dir.glob("*.csv"):
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if "disclosure_date" not in frame:
            continue
        parsed = pd.to_datetime(frame["disclosure_date"], errors="coerce")
        invalid = frame[parsed.isna() | (parsed < pd.Timestamp("1991-01-01"))].copy()
        if invalid.empty:
            continue
        invalid.insert(0, "cache_file", str(path))
        invalid.insert(1, "stock_code_from_file", path.stem)
        findings.append(invalid)
    return pd.concat(findings, ignore_index=True) if findings else pd.DataFrame(columns=["cache_file", "stock_code_from_file", "disclosure_date"])


def main() -> None:
    argparse.ArgumentParser(description="Audit invalid disclosure dates in the isolated V3 PIT cache.").parse_args()
    findings = audit_invalid_disclosure_dates()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    findings.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"invalid_disclosure_rows={len(findings)} output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
