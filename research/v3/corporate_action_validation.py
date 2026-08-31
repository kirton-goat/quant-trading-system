"""Representative corporate-action continuity checks for V3 HFQ return prices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import akshare as ak
import pandas as pd

from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight"
SUMMARY_PATH = OUTPUT_DIR / "corporate_action_validation_summary.json"
DETAIL_PATH = OUTPUT_DIR / "corporate_action_validation.csv"
SAMPLE_CODES = ("000001", "000333", "600519")


def ex_date_series(events: pd.DataFrame) -> pd.Series:
    if events is None or events.empty:
        return pd.Series(dtype="datetime64[ns]")
    candidates = [column for column in events.columns if "除权" in str(column) or "除息" in str(column)]
    for column in candidates:
        parsed = pd.to_datetime(events[column], errors="coerce").dropna()
        if not parsed.empty:
            return parsed
    return pd.Series(dtype="datetime64[ns]")


def adjusted_return_on_or_after(price: pd.DataFrame, event_date: pd.Timestamp) -> tuple[str | None, float | None]:
    frame = price.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    positions = frame.index[frame["date"] >= event_date]
    if positions.empty or positions[0] == 0:
        return None, None
    index = int(positions[0])
    previous, current = float(frame.iloc[index - 1]["close"]), float(frame.iloc[index]["close"])
    if previous == 0:
        return None, None
    return frame.iloc[index]["date"].date().isoformat(), current / previous - 1


def validate_corporate_actions(codes: tuple[str, ...] = SAMPLE_CODES, price_cache: Path = HFQ_BAOSTOCK_CACHE) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for code in codes:
        price_path = price_cache / f"{code}.csv"
        if not price_path.exists():
            rows.append({"stock_code": code, "status": "missing_hfq_price", "event_date": "", "trade_date": "", "adjusted_return": None})
            continue
        try:
            price = pd.read_csv(price_path, encoding="utf-8-sig")
            events = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
        except Exception as exc:
            rows.append({"stock_code": code, "status": "event_source_error", "event_date": "", "trade_date": "", "adjusted_return": None, "error": str(exc)})
            continue
        dates = ex_date_series(events)
        dates = dates[(dates >= pd.Timestamp("2015-01-01")) & (dates <= pd.Timestamp("2025-12-31"))]
        if dates.empty:
            rows.append({"stock_code": code, "status": "no_event_in_sample", "event_date": "", "trade_date": "", "adjusted_return": None})
            continue
        # The latest in-sample action is sufficient as an auditable continuity sample.
        event_date = dates.max()
        trade_date, return_value = adjusted_return_on_or_after(price, event_date)
        status = "checked" if return_value is not None else "missing_trade_price"
        rows.append({"stock_code": code, "status": status, "event_date": event_date.date().isoformat(), "trade_date": trade_date, "adjusted_return": return_value})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    checked = detail[detail["status"].eq("checked")]
    large_gaps = int((checked["adjusted_return"].abs() > 0.30).sum()) if not checked.empty else 0
    passed = len(checked) >= 3 and large_gaps == 0
    summary = {
        "classification": "research_experiment", "price_mode": "hfq_total_return", "price_source": "BaoStock query_history_k_data_plus(adjustflag=1)", "samples_requested": len(codes),
        "samples_checked": int(len(checked)), "large_gap_count": large_gaps, "passed": passed,
        "limitation": "Representative continuity check only. It does not replace vendor corporate-action records for every stock.",
        "detail_file": str(DETAIL_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate representative HFQ corporate-action continuity.")
    parser.add_argument("--price-cache", type=Path, default=HFQ_BAOSTOCK_CACHE)
    args = parser.parse_args()
    print(json.dumps(validate_corporate_actions(price_cache=args.price_cache), ensure_ascii=False))


if __name__ == "__main__":
    main()
