from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FUNDAMENTAL_AUDIT_LOG = BASE_DIR / "fundamental_future_data_check.log"


class FundamentalFutureDataError(RuntimeError):
    pass


@dataclass
class FundamentalValidationResult:
    as_of_date: str
    checked: int = 0
    future_records: int = 0
    missing_disclosure_date: int = 0

    @property
    def valid(self) -> bool:
        return self.future_records == 0 and self.missing_disclosure_date == 0


def normalize_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise FundamentalFutureDataError(f"无效日期：{value}")
    return parsed.strftime("%Y-%m-%d")


def validate_visible_record(record: dict[str, Any], as_of_date: str, strict: bool = True) -> bool:
    cutoff = normalize_date(as_of_date)
    disclosure = record.get("disclosure_date")
    if not disclosure:
        if strict:
            raise FundamentalFutureDataError(
                f"{record.get('code', '')} {record.get('report_period', '')} 缺少披露日期"
            )
        return False
    disclosure_date = normalize_date(disclosure)
    if disclosure_date > cutoff:
        if strict:
            raise FundamentalFutureDataError(
                f"未来财务数据：{record.get('code', '')} report_period={record.get('report_period', '')} "
                f"disclosure_date={disclosure_date} > as_of_date={cutoff}"
            )
        return False
    return True


def validate_records(
    records: Iterable[dict[str, Any]],
    as_of_date: str,
    audit_log: Path = DEFAULT_FUNDAMENTAL_AUDIT_LOG,
    strict: bool = True,
) -> FundamentalValidationResult:
    result = FundamentalValidationResult(as_of_date=normalize_date(as_of_date))
    for record in records:
        result.checked += 1
        if not record.get("disclosure_date"):
            result.missing_disclosure_date += 1
            continue
        if normalize_date(record["disclosure_date"]) > result.as_of_date:
            result.future_records += 1
    status = "PASS" if result.valid else "BLOCK"
    append_audit(
        audit_log,
        result.as_of_date,
        status,
        f"checked={result.checked} future_records={result.future_records} "
        f"missing_disclosure_date={result.missing_disclosure_date}",
    )
    if strict and not result.valid:
        raise FundamentalFutureDataError(
            f"基本面时点检查失败：future={result.future_records}, "
            f"missing_disclosure={result.missing_disclosure_date}"
        )
    return result


def determine_backtest_integrity(
    universe_mode: str,
    fundamental_mode: str,
    fundamental_validation_passed: bool,
) -> str:
    if (
        universe_mode == "historical_point_in_time"
        and fundamental_mode == "historical_point_in_time"
        and fundamental_validation_passed
    ):
        return "point_in_time_validated"
    return "incomplete"


def public_integrity_status(backtest_integrity: str) -> str:
    """Return a stable external status while retaining the detailed internal value."""
    return "validated" if backtest_integrity == "point_in_time_validated" else "incomplete"


def append_audit(path: Path, as_of_date: str, status: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as file:
        file.write(
            f"{timestamp}\tas_of_date={as_of_date}\tresult={status}\t{detail}\n"
        )
