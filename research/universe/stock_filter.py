from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .index_members import DEFAULT_CACHE_DIR, HistoricalIndexDataError, normalize_code, normalize_date
from .liquidity_filter import has_minimum_liquidity, is_long_suspended, trading_day_count


@dataclass(frozen=True)
class UniverseFilterConfig:
    min_listing_trading_days: int = 180
    liquidity_window: int = 20
    min_average_amount: float = 20_000_000.0
    max_calendar_days_without_trade: int = 20
    require_market_history: bool = True


@dataclass
class UniverseFilterStats:
    input_count: int = 0
    output_count: int = 0
    future_membership: int = 0
    future_listing: int = 0
    insufficient_listing: int = 0
    st: int = 0
    delisted: int = 0
    suspended: int = 0
    illiquid: int = 0
    missing_history: int = 0
    reasons: dict[str, list[str]] = field(default_factory=dict)

    def reject(self, reason: str, code: str) -> None:
        setattr(self, reason, getattr(self, reason) + 1)
        self.reasons.setdefault(reason, []).append(code)


def load_security_metadata(
    date: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    allow_network: bool = True,
) -> dict[str, dict[str, str]]:
    requested_date = normalize_date(date)
    status_path = cache_dir / "security_status" / f"{requested_date}.csv"
    basics_path = cache_dir / "security_basics.csv"
    status = read_keyed_csv(status_path)
    basics = read_keyed_csv(basics_path)
    if (not status or not basics) and allow_network:
        fetched_status, fetched_basics = fetch_baostock_security_metadata(requested_date)
        if fetched_status:
            write_rows(status_path, fetched_status.values())
            status = fetched_status
        if fetched_basics:
            write_rows(basics_path, fetched_basics.values())
            basics = fetched_basics
    metadata: dict[str, dict[str, str]] = {}
    for code in set(status) | set(basics):
        metadata[code] = {**basics.get(code, {}), **status.get(code, {})}
    return metadata


def fetch_baostock_security_metadata(date: str) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    try:
        import baostock as bs
    except ImportError as exc:
        raise HistoricalIndexDataError("缺少 baostock 依赖，无法获取历史交易状态。") from exc
    login = bs.login()
    if login.error_code != "0":
        raise HistoricalIndexDataError(f"BaoStock 登录失败：{login.error_msg}")
    try:
        status_result = bs.query_all_stock(day=date)
        status_rows = result_rows(status_result, "历史交易状态")
        basic_result = bs.query_stock_basic()
        basic_rows = result_rows(basic_result, "证券基础资料")
    finally:
        bs.logout()

    status: dict[str, dict[str, str]] = {}
    for row in status_rows:
        code = normalize_code(row.get("code"))
        if code:
            status[code] = {
                "code": code,
                "status_date": date,
                "name": str(row.get("code_name") or "").strip(),
                "trade_status": str(row.get("tradeStatus") or ""),
                "source": f"BaoStock query_all_stock(day={date})",
            }
    basics: dict[str, dict[str, str]] = {}
    for row in basic_rows:
        code = normalize_code(row.get("code"))
        if code:
            basics[code] = {
                "code": code,
                "name": str(row.get("code_name") or "").strip(),
                "list_date": normalize_optional_date(row.get("ipoDate")),
                "delist_date": normalize_optional_date(row.get("outDate")),
                "security_type": str(row.get("type") or ""),
                "listing_status": str(row.get("status") or ""),
                "source": "BaoStock query_stock_basic()",
            }
    return status, basics


def filter_universe(
    members: list[dict[str, Any]],
    date: str,
    histories: dict[str, pd.DataFrame] | None,
    metadata: dict[str, dict[str, str]] | None,
    config: UniverseFilterConfig,
) -> tuple[list[dict[str, Any]], UniverseFilterStats]:
    requested_date = normalize_date(date)
    histories = histories or {}
    metadata = metadata or {}
    stats = UniverseFilterStats(input_count=len(members))
    accepted: list[dict[str, Any]] = []
    for member in members:
        code = normalize_code(member.get("code"))
        if not code:
            continue
        info = metadata.get(code, {})
        snapshot_date = normalize_optional_date(member.get("snapshot_date"))
        list_date = normalize_optional_date(info.get("list_date") or member.get("list_date"))
        delist_date = normalize_optional_date(info.get("delist_date") or member.get("delist_date"))
        name = str(info.get("name") or member.get("name") or "").strip()

        if snapshot_date and snapshot_date > requested_date:
            stats.reject("future_membership", code)
            continue
        if list_date and list_date > requested_date:
            stats.reject("future_listing", code)
            continue
        if delist_date and delist_date < requested_date:
            stats.reject("delisted", code)
            continue
        normalized_name = name.upper().replace(" ", "")
        if normalized_name.startswith(("ST", "*ST", "S*ST", "SST")):
            stats.reject("st", code)
            continue
        if info.get("trade_status") == "0":
            stats.reject("suspended", code)
            continue

        history = histories.get(code)
        if config.require_market_history and (history is None or history.empty):
            stats.reject("missing_history", code)
            continue
        if history is not None and not history.empty:
            if trading_day_count(history, requested_date) < config.min_listing_trading_days:
                stats.reject("insufficient_listing", code)
                continue
            if is_long_suspended(history, requested_date, config.max_calendar_days_without_trade):
                stats.reject("suspended", code)
                continue
            liquid, average_amount = has_minimum_liquidity(
                history,
                requested_date,
                config.min_average_amount,
                config.liquidity_window,
            )
            if not liquid:
                stats.reject("illiquid", code)
                continue
        else:
            average_amount = None

        accepted.append(
            {
                **member,
                "code": code,
                "name": name,
                "list_date": list_date,
                "delist_date": delist_date,
                "status": "tradeable",
                "average_amount_20d": average_amount,
            }
        )
    stats.output_count = len(accepted)
    return accepted, stats


def result_rows(result: Any, label: str) -> list[dict[str, str]]:
    if result.error_code != "0":
        raise HistoricalIndexDataError(f"BaoStock 获取{label}失败：{result.error_msg}")
    rows: list[dict[str, str]] = []
    while result.next():
        rows.append(dict(zip(result.fields, result.get_row_data())))
    return rows


def read_keyed_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return {
                code: row
                for row in csv.DictReader(file)
                if (code := normalize_code(row.get("code")))
            }
    except OSError:
        return {}


def write_rows(path: Path, rows: Any) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def normalize_optional_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return normalize_date(text)
    except HistoricalIndexDataError:
        return ""
