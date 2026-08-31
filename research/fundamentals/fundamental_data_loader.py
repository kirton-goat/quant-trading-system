from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable
import datetime as dt

import pandas as pd

from .fundamental_cache import (
    DEFAULT_FUNDAMENTAL_CACHE,
    read_fundamental_cache,
    write_fundamental_cache,
)


DATA_SOURCE = "Eastmoney historical statements via AKShare"
_HISTORY_MEMORY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_PRICE_MEMORY_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def load_fundamental_history(
    code: str,
    cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE,
    allow_network: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    normalized_code = normalize_code(code)
    memory_key = (normalized_code, str(cache_dir.resolve()))
    if not force_refresh and memory_key in _HISTORY_MEMORY_CACHE:
        return _HISTORY_MEMORY_CACHE[memory_key]
    cached = read_fundamental_cache(normalized_code, cache_dir)
    if not force_refresh and not cached.empty:
        normalized = normalize_cached_history(cached)
        _HISTORY_MEMORY_CACHE[memory_key] = normalized
        return normalized
    if not allow_network:
        normalized = normalize_cached_history(cached)
        if not normalized.empty:
            _HISTORY_MEMORY_CACHE[memory_key] = normalized
        return normalized
    fetched = fetch_fundamental_history(normalized_code)
    if fetched.empty:
        return normalize_cached_history(cached)
    write_fundamental_cache(normalized_code, fetched, cache_dir)
    normalized = normalize_cached_history(fetched)
    _HISTORY_MEMORY_CACHE[memory_key] = normalized
    return normalized


def fetch_fundamental_history(code: str) -> pd.DataFrame:
    import akshare as ak

    symbol = eastmoney_symbol(code)
    statement_loaders: dict[str, Callable[..., pd.DataFrame]] = {
        "profit": ak.stock_profit_sheet_by_report_em,
        "balance": ak.stock_balance_sheet_by_report_em,
        "cashflow": ak.stock_cash_flow_sheet_by_report_em,
    }
    statements: dict[str, pd.DataFrame] = {}
    for name, loader in statement_loaders.items():
        try:
            statements[name] = loader(symbol=symbol)
        except Exception:
            statements[name] = fetch_delisted_statement(ak, name, symbol)
    return combine_statements(code, statements)


def load_unadjusted_price_history(
    code: str,
    cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE,
    allow_network: bool = True,
) -> pd.DataFrame:
    normalized_code = normalize_code(code)
    memory_key = (normalized_code, str(cache_dir.resolve()))
    if memory_key in _PRICE_MEMORY_CACHE:
        return _PRICE_MEMORY_CACHE[memory_key]
    path = cache_dir / "valuation_prices" / f"{normalized_code}.csv"
    if path.exists():
        try:
            cached = pd.read_csv(path, encoding="utf-8-sig")
            _PRICE_MEMORY_CACHE[memory_key] = cached
            return cached
        except (OSError, pd.errors.ParserError):
            pass
    if not allow_network:
        return pd.DataFrame()
    import akshare as ak

    price_source = "Eastmoney unadjusted daily price via AKShare"
    try:
        raw = ak.stock_zh_a_hist(
            symbol=normalized_code,
            period="daily",
            start_date="19900101",
            end_date=dt.date.today().strftime("%Y%m%d"),
            adjust="",
        )
    except Exception:
        raw = fetch_baostock_unadjusted_prices(normalized_code)
        price_source = "BaoStock unadjusted daily price"
    if raw is None or raw.empty:
        return pd.DataFrame()
    date_column = "日期" if "日期" in raw.columns else "date" if "date" in raw.columns else None
    close_column = "收盘" if "收盘" in raw.columns else "close" if "close" in raw.columns else None
    if not date_column or not close_column:
        return pd.DataFrame()
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_column], errors="coerce").dt.strftime("%Y-%m-%d"),
            "close": pd.to_numeric(raw[close_column], errors="coerce"),
            "adjustment": "none",
            "source": price_source,
        }
    ).dropna(subset=["date", "close"])
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    _PRICE_MEMORY_CACHE[memory_key] = output
    return output


def fetch_baostock_unadjusted_prices(code: str) -> pd.DataFrame:
    try:
        import baostock as bs
    except ImportError:
        return pd.DataFrame()
    market_code = f"sh.{code}" if code.startswith(("5", "6", "9")) else f"sz.{code}"
    login = bs.login()
    if login.error_code != "0":
        return pd.DataFrame()
    try:
        result = bs.query_history_k_data_plus(
            market_code,
            "date,close",
            start_date="1990-01-01",
            end_date=dt.date.today().isoformat(),
            frequency="d",
            adjustflag="3",
        )
        if result.error_code != "0":
            return pd.DataFrame()
        rows: list[list[str]] = []
        while result.next():
            rows.append(result.get_row_data())
        return pd.DataFrame(rows, columns=result.fields)
    finally:
        bs.logout()


def fetch_delisted_statement(ak: Any, name: str, symbol: str) -> pd.DataFrame:
    loader_name = {
        "profit": "stock_profit_sheet_by_report_delisted_em",
        "balance": "stock_balance_sheet_by_report_delisted_em",
        "cashflow": "stock_cash_flow_sheet_by_report_delisted_em",
    }[name]
    try:
        return getattr(ak, loader_name)(symbol=symbol)
    except Exception:
        return pd.DataFrame()


def combine_statements(code: str, statements: dict[str, pd.DataFrame]) -> pd.DataFrame:
    indexed = {
        name: statement_rows_by_period(frame)
        for name, frame in statements.items()
    }
    periods = sorted({period for rows in indexed.values() for period in rows})
    rows: list[dict[str, Any]] = []
    for report_period in periods:
        profit = indexed["profit"].get(report_period, {})
        balance = indexed["balance"].get(report_period, {})
        cashflow = indexed["cashflow"].get(report_period, {})
        source_rows = [row for row in (profit, balance, cashflow) if row]
        disclosure_dates = [row["notice_date"] for row in source_rows if row.get("notice_date")]
        update_dates = [row["update_date"] for row in source_rows if row.get("update_date")]
        if not disclosure_dates:
            continue
        disclosure_date = max(disclosure_dates)
        update_date = max(update_dates) if update_dates else disclosure_date
        is_revised = any(
            row.get("update_date") and row.get("notice_date") and row["update_date"] > row["notice_date"]
            for row in source_rows
        )
        rows.append(
            {
                "code": normalize_code(code),
                "report_period": report_period,
                "disclosure_date": disclosure_date,
                "original_disclosure_date": min(disclosure_dates),
                "update_date": update_date,
                "revenue": value(profit, "TOTAL_OPERATE_INCOME", "OPERATE_INCOME"),
                "operating_cost": value(profit, "OPERATE_COST", "TOTAL_OPERATE_COST"),
                "net_profit": value(profit, "PARENT_NETPROFIT", "NETPROFIT"),
                "operating_cash_flow": value(cashflow, "NETCASH_OPERATE"),
                "total_assets": value(balance, "TOTAL_ASSETS"),
                "total_liabilities": value(balance, "TOTAL_LIABILITIES"),
                "total_equity": value(balance, "TOTAL_PARENT_EQUITY", "TOTAL_EQUITY"),
                "share_capital": value(balance, "SHARE_CAPITAL"),
                "data_source": DATA_SOURCE,
                "is_revised": bool(is_revised),
                "revision_note": (
                    "源接口仅提供当前可见版本；UPDATE_DATE晚于NOTICE_DATE，历史原始版本不可恢复，存在修订偏差。"
                    if is_revised
                    else ""
                ),
                "unsupported_for_point_in_time": "",
            }
        )
    return pd.DataFrame(rows)


def statement_rows_by_period(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame is None or frame.empty or "REPORT_DATE" not in frame.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    data = frame.copy()
    data["_report_period"] = data["REPORT_DATE"].map(optional_date)
    data["_notice_date"] = data.get("NOTICE_DATE", pd.Series(index=data.index, dtype=object)).map(optional_date)
    data["_update_date"] = data.get("UPDATE_DATE", pd.Series(index=data.index, dtype=object)).map(optional_date)
    data = data[data["_report_period"].ne("") & data["_notice_date"].ne("")]
    data = data.sort_values(["_report_period", "_notice_date", "_update_date"], ascending=True)
    for report_period, group in data.groupby("_report_period", sort=False):
        # Prefer the earliest published row when the endpoint exposes duplicates.
        row = group.iloc[0]
        payload = row.to_dict()
        payload["notice_date"] = row["_notice_date"]
        payload["update_date"] = row["_update_date"] or row["_notice_date"]
        rows[str(report_period)] = payload
    return rows


def normalize_cached_history(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    output = data.copy()
    for column in ("report_period", "disclosure_date", "original_disclosure_date", "update_date"):
        if column in output.columns:
            output[column] = output[column].map(optional_date)
    if "code" in output.columns:
        output["code"] = output["code"].astype(str).map(normalize_code)
    return output.sort_values(["report_period", "disclosure_date"]).reset_index(drop=True)


def value(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        numeric = safe_number(row.get(name))
        if numeric is not None:
            return numeric
    return None


def safe_number(raw: Any) -> float | None:
    try:
        if raw in (None, "", "-") or pd.isna(raw):
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def optional_date(raw: Any) -> str:
    parsed = pd.to_datetime(raw, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def normalize_code(code: str) -> str:
    digits = "".join(character for character in str(code) if character.isdigit())
    return digits[-6:].zfill(6)


def eastmoney_symbol(code: str) -> str:
    normalized = normalize_code(code)
    prefix = "SH" if normalized.startswith(("5", "6", "9")) else "BJ" if normalized.startswith(("4", "8")) else "SZ"
    return f"{prefix}{normalized}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="缓存历史基本面原始数据")
    parser.add_argument("--codes", required=True, help="逗号分隔股票代码")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for item in args.codes.split(","):
        code = normalize_code(item)
        data = load_fundamental_history(code, force_refresh=args.force_refresh)
        print(f"{code}: {len(data)} records")
