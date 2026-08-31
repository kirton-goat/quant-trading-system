from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .fundamental_cache import DEFAULT_FUNDAMENTAL_CACHE
from .fundamental_data_loader import (
    load_fundamental_history,
    load_unadjusted_price_history,
    normalize_code,
    safe_number,
)
from .fundamental_validation import (
    DEFAULT_FUNDAMENTAL_AUDIT_LOG,
    FundamentalFutureDataError,
    normalize_date,
    validate_records,
    validate_visible_record,
)


@dataclass
class PointInTimeFundamentals:
    code: str
    as_of_date: str
    report_period: str
    disclosure_date: str
    revenue: float | None = None
    net_profit: float | None = None
    roe: float | None = None
    gross_margin: float | None = None
    operating_cash_flow: float | None = None
    operating_cash_flow_to_net_profit: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    debt_to_assets: float | None = None
    revenue_growth: float | None = None
    net_profit_growth: float | None = None
    pe: float | None = None
    pb: float | None = None
    valuation_price_source: str = ""
    data_source: str = ""
    is_point_in_time: bool = True
    is_revised: bool = False
    revision_note: str = ""
    missing_fields: list[str] = field(default_factory=list)
    unsupported_for_point_in_time: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FundamentalBatchResult:
    scores: dict[str, float] = field(default_factory=dict)
    records: dict[str, PointInTimeFundamentals] = field(default_factory=dict)
    missing_codes: list[str] = field(default_factory=list)
    revised_codes: list[str] = field(default_factory=list)
    future_records: int = 0
    validation_passed: bool = True


def get_fundamentals(
    code: str,
    as_of_date: str,
    price_history: pd.DataFrame | None = None,
    cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE,
    allow_network: bool = True,
    strict: bool = True,
) -> PointInTimeFundamentals | None:
    normalized_code = normalize_code(code)
    cutoff = normalize_date(as_of_date)
    history = load_fundamental_history(
        normalized_code,
        cache_dir=cache_dir,
        allow_network=allow_network,
    )
    if history.empty:
        return None
    history = history[history["disclosure_date"].astype(str).ne("")].copy()
    visible = history[history["disclosure_date"].astype(str) <= cutoff].copy()
    if visible.empty:
        return None
    selected = visible.sort_values(["report_period", "disclosure_date"]).iloc[-1].to_dict()
    validate_visible_record(selected, cutoff, strict=strict)

    report_period = str(selected["report_period"])
    previous_same_period = previous_period_record(visible, report_period)
    revenue = safe_number(selected.get("revenue"))
    net_profit = safe_number(selected.get("net_profit"))
    operating_cost = safe_number(selected.get("operating_cost"))
    operating_cash_flow = safe_number(selected.get("operating_cash_flow"))
    assets = safe_number(selected.get("total_assets"))
    liabilities = safe_number(selected.get("total_liabilities"))
    equity = safe_number(selected.get("total_equity"))
    shares = safe_number(selected.get("share_capital"))

    previous_revenue = safe_number(previous_same_period.get("revenue")) if previous_same_period else None
    previous_profit = safe_number(previous_same_period.get("net_profit")) if previous_same_period else None
    previous_equity = safe_number(previous_same_period.get("total_equity")) if previous_same_period else None
    revenue_growth = growth_rate(revenue, previous_revenue)
    net_profit_growth = growth_rate(net_profit, previous_profit)
    gross_margin = ratio(revenue - operating_cost, revenue, percent=True) if revenue is not None and operating_cost is not None else None
    cashflow_to_profit = ratio(operating_cash_flow, net_profit) if operating_cash_flow is not None else None
    debt_to_assets = ratio(liabilities, assets, percent=True)
    roe = annualized_roe(net_profit, equity, previous_equity, report_period)

    valuation_prices = price_history
    if valuation_prices is None or valuation_prices.empty:
        valuation_prices = load_unadjusted_price_history(normalized_code, cache_dir, allow_network=allow_network)
    price = close_as_of(valuation_prices, cutoff)
    cached_price_source = (
        str(valuation_prices["source"].dropna().iloc[-1])
        if valuation_prices is not None and not valuation_prices.empty and "source" in valuation_prices.columns and not valuation_prices["source"].dropna().empty
        else ""
    )
    ttm_profit = trailing_twelve_month_profit(visible, report_period)
    market_cap = price * shares if price is not None and shares is not None else None
    pe = ratio(market_cap, ttm_profit) if market_cap is not None and ttm_profit is not None and ttm_profit > 0 else None
    pb = ratio(market_cap, equity) if market_cap is not None and equity is not None and equity > 0 else None

    values = {
        "revenue": revenue,
        "net_profit": net_profit,
        "roe": roe,
        "gross_margin": gross_margin,
        "operating_cash_flow": operating_cash_flow,
        "operating_cash_flow_to_net_profit": cashflow_to_profit,
        "total_assets": assets,
        "total_liabilities": liabilities,
        "debt_to_assets": debt_to_assets,
        "revenue_growth": revenue_growth,
        "net_profit_growth": net_profit_growth,
        "pe": pe,
        "pb": pb,
    }
    unsupported = [
        item for item in str(selected.get("unsupported_for_point_in_time") or "").split(",") if item
    ]
    if price is None:
        unsupported.append("historical_unadjusted_price")
    return PointInTimeFundamentals(
        code=normalized_code,
        as_of_date=cutoff,
        report_period=report_period,
        disclosure_date=str(selected["disclosure_date"]),
        data_source=str(selected.get("data_source") or ""),
        is_point_in_time=True,
        is_revised=as_bool(selected.get("is_revised")),
        revision_note=str(selected.get("revision_note") or ""),
        missing_fields=[name for name, value_ in values.items() if value_ is None],
        unsupported_for_point_in_time=unsupported,
        valuation_price_source=(
            "caller-provided historical price"
            if price_history is not None and not price_history.empty
            else cached_price_source if price is not None else ""
        ),
        **values,
    )


def get_fundamental_scores(
    codes: list[str],
    as_of_date: str,
    price_histories: dict[str, pd.DataFrame] | None = None,
    cache_dir: Path = DEFAULT_FUNDAMENTAL_CACHE,
    allow_network: bool = True,
    strict: bool = True,
    audit_log: Path = DEFAULT_FUNDAMENTAL_AUDIT_LOG,
) -> FundamentalBatchResult:
    from fundamental_factor import score_point_in_time_fundamentals

    cutoff = normalize_date(as_of_date)
    price_histories = price_histories or {}
    result = FundamentalBatchResult()
    for code in codes:
        try:
            record = get_fundamentals(
                code,
                cutoff,
                price_history=price_histories.get(code),
                cache_dir=cache_dir,
                allow_network=allow_network,
                strict=strict,
            )
        except FundamentalFutureDataError:
            result.future_records += 1
            result.validation_passed = False
            if strict:
                raise
            continue
        if record is None:
            result.missing_codes.append(normalize_code(code))
            continue
        scored = score_point_in_time_fundamentals(record)
        if scored.fundamental_score is None:
            result.missing_codes.append(normalize_code(code))
            continue
        result.records[record.code] = record
        result.scores[record.code] = scored.fundamental_score
        if record.is_revised:
            result.revised_codes.append(record.code)
    validation = validate_records(
        [record.to_dict() for record in result.records.values()],
        cutoff,
        audit_log=audit_log,
        strict=strict,
    )
    result.validation_passed = result.validation_passed and validation.valid
    result.future_records += validation.future_records
    return result


def previous_period_record(visible: pd.DataFrame, report_period: str) -> dict[str, Any] | None:
    current = pd.to_datetime(report_period)
    target = (current - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    matches = visible[visible["report_period"].astype(str).eq(target)]
    if matches.empty:
        return None
    return matches.sort_values("disclosure_date").iloc[-1].to_dict()


def trailing_twelve_month_profit(visible: pd.DataFrame, report_period: str) -> float | None:
    current_date = pd.to_datetime(report_period)
    current = record_for_period(visible, report_period)
    current_profit = safe_number(current.get("net_profit")) if current else None
    if current_profit is None:
        return None
    if current_date.month == 12:
        return current_profit
    previous_annual_period = f"{current_date.year - 1}-12-31"
    previous_same_period = (current_date - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    previous_annual = record_for_period(visible, previous_annual_period)
    previous_same = record_for_period(visible, previous_same_period)
    annual_profit = safe_number(previous_annual.get("net_profit")) if previous_annual else None
    previous_ytd_profit = safe_number(previous_same.get("net_profit")) if previous_same else None
    if annual_profit is None or previous_ytd_profit is None:
        return None
    return annual_profit + current_profit - previous_ytd_profit


def record_for_period(visible: pd.DataFrame, report_period: str) -> dict[str, Any] | None:
    matches = visible[visible["report_period"].astype(str).eq(report_period)]
    if matches.empty:
        return None
    return matches.sort_values("disclosure_date").iloc[-1].to_dict()


def annualized_roe(
    net_profit: float | None,
    equity: float | None,
    previous_equity: float | None,
    report_period: str,
) -> float | None:
    if net_profit is None or equity in (None, 0):
        return None
    average_equity = (equity + previous_equity) / 2 if previous_equity not in (None, 0) else equity
    month = pd.to_datetime(report_period).month
    annualizer = {3: 4.0, 6: 2.0, 9: 4.0 / 3.0, 12: 1.0}.get(month, 1.0)
    return ratio(net_profit * annualizer, average_equity, percent=True)


def growth_rate(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in (None, 0):
        return None
    return (current / abs(previous) - 1) * 100


def ratio(numerator: float | None, denominator: float | None, percent: bool = False) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    value = numerator / denominator
    return value * 100 if percent else value


def close_as_of(history: pd.DataFrame | None, as_of_date: str) -> float | None:
    if history is None or history.empty:
        return None
    date_column = "date" if "date" in history.columns else "日期" if "日期" in history.columns else None
    close_column = "close" if "close" in history.columns else "收盘" if "收盘" in history.columns else None
    if not date_column or not close_column:
        return None
    data = history.copy()
    dates = pd.to_datetime(data[date_column], errors="coerce")
    data = data[dates <= pd.to_datetime(as_of_date)].copy()
    if data.empty:
        return None
    closes = pd.to_numeric(data[close_column], errors="coerce").dropna()
    return safe_number(closes.iloc[-1]) if not closes.empty else None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}
