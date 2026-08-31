from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from factor_engine import fetch_history, safe_float


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG = BASE_DIR / "logs" / "trading_log.csv"


@dataclass
class FactorObservation:
    date: str
    code: str
    factor_name: str
    factor_score: float
    forward_return: float


@dataclass
class FactorTestResult:
    factor_name: str
    observations: int
    average_forward_return: float | None
    top_quantile_return: float | None
    bottom_quantile_return: float | None
    spread_return: float | None
    win_rate: float | None
    note: str = ""
    details: dict[str, Any] = field(default_factory=dict)


FACTOR_NAMES = ("momentum", "money_flow", "fundamental")


def load_stock_codes_from_log(log_file: Path = DEFAULT_LOG, limit: int = 200) -> list[str]:
    if not log_file.exists():
        return []
    codes: list[str] = []
    try:
        with log_file.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                code = (row.get("关联股票") or "").strip()
                if code.isdigit() and len(code) == 6 and code not in codes:
                    codes.append(code)
                if len(codes) >= limit:
                    break
    except Exception:
        return []
    return codes


def parse_codes(text: str | None) -> list[str]:
    if not text:
        return []
    result: list[str] = []
    for raw in text.replace("，", ",").replace(" ", ",").split(","):
        code = raw.strip()
        if code.isdigit() and len(code) == 6 and code not in result:
            result.append(code)
    return result


def run_factor_tests(
    stock_codes: list[str],
    holding_days: int = 20,
    history_days: int = 260,
    sample_step: int = 5,
) -> tuple[list[FactorTestResult], list[FactorObservation]]:
    observations: list[FactorObservation] = []
    for code in stock_codes:
        history = fetch_history(code, days=history_days)
        if history is None or history.empty:
            continue
        observations.extend(build_observations_for_stock(code, history, holding_days=holding_days, sample_step=sample_step))

    results = [summarize_factor(name, observations) for name in FACTOR_NAMES]
    return results, observations


def build_observations_for_stock(
    code: str,
    history: pd.DataFrame,
    holding_days: int = 20,
    sample_step: int = 5,
) -> list[FactorObservation]:
    df = normalize_history(history)
    if df is None or len(df) < max(80, holding_days + 61):
        return []

    items: list[FactorObservation] = []
    start_index = 60
    end_index = len(df) - holding_days - 1
    for index in range(start_index, end_index + 1, max(1, sample_step)):
        window = df.iloc[: index + 1]
        date = str(df.iloc[index]["日期"])
        forward_return = calculate_forward_return(df, index, holding_days)
        if forward_return is None:
            continue
        scores = {
            "momentum": calculate_momentum_score(window),
            "money_flow": calculate_money_flow_score(window),
            "fundamental": 50.0,
        }
        for name, score in scores.items():
            items.append(
                FactorObservation(
                    date=date,
                    code=code,
                    factor_name=name,
                    factor_score=score,
                    forward_return=forward_return,
                )
            )
    return items


def summarize_factor(factor_name: str, observations: list[FactorObservation]) -> FactorTestResult:
    rows = [item for item in observations if item.factor_name == factor_name]
    if not rows:
        return FactorTestResult(
            factor_name=factor_name,
            observations=0,
            average_forward_return=None,
            top_quantile_return=None,
            bottom_quantile_return=None,
            spread_return=None,
            win_rate=None,
            note="没有可测试样本",
        )

    df = pd.DataFrame([item.__dict__ for item in rows])
    avg = round(float(df["forward_return"].mean()), 4)
    win_rate = round(float((df["forward_return"] > 0).mean()), 4)
    top = quantile_return(df, high=True)
    bottom = quantile_return(df, high=False)
    spread = round(top - bottom, 4) if top is not None and bottom is not None else None
    note = "高分组优于低分组" if spread is not None and spread > 0 else "暂未看到高分组优势"
    return FactorTestResult(
        factor_name=factor_name,
        observations=len(rows),
        average_forward_return=avg,
        top_quantile_return=top,
        bottom_quantile_return=bottom,
        spread_return=spread,
        win_rate=win_rate,
        note=note,
        details={"holding_days": int(df.attrs.get("holding_days", 0) or 0)},
    )


def quantile_return(df: pd.DataFrame, high: bool) -> float | None:
    if df.empty:
        return None
    q = 0.8 if high else 0.2
    threshold = df["factor_score"].quantile(q)
    part = df[df["factor_score"] >= threshold] if high else df[df["factor_score"] <= threshold]
    if part.empty:
        return None
    return round(float(part["forward_return"].mean()), 4)


def normalize_history(history: pd.DataFrame) -> pd.DataFrame | None:
    required = {"日期", "收盘", "成交量"}
    if not required.issubset(set(history.columns)):
        return None
    df = history.copy()
    for column in ("收盘", "开盘", "最高", "最低", "成交量", "成交额", "换手率"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["收盘", "成交量"])
    if df.empty:
        return None
    return df.sort_values("日期").reset_index(drop=True)


def calculate_forward_return(df: pd.DataFrame, index: int, holding_days: int) -> float | None:
    entry = safe_float(df.iloc[index]["收盘"])
    exit_index = index + holding_days
    if entry in (None, 0) or exit_index >= len(df):
        return None
    exit_price = safe_float(df.iloc[exit_index]["收盘"])
    if exit_price is None:
        return None
    return round((exit_price - entry) / entry * 100, 4)


def calculate_momentum_score(window: pd.DataFrame) -> float:
    closes = window["收盘"]
    score = 50.0
    pct20 = pct_return(closes, 20)
    pct60 = pct_return(closes, 60)
    ma20 = closes.tail(20).mean()
    ma60 = closes.tail(60).mean()
    latest = closes.iloc[-1]
    if pct20 is not None:
        score += max(-18, min(18, pct20 * 1.2))
    if pct60 is not None:
        score += max(-18, min(18, pct60 * 0.6))
    if latest > ma20:
        score += 8
    if ma20 > ma60:
        score += 8
    return clamp(score)


def calculate_money_flow_score(window: pd.DataFrame) -> float:
    if len(window) < 25:
        return 50.0
    latest = window.iloc[-1]
    score = 50.0
    vol5 = window["成交量"].tail(5).mean()
    vol20 = window["成交量"].tail(20).mean()
    if vol20:
        score += max(-18, min(18, (vol5 / vol20 - 1) * 22))
    if "成交额" in window.columns:
        amount = safe_float(latest.get("成交额"))
        amount20 = window["成交额"].tail(20).mean()
        if amount and amount20:
            score += max(-12, min(12, (amount / amount20 - 1) * 12))
    turnover = safe_float(latest.get("换手率"))
    if turnover is not None:
        if 2 <= turnover <= 12:
            score += 8
        elif turnover > 20:
            score -= 10
    return clamp(score)


def pct_return(closes: pd.Series, days: int) -> float | None:
    if len(closes) <= days:
        return None
    base = safe_float(closes.iloc[-days - 1])
    latest = safe_float(closes.iloc[-1])
    if base in (None, 0) or latest is None:
        return None
    return (latest - base) / base * 100


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))
