from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import akshare as ak
import pandas as pd

from factor_engine import FactorResult, clamp_score, safe_float


@dataclass
class MarketRegime:
    regime: str
    score: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    risk_tags: list[str] = field(default_factory=list)

    @property
    def risk_on(self) -> bool:
        return self.regime == "risk_on"


class MarketRegimeFactor:
    name = "market_regime"
    weight = 0.05

    def calculate(self, code: str = "", history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        regime = evaluate_market_regime()
        return FactorResult(
            self.name,
            regime.score,
            self.weight,
            regime.reason,
            details={**regime.details, "regime": regime.regime},
            risk_tags=regime.risk_tags,
        )


def evaluate_market_regime() -> MarketRegime:
    indexes = {
        "上证指数": "000001",
        "沪深300": "000300",
        "中证500": "000905",
        "创业板": "399006",
    }
    details: dict[str, Any] = {}
    scores: list[float] = []

    for name, code in indexes.items():
        trend_score, metrics = fetch_index_trend(code)
        if trend_score is not None:
            scores.append(trend_score)
        details[name] = metrics

    market_amount = fetch_market_amount()
    if market_amount is not None:
        details["市场成交额"] = market_amount
        if market_amount >= 9000:
            scores.append(70)
        elif market_amount < 6500:
            scores.append(42)

    if not scores:
        return MarketRegime("neutral", 50, "市场环境数据不足，默认中性", details)

    score = round(sum(scores) / len(scores), 2)
    risks: list[str] = []
    if score >= 62:
        regime = "risk_on"
        reason = "主要指数趋势和成交额支持风险偏好"
    elif score <= 45:
        regime = "risk_off"
        reason = "市场环境偏弱，降低交易强度"
        risks.append("市场风险偏好不足")
    else:
        regime = "neutral"
        reason = "市场环境中性"
    return MarketRegime(regime, clamp_score(score), reason, details, risks)


def fetch_index_trend(index_code: str) -> tuple[float | None, dict[str, Any]]:
    end_date = dt.datetime.now().strftime("%Y%m%d")
    start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_index_daily_em(symbol=index_code)
    except Exception:
        return None, {"error": "指数接口不可用"}
    if df is None or df.empty or len(df) < 60:
        return None, {"error": "指数数据不足"}
    df = df.tail(120).copy()
    close_col = "close" if "close" in df.columns else "收盘"
    closes = df[close_col]
    latest = safe_float(closes.iloc[-1])
    ma20 = safe_float(closes.tail(20).mean())
    ma60 = safe_float(closes.tail(60).mean())
    pct20 = round((latest - safe_float(closes.iloc[-21])) / safe_float(closes.iloc[-21]) * 100, 2) if latest and safe_float(closes.iloc[-21]) else None
    score = 50.0
    if latest and ma20 and latest > ma20:
        score += 10
    if ma20 and ma60 and ma20 > ma60:
        score += 10
    if pct20 is not None:
        score += max(-15, min(15, pct20 * 1.5))
    return clamp_score(score), {"pct20": pct20, "above_ma20": latest > ma20 if latest and ma20 else None, "ma20_above_ma60": ma20 > ma60 if ma20 and ma60 else None}


def fetch_market_amount() -> float | None:
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception:
        return None
    if df is None or df.empty or "成交额" not in df.columns:
        return None
    amount = pd.to_numeric(df["成交额"], errors="coerce").sum()
    return round(float(amount) / 100000000, 2)
