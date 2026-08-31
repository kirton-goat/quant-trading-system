from __future__ import annotations

from typing import Any

import pandas as pd

from factor_engine import FactorResult, clamp_score, ensure_history, safe_float


class MomentumFactor:
    name = "momentum"
    weight = 0.25

    def calculate(self, code: str, history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        df = ensure_history(code, history)
        if df is None or len(df) < 60:
            return FactorResult(self.name, 50, self.weight, "历史行情不足，动量因子中性")

        closes = df["收盘"]
        latest = safe_float(closes.iloc[-1])
        returns = {
            "5日收益率": pct_return(closes, 5),
            "20日收益率": pct_return(closes, 20),
            "60日收益率": pct_return(closes, 60),
            "120日收益率": pct_return(closes, 120),
        }
        ma20 = closes.rolling(20).mean().iloc[-1]
        ma60 = closes.rolling(60).mean().iloc[-1]
        recent_high = closes.tail(60).max()
        breakout = bool(latest is not None and recent_high and latest >= recent_high * 0.995)

        score = 45.0
        score += score_return(returns["5日收益率"], -8, 10) * 0.12
        score += score_return(returns["20日收益率"], -15, 25) * 0.22
        score += score_return(returns["60日收益率"], -25, 45) * 0.24
        score += score_return(returns["120日收益率"], -35, 70) * 0.12
        if latest is not None and latest > ma20:
            score += 8
        if latest is not None and latest > ma60:
            score += 8
        if ma20 > ma60:
            score += 6
        if breakout:
            score += 8

        risks: list[str] = []
        if returns["20日收益率"] is not None and returns["20日收益率"] > 35:
            risks.append("短期涨幅过高")
        if returns["60日收益率"] is not None and returns["60日收益率"] > 70:
            risks.append("中期高位风险")

        reason = "趋势向上" if score >= 70 else "趋势一般" if score >= 50 else "趋势偏弱"
        return FactorResult(
            self.name,
            clamp_score(score),
            self.weight,
            reason,
            details={**returns, "MA20趋势": latest > ma20 if latest is not None else None, "MA60趋势": latest > ma60 if latest is not None else None, "突破近期高点": breakout},
            risk_tags=risks,
        )


def pct_return(closes: pd.Series, days: int) -> float | None:
    if len(closes) <= days:
        return None
    base = safe_float(closes.iloc[-days - 1])
    latest = safe_float(closes.iloc[-1])
    if base in (None, 0) or latest is None:
        return None
    return round((latest - base) / base * 100, 2)


def score_return(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0
    return clamp_score((value - low) / (high - low) * 20 - 10, -10, 20)
