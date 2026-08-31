from __future__ import annotations

from typing import Any

import pandas as pd

from factor_engine import FactorResult, clamp_score, ensure_history, safe_float
from market import calculate_kdj


class TechnicalFactor:
    name = "technical"
    weight = 0.15

    def calculate(self, code: str, history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        df = ensure_history(code, history)
        if df is None or len(df) < 35:
            return FactorResult(self.name, 50, self.weight, "技术数据不足，技术因子中性")

        df = df.copy()
        close = df["收盘"]
        df["MA5"] = close.rolling(5).mean()
        df["MA20"] = close.rolling(20).mean()
        df["MA60"] = close.rolling(60).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["DIF"] = ema12 - ema26
        df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
        df["MACD"] = 2 * (df["DIF"] - df["DEA"])
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        df["BOLL_MID"] = close.rolling(20).mean()
        df["BOLL_STD"] = close.rolling(20).std()
        df["BOLL_UP"] = df["BOLL_MID"] + 2 * df["BOLL_STD"]
        df["BOLL_LOW"] = df["BOLL_MID"] - 2 * df["BOLL_STD"]
        df = calculate_kdj(df)

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        score = 50.0
        signals: list[str] = []
        risks: list[str] = []

        if latest["MA5"] > latest["MA20"]:
            score += 10
            signals.append("MA多头")
        else:
            score -= 8
        if len(df) >= 60 and latest["MA20"] > latest["MA60"]:
            score += 8
            signals.append("MA20上穿MA60结构")
        if latest["DIF"] > latest["DEA"]:
            score += 10
            signals.append("MACD多头")
        if latest["MACD"] > prev["MACD"]:
            score += 6
            signals.append("MACD动能改善")
        rsi = safe_float(latest["RSI"])
        if rsi is not None:
            if 45 <= rsi <= 70:
                score += 8
            elif rsi > 82:
                score -= 14
                risks.append("RSI超买")
            elif rsi < 30:
                score -= 6
                risks.append("弱势超跌")
        if latest["K"] > latest["D"]:
            score += 5
            signals.append("KDJ偏强")
        if latest["收盘"] > latest["BOLL_UP"]:
            score -= 8
            risks.append("突破布林上轨后回落风险")
        elif latest["收盘"] > latest["BOLL_MID"]:
            score += 5

        reason = "技术状态偏强" if score >= 70 else "技术状态中性" if score >= 50 else "技术状态偏弱"
        return FactorResult(
            self.name,
            clamp_score(score),
            self.weight,
            reason,
            details={"signals": signals, "RSI": round(rsi, 2) if rsi is not None else None},
            risk_tags=risks,
        )
