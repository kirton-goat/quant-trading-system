from __future__ import annotations

from typing import Any

import pandas as pd

from factor_engine import FactorResult, clamp_score, ensure_history, safe_float


class MoneyFlowFactor:
    name = "money_flow"
    weight = 0.20

    def calculate(self, code: str, history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        df = ensure_history(code, history)
        if df is None or len(df) < 25:
            return FactorResult(self.name, 50, self.weight, "历史成交数据不足，资金因子中性")

        latest = df.iloc[-1]
        close = safe_float(latest.get("收盘"))
        prev_close = safe_float(df.iloc[-2].get("收盘"))
        volume = safe_float(latest.get("成交量"))
        amount = safe_float(latest.get("成交额"))
        turnover = safe_float(latest.get("换手率"))
        vol_ma5 = safe_float(df["成交量"].tail(5).mean())
        vol_ma20 = safe_float(df["成交量"].tail(20).mean())
        amount_ma20 = safe_float(df["成交额"].tail(20).mean()) if "成交额" in df.columns else None
        volume_ratio = round(vol_ma5 / vol_ma20, 2) if vol_ma20 else None
        amount_ratio = round(amount / amount_ma20, 2) if amount_ma20 and amount else None
        price_up = close is not None and prev_close is not None and close > prev_close

        score = 50.0
        if volume_ratio is not None:
            score += (volume_ratio - 1) * 18
        if amount_ratio is not None:
            score += (amount_ratio - 1) * 12
        if turnover is not None:
            if 2 <= turnover <= 12:
                score += 10
            elif turnover > 20:
                score -= 12
        if price_up and volume_ratio is not None and volume_ratio >= 1.2:
            score += 12
        if not price_up and volume_ratio is not None and volume_ratio >= 1.8:
            score -= 15

        risks: list[str] = []
        if turnover is not None and turnover > 20:
            risks.append("高换手博弈风险")
        if not price_up and volume_ratio is not None and volume_ratio >= 1.8:
            risks.append("放量下跌风险")

        reason = "资金确认较好" if score >= 70 else "资金确认一般" if score >= 50 else "资金状态偏弱"
        return FactorResult(
            self.name,
            clamp_score(score),
            self.weight,
            reason,
            details={
                "成交量": volume,
                "成交额": amount,
                "成交量变化": volume_ratio,
                "成交额变化": amount_ratio,
                "换手率": turnover,
                "量价关系": "价涨量增" if price_up and volume_ratio and volume_ratio >= 1.2 else "未确认",
            },
            risk_tags=risks,
        )
