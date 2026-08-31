from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from factor_engine import history_until


@dataclass(frozen=True)
class ResearchRiskDecision:
    allowed: bool
    reason: str
    risk_tags: list[str] = field(default_factory=list)


def evaluate_research_risk(
    history: pd.DataFrame,
    as_of_date: str,
    max_20d_volatility: float = 0.12,
    max_60d_drawdown: float = -0.35,
    min_avg_amount: float = 20_000_000,
) -> ResearchRiskDecision:
    """Risk gate for backtests. Every metric is limited to the signal date."""
    window = history_until(history, as_of_date)
    if len(window) < 60:
        return ResearchRiskDecision(False, "历史数据不足，风险过滤不放行", ["历史数据不足"])
    close = pd.to_numeric(window["close"], errors="coerce").dropna()
    if close.empty or float(close.iloc[-1]) <= 0:
        return ResearchRiskDecision(False, "价格数据异常", ["价格异常"])
    returns = close.pct_change().dropna().tail(20)
    volatility = float(returns.std()) if len(returns) >= 10 else 0.0
    recent = close.tail(60)
    drawdown = float((recent / recent.cummax() - 1).min()) if not recent.empty else 0.0
    raw_amount = window["amount"] if "amount" in window.columns else pd.Series(dtype=float)
    amount = pd.to_numeric(raw_amount, errors="coerce").dropna().tail(20)
    avg_amount = float(amount.mean()) if not amount.empty else None
    tags: list[str] = []
    if volatility > max_20d_volatility:
        tags.append("短期波动过高")
    if drawdown < max_60d_drawdown:
        tags.append("近60日回撤过大")
    if avg_amount is not None and avg_amount < min_avg_amount:
        tags.append("流动性不足")
    if tags:
        return ResearchRiskDecision(False, "；".join(tags), tags)
    return ResearchRiskDecision(True, "通过波动、回撤和流动性风险过滤")
