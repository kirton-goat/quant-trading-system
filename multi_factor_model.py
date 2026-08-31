from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from event_factor import EventFactor
from factor_engine import FactorResult, fetch_history, weighted_average
from fundamental_factor import FundamentalFactor
from market_factor import MarketRegimeFactor
from momentum_factor import MomentumFactor
from money_flow_factor import MoneyFlowFactor
from technical_factor import TechnicalFactor


DEFAULT_WEIGHTS = {
    "market_regime": 0.20,
    "momentum": 0.25,
    "money_flow": 0.20,
    "fundamental": 0.25,
    "technical": 0.10,
    "event": 0.0,
}


@dataclass
class MultiFactorDecision:
    code: str
    total_score: float
    factors: list[FactorResult]
    risk_tags: list[str] = field(default_factory=list)
    allowed_to_trade: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "total_score": self.total_score,
            "allowed_to_trade": self.allowed_to_trade,
            "reason": self.reason,
            "risk_tags": self.risk_tags,
            "factors": [factor.to_dict() for factor in self.factors],
        }

    def factor_score(self, factor_name: str) -> float:
        for factor in self.factors:
            if factor.factor_name == factor_name:
                return factor.score
        return 0.0


class MultiFactorModel:
    def __init__(self, weights: dict[str, float] | None = None, event_boost_weight: float = 0.0) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.event_boost_weight = max(0.0, min(float(event_boost_weight), 0.05))
        self.factors = [
            MomentumFactor(),
            MoneyFlowFactor(),
            EventFactor(),
            TechnicalFactor(),
            FundamentalFactor(),
            MarketRegimeFactor(),
        ]
        for factor in self.factors:
            factor.weight = self.weights.get(factor.name, factor.weight)

    def evaluate(
        self,
        code: str,
        history: pd.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> MultiFactorDecision:
        context = context or {}
        shared_history = history if history is not None else fetch_history(code)
        results = [factor.calculate(code, shared_history, context) for factor in self.factors]
        base_results = [result for result in results if result.factor_name != "event"]
        total_score = weighted_average(base_results)
        event_score = next((result.score for result in results if result.factor_name == "event"), 50.0)
        total_score = max(0.0, min(100.0, round(total_score + (event_score - 50.0) * self.event_boost_weight, 2)))
        risks = collect_risks(results)
        allowed, reason = evaluate_trade_gate(total_score, results)
        return MultiFactorDecision(
            code=code,
            total_score=total_score,
            factors=results,
            risk_tags=risks,
            allowed_to_trade=allowed,
            reason=reason,
        )


def evaluate_trade_gate(total_score: float, factors: list[FactorResult]) -> tuple[bool, str]:
    scores = {factor.factor_name: factor.score for factor in factors}
    market_score = scores.get("market_regime", 50)
    if total_score <= 80:
        return False, f"综合评分{total_score}未超过80"
    if market_score < 45:
        return False, f"市场环境因子{market_score:.1f}偏弱"
    if has_blocking_risk(factors):
        return False, "存在高位、重复消息或市场风险等阻断标签"
    return True, "满足量化评分、市场环境和风险过滤条件；事件仅可辅助加减分"


def has_blocking_risk(factors: list[FactorResult]) -> bool:
    blocking = {"利好兑现风险", "中期高位风险", "消息重复风险", "市场风险偏好不足", "放量下跌风险"}
    return any(risk in blocking for factor in factors for risk in factor.risk_tags)


def collect_risks(results: list[FactorResult]) -> list[str]:
    seen: set[str] = set()
    risks: list[str] = []
    for result in results:
        for risk in result.risk_tags:
            if risk not in seen:
                risks.append(risk)
                seen.add(risk)
    return risks
