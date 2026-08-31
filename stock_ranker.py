from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from multi_factor_model import MultiFactorDecision, MultiFactorModel


@dataclass
class RankedStock:
    rank: int
    code: str
    total_score: float
    factor_contributions: dict[str, float]
    factor_scores: dict[str, float]
    risk_tags: list[str] = field(default_factory=list)
    allowed_to_trade: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "code": self.code,
            "total_score": self.total_score,
            "factor_contributions": self.factor_contributions,
            "factor_scores": self.factor_scores,
            "risk_tags": self.risk_tags,
            "allowed_to_trade": self.allowed_to_trade,
            "reason": self.reason,
        }


class StockRanker:
    def __init__(self, model: MultiFactorModel | None = None) -> None:
        self.model = model or MultiFactorModel()

    def rank(
        self,
        stock_codes: list[str],
        top_n: int = 20,
        context_by_code: dict[str, dict[str, Any]] | None = None,
    ) -> list[RankedStock]:
        context_by_code = context_by_code or {}
        decisions: list[MultiFactorDecision] = []
        for code in stock_codes:
            if not is_stock_code(code):
                continue
            context = context_by_code.get(code, {})
            decisions.append(self.model.evaluate(code, context=context))

        decisions.sort(key=lambda item: item.total_score, reverse=True)
        return [to_ranked_stock(idx + 1, decision) for idx, decision in enumerate(decisions[:top_n])]


def to_ranked_stock(rank: int, decision: MultiFactorDecision) -> RankedStock:
    return RankedStock(
        rank=rank,
        code=decision.code,
        total_score=decision.total_score,
        factor_contributions={factor.factor_name: round(factor.contribution(), 2) for factor in decision.factors},
        factor_scores={factor.factor_name: round(factor.score, 2) for factor in decision.factors},
        risk_tags=decision.risk_tags,
        allowed_to_trade=decision.allowed_to_trade,
        reason=decision.reason,
    )


def is_stock_code(code: str) -> bool:
    return code.isdigit() and len(code) == 6
