from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from factor_engine import FactorResult, clamp_score, safe_float
from fundamental_check import check_fundamentals


class FundamentalFactor:
    name = "fundamental"
    weight = 0.15

    def calculate(self, code: str, history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        context = context or {}
        point_in_time = context.get("point_in_time_fundamentals")
        if point_in_time is not None:
            scored = score_point_in_time_fundamentals(point_in_time)
            if scored.fundamental_score is None:
                return FactorResult(
                    self.name,
                    0.0,
                    self.weight,
                    "missing_fundamental_data",
                    details=scored.to_dict(),
                    risk_tags=["missing_fundamental_data"],
                )
            return FactorResult(
                self.name,
                scored.fundamental_score,
                self.weight,
                scored.reason,
                details=scored.to_dict(),
                risk_tags=scored.risk_tags,
            )
        if context.get("formal_backtest"):
            return FactorResult(
                self.name,
                0.0,
                self.weight,
                "正式回测缺少历史时点基本面，禁止使用最新数据或中性50分。",
                details={"status": "missing_fundamental_data"},
                risk_tags=["missing_fundamental_data"],
            )
        news_text = context.get("news_text", "")
        result = check_fundamentals(code, news_text=news_text, timeout=context.get("fundamental_timeout", 12))

        score = result.score * 100
        roe = safe_float(result.roe)
        net_profit = safe_float(result.net_profit)
        if roe is not None:
            if roe >= 12:
                score += 10
            elif roe < 3:
                score -= 10
        if net_profit is not None and net_profit <= 0:
            score -= 12

        reason = result.note or "基本面数据接口预留，当前使用中性评分"
        return FactorResult(
            self.name,
            clamp_score(score),
            self.weight,
            reason,
            details={
                "PE": context.get("pe"),
                "PB": context.get("pb"),
                "ROE": result.roe,
                "营收": result.revenue,
                "净利润": result.net_profit,
                "毛利率": result.gross_margin,
                "营收增长": context.get("revenue_growth"),
                "净利润增长": context.get("net_profit_growth"),
                "现金流": context.get("cash_flow"),
                "business_relevance": result.business_relevance,
            },
            risk_tags=result.risk_tags,
        )


@dataclass
class FundamentalScoreResult:
    quality_score: float | None
    growth_score: float | None
    valuation_score: float | None
    cashflow_score: float | None
    fundamental_score: float | None
    reason: str
    report_period: str = ""
    disclosure_date: str = ""
    risk_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "growth_score": self.growth_score,
            "valuation_score": self.valuation_score,
            "cashflow_score": self.cashflow_score,
            "fundamental_score": self.fundamental_score,
            "report_period": self.report_period,
            "disclosure_date": self.disclosure_date,
            "reason": self.reason,
            "risk_tags": self.risk_tags,
        }


def score_point_in_time_fundamentals(data: Any) -> FundamentalScoreResult:
    get = data.get if isinstance(data, dict) else lambda name, default=None: getattr(data, name, default)
    if not get("is_point_in_time", False) or not get("disclosure_date"):
        return FundamentalScoreResult(None, None, None, None, None, "missing_fundamental_data")

    quality_components = compact_scores(
        scale(get("roe"), -5, 25),
        scale(get("gross_margin"), 0, 60),
        inverse_scale(get("debt_to_assets"), 20, 90),
    )
    growth_components = compact_scores(
        scale(get("revenue_growth"), -30, 50),
        scale(get("net_profit_growth"), -30, 50),
    )
    cashflow_components = compact_scores(
        scale(get("operating_cash_flow_to_net_profit"), -0.5, 1.5),
        70.0 if numeric(get("operating_cash_flow")) is not None and numeric(get("operating_cash_flow")) > 0 else 20.0
        if numeric(get("operating_cash_flow")) is not None
        else None,
    )
    valuation_components = compact_scores(
        inverse_scale(get("pe"), 5, 60, require_positive=True),
        inverse_scale(get("pb"), 0.5, 8, require_positive=True),
    )
    quality = average(quality_components)
    growth = average(growth_components)
    cashflow = average(cashflow_components)
    valuation = average(valuation_components)
    available = [item for item in (quality, growth, cashflow, valuation) if item is not None]
    if quality is None or len(available) < 2:
        return FundamentalScoreResult(
            quality,
            growth,
            valuation,
            cashflow,
            None,
            "missing_fundamental_data：基本面子因子覆盖不足",
            str(get("report_period") or ""),
            str(get("disclosure_date") or ""),
            ["missing_fundamental_data"],
        )
    weighted = [(quality, 0.30), (growth, 0.25), (cashflow, 0.20), (valuation, 0.25)]
    usable = [(score, weight) for score, weight in weighted if score is not None]
    total_weight = sum(weight for _, weight in usable)
    fundamental_score = round(sum(score * weight for score, weight in usable) / total_weight, 2)
    risks: list[str] = []
    if get("is_revised", False):
        risks.append("revised_statement_bias")
    return FundamentalScoreResult(
        round(quality, 2) if quality is not None else None,
        round(growth, 2) if growth is not None else None,
        round(valuation, 2) if valuation is not None else None,
        round(cashflow, 2) if cashflow is not None else None,
        fundamental_score,
        f"使用{get('report_period')}财报，披露日{get('disclosure_date')}，仅采用回测日已公开数据。",
        str(get("report_period") or ""),
        str(get("disclosure_date") or ""),
        risks,
    )


def numeric(value: Any) -> float | None:
    return safe_float(value)


def scale(value: Any, low: float, high: float) -> float | None:
    number = numeric(value)
    if number is None or high <= low:
        return None
    return clamp_score((number - low) / (high - low) * 100)


def inverse_scale(value: Any, low: float, high: float, require_positive: bool = False) -> float | None:
    number = numeric(value)
    if number is None or (require_positive and number <= 0) or high <= low:
        return None
    return clamp_score(100 - (number - low) / (high - low) * 100)


def compact_scores(*values: float | None) -> list[float]:
    return [value for value in values if value is not None]


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
