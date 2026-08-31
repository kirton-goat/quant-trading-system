from __future__ import annotations

from typing import Any

import pandas as pd

from factor_engine import FactorResult, clamp_score
from market import MarketSnapshot, NewsItem
from news_quality_filter import NewsQualityFilter, NewsQualityReport


class EventFactor:
    """Turn only policy/announcement inputs into a bounded research signal.

    Ordinary finance news is deliberately neutral. It remains available for
    sentiment observation and human research, but cannot affect trading score.
    """

    name = "event"
    weight = 0.0

    def calculate(self, code: str, history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        context = context or {}
        quality = context.get("news_quality")
        news = context.get("news")
        snapshot = context.get("snapshot")
        event_type = str(context.get("event_type", ""))

        if event_type and not event_type.startswith(("政策信息/", "公司公告/")):
            return FactorResult(
                self.name,
                50,
                self.weight,
                "普通财经新闻仅用于市场观察和研究记录，不进入交易评分",
                details={"event_type": event_type, "research_only": True},
            )

        if not isinstance(quality, NewsQualityReport):
            if isinstance(news, NewsItem):
                quality = NewsQualityFilter(context.get("log_file")).evaluate(
                    news,
                    mapped_target=code,
                    snapshot=snapshot if isinstance(snapshot, MarketSnapshot) else None,
                )
            else:
                return FactorResult(self.name, 50, self.weight, "无政策或公告事件，事件因子中性")

        score = quality.trade_value_score * 100
        if quality.source_level <= 2:
            return FactorResult(
                self.name,
                50,
                self.weight,
                "Level 1-2 普通财经信息不作为事件因子",
                details={"source_level": quality.source_level, "research_only": True},
                risk_tags=quality.risk,
            )
        if quality.is_repeat:
            score -= 18
        if not quality.has_financial_impact and quality.source_level < 5:
            score -= 10
        if quality.market_already_priced:
            score -= 15

        reason = (
            f"来源L{quality.source_level}，新颖性{quality.novelty_score:.2f}，"
            f"基本面{quality.fundamental_score:.2f}，市场反应{quality.market_reaction_score:.2f}"
        )
        return FactorResult(
            self.name,
            clamp_score(score),
            self.weight,
            reason,
            details=quality.to_dict(),
            risk_tags=quality.risk,
        )
