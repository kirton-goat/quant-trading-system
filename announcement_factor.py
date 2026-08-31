from __future__ import annotations

from typing import Any

import pandas as pd

from event_research_data import event_scores_as_of
from factor_engine import FactorResult


class AnnouncementFactor:
    """Company announcements are retained as an auditable auxiliary factor."""

    name = "announcement"
    weight = 0.0

    def calculate(self, code: str, history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        context = context or {}
        as_of_date = str(context.get("as_of_date", ""))
        score, stats = event_scores_as_of(as_of_date, [code])
        return FactorResult(
            self.name,
            score[code],
            self.weight,
            "公告只作为0-5%辅助研究信息，不可单独触发交易",
            details={"announcement_event_rows": stats.announcement_rows, "as_of_date": as_of_date},
        )
