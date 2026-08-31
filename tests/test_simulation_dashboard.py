from __future__ import annotations

from dashboard.api.repository import _normalise_sim_status
from market import NewsItem
from simulation_engine import build_simulated_trade_plan


def _news() -> NewsItem:
    return NewsItem(
        title="测试事件",
        content="仅用于状态分类测试",
        published_at="2026-08-09 10:00:00",
        source="测试源",
    )


def test_research_observation_is_not_market_data_failure() -> None:
    plan = build_simulated_trade_plan(_news(), "600000", None, "观望")

    assert plan.sim_status == "research_only"
    assert plan.sim_result == "no_trade"
    assert "多因子交易许可" in plan.sim_note


def test_trade_intent_without_price_is_market_data_failure() -> None:
    plan = build_simulated_trade_plan(_news(), "600000", None, "买入")

    assert plan.sim_status == "market_data_missing"
    assert plan.sim_direction == "long"
    assert plan.sim_result == "data_error"


def test_legacy_no_trade_row_is_normalised_as_research_only() -> None:
    row = {"sim_status": "not_opened", "sim_direction": "none"}

    assert _normalise_sim_status(row) == "research_only"


def test_legacy_trade_intent_without_open_is_market_data_missing() -> None:
    row = {"sim_status": "not_opened", "sim_direction": "long"}

    assert _normalise_sim_status(row) == "market_data_missing"
