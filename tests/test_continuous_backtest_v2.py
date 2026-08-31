from __future__ import annotations

import pandas as pd

from research.continuous_backtest_v2 import calculate_turnover, validate_continuous_timeline


def test_equal_weight_turnover_only_trades_replaced_positions() -> None:
    old = [f"A{i}" for i in range(20)]
    new = old[:12] + [f"B{i}" for i in range(8)]
    buy, sell = calculate_turnover(old, new)
    assert round(buy, 8) == 0.4
    assert round(sell, 8) == 0.4


def test_initial_and_liquidation_turnover_are_one_sided() -> None:
    codes = [f"A{i}" for i in range(20)]
    assert calculate_turnover([], codes) == (1.0, 0.0)
    assert calculate_turnover(codes, []) == (0.0, 1.0)


def test_continuous_timeline_accepts_adjacent_successful_rebalances() -> None:
    curve = pd.DataFrame([
        {"date": "2020-01-01", "portfolio_exposure": 0.0, "cash_reason": "warmup_or_no_signal"},
        {"date": "2020-01-02", "portfolio_exposure": 1.0, "cash_reason": "normal_holding"},
        {"date": "2020-01-03", "portfolio_exposure": 1.0, "cash_reason": "normal_holding"},
        {"date": "2020-01-06", "portfolio_exposure": 1.0, "cash_reason": "normal_holding"},
    ])
    audits = [
        {"execution_date": "2020-01-02", "selected_count": 20},
        {"execution_date": "2020-01-06", "selected_count": 20},
    ]
    validation = validate_continuous_timeline(curve, audits)
    assert validation["passed"] is True
    assert validation["cash_days_between_successful_rebalances"] == 0


def test_continuous_timeline_rejects_schedule_gap_label() -> None:
    curve = pd.DataFrame([
        {"date": "2020-01-01", "portfolio_exposure": 0.0, "cash_reason": "rebalance_schedule_gap"},
    ])
    assert validate_continuous_timeline(curve, [])["passed"] is False
