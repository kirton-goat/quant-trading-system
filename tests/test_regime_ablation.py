from __future__ import annotations

import pandas as pd

from research.experiments.regime_ablation import build_cash_exposure_audit


def test_cash_audit_separates_market_block_from_holdings() -> None:
    curve = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
            "holding_codes": ["", "000001,000002", ""],
        }
    )
    decisions = [
        {
            "entry_date": "2020-01-02",
            "exit_date": "2020-01-02",
            "reason": "market_regime_block",
        },
        {
            "entry_date": "2020-01-03",
            "exit_date": "2020-01-03",
            "reason": "normal_holding",
        },
    ]

    audit = build_cash_exposure_audit(curve, decisions, "baseline")

    assert audit["reason"].tolist() == ["market_regime_block", "normal_holding", "rebalance_schedule_gap"]
    assert audit["cash_ratio"].tolist() == [1.0, 0.0, 1.0]
