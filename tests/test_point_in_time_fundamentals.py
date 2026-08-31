from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.fundamentals.fundamental_cache import write_fundamental_cache
from research.fundamentals.fundamental_validation import (
    FundamentalFutureDataError,
    determine_backtest_integrity,
    public_integrity_status,
    validate_visible_record,
)
from research.fundamentals.point_in_time_fundamentals import get_fundamentals
from factor_engine import calculate_factor_scores_for_universe


CODE = "600000"


def fixture_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            row("2022-09-30", "2022-10-28", 80, 12, 8, 10, 180, 80, 100, 10),
            row("2023-09-30", "2023-10-27", 100, 15, 10, 18, 200, 90, 110, 10),
            row("2023-12-31", "2024-03-28", 150, 25, 17, 30, 220, 95, 120, 10),
        ]
    )


def row(
    report_period: str,
    disclosure_date: str,
    revenue: float,
    operating_cost: float,
    net_profit: float,
    operating_cash_flow: float,
    assets: float,
    liabilities: float,
    equity: float,
    shares: float,
) -> dict:
    return {
        "code": CODE,
        "report_period": report_period,
        "disclosure_date": disclosure_date,
        "original_disclosure_date": disclosure_date,
        "update_date": disclosure_date,
        "revenue": revenue,
        "operating_cost": operating_cost,
        "net_profit": net_profit,
        "operating_cash_flow": operating_cash_flow,
        "total_assets": assets,
        "total_liabilities": liabilities,
        "total_equity": equity,
        "share_capital": shares,
        "data_source": "test fixture",
        "is_revised": False,
        "revision_note": "",
        "unsupported_for_point_in_time": "",
    }


def prepare_cache(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "fundamentals"
    write_fundamental_cache(CODE, fixture_rows(), cache_dir)
    return cache_dir


def price_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-02-01", "2024-04-01"],
            "close": [20.0, 21.0],
        }
    )


def test_future_annual_report_is_not_visible_before_disclosure(tmp_path: Path) -> None:
    cache_dir = prepare_cache(tmp_path)
    result = get_fundamentals(CODE, "2024-02-01", price_history(), cache_dir, allow_network=False)
    assert result is not None
    assert result.report_period == "2023-09-30"
    assert result.disclosure_date == "2023-10-27"


def test_annual_report_becomes_visible_after_disclosure(tmp_path: Path) -> None:
    cache_dir = prepare_cache(tmp_path)
    result = get_fundamentals(CODE, "2024-04-01", price_history(), cache_dir, allow_network=False)
    assert result is not None
    assert result.report_period == "2023-12-31"
    assert result.disclosure_date == "2024-03-28"


def test_previous_disclosed_report_remains_effective_until_new_disclosure(tmp_path: Path) -> None:
    cache_dir = prepare_cache(tmp_path)
    result = get_fundamentals(CODE, "2024-03-27", price_history(), cache_dir, allow_network=False)
    assert result is not None
    assert result.report_period == "2023-09-30"


def test_future_record_cannot_enter_formal_factor_validation() -> None:
    with pytest.raises(FundamentalFutureDataError):
        validate_visible_record(
            {
                "code": CODE,
                "report_period": "2023-12-31",
                "disclosure_date": "2024-03-28",
            },
            "2024-02-01",
            strict=True,
        )


def test_failed_fundamental_validation_cannot_claim_integrity() -> None:
    assert (
        determine_backtest_integrity(
            "historical_point_in_time",
            "historical_point_in_time",
            fundamental_validation_passed=False,
        )
        == "incomplete"
    )


def test_formal_factor_engine_uses_supplied_point_in_time_score() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=90, freq="D").strftime("%Y-%m-%d"),
            "close": range(10, 100),
            "volume": [1000 + index for index in range(90)],
            "amount": [1_000_000 + index * 1000 for index in range(90)],
        }
    )
    scores = calculate_factor_scores_for_universe(
        {CODE: history},
        "2023-03-31",
        fundamental_scores={CODE: 63.25},
        require_fundamentals=True,
    )
    assert len(scores) == 1
    assert scores[0].fundamental_score == 63.25


def test_formal_factor_engine_drops_missing_fundamentals() -> None:
    history = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=90, freq="D").strftime("%Y-%m-%d"),
            "close": range(10, 100),
            "volume": [1000] * 90,
            "amount": [1_000_000] * 90,
        }
    )
    scores = calculate_factor_scores_for_universe(
        {CODE: history},
        "2023-03-31",
        fundamental_scores={},
        require_fundamentals=True,
    )
    assert scores == []


def test_public_integrity_status_requires_full_point_in_time_validation() -> None:
    assert public_integrity_status("point_in_time_validated") == "validated"
    assert public_integrity_status("incomplete") == "incomplete"