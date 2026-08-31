from __future__ import annotations

from research.data_integrity.price_gap_resolution import _reason
from research.data_integrity.price_quality_v2 import _boundary_status


def test_delisted_price_end_is_classified_as_valid_no_trade_not_fillable_gap() -> None:
    cause, detail = _reason({"first_date": "2014-01-01", "last_date": "2019-01-10"}, "2000-01-01", "2019-01-10", "2014-01-01", "2019-02-01")
    assert cause.startswith("A.")
    assert detail == "valid_no_trade_after_delisting"


def test_unexplained_missing_end_remains_upstream_gap() -> None:
    cause, detail = _reason({"first_date": "2014-01-01", "last_date": "2020-01-01"}, "2000-01-01", "", "2014-01-01", "2020-02-01")
    assert cause.startswith("D.")
    assert detail == "missing_end_coverage"


def test_delisting_boundary_is_not_reclassified_as_a_true_price_gap() -> None:
    status, reason = _boundary_status(
        {"first_date": "2014-01-01", "last_date": "2019-01-10", "reason": "missing_end_coverage"},
        "2014-01-01", "2019-01-10", "2000-01-01", "2019-01-10",
    )
    assert status == "VALID_END_OF_TRADING"
    assert reason == "history_ends_at_delisting_or_termination"


def test_missing_price_inside_tradeable_life_remains_a_true_gap() -> None:
    status, _ = _boundary_status(
        {"first_date": "2014-01-01", "last_date": "2020-01-01", "reason": "missing_end_coverage"},
        "2014-01-01", "2020-02-01", "2000-01-01", "",
    )
    assert status == "TRUE_PRICE_GAP"
