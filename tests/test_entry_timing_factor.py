import pandas as pd

from factor_engine import calculate_technical_score_from_standard_history
from research.v3.preflight import default_v3_config
from research.v4.config import default_v4_config


def _history(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=len(prices), freq="B").strftime("%Y-%m-%d"),
        "open": prices, "high": prices, "low": prices, "close": prices,
        "volume": 1_000_000, "amount": 10_000_000,
    })


def test_legacy_variant_remains_the_default():
    prices = [100 + index * .2 for index in range(80)]
    history = _history(prices)
    assert calculate_technical_score_from_standard_history(history) == calculate_technical_score_from_standard_history(history, "legacy")


def test_entry_timing_penalises_a_three_day_chase():
    controlled = _history([100 + index * .2 for index in range(77)] + [115, 116, 117])
    extended = _history([100 + index * .2 for index in range(77)] + [120, 129, 140])
    assert calculate_technical_score_from_standard_history(extended, "entry_timing") < calculate_technical_score_from_standard_history(controlled, "entry_timing")


def test_v4_candidate_defaults_to_entry_timing_without_changing_v3_legacy_default():
    assert default_v3_config()["technical_variant"] == "legacy"
    assert default_v4_config()["technical_variant"] == "entry_timing"
