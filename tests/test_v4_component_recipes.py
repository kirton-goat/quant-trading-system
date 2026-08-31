from __future__ import annotations

import pandas as pd

from research.v4.component_panel import _component_score, _entry_timing_components, _momentum_components


def _history(rows: int = 70) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="B").strftime("%Y-%m-%d"),
        "open": range(10, 10 + rows), "close": range(10, 10 + rows),
        "volume": range(100, 100 + rows), "amount": range(1_000, 1_000 + rows),
    })


def test_component_score_reweights_only_enabled_items() -> None:
    recipe = {"components": [{"id": "a", "enabled": True, "weight": .8}, {"id": "b", "enabled": True, "weight": .2}, {"id": "c", "enabled": False, "weight": 99}]}
    assert _component_score({"a": 80, "b": 20, "c": 0}, recipe) == 68


def test_momentum_component_panel_uses_historical_window() -> None:
    values = _momentum_components(_history())
    assert set(values) == {"return_20d", "return_60d", "close_above_ma20", "ma20_above_ma60"}
    assert values["close_above_ma20"] == 100.0


def test_entry_timing_does_not_include_medium_trend_component() -> None:
    values = _entry_timing_components(_history())
    assert "close_above_ma20" not in values
    assert "ma20_above_ma60" not in values
