from __future__ import annotations

import pandas as pd

from research.backtest_v1 import benchmark_metrics, drawdown_duration


def test_drawdown_duration_counts_underwater_days() -> None:
    assert drawdown_duration(pd.Series([100, 110, 108, 107, 111, 109])) == 2


def test_benchmark_metrics_outputs_required_statistics() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="B").strftime("%Y-%m-%d")
    curve = pd.DataFrame({"date": dates, "equity": [100 + i * 1.2 for i in range(40)]})
    benchmark = pd.DataFrame({"date": dates, "close": [100 + i * 0.7 for i in range(40)]})
    result = benchmark_metrics(curve, benchmark)
    assert result["status"] == "ok"
    assert result["beta"] is not None
    assert "information_ratio" in result
