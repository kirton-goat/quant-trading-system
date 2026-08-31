import pandas as pd

from research.v3.factor_research import _daily_ic, _quantile_returns


def _observations() -> pd.DataFrame:
    rows = []
    for code, score in enumerate(range(1, 11), start=1):
        rows.append({
            "signal_date": "2020-01-31",
            "stock_code": f"000{code:03d}",
            "momentum_score": score,
            "money_flow_score": 11 - score,
            "fundamental_score": score,
            "technical_score": score,
            "total_score": score,
            "forward_return": score / 100,
        })
    return pd.DataFrame(rows)


def test_daily_ic_uses_cross_sectional_scores_and_subsequent_returns():
    result = _daily_ic(_observations())
    momentum = result[result["factor"] == "momentum_score"].iloc[0]
    assert momentum["observations"] == 10
    assert abs(momentum["spearman_ic"] - 1.0) < 1e-12


def test_quantile_returns_keep_high_and_low_score_groups_separate():
    result = _quantile_returns(_observations())
    momentum = result[result["factor"] == "momentum_score"].set_index("quantile")
    assert momentum.loc[5, "mean_forward_return"] > momentum.loc[1, "mean_forward_return"]
