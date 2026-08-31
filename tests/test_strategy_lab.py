from research.v3.strategy_lab import StrategyLabError, StrategySpec, validate_spec


def test_strategy_lab_normalizes_enabled_weights():
    spec = validate_spec(StrategySpec(
        name="test", hypothesis="Test a factor configuration.",
        factor_weights={"momentum": 2, "fundamental": 2}, top_n=20,
    ))
    assert spec.factor_weights["momentum"] == 0.5
    assert spec.factor_weights["fundamental"] == 0.5
    assert sum(spec.factor_weights.values()) == 1.0


def test_strategy_lab_requires_hypothesis_and_valid_top_n():
    try:
        validate_spec(StrategySpec(name="test", hypothesis="", factor_weights={"momentum": 1}, top_n=20))
    except StrategyLabError as error:
        assert "假设" in str(error)
    else:
        raise AssertionError("Expected hypothesis validation error")

    try:
        validate_spec(StrategySpec(name="test", hypothesis="x", factor_weights={"momentum": 1}, top_n=7))
    except StrategyLabError as error:
        assert "持仓数量" in str(error)
    else:
        raise AssertionError("Expected top_n validation error")
