from research.v3.preflight import default_v3_config
from research.v4.config import V4_OUTPUT_DIR, default_v4_config
from research.v4.strategy_lab import V4StrategySpec, validate_spec


def test_v3_remains_legacy_and_v4_isolated_entry_timing_candidate():
    v3 = default_v3_config()
    v4 = default_v4_config()
    assert v3["research_version"] == "v3_long_sample_research"
    assert v3["technical_variant"] == "legacy"
    assert v4["research_version"] == "v4_entry_timing_candidate"
    assert v4["technical_variant"] == "entry_timing"
    assert "v4_entry_timing" in str(V4_OUTPUT_DIR)


def test_v4_strategy_lab_normalizes_the_entry_timing_five_factor_request():
    spec = validate_spec(V4StrategySpec(
        name="entry timing five factor check",
        hypothesis="Validate the isolated V4 Entry Timing five-factor research input.",
        factor_weights={
            "market_regime": 20,
            "momentum": 25,
            "money_flow": 20,
            "fundamental": 25,
            "technical": 10,
        },
        top_n=20,
        market_regime_gate=False,
    ))
    assert abs(sum(spec.factor_weights.values()) - 1) < 1e-9
    assert spec.factor_weights["technical"] == 0.10
