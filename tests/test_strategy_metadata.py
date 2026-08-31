from research.strategy_metadata import get_strategy_version, get_strategy_versions


def test_strategy_explorer_exposes_all_known_versions():
    catalog = get_strategy_versions()
    assert catalog["default_version"] == "v2_continuous_rebalance"
    assert {item["version"] for item in catalog["versions"]} == {
        "v1_historical_point_in_time",
        "v2_continuous_rebalance",
        "v3_long_sample_research",
        "v4_entry_timing_candidate",
    }


def test_v4_entry_timing_metadata_matches_the_distinct_formula():
    version = get_strategy_version("v4_entry_timing_candidate")
    technical = next(item for item in version["factors"] if item["id"] == "technical")
    assert technical["name"] == "Entry Timing"
    assert technical["status"] == "active"
    assert "MA5 deviation" in technical["formula"]
    assert "MA20 > MA60" not in technical["formula"]


def test_legacy_versions_keep_legacy_technical_and_v2_timeline():
    v2 = get_strategy_version("v2_continuous_rebalance")
    technical = next(item for item in v2["factors"] if item["id"] == "technical")
    assert technical["name"] == "Legacy Technical"
    assert "T+1 close" in v2["timeline"]
    assert v2["summary"]["pit_universe"] is True
    assert v2["summary"]["pit_fundamentals"] is True


def test_market_soft_factor_and_hard_gate_are_separate():
    version = get_strategy_version("v3_long_sample_research")
    market = next(item for item in version["factors"] if item["id"] == "market_regime")
    assert market["type"] == "time_series_common_market_score"
    assert version["market_hard_gate"]["trigger"] == "market_score < threshold"
    assert market["score_display"].startswith("Historical score")
    assert market["evidence_status"] == "not_validated"
