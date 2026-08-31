from __future__ import annotations

import pandas as pd

from research.v3.preflight import default_v3_config
from research.v4.config import default_v4_config
from research.v3.strategy_lab import snapshot_technical_variant
from dashboard.api.repository import get_backtest_version_catalog


def test_snapshot_technical_variant_is_read_from_the_panel_metadata(tmp_path):
    panel = tmp_path / "factor_score_panel.csv"
    pd.DataFrame({"technical_variant": ["entry_timing", "entry_timing"]}).to_csv(panel, index=False)
    assert snapshot_technical_variant(panel) == "entry_timing"


def test_v3_v4_contract_keeps_legacy_and_entry_timing_separate():
    assert default_v3_config()["technical_variant"] == "legacy"
    assert default_v4_config()["technical_variant"] == "entry_timing"


def test_dashboard_version_catalog_keeps_frozen_backtests_and_research_separate():
    catalog = get_backtest_version_catalog()
    assert catalog["official_current_version"] == "v2_continuous_rebalance"
    assert catalog["data_version"] is None
    assert {item["version"] for item in catalog["official_versions"]} >= {
        "v1_historical_point_in_time", "v2_continuous_rebalance",
    }
    assert {item["version"] for item in catalog["research_versions"]} >= {
        "v3_long_sample_research", "v4_entry_timing_candidate",
    }
