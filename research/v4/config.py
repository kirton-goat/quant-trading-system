"""Immutable default configuration for the V4 Entry Timing candidate."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from research.v3.preflight import default_v3_config


BASE_DIR = Path(__file__).resolve().parents[2]
V4_OUTPUT_DIR = BASE_DIR / "research" / "experiments" / "output" / "v4_entry_timing"
V4_CONFIG_PATH = BASE_DIR / "backtest_config_v4_entry_timing.json"


def default_v4_config() -> dict[str, Any]:
    """Return V4 without mutating the frozen V3 configuration."""
    config = deepcopy(default_v3_config())
    config.update({
        "research_version": "v4_entry_timing_candidate",
        "result_classification": "research_experiment",
        "research_state": "in_sample_research",
        "technical_variant": "entry_timing",
    })
    return config
