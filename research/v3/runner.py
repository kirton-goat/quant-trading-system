"""Guarded V3 experiment launcher.

It intentionally refuses to invoke the expensive strategy engine until the
offline integrity preflight succeeds. This protects the project from creating
an attractive but invalid 2015-2025 curve from partial data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.experiments.registry import ExperimentRequest, create_experiment, update_experiment_status
from research.v3.preflight import DEFAULT_CONFIG_PATH, default_v3_config, run_preflight, write_preflight


class V3DataIntegrityError(RuntimeError):
    pass


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    if not path.exists():
        return default_v3_config()
    return json.loads(path.read_text(encoding="utf-8"))


def queue_v3_experiment(hypothesis_note: str, config: dict | None = None) -> str:
    config = config or default_v3_config()
    request = ExperimentRequest(
        hypothesis_note=hypothesis_note,
        strategy_version=config["research_version"],
        data_version="v3_cache_preflight_pending",
        sample_period=dict(config["sample_period"]),
        factor_weights=dict(config["factor_weights"]),
        enabled_factors=sorted(config["factor_weights"]),
        market_regime_gate=bool(config["market_regime_gate"]),
        execution_assumptions={
            "execution_price_convention": "next_trading_day_close",
            "holding_days": config["holding_days"], "rebalance_days": config["rebalance_days"], "top_n": config["top_n"],
        },
        fee_assumptions={"fee_rate": config["fee_rate"], "slippage_rate": config["slippage_rate"]},
        benchmark_setup=dict(config["benchmark_setup"]),
        research_state=config["research_state"],
    )
    record = create_experiment(request)
    result = run_preflight(config)
    write_preflight(result)
    if result.integrity_status != "validated":
        update_experiment_status(record.experiment_id, "incomplete", {
            "integrity_status": result.integrity_status,
            "blocking_issues": result.issues,
            "backtest_executed": False,
        })
        raise V3DataIntegrityError(
            f"V3 experiment {record.experiment_id} recorded as incomplete. Resolve data preflight before any 2015-2025 run."
        )
    update_experiment_status(record.experiment_id, "running")
    # The research engine is intentionally not called here until the strict data
    # adapters and corporate-action audit have passed. It must be added as a
    # separate implementation, never as a fallback path to v2 formal outputs.
    return record.experiment_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Queue a guarded V3 research experiment.")
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args()
    print(queue_v3_experiment(args.hypothesis, load_config()))


if __name__ == "__main__":
    main()
