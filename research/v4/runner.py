"""V4 Entry Timing runner with isolated output paths.

V4 deliberately reuses the validated V3 data adapters and continuous
rebalancing engine, but never writes into V3's frozen output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.v3.engine import metrics, run_v3, write_result
from research.v3.preflight import run_preflight
from research.v4.config import V4_CONFIG_PATH, V4_OUTPUT_DIR, default_v4_config


V4_RUN_DIR = V4_OUTPUT_DIR / "runs"
V4_MANIFEST = V4_OUTPUT_DIR / "v4_manifest.json"


def write_v4_manifest(status: str, payload: dict[str, Any] | None = None) -> Path:
    V4_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    content = {
        "version": "v4_entry_timing_candidate",
        "technical_variant": "entry_timing",
        "status": status,
        "config_path": str(V4_CONFIG_PATH),
        "output_dir": str(V4_OUTPUT_DIR),
        **(payload or {}),
    }
    V4_MANIFEST.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return V4_MANIFEST


def run_v4(
    model_label: str = "model_a",
    event_boost_weight: float = 0.0,
    market_regime_gate: bool | None = None,
) -> dict[str, Any]:
    """Run a new V4 result without altering V3 records or snapshots."""
    config = default_v4_config()
    preflight = run_preflight()
    if preflight.integrity_status != "validated":
        write_v4_manifest("blocked_preflight", {"issues": preflight.issues})
        raise RuntimeError("V4 preflight is not validated; refusing to generate a performance result.")
    result = run_v3(
        model_label=model_label,
        event_boost_weight=event_boost_weight,
        market_regime_gate=market_regime_gate,
        technical_variant="entry_timing",
        config_overrides=config,
    )
    output = write_result(result, V4_RUN_DIR)
    summary = metrics(result)
    summary["config"] = result.config
    summary["output"] = str(output)
    write_v4_manifest("completed", {"latest_model": model_label, "summary": summary})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated V4 Entry Timing research.")
    parser.add_argument("--model", choices=["model_a", "model_b"], default="model_a")
    parser.add_argument("--no-market-gate", action="store_true")
    args = parser.parse_args()
    summary = run_v4(
        model_label=f"{args.model}_no_gate" if args.no_market_gate else args.model,
        event_boost_weight=0.0 if args.model == "model_a" else 0.05,
        market_regime_gate=not args.no_market_gate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
