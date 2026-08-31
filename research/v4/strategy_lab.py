"""User-configurable V4 Entry Timing snapshot experiments.

V4 experiments never read the V3 Legacy panel.  They become available only
after the V4 no-gate baseline has generated an Entry Timing score snapshot.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from research.experiments.registry import ExperimentRequest, create_experiment, list_experiments, update_experiment_status
from research.v3.ablation_replay import normalize_weights, run_replay
from research.v3.preflight import run_preflight
from research.v4.config import V4_OUTPUT_DIR, default_v4_config
from research.v4.factor_recipes import active_recipes


V4_PANEL_DIR = V4_OUTPUT_DIR / "runs" / "model_a_no_gate"
V4_PANEL_PATH = V4_PANEL_DIR / "factor_score_panel.csv"
V4_LAB_OUTPUT_DIR = V4_OUTPUT_DIR / "strategy_lab"
FACTORS = ("market_regime", "momentum", "money_flow", "fundamental", "technical")
TOP_N_OPTIONS = {5, 10, 20, 50}


class V4StrategyLabError(ValueError):
    pass


@dataclass(frozen=True)
class V4StrategySpec:
    name: str
    hypothesis: str
    factor_weights: dict[str, float]
    top_n: int = 20
    market_regime_gate: bool = False
    market_min_score: float = 40.0


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")[:48] or "strategy"


def _snapshot_variant() -> str:
    if not V4_PANEL_PATH.exists():
        return "entry_timing"
    try:
        values = pd.read_csv(V4_PANEL_PATH, usecols=["technical_variant"])["technical_variant"].dropna().astype(str).unique()
    except (OSError, ValueError, pd.errors.ParserError):
        return "unknown"
    return values[0] if len(values) == 1 else "unknown"


def validate_spec(spec: V4StrategySpec) -> V4StrategySpec:
    if not spec.name.strip():
        raise V4StrategyLabError("请为实验填写名称。")
    if not spec.hypothesis.strip():
        raise V4StrategyLabError("请先写下需要验证的假设，避免无目的调参。")
    if int(spec.top_n) not in TOP_N_OPTIONS:
        raise V4StrategyLabError("持仓数量只支持 5、10、20 或 50。")
    raw = {factor: float(spec.factor_weights.get(factor, 0.0)) for factor in FACTORS}
    if any(weight < 0 for weight in raw.values()) or sum(raw.values()) <= 0:
        raise V4StrategyLabError("至少启用一个非负因子。")
    if not 0 <= float(spec.market_min_score) <= 100:
        raise V4StrategyLabError("市场环境阈值必须在 0 到 100 之间。")
    return V4StrategySpec(
        name=spec.name.strip(), hypothesis=spec.hypothesis.strip(), factor_weights=normalize_weights(raw),
        top_n=int(spec.top_n), market_regime_gate=bool(spec.market_regime_gate), market_min_score=float(spec.market_min_score),
    )


def default_spec() -> dict[str, Any]:
    config = default_v4_config()
    return {
        "name": "V4 Entry Timing 多因子实验",
        "hypothesis": "在固定 V4 Entry Timing 历史快照上，检验五因子配置是否改善风险调整收益。",
        "factor_weights": config["factor_weights"],
        "top_n": int(config["top_n"]),
        "market_regime_gate": False,
        "market_min_score": float(config["market_min_score"]),
        "snapshot": str(V4_PANEL_PATH),
        "snapshot_status": "ready" if V4_PANEL_PATH.exists() else "pending_v4_baseline",
        "snapshot_technical_variant": _snapshot_variant(),
        "scope_note": "V4 实验只使用 Entry Timing 因子快照；不会读写 V3 Legacy 快照或任何正式回测结果。",
    }


def run_strategy_lab(spec: V4StrategySpec) -> dict[str, Any]:
    spec = validate_spec(spec)
    if not V4_PANEL_PATH.exists():
        raise V4StrategyLabError("V4 Entry Timing 基线尚未生成因子快照，请先完成 V4 No-Gate 基线运行。")
    if _snapshot_variant() != "entry_timing":
        raise V4StrategyLabError("V4 快照技术定义不为 entry_timing，拒绝运行以避免版本混用。")
    preflight = run_preflight()
    if preflight.integrity_status != "validated":
        raise V4StrategyLabError("V4 共享的数据完整性预检未通过，拒绝运行策略实验。")
    config = default_v4_config()
    recipe_snapshot = active_recipes()
    request = ExperimentRequest(
        hypothesis_note=spec.hypothesis,
        strategy_version="v4_entry_timing_snapshot_replay",
        data_version="v4_entry_timing_hfq_baostock_pit_fundamentals_score_panel_v1",
        sample_period=dict(config["sample_period"]),
        factor_weights=spec.factor_weights,
        enabled_factors=[name for name, weight in spec.factor_weights.items() if weight > 0],
        market_regime_gate=spec.market_regime_gate,
        execution_assumptions={
            "top_n": spec.top_n, "rebalance_days": config["rebalance_days"],
            "execution_price_convention": "next_trading_day_close", "input_snapshot": str(V4_PANEL_PATH),
            "technical_variant": "entry_timing",
            "factor_recipe_snapshot": recipe_snapshot,
        },
        fee_assumptions={"fee_rate": config["fee_rate"], "slippage_rate": config["slippage_rate"]},
        benchmark_setup=dict(config["benchmark_setup"]), research_state="in_sample_research",
    )
    record = create_experiment(request)
    update_experiment_status(record.experiment_id, "running")
    output_dir = V4_LAB_OUTPUT_DIR / record.experiment_id
    try:
        result = run_replay(
            name=_safe_name(spec.name), weights=spec.factor_weights, panel_path=V4_PANEL_PATH,
            top_n=spec.top_n, market_regime_gate=spec.market_regime_gate,
            market_min_score=spec.market_min_score, config_overrides=config,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        result.curve.to_csv(output_dir / "daily_equity.csv", index=False, encoding="utf-8-sig")
        result.audit.to_csv(output_dir / "rebalance_audit.csv", index=False, encoding="utf-8-sig")
        payload = {
            "experiment_id": record.experiment_id, "strategy": asdict(spec), "metrics": result.metrics,
            "classification": "research_experiment", "integrity_status": "validated_replay_snapshot",
            "technical_variant": "entry_timing", "factor_recipes": recipe_snapshot, "scope_note": "固定 V4 Entry Timing 快照上的样本内重放研究。",
        }
        (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        update_experiment_status(record.experiment_id, "completed", {**result.metrics, "output_dir": str(output_dir), "strategy_name": spec.name, "technical_variant": "entry_timing"})
        return payload
    except Exception as error:
        update_experiment_status(record.experiment_id, "failed", {"error": str(error), "strategy_name": spec.name, "technical_variant": "entry_timing"})
        raise


def list_strategy_lab_experiments(limit: int = 30) -> list[dict[str, Any]]:
    latest: dict[str, Any] = {}
    for record in list_experiments():
        if record.request.get("strategy_version") == "v4_entry_timing_snapshot_replay":
            latest[record.experiment_id] = record
    return [
        {
            "experiment_id": record.experiment_id, "created_at": record.created_at, "status": record.status,
            "hypothesis": record.request.get("hypothesis_note", ""), "weights": record.request.get("factor_weights", {}),
            "top_n": record.request.get("execution_assumptions", {}).get("top_n"),
            "market_regime_gate": record.request.get("market_regime_gate", False),
            "technical_variant": "entry_timing", "technical_variant_source": "snapshot_metadata",
            "factor_recipes": record.request.get("execution_assumptions", {}).get("factor_recipe_snapshot"),
            "recipe_snapshot_status": "recorded" if record.request.get("execution_assumptions", {}).get("factor_recipe_snapshot") else "unavailable_predates_recipe_tracking",
            "metrics": record.result_summary or {}, "scope": "v4_entry_timing_snapshot_replay",
        }
        for record in list(reversed(list(latest.values())))[:max(1, min(int(limit), 100))]
    ]
