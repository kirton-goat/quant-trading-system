"""User-configurable, snapshot-based V3 research experiments.

This module deliberately replays a frozen factor-score panel.  It gives the
dashboard a safe way to compare factor weights and inclusion rules without
mutating the formal V1/V2 strategy or pretending that an in-sample experiment
is out-of-sample evidence.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from research.experiments.registry import ExperimentRequest, list_experiments, create_experiment, update_experiment_status
from research.v3.ablation_replay import PANEL_RUN_DIR, normalize_weights, run_replay
from research.v3.preflight import OUTPUT_DIR, default_v3_config, run_preflight


LAB_OUTPUT_DIR = OUTPUT_DIR / "strategy_lab"
FACTORS = ("market_regime", "momentum", "money_flow", "fundamental", "technical")
TOP_N_OPTIONS = {5, 10, 20, 50}


class StrategyLabError(ValueError):
    pass


def snapshot_technical_variant_details(panel_path: Path = PANEL_RUN_DIR / "factor_score_panel.csv") -> tuple[str, str]:
    """Return the panel definition and whether it is stored or historically inferred."""
    if not panel_path.exists():
        return "unknown", "missing_snapshot"
    try:
        values = pd.read_csv(panel_path, usecols=["technical_variant"])["technical_variant"].dropna().astype(str).unique()
    except (OSError, ValueError, pd.errors.ParserError):
        # This immutable panel predates the field. Its paired summary is the
        # recorded `v3_long_sample_research` Legacy Technical baseline, and
        # the Entry Timing experiment used it as that explicit control.
        if panel_path.resolve() == (PANEL_RUN_DIR / "factor_score_panel.csv").resolve():
            return "legacy", "historical_inference"
        return "unknown", "missing_metadata"
    return (
        (values[0], "snapshot_metadata")
        if len(values) == 1 and values[0] in {"legacy", "entry_timing"}
        else ("unknown", "invalid_metadata")
    )


def snapshot_technical_variant(panel_path: Path = PANEL_RUN_DIR / "factor_score_panel.csv") -> str:
    return snapshot_technical_variant_details(panel_path)[0]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    hypothesis: str
    factor_weights: dict[str, float]
    top_n: int = 20
    market_regime_gate: bool = False
    market_min_score: float = 40.0


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return normalized[:48] or "strategy"


def validate_spec(spec: StrategySpec) -> StrategySpec:
    if not spec.name.strip():
        raise StrategyLabError("请为实验填写名称。")
    if not spec.hypothesis.strip():
        raise StrategyLabError("请先写下需要验证的假设，避免无目的调参。")
    if int(spec.top_n) not in TOP_N_OPTIONS:
        raise StrategyLabError("持仓数量只支持 5、10、20 或 50。")
    raw = {factor: float(spec.factor_weights.get(factor, 0.0)) for factor in FACTORS}
    if any(weight < 0 for weight in raw.values()):
        raise StrategyLabError("因子权重不能为负数。")
    if sum(raw.values()) <= 0:
        raise StrategyLabError("至少启用一个因子。")
    if not 0 <= float(spec.market_min_score) <= 100:
        raise StrategyLabError("市场环境阈值必须在 0 到 100 之间。")
    return StrategySpec(
        name=spec.name.strip(), hypothesis=spec.hypothesis.strip(), factor_weights=normalize_weights(raw),
        top_n=int(spec.top_n), market_regime_gate=bool(spec.market_regime_gate),
        market_min_score=float(spec.market_min_score),
    )


def default_spec() -> dict[str, Any]:
    config = default_v3_config()
    snapshot_variant, snapshot_variant_source = snapshot_technical_variant_details()
    return {
        "name": "我的多因子实验",
        "hypothesis": "在不改变数据、交易成本和历史可见性约束的前提下，检验该因子组合是否改善风险调整收益。",
        "factor_weights": config["factor_weights"],
        "top_n": int(config["top_n"]),
        "market_regime_gate": False,
        "market_min_score": float(config["market_min_score"]),
        "snapshot": str(PANEL_RUN_DIR / "factor_score_panel.csv"),
        "snapshot_technical_variant": snapshot_variant,
        "snapshot_technical_variant_source": snapshot_variant_source,
        "next_v3_default_technical_variant": str(config.get("technical_variant", "legacy")),
        "scope_note": "当前实验室重放冻结的 Legacy Technical 因子快照；V3 后续默认已切换为 Entry Timing，须完成独立全样本重跑并生成新快照后才会进入实验室。",
    }


def run_strategy_lab(spec: StrategySpec) -> dict[str, Any]:
    spec = validate_spec(spec)
    preflight = run_preflight()
    if preflight.integrity_status != "validated":
        raise StrategyLabError("V3 数据完整性预检未通过，拒绝运行策略实验。")
    panel_path = PANEL_RUN_DIR / "factor_score_panel.csv"
    if not panel_path.exists():
        raise StrategyLabError("缺少冻结的 V3 因子快照，无法保证实验可复现。")

    config = default_v3_config()
    request = ExperimentRequest(
        hypothesis_note=spec.hypothesis,
        strategy_version="v3_strategy_lab_snapshot_replay",
        data_version="v3_hfq_baostock_pit_fundamentals_score_panel_v1",
        sample_period=dict(config["sample_period"]),
        factor_weights=spec.factor_weights,
        enabled_factors=[name for name, weight in spec.factor_weights.items() if weight > 0],
        market_regime_gate=spec.market_regime_gate,
        execution_assumptions={
            "top_n": spec.top_n,
            "rebalance_days": config["rebalance_days"],
            "execution_price_convention": "next_trading_day_close",
            "input_snapshot": str(panel_path),
            "technical_variant": snapshot_technical_variant(panel_path),
        },
        fee_assumptions={"fee_rate": config["fee_rate"], "slippage_rate": config["slippage_rate"]},
        benchmark_setup=dict(config["benchmark_setup"]),
        research_state="in_sample_research",
    )
    record = create_experiment(request)
    update_experiment_status(record.experiment_id, "running")
    output_dir = LAB_OUTPUT_DIR / record.experiment_id
    try:
        result = run_replay(
            name=_safe_name(spec.name), weights=spec.factor_weights, panel_path=panel_path,
            top_n=spec.top_n, market_regime_gate=spec.market_regime_gate,
            market_min_score=spec.market_min_score,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        result.curve.to_csv(output_dir / "daily_equity.csv", index=False, encoding="utf-8-sig")
        result.audit.to_csv(output_dir / "rebalance_audit.csv", index=False, encoding="utf-8-sig")
        payload = {
            "experiment_id": record.experiment_id,
            "strategy": asdict(spec),
            "metrics": result.metrics,
            "classification": "research_experiment",
            "integrity_status": "validated_replay_snapshot",
            "technical_variant": snapshot_technical_variant(panel_path),
            "scope_note": "固定 V3 因子快照上的样本内重放；不构成样本外验证或交易建议。",
        }
        (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        update_experiment_status(record.experiment_id, "completed", {**result.metrics, "output_dir": str(output_dir), "strategy_name": spec.name})
        return payload
    except Exception as error:
        update_experiment_status(record.experiment_id, "failed", {"error": str(error), "strategy_name": spec.name})
        raise


def list_strategy_lab_experiments(limit: int = 30) -> list[dict[str, Any]]:
    latest: dict[str, Any] = {}
    for record in list_experiments():
        if record.request.get("strategy_version") == "v3_strategy_lab_snapshot_replay":
            latest[record.experiment_id] = record
    rows: list[dict[str, Any]] = []
    for record in reversed(list(latest.values())):
        summary = record.result_summary or {}
        execution = record.request.get("execution_assumptions", {})
        # All pre-metadata strategy-lab records were created from the frozen
        # Legacy Technical panel.  Preserve that historical meaning rather
        # than relabelling old results as the later Entry Timing definition.
        technical_variant = execution.get("technical_variant") or "legacy"
        rows.append({
            "experiment_id": record.experiment_id, "created_at": record.created_at,
            "status": record.status, "hypothesis": record.request.get("hypothesis_note", ""),
            "weights": record.request.get("factor_weights", {}),
            "top_n": execution.get("top_n"),
            "market_regime_gate": record.request.get("market_regime_gate", False),
            "technical_variant": technical_variant,
            "technical_variant_source": "stored_metadata" if execution.get("technical_variant") else "historical_inference",
            "metrics": summary, "scope": "snapshot_replay_in_sample",
        })
    return rows[:max(1, min(int(limit), 100))]
