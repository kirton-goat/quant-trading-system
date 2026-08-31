"""Append-only V4 factor recipes and single-factor research records.

Recipes are research assets, not strategy code.  The active recipe pointer is
explicit and every experiment embeds a full recipe copy, so later edits cannot
change historical research records.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from research.v4.config import V4_OUTPUT_DIR


STORE = V4_OUTPUT_DIR / "factor_validation" / "recipe_store.json"
EXPERIMENTS = V4_OUTPUT_DIR / "factor_validation" / "experiments.jsonl"
FACTOR_IDS = ("market_regime", "momentum", "money_flow", "fundamental", "technical")

DEFAULTS: dict[str, dict[str, Any]] = {
    "market_regime": {"label": "Market", "kind": "time_series", "components": [
        {"id": "return_20d", "label": "20D CSI300 Return", "weight": .4, "enabled": True},
        {"id": "return_60d", "label": "60D CSI300 Return", "weight": .4, "enabled": True},
        {"id": "close_above_ma20", "label": "Price > MA20", "weight": .1, "enabled": True},
        {"id": "ma20_above_ma60", "label": "MA20 > MA60", "weight": .1, "enabled": True},
    ], "parameters": {"gate_threshold": 40}},
    "momentum": {"label": "Momentum", "kind": "cross_sectional", "components": [
        {"id": "return_20d", "label": "20D Return", "weight": .45, "enabled": True},
        {"id": "return_60d", "label": "60D Return", "weight": .45, "enabled": True},
        {"id": "close_above_ma20", "label": "Price > MA20", "weight": .05, "enabled": True},
        {"id": "ma20_above_ma60", "label": "MA20 > MA60", "weight": .05, "enabled": True},
    ], "parameters": {}},
    "money_flow": {"label": "Money Flow", "kind": "cross_sectional", "components": [
        {"id": "volume_ratio_5_20", "label": "5D / 20D Volume", "weight": .5, "enabled": True},
        {"id": "amount_ratio", "label": "Current / 20D Amount", "weight": .3, "enabled": True},
        {"id": "price_volume_confirmation", "label": "Price + Volume Confirmation", "weight": .2, "enabled": True},
    ], "parameters": {}},
    "fundamental": {"label": "Fundamental", "kind": "cross_sectional", "components": [
        {"id": "quality", "label": "Quality", "weight": .30, "enabled": True}, {"id": "growth", "label": "Growth", "weight": .25, "enabled": True},
        {"id": "cashflow", "label": "Cash Flow", "weight": .20, "enabled": True}, {"id": "valuation", "label": "Valuation", "weight": .25, "enabled": True},
    ], "parameters": {"pit_required": True}},
    "technical": {"label": "Entry Timing", "kind": "cross_sectional", "components": [
        {"id": "ma5_deviation", "label": "Short-term MA Deviation", "weight": .25, "enabled": True}, {"id": "three_day_chase", "label": "Overheat Penalty", "weight": .15, "enabled": True},
        {"id": "pullback", "label": "Short-term Pullback", "weight": .15, "enabled": True}, {"id": "macd_change", "label": "MACD Marginal Change", "weight": .15, "enabled": True},
        {"id": "rsi", "label": "RSI", "weight": .15, "enabled": True}, {"id": "volatility_gap", "label": "Gap / Volatility", "weight": .15, "enabled": True},
    ], "parameters": {}},
}


def _now() -> str: return dt.datetime.now(dt.timezone.utc).isoformat()
def _hash(value: Any) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
def _read() -> dict[str, Any]:
    if STORE.exists():
        try: return json.loads(STORE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): pass
    recipes, active = {}, {}
    for factor, definition in DEFAULTS.items():
        recipe = _new_recipe(factor, definition, source_experiment_id=None, version=1)
        recipes[recipe["recipe_id"]] = recipe; active[factor] = recipe["recipe_id"]
    state = {"recipes": recipes, "active": active, "history": [], "approvals": {}}; _write(state); return state
def _write(state: dict[str, Any]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True); STORE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
def _new_recipe(factor: str, definition: dict[str, Any], source_experiment_id: str | None, version: int) -> dict[str, Any]:
    payload = {"factor_name": factor, "label": definition["label"], "kind": definition["kind"], "components": deepcopy(definition["components"]), "parameters": deepcopy(definition.get("parameters", {}))}
    return {**payload, "recipe_id": f"{factor}_v{version}_{uuid.uuid4().hex[:6]}", "recipe_version": version, "recipe_hash": _hash(payload), "created_at": _now(), "source_experiment_id": source_experiment_id, "status": "active_candidate"}


def active_recipes() -> dict[str, Any]:
    state = _read(); return {factor: state["recipes"][recipe_id] for factor, recipe_id in state["active"].items()}
def recipe_history(factor: str) -> list[dict[str, Any]]:
    state = _read(); return sorted([r for r in state["recipes"].values() if r["factor_name"] == factor], key=lambda x: x["recipe_version"], reverse=True)
def create_recipe(factor: str, components: list[dict[str, Any]], parameters: dict[str, Any], source_experiment_id: str | None = None) -> dict[str, Any]:
    if factor not in FACTOR_IDS: raise ValueError("Unsupported factor.")
    allowed = {item["id"] for item in DEFAULTS[factor]["components"]}
    if not components or any(str(item.get("id")) not in allowed for item in components): raise ValueError("Recipe contains an unsupported indicator.")
    enabled = [item for item in components if item.get("enabled")]
    if not enabled or sum(float(item.get("weight", 0)) for item in enabled) <= 0: raise ValueError("Enable at least one component with positive weight.")
    definition = {"label": DEFAULTS[factor]["label"], "kind": DEFAULTS[factor]["kind"], "components": components, "parameters": parameters}
    state = _read(); version = max([r["recipe_version"] for r in recipe_history(factor)] or [0]) + 1
    recipe = _new_recipe(factor, definition, source_experiment_id, version); state["recipes"][recipe["recipe_id"]] = recipe; _write(state); return recipe
def set_active_recipe(factor: str, recipe_id: str, source_experiment_id: str, note: str = "") -> dict[str, Any]:
    state = _read(); recipe = state["recipes"].get(recipe_id)
    if not recipe or recipe["factor_name"] != factor: raise ValueError("Recipe does not belong to factor.")
    prior = state["active"].get(factor); state["active"][factor] = recipe_id
    state["history"].append({"factor_name": factor, "previous_recipe_id": prior, "new_recipe_id": recipe_id, "source_experiment_id": source_experiment_id, "switched_at": _now(), "user_note": note}); _write(state)
    return {"previous_recipe_id": prior, "new_recipe_id": recipe_id}
def append_experiment(record: dict[str, Any]) -> None:
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, ensure_ascii=False) + "\n")
def list_factor_experiments(factor: str | None = None) -> list[dict[str, Any]]:
    if not EXPERIMENTS.exists(): return []
    result = [json.loads(line) for line in EXPERIMENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    approvals = _read().get("approvals", {})
    return [{**item, **approvals.get(item["experiment_id"], {"approved": False, "approval_note": ""})} for item in result if factor is None or item["factor_name"] == factor]
def set_experiment_approval(experiment_id: str, approved: bool, note: str = "") -> dict[str, Any]:
    state = _read(); approvals = state.setdefault("approvals", {})
    approvals[experiment_id] = {"approved": bool(approved), "approved_at": _now() if approved else None, "approval_note": note}; _write(state)
    return approvals[experiment_id]
def approval_for(experiment_id: str) -> dict[str, Any]: return _read().get("approvals", {}).get(experiment_id, {"approved": False, "approval_note": ""})
