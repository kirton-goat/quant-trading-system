"""Append-only metadata registry for isolated quantitative research experiments.

The registry deliberately knows nothing about the formal v1/v2 result folders.
It preserves the hypothesis and immutable configuration of every experiment so a
poor result cannot be silently replaced after it has been observed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = BASE_DIR / "research" / "experiments" / "registry" / "experiments.jsonl"
EXPERIMENT_STATUSES = {"queued", "running", "completed", "incomplete", "failed", "archived"}
RESEARCH_STATES = {"in_sample_research", "historical_holdout", "walk_forward_simulation", "future_oos"}


class ExperimentRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentRequest:
    hypothesis_note: str
    strategy_version: str
    data_version: str
    sample_period: dict[str, str]
    factor_weights: dict[str, float]
    enabled_factors: list[str]
    market_regime_gate: bool
    execution_assumptions: dict[str, Any]
    fee_assumptions: dict[str, Any]
    benchmark_setup: dict[str, Any]
    research_state: str = "in_sample_research"
    parent_experiment_id: str | None = None


@dataclass
class ExperimentRecord:
    experiment_id: str
    created_at: str
    configuration_hash: str
    cache_key: str
    status: str
    request: dict[str, Any]
    result_summary: dict[str, Any] = field(default_factory=dict)
    archived: bool = False


def canonical_configuration(request: ExperimentRequest | dict[str, Any]) -> dict[str, Any]:
    payload = asdict(request) if isinstance(request, ExperimentRequest) else dict(request)
    payload["enabled_factors"] = sorted(str(item) for item in payload.get("enabled_factors", []))
    payload["factor_weights"] = {
        str(key): round(float(value), 10) for key, value in sorted(dict(payload.get("factor_weights", {})).items())
    }
    return payload


def configuration_hash(request: ExperimentRequest | dict[str, Any]) -> str:
    encoded = json.dumps(canonical_configuration(request), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_request(request: ExperimentRequest) -> None:
    if not request.hypothesis_note or not request.hypothesis_note.strip():
        raise ExperimentRegistryError("Research experiments require a hypothesis_note before execution.")
    if request.research_state not in RESEARCH_STATES:
        raise ExperimentRegistryError(f"Unsupported research_state: {request.research_state}")
    if not request.strategy_version or not request.data_version:
        raise ExperimentRegistryError("strategy_version and data_version are required.")
    start, end = request.sample_period.get("start", ""), request.sample_period.get("end", "")
    if not start or not end or start > end:
        raise ExperimentRegistryError("sample_period must have an ordered start and end date.")
    weights = request.factor_weights
    if any(value < 0 for value in weights.values()):
        raise ExperimentRegistryError("factor weights cannot be negative.")
    if request.enabled_factors:
        enabled_total = sum(float(weights.get(name, 0.0)) for name in request.enabled_factors)
        if abs(enabled_total - 1.0) > 1e-8:
            raise ExperimentRegistryError(f"Enabled factor weights must sum to 1.0; got {enabled_total}.")


def create_experiment(request: ExperimentRequest, registry_path: Path = DEFAULT_REGISTRY_PATH) -> ExperimentRecord:
    validate_request(request)
    digest = configuration_hash(request)
    record = ExperimentRecord(
        experiment_id=f"exp_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        configuration_hash=digest,
        cache_key=digest,
        status="queued",
        request=canonical_configuration(request),
    )
    append_record(record, registry_path)
    return record


def append_record(record: ExperimentRecord, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def list_experiments(registry_path: Path = DEFAULT_REGISTRY_PATH) -> list[ExperimentRecord]:
    if not registry_path.exists():
        return []
    records: list[ExperimentRecord] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        records.append(ExperimentRecord(**raw))
    return records


def update_experiment_status(
    experiment_id: str,
    status: str,
    result_summary: dict[str, Any] | None = None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> ExperimentRecord:
    """Append a status event; existing records are intentionally never rewritten."""
    if status not in EXPERIMENT_STATUSES:
        raise ExperimentRegistryError(f"Unsupported experiment status: {status}")
    records = list_experiments(registry_path)
    prior = next((item for item in reversed(records) if item.experiment_id == experiment_id), None)
    if prior is None:
        raise ExperimentRegistryError(f"Unknown experiment_id: {experiment_id}")
    event = ExperimentRecord(
        experiment_id=prior.experiment_id,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        configuration_hash=prior.configuration_hash,
        cache_key=prior.cache_key,
        status=status,
        request=prior.request,
        result_summary=result_summary or prior.result_summary,
        archived=status == "archived" or prior.archived,
    )
    append_record(event, registry_path)
    return event
