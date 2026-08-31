"""Research-only test of de-overlapping Momentum and Entry Timing.

The control uses the immutable V3 no-gate score panel. The candidate changes
only the technical score inside a copied panel, then replays the same stored
risk/execution eligibility and fee logic. It never rewrites V1/V2/V3 outputs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from factor_engine import calculate_entry_timing_scores_from_standard_history
from market_data_manager import MarketDataManager
from research.v3.ablation_replay import PANEL_RUN_DIR, ReplayResult, run_replay
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE
from research.v3.preflight import OUTPUT_DIR, default_v3_config


EXPERIMENT_DIR = OUTPUT_DIR / "entry_timing_refactor"
BASE_PANEL = PANEL_RUN_DIR / "factor_score_panel.csv"


def _write_result(label: str, result: ReplayResult) -> dict:
    folder = EXPERIMENT_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    result.curve.to_csv(folder / "daily_equity.csv", index=False, encoding="utf-8-sig")
    result.audit.to_csv(folder / "rebalance_audit.csv", index=False, encoding="utf-8-sig")
    payload = {"weights": result.weights, **result.metrics}
    (folder / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _build_entry_timing_panel() -> pd.DataFrame:
    if not BASE_PANEL.exists():
        raise FileNotFoundError(f"Missing immutable V3 panel: {BASE_PANEL}")
    panel = pd.read_csv(BASE_PANEL, dtype={"stock_code": str})
    panel["stock_code"] = panel["stock_code"].str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    manager = MarketDataManager(cache_dir=HFQ_BAOSTOCK_CACHE)
    histories = manager.load_histories(
        sorted(panel["stock_code"].unique()), "2014-01-01", "2025-12-31", min_rows=1, allow_network=False,
    )
    replacement: dict[int, float] = {}
    for code, frame in panel.groupby("stock_code", sort=False):
        history = histories.get(code)
        if history is None or history.empty:
            continue
        history = history.copy()
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history = history.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        score_by_date = pd.Series(
            calculate_entry_timing_scores_from_standard_history(history).to_numpy(),
            index=history["date"].dt.strftime("%Y-%m-%d"),
        )
        for row_index, signal_date in frame["signal_date"].items():
            value = score_by_date.get(str(signal_date))
            if value is not None and pd.notna(value):
                replacement[row_index] = float(value)
    candidate = panel.copy()
    candidate["raw_technical_score"] = candidate.index.map(replacement).fillna(50.0)
    candidate["technical_score"] = candidate["raw_technical_score"]
    # The replayer uses base_rank only for the frozen legacy ordering. Remove
    # it so the candidate is genuinely ranked from the altered factor.
    return candidate.drop(columns=["base_rank"], errors="ignore")


def _average_top_n_overlap(control_audit: pd.DataFrame, candidate_audit: pd.DataFrame) -> float | None:
    left = control_audit.set_index("execution_date")
    right = candidate_audit.set_index("execution_date")
    overlaps: list[float] = []
    for date in left.index.intersection(right.index):
        a = {code for code in str(left.loc[date, "end_codes"]).split(",") if code}
        b = {code for code in str(right.loc[date, "end_codes"]).split(",") if code}
        if a or b:
            overlaps.append(len(a & b) / len(a | b))
    return round(sum(overlaps) / len(overlaps), 4) if overlaps else None


def _average_momentum_technical_correlation(panel: pd.DataFrame) -> float | None:
    values: list[float] = []
    for _, frame in panel.groupby("signal_date"):
        momentum = pd.to_numeric(frame["raw_momentum_score"], errors="coerce")
        technical = pd.to_numeric(frame["raw_technical_score"], errors="coerce")
        # Spearman equals the Pearson correlation of ranks. This keeps the
        # research runner free of an optional scipy dependency.
        correlation = momentum.rank().corr(technical.rank(), method="pearson")
        if pd.notna(correlation):
            values.append(float(correlation))
    return round(sum(values) / len(values), 4) if values else None


def _report(control: dict, candidate: dict, overlap: float | None, control_corr: float | None, candidate_corr: float | None) -> str:
    lines = [
        "# Entry Timing Refactor Research", "", "## Scope", "",
        "- Classification: `research_experiment`, in-sample snapshot replay.",
        "- Same immutable V3 no-gate panel, Historical Universe, PIT fundamentals, risk/execution eligibility, Top20, fees, and HFQ return prices.",
        "- Only the technical score differs: frozen legacy definition versus Entry Timing definition.",
        "- No V1, V2, or V3 baseline artifact was changed or replaced.", "", "## Results", "",
        "| Metric | Legacy Technical | Entry Timing |", "| --- | ---: | ---: |",
    ]
    for key, label in (("total_return_pct", "Total Return"), ("cagr_pct", "CAGR"), ("annualized_volatility_pct", "Annual Volatility"), ("sharpe_ratio", "Sharpe"), ("max_drawdown_pct", "Max Drawdown"), ("turnover_pct", "Turnover"), ("transaction_cost", "Transaction Cost"), ("cash_days", "Cash Days")):
        suffix = "%" if key.endswith("_pct") else ""
        lines.append(f"| {label} | {control.get(key)}{suffix} | {candidate.get(key)}{suffix} |")
    lines.extend([
        "", "## Structural Diagnostics", "",
        f"- Average Top20 Jaccard overlap by execution date: `{overlap}`.",
        f"- Average within-date Spearman correlation, Momentum vs legacy Technical: `{control_corr}`.",
        f"- Average within-date Spearman correlation, Momentum vs Entry Timing: `{candidate_corr}`.",
        "", "## Interpretation Boundary", "",
        "- Lower Momentum/Entry-Timing correlation means less trend overlap, not proof of alpha.",
        "- The result is in-sample and must not replace a formal baseline without robustness and out-of-sample work.",
        "- Public-event data remains excluded from both runs.",
    ])
    return "\n".join(lines) + "\n"


def run_entry_timing_refactor() -> dict:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    config = default_v3_config()
    weights = dict(config["factor_weights"])
    control_panel = pd.read_csv(BASE_PANEL, dtype={"stock_code": str})
    candidate_panel = _build_entry_timing_panel()
    candidate_path = EXPERIMENT_DIR / "candidate_entry_timing_score_panel.csv"
    candidate_panel.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    common = {"weights": weights, "top_n": int(config["top_n"]), "market_regime_gate": False}
    control_result = run_replay("entry_timing_control_legacy", panel_path=BASE_PANEL, **common)
    candidate_result = run_replay("entry_timing_candidate", panel_path=candidate_path, **common)
    control = _write_result("control_legacy", control_result)
    candidate = _write_result("candidate_entry_timing", candidate_result)
    overlap = _average_top_n_overlap(control_result.audit, candidate_result.audit)
    control_corr = _average_momentum_technical_correlation(control_panel)
    candidate_corr = _average_momentum_technical_correlation(candidate_panel)
    payload = {
        "experiment": "entry_timing_refactor", "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "research_experiment", "integrity_status": "validated_replay_snapshot",
        "only_changed_input": "technical_factor_definition", "control": control, "candidate": candidate,
        "average_top20_jaccard_overlap": overlap,
        "momentum_technical_spearman": {"legacy": control_corr, "entry_timing": candidate_corr},
    }
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXPERIMENT_DIR / "entry_timing_refactor_report.md").write_text(_report(control, candidate, overlap, control_corr, candidate_corr), encoding="utf-8")
    return payload


def finalize_existing_output() -> dict:
    """Finish a written experiment after a non-data/reporting interruption."""
    control_folder = EXPERIMENT_DIR / "control_legacy"
    candidate_folder = EXPERIMENT_DIR / "candidate_entry_timing"
    control = json.loads((control_folder / "summary.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_folder / "summary.json").read_text(encoding="utf-8"))
    control_panel = pd.read_csv(BASE_PANEL, dtype={"stock_code": str})
    candidate_panel = pd.read_csv(EXPERIMENT_DIR / "candidate_entry_timing_score_panel.csv", dtype={"stock_code": str})
    overlap = _average_top_n_overlap(
        pd.read_csv(control_folder / "rebalance_audit.csv"),
        pd.read_csv(candidate_folder / "rebalance_audit.csv"),
    )
    control_corr = _average_momentum_technical_correlation(control_panel)
    candidate_corr = _average_momentum_technical_correlation(candidate_panel)
    payload = {
        "experiment": "entry_timing_refactor", "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "research_experiment", "integrity_status": "validated_replay_snapshot",
        "only_changed_input": "technical_factor_definition", "control": control, "candidate": candidate,
        "average_top20_jaccard_overlap": overlap,
        "momentum_technical_spearman": {"legacy": control_corr, "entry_timing": candidate_corr},
    }
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXPERIMENT_DIR / "entry_timing_refactor_report.md").write_text(_report(control, candidate, overlap, control_corr, candidate_corr), encoding="utf-8")
    return payload


if __name__ == "__main__":
    print(json.dumps(run_entry_timing_refactor(), ensure_ascii=False, indent=2))
