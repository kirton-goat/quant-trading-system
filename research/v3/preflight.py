"""Strict offline data-integrity preflight for the 2015-2025 V3 research plan."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark import load_benchmark
from market_data_manager import MARKET_DATA_CACHE_DIR
from research.universe.index_members import DEFAULT_CACHE_DIR, HistoricalIndexDataError, get_index_snapshot
from research.v3.execution_policy import POLICY_ID


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight"
DEFAULT_CONFIG_PATH = BASE_DIR / "backtest_config_v3_research.json"
# Keep this path local: importing the acquisition module here would create a
# configuration/downloader import cycle during a command-line preflight.
V3_STRATEGY_PRICE_CACHE = BASE_DIR / "data_cache" / "v3_strategy_prices_hfq_baostock"
V3_FUNDAMENTAL_CACHE = BASE_DIR / "data_cache" / "v3_fundamentals"
CORPORATE_ACTION_VALIDATION = OUTPUT_DIR / "corporate_action_validation_summary.json"
EXECUTION_ELIGIBILITY_SUMMARY = OUTPUT_DIR / "execution_eligibility_summary.json"


@dataclass
class V3PreflightResult:
    sample_start: str
    sample_end: str
    required_history_start: str
    rebalance_dates: int
    historical_universe_passed: bool
    market_data_passed: bool
    pit_fundamentals_passed: bool
    corporate_action_price_passed: bool
    benchmark_passed: bool
    integrity_status: str
    issues: list[str] = field(default_factory=list)
    universe_failures: list[dict[str, str]] = field(default_factory=list)
    market_cache_summary: dict[str, Any] = field(default_factory=dict)
    fundamentals_summary: dict[str, Any] = field(default_factory=dict)


def required_history_start(sample_start: str) -> str:
    """Use a conservative history window for 120d prices plus PIT YoY/TTM statements.

    900 calendar days is intentionally longer than the current technical 120d
    lookback; it allows a prior-year same-period filing and its disclosure lag.
    """
    return (pd.Timestamp(sample_start) - pd.Timedelta(days=900)).date().isoformat()


def default_v3_config() -> dict[str, Any]:
    return {
        "research_version": "v3_long_sample_research",
        "result_classification": "research_experiment",
        "research_state": "in_sample_research",
        "sample_period": {"start": "2015-01-01", "end": "2025-12-31"},
        "universe": "CSI300_CSI500",
        "top_n": 20,
        "holding_days": 20,
        "rebalance_days": 20,
        "initial_capital": 1000000,
        "fee_rate": 0.0015,
        "slippage_rate": 0.0,
        "market_min_score": 40.0,
        "factor_weights": {
            "market_regime": 0.20,
            "momentum": 0.25,
            "money_flow": 0.20,
            "fundamental": 0.25,
            "technical": 0.10,
        },
        "market_regime_gate": True,
        "technical_variant": "legacy",
        "benchmark_setup": {"CSI300": "sh000300", "CSI500": "sh000905", "combined": "undefined"},
        "validation_required": [
            "historical_universe", "point_in_time_fundamentals", "future_price_leakage_zero",
            "corporate_action_price_audit", "continuous_rebalance_timeline", "benchmark_definition",
        ],
    }


def trading_rebalance_dates(start: str, end: str, rebalance_days: int = 20) -> list[str]:
    benchmark = load_benchmark("sh000300", "CSI 300", start, end)
    if benchmark.data.empty:
        return []
    dates = benchmark.data["date"].astype(str).tolist()
    return dates[60::max(1, rebalance_days)]


def _snapshot_failures(dates: list[str]) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for date in dates:
        for universe in ("CSI300", "CSI500"):
            try:
                get_index_snapshot(date, universe, cache_dir=DEFAULT_CACHE_DIR, allow_network=False)
            except HistoricalIndexDataError as exc:
                failures.append({"date": date, "universe": universe, "reason": str(exc)})
    return failures


def _market_cache_summary() -> dict[str, Any]:
    price_quality_path = OUTPUT_DIR / "price_quality_summary.json"
    price_quality: dict[str, Any] = {}
    if price_quality_path.exists():
        try:
            price_quality = json.loads(price_quality_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            price_quality = {}
    files = list(V3_STRATEGY_PRICE_CACHE.glob("*.csv"))
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    missing_metadata = 0
    for path in files:
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {"date", "source", "adjustment"})
            dates = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
            if dates.empty:
                continue
            ranges.append((dates.min(), dates.max()))
            if "source" not in frame.columns or "adjustment" not in frame.columns:
                missing_metadata += 1
        except Exception:
            continue
    return {
        "files": len(files),
        "earliest_date": min((item[0] for item in ranges), default=pd.NaT).date().isoformat() if ranges else None,
        "latest_date": max((item[1] for item in ranges), default=pd.NaT).date().isoformat() if ranges else None,
        "files_missing_price_semantics_metadata": missing_metadata,
        "cache_path": str(V3_STRATEGY_PRICE_CACHE),
        "primary_price_semantics": "hfq_total_return / BaoStock adjustflag=1",
        "price_quality_audit": price_quality,
    }


def _v3_price_coverage(windows: dict[str, tuple[str, str]], end: str) -> dict[str, Any]:
    complete = 0
    missing: list[str] = []
    incomplete: list[str] = []
    for code, (first_signal, last_signal) in sorted(windows.items()):
        path = V3_STRATEGY_PRICE_CACHE / f"{code}.csv"
        if not path.exists():
            missing.append(code)
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {"date", "source", "adjustment"})
            dates = pd.to_datetime(frame.get("date"), errors="coerce").dropna()
            metadata_ok = {"source", "adjustment"}.issubset(frame.columns) and set(frame["adjustment"].dropna()) == {"hfq_total_return"}
            required_start = pd.Timestamp(first_signal) - pd.Timedelta(days=120)
            required_end = min(pd.Timestamp(end), pd.Timestamp(last_signal) + pd.Timedelta(days=45))
            dates_ok = not dates.empty and dates.min() <= required_start + pd.Timedelta(days=7) and dates.max() >= required_end - pd.Timedelta(days=7)
            if metadata_ok and dates_ok:
                complete += 1
            else:
                incomplete.append(code)
        except Exception:
            incomplete.append(code)
    return {"required_codes": len(windows), "complete_codes": complete, "missing_codes": missing, "incomplete_codes": incomplete}


def _fundamental_cache_summary() -> dict[str, Any]:
    cache_dir = V3_FUNDAMENTAL_CACHE
    files = list(cache_dir.glob("*.csv"))
    disclosures: list[pd.Timestamp] = []
    disclosed_files = 0
    invalid_disclosure_dates = 0
    for path in files:
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in {"disclosure_date", "report_period"})
            values = pd.to_datetime(frame.get("disclosure_date"), errors="coerce").dropna()
            invalid_disclosure_dates += int((values < pd.Timestamp("1991-01-01")).sum())
            values = values[values >= pd.Timestamp("1991-01-01")]
            if not values.empty:
                disclosures.extend(values.tolist())
                disclosed_files += 1
        except Exception:
            continue
    return {
        "files": len(files), "files_with_disclosure_dates": disclosed_files,
        "earliest_valid_disclosure_date": min(disclosures).date().isoformat() if disclosures else None,
        "latest_disclosure_date": max(disclosures).date().isoformat() if disclosures else None,
        "invalid_disclosure_date_records": invalid_disclosure_dates,
        "cache_path": str(cache_dir),
    }


def run_preflight(config: dict[str, Any] | None = None) -> V3PreflightResult:
    config = config or default_v3_config()
    period = config["sample_period"]
    start, end = period["start"], period["end"]
    dates = trading_rebalance_dates(start, end, int(config.get("rebalance_days", 20)))
    failures = _snapshot_failures(dates)
    market = _market_cache_summary()
    fundamentals = _fundamental_cache_summary()
    issues: list[str] = []
    universe_ok = bool(dates) and not failures
    benchmark_ok = bool(dates)
    required_codes: set[str] = set()
    membership_windows: dict[str, list[str]] = {}
    if universe_ok:
        for date in dates:
            for universe in ("CSI300", "CSI500"):
                for member in get_index_snapshot(date, universe, cache_dir=DEFAULT_CACHE_DIR, allow_network=False).members:
                    required_codes.add(member.code)
                    membership_windows.setdefault(member.code, []).append(date)
        windows = {code: (min(values), max(values)) for code, values in membership_windows.items()}
        market["coverage"] = _v3_price_coverage(windows, end)
    else:
        market["coverage"] = {"required_codes": None, "complete_codes": 0, "missing_codes": [], "incomplete_codes": [], "blocked_by_universe": True}
    if not universe_ok:
        issues.append("Historical CSI300/CSI500 snapshots are missing for one or more V3 rebalance dates; strict research cannot fall back to current constituents.")
    coverage = market["coverage"]
    if not coverage.get("required_codes"):
        issues.append("V3 BaoStock HFQ strategy-price cache has no historical-membership coverage evidence.")
    execution_audit: dict[str, Any] = {}
    if EXECUTION_ELIGIBILITY_SUMMARY.exists():
        try:
            execution_audit = json.loads(EXECUTION_ELIGIBILITY_SUMMARY.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            execution_audit = {}
    # Raw all-member price gaps are diagnostic only. V3 uses an explicit
    # point-in-time suspension policy for eligible holdings, so the policy
    # must be present and the audit must cover the planned signal dates.
    execution_policy_ok = (
        execution_audit.get("signal_dates") == len(dates)
        and execution_audit.get("eligible_candidate_rows", 0) > 0
        and POLICY_ID == "stale_mark_and_rebalance_deferral_v1"
    )
    if not execution_policy_ok:
        issues.append("V3 execution-eligibility audit or the conservative missing-quote policy is unavailable.")
    if market.get("files_missing_price_semantics_metadata", 0):
        issues.append("Strategy-price cache lacks source/adjustment metadata; corporate-action return treatment is not auditable yet.")
    # ``coverage`` is a deliberately stricter raw-membership diagnostic: it
    # includes securities the strategy would reject as-of-date. The formal
    # gate is the filtered execution audit plus the no-look-ahead suspension
    # policy, rather than future-aware deletion of every diagnostic gap.
    market_ok = bool(coverage.get("required_codes")) and execution_policy_ok
    corporate_validation = {}
    if CORPORATE_ACTION_VALIDATION.exists():
        try:
            corporate_validation = json.loads(CORPORATE_ACTION_VALIDATION.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            corporate_validation = {}
    market_semantics_ok = (
        market_ok
        and market.get("files_missing_price_semantics_metadata", 0) == 0
        and execution_policy_ok
    )
    corporate_ok = market_semantics_ok and bool(corporate_validation.get("passed"))
    market["corporate_action_validation"] = corporate_validation
    market["execution_eligibility_audit"] = execution_audit
    market["execution_policy"] = POLICY_ID
    if universe_ok:
        fundamental_codes = {path.stem for path in V3_FUNDAMENTAL_CACHE.glob("*.csv")}
        fundamentals["coverage"] = {"required_codes": len(required_codes), "complete_codes": len(required_codes & fundamental_codes), "missing_codes": sorted(required_codes - fundamental_codes)}
    else:
        fundamentals["coverage"] = {"required_codes": None, "complete_codes": 0, "missing_codes": [], "blocked_by_universe": True}
    fundamental_coverage = fundamentals["coverage"]
    fundamental_ok = (
        bool(fundamentals.get("earliest_valid_disclosure_date"))
        and str(fundamentals["earliest_valid_disclosure_date"]) <= required_history_start(start)
        and int(fundamentals.get("invalid_disclosure_date_records", 0)) == 0
        and fundamental_coverage.get("required_codes") == fundamental_coverage.get("complete_codes")
    )
    if not fundamental_ok:
        issues.append("PIT fundamental cache has incomplete pre-sample coverage or invalid disclosure dates; it cannot yet validate a 2015 signal date.")
    if not benchmark_ok:
        issues.append("CSI300 calendar is unavailable for the requested V3 sample.")
    validated = universe_ok and market_ok and fundamental_ok and corporate_ok and benchmark_ok
    return V3PreflightResult(
        sample_start=start, sample_end=end, required_history_start=required_history_start(start),
        rebalance_dates=len(dates), historical_universe_passed=universe_ok, market_data_passed=market_ok,
        pit_fundamentals_passed=fundamental_ok, corporate_action_price_passed=corporate_ok,
        benchmark_passed=benchmark_ok, integrity_status="validated" if validated else "incomplete",
        issues=issues, universe_failures=failures, market_cache_summary=market, fundamentals_summary=fundamentals,
    )


def write_preflight(result: V3PreflightResult, output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "v3_preflight_summary.json"
    summary.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    failures = output_dir / "historical_universe_missing.csv"
    pd.DataFrame(result.universe_failures, columns=["date", "universe", "reason"]).to_csv(failures, index=False, encoding="utf-8-sig")
    report = output_dir / "v3_preflight_report.md"
    lines = [
        "# V3 Long-Sample Data Preflight", "",
        "- Classification: `research_experiment`", f"- Integrity status: `{result.integrity_status}`",
        f"- Sample: `{result.sample_start}` to `{result.sample_end}`", f"- Required raw-history start: `{result.required_history_start}`",
        f"- Planned rebalance dates: {result.rebalance_dates}", "",
        "## Checks", "",
        f"| Check | Passed |", "| --- | --- |",
        f"| Historical universe | {result.historical_universe_passed} |",
        f"| Market-data coverage | {result.market_data_passed} |",
        f"| PIT fundamentals coverage | {result.pit_fundamentals_passed} |",
        f"| Corporate-action/price semantics audit | {result.corporate_action_price_passed} |",
        f"| Benchmark calendar | {result.benchmark_passed} |", "",
        "## Current Cache Evidence", "",
        f"- Market cache: {json.dumps(result.market_cache_summary, ensure_ascii=False)}",
        f"- Fundamental cache: {json.dumps(result.fundamentals_summary, ensure_ascii=False)}", "",
        "## Blocking Issues", "",
    ]
    lines.extend([f"- {issue}" for issue in result.issues] or ["- None."])
    lines.extend([
        "", "## Decision", "",
        "This preflight does not execute a 2015-2025 strategy backtest. A result is only allowed after every check passes; otherwise any output remains `research_experiment / incomplete` and must not be compared with the formal dashboard baseline.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, failures, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict V3 long-sample data preflight.")
    parser.add_argument("--write-config", action="store_true", help="Write the isolated V3 research configuration.")
    args = parser.parse_args()
    config = default_v3_config()
    if args.write_config:
        DEFAULT_CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = run_preflight(config)
    _, _, report = write_preflight(result)
    print(f"V3 preflight: {result.integrity_status}; report={report}")


if __name__ == "__main__":
    main()
