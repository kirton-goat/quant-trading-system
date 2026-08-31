from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from factor_engine import calculate_factor_scores_for_universe, calculate_market_regime_score_from_history
from market_data_manager import normalize_market_data
from research.v3.preflight import default_v3_config
from research.v4.config import V4_OUTPUT_DIR, default_v4_config
from research.strategy_metadata import get_strategy_version as _get_strategy_version
from research.strategy_metadata import get_strategy_versions as _get_strategy_versions


BASE_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = BASE_DIR / "data_cache"
MARKET_DATA_DIR = CACHE_DIR / "market_data"
UNIVERSE_DIR = CACHE_DIR / "universe"
EVENT_DIR = CACHE_DIR / "events"
BENCHMARK_FILE = CACHE_DIR / "benchmark" / "sh000300.csv"
LOG_FILE = BASE_DIR / "logs" / "trading_log.csv"
BACKTEST_V1_SUMMARY = BASE_DIR / "backtest_v1_summary.json"
BACKTEST_V1_VERSION = "v1_historical_point_in_time"
BACKTEST_V2_SUMMARY = BASE_DIR / "backtest_v2_summary.json"
BACKTEST_V2_VERSION = "v2_continuous_rebalance"
V3_OUTPUT_DIR = BASE_DIR / "research" / "experiments" / "output" / "v3_long_sample_preflight"
V3_HFQ_PRICE_DIR = CACHE_DIR / "v3_strategy_prices_hfq_total_return"
V3_HFQ_BAOSTOCK_PRICE_DIR = CACHE_DIR / "v3_strategy_prices_hfq_baostock"
V3_FUNDAMENTAL_DIR = CACHE_DIR / "v3_fundamentals"
V3_QFQ_PRICE_DIR = CACHE_DIR / "v3_strategy_prices"
V3_LIQUIDITY_DIR = CACHE_DIR / "v3_liquidity_baostock"
V3_EXPECTED_CODES = 1522


def get_strategy_versions() -> dict[str, Any]:
    """Return strategy logic metadata without running or changing a strategy."""
    return _get_strategy_versions()


def get_strategy_version(version: str) -> dict[str, Any] | None:
    """Return one version's code-audited metadata for Strategy Explorer."""
    return _get_strategy_version(version)


def get_market_status() -> dict[str, Any]:
    benchmark = _read_csv(BENCHMARK_FILE)
    if benchmark.empty or "close" not in benchmark.columns:
        return {
            "regime": "neutral", "risk_score": 50.0, "trend_score": 50.0,
            "label": "数据不足", "as_of": None, "source": "本地缓存不可用",
            "note": "暂无沪深300本地缓存，已降级为中性状态。",
        }
    benchmark = _normalise_benchmark(benchmark)
    as_of = pd.to_datetime(benchmark.iloc[-1]["date"]).strftime("%Y-%m-%d")
    score = calculate_market_regime_score_from_history(benchmark.assign(stock_code="000300", open=benchmark["close"], high=benchmark["close"], low=benchmark["close"], volume=0, amount=0), as_of)
    regime = "risk_on" if score >= 62 else "risk_off" if score <= 45 else "neutral"
    label = {"risk_on": "风险偏好较高", "neutral": "市场中性", "risk_off": "风险偏好较低"}[regime]
    return {
        "regime": regime, "risk_score": round(score, 2), "trend_score": round(score, 2),
        "label": label, "as_of": as_of, "source": "沪深300本地日线缓存",
        "note": "研究状态由本地缓存计算，不代表实时交易指令。",
    }


def get_cached_ranking(limit: int = 20) -> list[dict[str, Any]]:
    histories = _load_cached_histories()
    if not histories:
        return []
    as_of = min(str(history.iloc[-1]["date"]) for history in histories.values() if not history.empty)
    status = get_market_status()
    scores = calculate_factor_scores_for_universe(histories, as_of, market_regime_score=status["risk_score"])
    names = _load_names()
    return [
        {
            "stock_code": item.stock_code,
            "stock_name": names.get(item.stock_code, item.stock_code),
            "as_of": item.date,
            "total_score": item.total_score,
            "momentum_score": item.momentum_score,
            "money_flow_score": item.money_flow_score,
            "fundamental_score": item.fundamental_score,
            "technical_score": item.technical_score,
            "market_regime_score": item.market_regime_score,
            "event_score": item.event_score,
            "source": "本地行情缓存横向计算",
        }
        for item in scores[: max(1, min(limit, 100))]
    ]


def get_backtest_results() -> dict[str, Any]:
    summary = _validated_backtest_v1_summary()
    if summary is None:
        return {
            "backtest_version": "legacy",
            "integrity_status": "incomplete",
            "models": [],
            "comparison": {},
            "note": "旧回测已隔离。请完成 Backtest v1.0 严格历史时点回测后再查看正式结果。",
        }
    models = [
        {
            **item,
            "historical_universe_verified": item.get("universe_mode") == "historical_point_in_time",
            "fundamental_point_in_time_verified": item.get("fundamental_mode") == "historical_point_in_time",
        }
        for item in summary.get("models", [])
    ]
    model_a = models[0] if len(models) > 0 else {}
    model_b = models[1] if len(models) > 1 else {}
    return {
        **summary,
        "models": models,
        "comparison": {
            "total_return_diff_pct": _difference(model_b.get("total_return_pct"), model_a.get("total_return_pct")),
            "annualized_return_diff_pct": _difference(model_b.get("annualized_return_pct"), model_a.get("annualized_return_pct")),
            "max_drawdown_diff_pct": _difference(model_b.get("max_drawdown_pct"), model_a.get("max_drawdown_pct")),
            "sharpe_diff": _difference(model_b.get("sharpe_ratio"), model_a.get("sharpe_ratio")),
        },
        "note": "仅展示已通过历史股票池与历史时点基本面检查的 Backtest v1.0。",
    }

def get_equity_curve(model: str = "a") -> list[dict[str, Any]]:
    if _validated_backtest_v1_summary() is None:
        return []
    suffix = "model_b" if model.lower() == "b" else "model_a"
    path = BASE_DIR / f"backtest_equity_curve_{suffix}.csv"
    curve = _read_csv(path)
    if curve.empty:
        return []
    curve["date"] = pd.to_datetime(curve["date"], errors="coerce")
    curve = curve.dropna(subset=["date", "equity"]).sort_values("date")
    benchmark = _normalise_benchmark(_read_csv(BENCHMARK_FILE))
    if not benchmark.empty:
        benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
        merged = curve.merge(benchmark[["date", "close"]], on="date", how="left")
        first = merged["close"].dropna().iloc[0] if not merged["close"].dropna().empty else None
        merged["benchmark_return_pct"] = ((merged["close"] / first - 1) * 100) if first else pd.NA
    else:
        merged = curve.copy()
        merged["benchmark_return_pct"] = pd.NA
    return [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "equity": _number(row.get("equity"), 0.0),
            "return_pct": _number(row.get("return_pct"), 0.0),
            "benchmark_return_pct": _nullable_number(row.get("benchmark_return_pct")),
        }
        for _, row in merged.iterrows()
    ]


def get_trade_history(limit: int = 200) -> list[dict[str, Any]]:
    data = _read_csv(LOG_FILE)
    if data.empty:
        return []
    data["_dashboard_status"] = data.apply(_normalise_sim_status, axis=1)
    actual_trade_statuses = {"pending_exit", "open", "completed", "closed", "market_data_missing"}
    data = data[data["_dashboard_status"].isin(actual_trade_statuses)]
    if data.empty:
        return []
    data = data.tail(max(1, min(limit, 500))).iloc[::-1]
    return [
        {
            "timestamp": str(row.get("时间", "")),
            "stock_code": str(row.get("关联股票", "")),
            "title": str(row.get("新闻标题", "")),
            "signal_source": str(row.get("新闻来源", "")),
            "signal_type": str(row.get("事件类型", "研究记录")),
            "entry_price": _display_value(row.get("sim_entry_price", row.get("当前价格", ""))),
            "status": str(row.get("_dashboard_status", "unknown")),
            "pnl_pct": _display_value(row.get("sim_pnl_pct", "")),
            "action": str(row.get("AI操作", "观望")),
            "note": str(row.get("sim_note", row.get("AI逻辑", ""))),
        }
        for _, row in data.iterrows()
    ]


def get_trade_summary() -> dict[str, Any]:
    data = _read_csv(LOG_FILE)
    if data.empty:
        return {
            "total_records": 0,
            "research_only_records": 0,
            "market_data_missing_records": 0,
            "active_records": 0,
            "completed_records": 0,
            "message": "当前没有模拟交易或研究观察记录。",
        }

    statuses = data.apply(_normalise_sim_status, axis=1)
    research_only = int((statuses == "research_only").sum())
    market_data_missing = int((statuses == "market_data_missing").sum())
    active = int(statuses.isin({"pending_exit", "open"}).sum())
    completed = int(statuses.isin({"completed", "closed"}).sum())
    return {
        "total_records": int(len(data)),
        "research_only_records": research_only,
        "market_data_missing_records": market_data_missing,
        "active_records": active,
        "completed_records": completed,
        "message": (
            f"{research_only} 条事件记录属于研究观察，未通过多因子交易许可；"
            f"真正缺少行情的交易意图为 {market_data_missing} 条。"
        ),
    }


def _normalise_sim_status(row: Any) -> str:
    status = str(row.get("sim_status", "") or "").strip()
    direction = str(row.get("sim_direction", "") or "").strip().lower()
    if status == "not_opened" and direction in {"", "none", "nan"}:
        return "research_only"
    if status == "not_opened":
        return "market_data_missing"
    return status or "unknown"


def get_events(event_type: str = "all", limit: int = 100, official_only: bool = True) -> list[dict[str, Any]]:
    files = []
    if event_type in {"all", "policy"}:
        files.append(EVENT_DIR / "policy_events.csv")
    if event_type in {"all", "announcement"}:
        files.append(EVENT_DIR / "announcement_events.csv")
    frames = [_read_csv(path) for path in files]
    data = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    if data.empty:
        return []
    if official_only:
        # Older aggregated-news records stay on disk for research traceability,
        # but the dashboard's event page is intentionally an official-source
        # research feed.
        data = data[data.get("is_official", pd.Series(False, index=data.index)).astype(str).str.lower().isin({"true", "1", "yes"})]
    if data.empty:
        return []
    data["published_at"] = pd.to_datetime(data["published_at"], errors="coerce")
    data = data.dropna(subset=["published_at"]).sort_values("published_at", ascending=False).head(max(1, min(limit, 500)))
    return [
        {
            "published_at": row["published_at"].strftime("%Y-%m-%d"),
            "event_type": str(row.get("event_type", "")),
            "stock_code": _normalise_stock_code(row.get("stock_code", "")),
            "industry": _text_or_blank(row.get("industry", "")),
            "score": _number(row.get("score"), 50.0),
            "source": _text_or_blank(row.get("source", "")),
            "publisher": _text_or_blank(row.get("publisher", "")),
            "source_url": _text_or_blank(row.get("source_url", "")),
            "source_kind": _text_or_blank(row.get("source_kind", "")),
            "is_official": str(row.get("is_official", "")).lower() in {"true", "1", "yes"},
            "fetched_at": _text_or_blank(row.get("fetched_at", "")),
            "title": _text_or_blank(row.get("title", "")),
            "note": _text_or_blank(row.get("note", "")),
        }
        for _, row in data.iterrows()
    ]


def get_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "market_cache_files": len(list(MARKET_DATA_DIR.glob("*.csv"))) if MARKET_DATA_DIR.exists() else 0,
        "has_model_a": (BASE_DIR / "backtest_equity_curve_model_a.csv").exists(),
        "has_model_b": (BASE_DIR / "backtest_equity_curve_model_b.csv").exists(),
        "event_files": len(list(EVENT_DIR.glob("*.csv"))) if EVENT_DIR.exists() else 0,
    }


def get_backtest_version_catalog() -> dict[str, Any]:
    """Read-only catalog used to label Dashboard data by version and purpose."""
    v1 = _validated_backtest_v1_summary()
    v2 = _read_json(BACKTEST_V2_SUMMARY)
    v2_validated = (
        v2.get("backtest_version") == BACKTEST_V2_VERSION
        and v2.get("integrity_status") == "validated"
    )

    def entry(summary: dict[str, Any], *, version: str, label: str, status: str, scope: str) -> dict[str, Any]:
        config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
        models = summary.get("models") if isinstance(summary.get("models"), list) else []
        model = models[0] if models else {}
        return {
            "version": version,
            "label": label,
            "status": status,
            "scope": scope,
            "period": f"{config.get('start_date', 'N/A')} to {config.get('end_date', 'N/A')}",
            "cagr_pct": model.get("annualized_return_pct"),
            "sharpe_ratio": model.get("sharpe_ratio"),
            "max_drawdown_pct": model.get("max_drawdown_pct"),
        }

    official_versions = []
    if v1:
        official_versions.append(entry(v1, version="v1_historical_point_in_time", label="V1 Historical PIT", status="frozen", scope="historical_reference"))
    if v2_validated:
        official_versions.append(entry(v2, version="v2_continuous_rebalance", label="V2 Continuous Rebalance", status="validated", scope="official_baseline"))

    v4_manifest = _read_json(V4_OUTPUT_DIR / "v4_manifest.json")
    v4_result = v4_manifest.get("summary") if isinstance(v4_manifest.get("summary"), dict) else {}
    return {
        "official_current_version": "v2_continuous_rebalance" if v2_validated else ("v1_historical_point_in_time" if v1 else None),
        "data_version": None,
        "official_versions": official_versions,
        "research_versions": [
            {
                "version": "v3_long_sample_research",
                "label": "V3 Long-History Research Platform",
                "status": "research",
                "scope": "research_platform",
                "period": "2015-01-01 to 2025-12-31",
                "cagr_pct": None,
                "sharpe_ratio": None,
                "max_drawdown_pct": None,
            },
            {
                "version": "v4_entry_timing_candidate",
                "label": "V4 Entry Timing Five-Factor Research",
                "status": "research",
                "scope": "historical_research_experiment",
                "period": "2015-01-01 to 2025-12-31",
                "cagr_pct": v4_result.get("cagr_pct"),
                "sharpe_ratio": v4_result.get("sharpe_ratio"),
                "max_drawdown_pct": v4_result.get("max_drawdown_pct"),
                "result_status": v4_manifest.get("status", "not_run"),
            },
        ],
    }


def get_v3_research_status() -> dict[str, Any]:
    """Expose only data-integrity progress, never unvalidated V3 performance."""
    config = default_v3_config()
    preflight = _read_json(V3_OUTPUT_DIR / "v3_preflight_summary.json")
    corporate = _read_json(V3_OUTPUT_DIR / "corporate_action_validation_summary.json")
    corporate_hfq_passed = bool(corporate.get("passed")) and corporate.get("price_mode") == "hfq_total_return"
    quality = _read_json(V3_OUTPUT_DIR / "price_quality_summary.json")
    hfq_files = len(list(V3_HFQ_PRICE_DIR.glob("*.csv"))) if V3_HFQ_PRICE_DIR.exists() else 0
    hfq_baostock_files = len(list(V3_HFQ_BAOSTOCK_PRICE_DIR.glob("*.csv"))) if V3_HFQ_BAOSTOCK_PRICE_DIR.exists() else 0
    fundamentals = len(list(V3_FUNDAMENTAL_DIR.glob("*.csv"))) if V3_FUNDAMENTAL_DIR.exists() else 0
    qfq_files = len(list(V3_QFQ_PRICE_DIR.glob("*.csv"))) if V3_QFQ_PRICE_DIR.exists() else 0
    liquidity_files = len(list(V3_LIQUIDITY_DIR.glob("*.csv"))) if V3_LIQUIDITY_DIR.exists() else 0
    checks = [
        {"name": "历史股票池", "status": "passed", "detail": "131 个计划调仓日已缓存，严格使用历史 CSI300/CSI500 成分。"},
        {"name": "策略收益价格", "status": "building" if hfq_baostock_files < V3_EXPECTED_CODES else "pending_audit", "detail": f"BaoStock HFQ 总回报序列 {hfq_baostock_files}/{V3_EXPECTED_CODES}；腾讯 HFQ 缓存仅保留作日期缺口审计，不用于 V3 收益。"},
        {"name": "历史时点基本面", "status": "building" if fundamentals < V3_EXPECTED_CODES else "pending_audit", "detail": f"PIT 财报缓存 {fundamentals}/{V3_EXPECTED_CODES}；仅允许披露日不晚于回测日的数据。"},
        {"name": "历史流动性数据", "status": "building" if liquidity_files < V3_EXPECTED_CODES else "pending_audit", "detail": f"BaoStock 成交额（CNY）缓存 {liquidity_files}/{V3_EXPECTED_CODES}；与 HFQ 收益价格分离，避免单位混用。"},
        {"name": "公司行动连续性", "status": "passed" if corporate_hfq_passed else "pending", "detail": "必须以 HFQ 总回报口径完成代表性分红/除权连续性审计；旧 QFQ 样本不作为 V3 通过依据。"},
        {"name": "全量价格质量", "status": "passed" if quality.get("passed") else "pending", "detail": "检查覆盖范围、复权口径、日期顺序、重复行与非正价格。"},
    ]
    ready = hfq_baostock_files >= V3_EXPECTED_CODES and fundamentals >= V3_EXPECTED_CODES and liquidity_files >= V3_EXPECTED_CODES and corporate_hfq_passed and bool(quality.get("passed")) and preflight.get("integrity_status") == "validated"
    return {
        "research_version": str(config["research_version"]),
        "technical_variant": str(config.get("technical_variant", "legacy")),
        "factor_weights": dict(config["factor_weights"]),
        "top_n": int(config["top_n"]),
        "rebalance_days": int(config["rebalance_days"]),
        "market_regime_gate": bool(config["market_regime_gate"]),
        "market_min_score": float(config["market_min_score"]),
        "classification": "research_experiment",
        "sample_period": "2015-01-01 to 2025-12-31",
        "integrity_status": "validated" if ready else "building",
        "performance_visible": False,
        "message": "V3 仅展示数据完整性进度；未通过预检前不生成或展示收益曲线。",
        "progress": {"expected_codes": V3_EXPECTED_CODES, "hfq_baostock_price_files": hfq_baostock_files, "pit_fundamental_files": fundamentals, "liquidity_files": liquidity_files, "legacy_hfq_audit_files": hfq_files, "legacy_qfq_audit_files": qfq_files},
        "checks": checks,
        "preflight": preflight,
    }


def get_v4_research_status() -> dict[str, Any]:
    """Expose the isolated V4 candidate without borrowing V3 result files."""
    default_config = default_v4_config()
    shared_data_status = get_v3_research_status()
    shared_progress = shared_data_status.get("progress", {})
    fundamentals = int(shared_progress.get("pit_fundamental_files", 0))
    liquidity_files = int(shared_progress.get("liquidity_files", 0))
    corporate_hfq_passed = next((item.get("status") == "passed" for item in shared_data_status.get("checks", []) if item.get("name") == "公司行动连续性"), False)
    manifest = _read_json(V4_OUTPUT_DIR / "v4_manifest.json")
    result = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    result_config = result.get("config") if isinstance(result.get("config"), dict) else {}
    config = {**default_config, **result_config}
    completed = manifest.get("status") == "completed"
    price_quality = _read_json(V3_OUTPUT_DIR / "price_quality_summary_v2.json")
    fundamental_quality = _read_json(V3_OUTPUT_DIR / "pit_fundamental_quality_audit.json")
    liquidity_quality = _read_json(V3_OUTPUT_DIR / "liquidity_quality_summary.json")
    original_price_flags = int(price_quality.get("original_flagged_stocks", 0))
    true_price_gaps = int(price_quality.get("true_price_gaps", 0))
    strategy_gaps = int(price_quality.get("special_exit_holding_records", 0))
    price_quality_passed = bool(price_quality.get("passed"))
    checks = [
        {"name": "V4 独立基线", "status": "passed" if completed else "pending_validation", "detail": f"{result.get('model', 'model_a_no_gate')} 已完成；future fundamental data={result.get('future_fundamental_data', '—')}。这是研究运行状态，不代表全量数据质量已经通过。"},
        {"name": "历史股票池", "status": "validated", "detail": "131 个计划调仓日已缓存，严格使用历史 CSI300/CSI500 成分。"},
        {"name": "策略收益价格", "status": "validated" if price_quality_passed else "issues_found" if true_price_gaps else "pending_validation", "detail": f"缓存文件完整：{len(list(V3_HFQ_BAOSTOCK_PRICE_DIR.glob('*.csv')))}/{V3_EXPECTED_CODES}；旧规则标记 {original_price_flags} 条边界记录，生命周期审计确认真实价格缺口 {true_price_gaps} 条；V4 基线持仓受特殊终止事件影响 {strategy_gaps} 条。"},
        {"name": "历史时点基本面", "status": "validated" if fundamental_quality.get("passed") else "pending_validation", "detail": f"PIT 财报缓存 {fundamentals}/{V3_EXPECTED_CODES}；全量元数据审计 {'通过' if fundamental_quality.get('passed') else '尚未完成'}，回测可见性仍强制 disclosure_date <= signal_date。"},
        {"name": "历史流动性数据", "status": "validated" if liquidity_quality.get("passed") else "pending_validation", "detail": f"BaoStock 成交额（CNY）缓存 {liquidity_files}/{V3_EXPECTED_CODES}；全量单位、日期和异常值审计 {'通过' if liquidity_quality.get('passed') else '尚未完成'}。"},
        {"name": "公司行动连续性", "status": "validated" if corporate_hfq_passed else "pending_validation", "detail": "代表性分红/除权连续性审计通过；它不替代缺口股票的逐段公司行动核验。"},
        {"name": "全量价格质量", "status": "validated" if price_quality_passed else "issues_found" if true_price_gaps else "pending_validation", "detail": f"覆盖范围、复权口径、日期顺序、重复行和非正价格已审计；正常终止交易 {price_quality.get('valid_end_of_trading', 0)} 条、正常上市起点 {price_quality.get('valid_listing_start', 0)} 条、真实未闭合缺口 {true_price_gaps} 条。"},
    ]
    data_integrity_status = "validated" if completed and price_quality_passed and fundamental_quality.get("passed") and liquidity_quality.get("passed") else "partial" if original_price_flags or true_price_gaps or strategy_gaps else "pending_validation"
    return {
        **shared_data_status,
        "research_version": str(config["research_version"]),
        "technical_variant": str(config["technical_variant"]),
        "factor_weights": dict(config["factor_weights"]),
        "top_n": int(config["top_n"]),
        "rebalance_days": int(config["rebalance_days"]),
        "market_regime_gate": bool(config["market_regime_gate"]),
        "market_min_score": float(config["market_min_score"]),
        "integrity_status": data_integrity_status,
        "data_integrity_status": data_integrity_status,
        "data_version": "v4_hfq_baostock_pit_price_lifecycle_audit_v2",
        "price_gap_summary": {"cache_files": len(list(V3_HFQ_BAOSTOCK_PRICE_DIR.glob("*.csv"))), "expected_files": V3_EXPECTED_CODES, "original_flags": original_price_flags, "valid_end_of_trading": int(price_quality.get("valid_end_of_trading", 0)), "valid_listing_start": int(price_quality.get("valid_listing_start", 0)), "valid_suspension": int(price_quality.get("valid_suspension", 0)), "total_gaps": true_price_gaps, "strategy_impacting_gaps": strategy_gaps, "quality_status": "validated" if price_quality_passed else "issues_found"},
        "checks": checks,
        "result_status": manifest.get("status", "not_run"),
        "manifest": manifest,
        "performance_visible": completed,
        "message": "V4 的历史价格已按实际可交易生命周期重新审计：退市后无行情与新股上市前无行情不计为缺口；当前真实价格缺口为 0。V4 仍是研究版本，不构成正式策略结论。" if completed and data_integrity_status == "validated" else "V4 使用 Entry Timing；它与 V3 Legacy 结果完全隔离，必须独立重跑后才会产生可展示的新结果。",
    }


def get_v4_price_gaps() -> dict[str, Any]:
    """Expose lifecycle price-audit records; boundary events are not gaps."""
    path = V3_OUTPUT_DIR / "price_quality_audit_v2.csv"
    summary = _read_json(V3_OUTPUT_DIR / "price_quality_summary_v2.json")
    if not path.exists():
        return {"items": [], "summary": {"total_gaps": 0, "strategy_impacting_gaps": 0}}
    data = _read_csv(path)
    data = data[data.get("passed", pd.Series(False, index=data.index)).astype(str).str.lower().ne("true")] if not data.empty else data
    items = data.to_dict("records") if not data.empty else []
    for item in items:
        for key in ("eligible_period_affected", "holding_period_affected", "execution_price_affected", "true_price_gap", "valid_end_of_trading"):
            item[key] = str(item.get(key, "")).lower() in {"true", "1", "yes"}
    return {"items": items, "summary": {"original_flags": int(summary.get("original_flagged_stocks", len(items))), "valid_end_of_trading": int(summary.get("valid_end_of_trading", 0)), "valid_listing_start": int(summary.get("valid_listing_start", 0)), "valid_suspension": int(summary.get("valid_suspension", 0)), "total_gaps": int(summary.get("true_price_gaps", 0)), "strategy_impacting_gaps": int(summary.get("special_exit_holding_records", 0)), "quality_status": "validated" if summary.get("passed") else "issues_found"}}


def _validated_backtest_v1_summary() -> dict[str, Any] | None:
    if not BACKTEST_V1_SUMMARY.exists():
        return None
    try:
        summary = json.loads(BACKTEST_V1_SUMMARY.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if summary.get("backtest_version") != BACKTEST_V1_VERSION:
        return None
    if summary.get("integrity_status") != "validated":
        return None
    return summary


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError, TypeError):
        return {}


def _backtest_metrics(model: str) -> dict[str, Any]:
    label = "模型B：政策公告增强" if model == "b" else "模型A：基础量化"
    raw_curve = _read_csv(BASE_DIR / f"backtest_equity_curve_model_{model}.csv")
    curve = pd.DataFrame(get_equity_curve(model))
    trades = _read_csv(BASE_DIR / f"backtest_trades_model_{model}.csv")
    if curve.empty:
        return {"model": label, "note": "尚未找到对应的回测输出。", "trade_count": 0}
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    returns = equity.pct_change().dropna()
    total = (equity.iloc[-1] / equity.iloc[0] - 1) * 100 if len(equity) > 1 else 0.0
    # The backtest engine applies trading fees at rebalance. Its trade ledger is
    # therefore the authoritative source for final portfolio equity.
    if not trades.empty and {"entry_equity", "exit_equity"}.issubset(trades.columns):
        entries = pd.to_numeric(trades["entry_equity"], errors="coerce").dropna()
        exits = pd.to_numeric(trades["exit_equity"], errors="coerce").dropna()
        if not entries.empty and not exits.empty and float(entries.iloc[0]) > 0:
            total = (float(exits.iloc[-1]) / float(entries.iloc[0]) - 1) * 100
    years = max(len(equity) / 252, 1 / 252)
    annualized = ((1 + total / 100) ** (1 / years) - 1) * 100
    drawdown = (equity / equity.cummax() - 1).min() * 100
    sharpe = (returns.mean() / returns.std() * math.sqrt(252)) if len(returns) > 1 and returns.std() else None
    trade_returns = pd.to_numeric(trades.get("portfolio_return_pct"), errors="coerce").dropna().drop_duplicates() if not trades.empty else pd.Series(dtype=float)
    positives = trade_returns[trade_returns > 0]
    negatives = trade_returns[trade_returns < 0]
    win_rate = float((trade_returns > 0).mean()) if not trade_returns.empty else None
    pl_ratio = float(positives.mean() / abs(negatives.mean())) if not positives.empty and not negatives.empty and negatives.mean() else None
    historical_universe_verified = (
        not raw_curve.empty
        and "universe_mode" in raw_curve.columns
        and raw_curve["universe_mode"].astype(str).eq("historical_point_in_time").all()
    )
    fundamental_point_in_time_verified = (
        not raw_curve.empty
        and "fundamental_mode" in raw_curve.columns
        and raw_curve["fundamental_mode"].astype(str).eq("historical_point_in_time").all()
    )
    point_in_time_validated = (
        historical_universe_verified
        and fundamental_point_in_time_verified
        and "backtest_integrity" in raw_curve.columns
        and raw_curve["backtest_integrity"].astype(str).eq("point_in_time_validated").all()
    )
    provenance_note = (
        "该结果已通过历史股票池与历史时点基本面检查。"
        if point_in_time_validated
        else "该结果尚未同时通过历史股票池和历史时点基本面检查，请重新运行严格历史回测。"
    )
    return {
        "model": label,
        "total_return_pct": round(float(total), 4),
        "annualized_return_pct": round(float(annualized), 4),
        "max_drawdown_pct": round(float(drawdown), 4),
        "sharpe_ratio": round(float(sharpe), 4) if sharpe is not None else None,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit_loss_ratio": round(pl_ratio, 4) if pl_ratio is not None else None,
        "trade_count": int(len(trade_returns)),
        "as_of": str(curve.iloc[-1]["date"]),
        "historical_universe_verified": historical_universe_verified,
        "fundamental_point_in_time_verified": fundamental_point_in_time_verified,
        "backtest_integrity": "point_in_time_validated" if point_in_time_validated else "incomplete",
        "integrity_status": "validated" if point_in_time_validated else "incomplete",
        "note": f"{provenance_note} 本结果仅用于研究，不是实盘表现。",
    }


def _load_cached_histories() -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    if not MARKET_DATA_DIR.exists():
        return histories
    for path in MARKET_DATA_DIR.glob("*.csv"):
        code = path.stem
        data = normalize_market_data(_read_csv(path), code)
        if len(data) >= 60:
            histories[code] = data
    return histories


def _load_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for path in UNIVERSE_DIR.glob("*.csv") if UNIVERSE_DIR.exists() else []:
        data = _read_csv(path)
        if {"stock_code", "stock_name"}.issubset(data.columns):
            names.update(dict(zip(data["stock_code"].astype(str).str.zfill(6), data["stock_name"].astype(str))))
    return names


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    return pd.DataFrame()


def _normalise_benchmark(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["date", "close"])
    data = data.copy()
    data["date"] = pd.to_datetime(data.get("date"), errors="coerce")
    data["close"] = pd.to_numeric(data.get("close"), errors="coerce")
    return data.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)


def _number(value: Any, default: float) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return default


def _normalise_stock_code(value: Any) -> str:
    """Preserve six-digit A-share codes when CSV inference turns them numeric."""
    text = _text_or_blank(value)
    match = pd.Series([text]).str.extract(r"(\d{1,6})", expand=False).iloc[0]
    return str(match).zfill(6) if pd.notna(match) else ""


def _text_or_blank(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _nullable_number(value: Any) -> float | None:
    try:
        return round(float(value), 4) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def _display_value(value: Any) -> str | float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return str(value)


def _difference(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return round(value - base, 4)
