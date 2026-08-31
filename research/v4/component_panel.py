"""Historical component-level score panels for V4 factor recipes.

The original V4 snapshot deliberately persisted only final factor scores.  A
recipe experiment cannot honestly change component weights by replaying those
final scores, so this module recomputes the requested factor from the frozen
HFQ price cache and the point-in-time fundamental cache.  It never fetches
network data and never alters V1/V2/V3/V4 baseline outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from factor_engine import clamp_score, history_until, pct_return
from fundamental_factor import score_point_in_time_fundamentals
from market_data_manager import MarketDataManager
from research.fundamentals.point_in_time_fundamentals import get_fundamentals
from research.v3.hfq_baostock_acquisition import HFQ_BAOSTOCK_CACHE
from research.v4.config import V4_OUTPUT_DIR
from research.v4.strategy_lab import V4_PANEL_PATH


COMPONENT_DIR = V4_OUTPUT_DIR / "factor_validation" / "component_panels"
FUNDAMENTAL_CACHE = Path(__file__).resolve().parents[2] / "data_cache" / "v3_fundamentals"
STOCK_FACTORS = {"momentum", "money_flow", "fundamental", "technical"}


def _number(value: Any) -> float | None:
    result = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(result) else float(result)


def _component_score(values: dict[str, float | None], recipe: dict[str, Any]) -> float | None:
    enabled = [item for item in recipe["components"] if item.get("enabled")]
    usable = [(float(item.get("weight", 0)), values.get(str(item["id"]))) for item in enabled]
    usable = [(weight, value) for weight, value in usable if weight > 0 and value is not None]
    if not usable:
        return None
    divisor = sum(weight for weight, _ in usable)
    return round(sum(weight * value for weight, value in usable) / divisor, 4)


def _momentum_components(window: pd.DataFrame) -> dict[str, float | None]:
    closes = pd.to_numeric(window.get("close"), errors="coerce").dropna()
    if len(closes) < 60:
        return {name: None for name in ("return_20d", "return_60d", "close_above_ma20", "ma20_above_ma60")}
    value20, value60 = pct_return(closes, 20), pct_return(closes, 60)
    ma20, ma60, latest = closes.tail(20).mean(), closes.tail(60).mean(), float(closes.iloc[-1])
    return {
        "return_20d": clamp_score(50 + max(-18, min(18, (value20 or 0) * 1.2))),
        "return_60d": clamp_score(50 + max(-18, min(18, (value60 or 0) * 0.6))),
        "close_above_ma20": 100.0 if latest > ma20 else 0.0,
        "ma20_above_ma60": 100.0 if ma20 > ma60 else 0.0,
    }


def _money_flow_components(window: pd.DataFrame) -> dict[str, float | None]:
    if len(window) < 25:
        return {name: None for name in ("volume_ratio_5_20", "amount_ratio", "price_volume_confirmation")}
    volume = pd.to_numeric(window.get("volume"), errors="coerce")
    amount = pd.to_numeric(window.get("amount"), errors="coerce")
    close = pd.to_numeric(window.get("close"), errors="coerce")
    vol5, vol20 = _number(volume.tail(5).mean()), _number(volume.tail(20).mean())
    amount_latest, amount20 = _number(amount.iloc[-1]), _number(amount.tail(20).mean())
    ratio_score = None if not vol20 else clamp_score(50 + max(-18, min(18, (vol5 / vol20 - 1) * 22)))
    amount_score = None if not amount_latest or not amount20 else clamp_score(50 + max(-12, min(12, (amount_latest / amount20 - 1) * 12)))
    confirm = 100.0 if len(close) >= 2 and close.iloc[-1] > close.iloc[-2] and vol20 and vol5 and vol5 / vol20 >= 1.2 else 0.0
    return {"volume_ratio_5_20": ratio_score, "amount_ratio": amount_score, "price_volume_confirmation": confirm}


def _entry_timing_components(window: pd.DataFrame) -> dict[str, float | None]:
    closes = pd.to_numeric(window.get("close"), errors="coerce").dropna()
    if len(closes) < 35:
        return {name: None for name in ("ma5_deviation", "three_day_chase", "pullback", "macd_change", "rsi", "volatility_gap")}
    latest = float(closes.iloc[-1]); ma5 = float(closes.tail(5).mean()); deviation = (latest / ma5 - 1) * 100
    deviation_score = 58.0 if -3 <= deviation <= 3 else 38.0 if deviation > 8 else 42.0 if deviation < -8 else 50.0
    return3 = (latest / float(closes.iloc[-4]) - 1) * 100
    chase_score = 38.0 if return3 > 10 else 44.0 if return3 > 6 else 50.0
    pullback = (latest / float(closes.tail(5).max()) - 1) * 100
    pullback_score = 56.0 if -5 <= pullback <= -1 else 42.0 if pullback < -8 else 50.0
    macd_line = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    histogram_change = (macd_line - macd_line.ewm(span=9, adjust=False).mean()).diff().iloc[-1]
    macd_score = 56.0 if histogram_change > 0 else 46.0
    delta = closes.diff(); gains = delta.clip(lower=0).rolling(6).mean(); losses = (-delta.clip(upper=0)).rolling(6).mean()
    rsi = 100 - 100 / (1 + gains.iloc[-1] / losses.iloc[-1]) if losses.iloc[-1] not in (0, None) else 100.0
    rsi_score = 56.0 if 45 <= rsi <= 70 else 38.0 if rsi > 82 else 44.0 if rsi < 25 else 50.0
    volatility = closes.pct_change().tail(10).std(ddof=1)
    gap = 0.0
    if "open" in window:
        opening = _number(pd.to_numeric(window["open"], errors="coerce").iloc[-1]); previous = _number(closes.iloc[-2])
        gap = abs(opening / previous - 1) if opening is not None and previous else 0.0
    stability_score = 34.0 if volatility > .06 or gap > .05 else 50.0
    return {"ma5_deviation": deviation_score, "three_day_chase": chase_score, "pullback": pullback_score, "macd_change": macd_score, "rsi": rsi_score, "volatility_gap": stability_score}


def _fundamental_components(code: str, signal_date: str, history: pd.DataFrame) -> dict[str, float | None]:
    # get_fundamentals enforces disclosure_date <= as_of_date.  Passing only
    # historical prices makes PE/PB point-in-time as well.
    record = get_fundamentals(code, signal_date, price_history=history, cache_dir=FUNDAMENTAL_CACHE, allow_network=False, strict=True)
    if record is None:
        return {name: None for name in ("quality", "growth", "cashflow", "valuation")}
    scored = score_point_in_time_fundamentals(record)
    return {"quality": scored.quality_score, "growth": scored.growth_score, "cashflow": scored.cashflow_score, "valuation": scored.valuation_score}


def build_component_score_panel(factor: str, recipe: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Build a factor-replaced score panel and return its immutable cache path."""
    if factor not in STOCK_FACTORS:
        raise ValueError("Only cross-sectional factors have component score panels.")
    if not V4_PANEL_PATH.exists():
        raise FileNotFoundError("V4 factor score panel is unavailable.")
    target = COMPONENT_DIR / factor / f"{recipe['recipe_id']}_{recipe['recipe_hash']}.csv"
    manifest = target.with_suffix(".json")
    if target.exists() and manifest.exists():
        return target, json.loads(manifest.read_text(encoding="utf-8"))
    panel = pd.read_csv(V4_PANEL_PATH, dtype={"stock_code": str})
    codes = sorted(panel["stock_code"].dropna().astype(str).unique())
    manager = MarketDataManager(cache_dir=HFQ_BAOSTOCK_CACHE)
    histories = manager.load_histories(codes, "2014-01-01", "2025-12-31", min_rows=1, allow_network=False)
    values: list[float | None] = []
    missing = 0
    for item in panel.itertuples(index=False):
        code, signal_date = str(item.stock_code), str(item.signal_date)
        history = histories.get(code, pd.DataFrame())
        window = history_until(history, signal_date)
        if factor == "momentum": components = _momentum_components(window)
        elif factor == "money_flow": components = _money_flow_components(window)
        elif factor == "technical": components = _entry_timing_components(window)
        else: components = _fundamental_components(code, signal_date, window)
        score = _component_score(components, recipe)
        values.append(score)
        missing += int(score is None)
    panel[f"raw_{factor}_score"] = values
    panel[f"{factor}_score"] = values
    # Missing recipe values are not treated as neutral.  The existing replay
    # marks them -inf, hence they cannot be selected until data is available.
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(target, index=False, encoding="utf-8-sig")
    metadata = {"factor_name": factor, "recipe_id": recipe["recipe_id"], "recipe_hash": recipe["recipe_hash"], "source_panel": str(V4_PANEL_PATH), "rows": len(panel), "missing_component_score_rows": missing, "data_visibility": "historical_prices_and_pit_fundamentals_only", "future_data_count": 0}
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return target, metadata
