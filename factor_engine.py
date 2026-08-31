from __future__ import annotations

from dataclasses import dataclass, field
import multiprocessing as mp
from typing import Any, Protocol

import pandas as pd


@dataclass
class FactorResult:
    factor_name: str
    score: float
    weight: float
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    risk_tags: list[str] = field(default_factory=list)

    def contribution(self) -> float:
        return clamp_score(self.score) * self.weight

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "score": round(clamp_score(self.score), 2),
            "weight": self.weight,
            "reason": self.reason,
            "details": self.details,
            "risk_tags": self.risk_tags,
            "contribution": round(self.contribution(), 2),
        }


@dataclass(frozen=True)
class StockFactorScore:
    stock_code: str
    date: str
    momentum_score: float
    money_flow_score: float
    total_score: float
    technical_score: float = 50.0
    fundamental_score: float = 50.0
    market_regime_score: float = 50.0
    event_score: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "date": self.date,
            "momentum_score": round(self.momentum_score, 2),
            "money_flow_score": round(self.money_flow_score, 2),
            "technical_score": round(self.technical_score, 2),
            "fundamental_score": round(self.fundamental_score, 2),
            "market_regime_score": round(self.market_regime_score, 2),
            "event_score": round(self.event_score, 2),
            "total_score": round(self.total_score, 2),
        }


class Factor(Protocol):
    name: str
    weight: float

    def calculate(self, code: str, history: pd.DataFrame | None = None, context: dict[str, Any] | None = None) -> FactorResult:
        ...


def clamp_score(score: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(score)))


def normalize_pct(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 50.0
    if high == low:
        return 50.0
    return clamp_score((value - low) / (high - low) * 100)


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, "", "-"):
            return default
        return float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return default


def weighted_average(results: list[FactorResult]) -> float:
    total_weight = sum(result.weight for result in results)
    if total_weight <= 0:
        return 0.0
    return round(sum(result.contribution() for result in results) / total_weight, 2)


def fetch_history(code: str, days: int = 260) -> pd.DataFrame | None:
    import datetime as dt

    end_date = dt.datetime.now().strftime("%Y%m%d")
    start_date = (dt.datetime.now() - dt.timedelta(days=days * 2)).strftime("%Y%m%d")
    df = fetch_history_with_timeout(code, start_date, end_date, timeout=18)
    if df is None or df.empty:
        return None
    return df.tail(days).copy()


def fetch_history_with_timeout(code: str, start_date: str, end_date: str, timeout: int = 18) -> pd.DataFrame | None:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_history_worker, args=(queue, code, start_date, end_date))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return None
    if queue.empty():
        return None
    status, payload = queue.get()
    return payload if status == "ok" else None


def _history_worker(queue: mp.Queue, code: str, start_date: str, end_date: str) -> None:
    try:
        from market import fetch_daily_history_with_fallback

        queue.put(("ok", fetch_daily_history_with_fallback(code, start_date, end_date)))
    except Exception as exc:
        queue.put(("error", str(exc)))


def ensure_history(code: str, history: pd.DataFrame | None) -> pd.DataFrame | None:
    if history is not None and not history.empty:
        return history
    return fetch_history(code)


def calculate_stock_factor_score(
    stock_code: str,
    history: pd.DataFrame,
    as_of_date: str,
    momentum_weight: float = 0.25,
    money_flow_weight: float = 0.20,
    fundamental_weight: float = 0.25,
    technical_weight: float = 0.10,
    market_regime_weight: float = 0.20,
    market_regime_score: float = 50.0,
    fundamental_score: float | None = 50.0,
    event_score: float = 50.0,
    event_boost_weight: float = 0.0,
    technical_variant: str = "legacy",
) -> StockFactorScore | None:
    window = history_until(history, as_of_date)
    if len(window) < 60 or fundamental_score is None:
        return None
    momentum = calculate_momentum_score_from_standard_history(window)
    money_flow = calculate_money_flow_score_from_standard_history(window)
    technical = calculate_technical_score_from_standard_history(window, variant=technical_variant)
    total_weight = momentum_weight + money_flow_weight + fundamental_weight + technical_weight + market_regime_weight
    if total_weight <= 0:
        total_weight = 1.0
    base_total = (
        momentum * momentum_weight
        + money_flow * money_flow_weight
        + clamp_score(fundamental_score) * fundamental_weight
        + technical * technical_weight
        + clamp_score(market_regime_score) * market_regime_weight
    ) / total_weight
    # Event information is deliberately an optional, bounded adjustment. It can
    # never create a signal by itself or outweigh the cross-sectional factors.
    total = clamp_score(base_total + (clamp_score(event_score) - 50.0) * max(0.0, min(event_boost_weight, 0.05)))
    return StockFactorScore(
        stock_code=stock_code,
        date=str(window.iloc[-1]["date"]),
        # Retain full precision for deterministic ranking and research replay.
        # ``to_dict`` remains rounded for presentation/API compatibility.
        momentum_score=momentum,
        money_flow_score=money_flow,
        technical_score=technical,
        fundamental_score=clamp_score(fundamental_score),
        market_regime_score=clamp_score(market_regime_score),
        event_score=clamp_score(event_score),
        # Preserve the established strategy tie-breaking/ordering contract.
        # Raw component scores are retained for research snapshots, while the
        # production total remains the historical two-decimal score.
        total_score=round(total, 2),
    )


def calculate_factor_scores_for_universe(
    histories: dict[str, pd.DataFrame],
    as_of_date: str,
    momentum_weight: float = 0.25,
    money_flow_weight: float = 0.20,
    fundamental_weight: float = 0.25,
    technical_weight: float = 0.10,
    market_regime_weight: float = 0.20,
    market_regime_score: float = 50.0,
    fundamental_scores: dict[str, float] | None = None,
    require_fundamentals: bool = False,
    event_scores: dict[str, float] | None = None,
    event_boost_weight: float = 0.0,
    technical_variant: str = "legacy",
) -> list[StockFactorScore]:
    scores: list[StockFactorScore] = []
    for code, history in histories.items():
        fundamental_score = (fundamental_scores or {}).get(code)
        if require_fundamentals and fundamental_score is None:
            continue
        score = calculate_stock_factor_score(
            code,
            history,
            as_of_date,
            momentum_weight=momentum_weight,
            money_flow_weight=money_flow_weight,
            fundamental_weight=fundamental_weight,
            technical_weight=technical_weight,
            market_regime_weight=market_regime_weight,
            market_regime_score=market_regime_score,
            fundamental_score=fundamental_score if fundamental_score is not None else 50.0,
            event_score=(event_scores or {}).get(code, 50.0),
            event_boost_weight=event_boost_weight,
            technical_variant=technical_variant,
        )
        if score is not None:
            scores.append(score)
    return sorted(scores, key=lambda item: item.total_score, reverse=True)


def history_until(history: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if history is None or history.empty or "date" not in history.columns:
        return pd.DataFrame()
    data = history.copy()
    dates = pd.to_datetime(data["date"], errors="coerce")
    cutoff = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(cutoff):
        return pd.DataFrame()
    return data[dates <= cutoff].sort_values("date").reset_index(drop=True)


def calculate_momentum_score_from_standard_history(window: pd.DataFrame) -> float:
    closes = pd.to_numeric(window["close"], errors="coerce").dropna()
    if len(closes) < 60:
        return 50.0
    score = 50.0
    pct20 = pct_return(closes, 20)
    pct60 = pct_return(closes, 60)
    ma20 = closes.tail(20).mean()
    ma60 = closes.tail(60).mean()
    latest = safe_float(closes.iloc[-1])
    if pct20 is not None:
        score += max(-18, min(18, pct20 * 1.2))
    if pct60 is not None:
        score += max(-18, min(18, pct60 * 0.6))
    if latest is not None and latest > ma20:
        score += 8
    if ma20 > ma60:
        score += 8
    return clamp_score(score)


def calculate_money_flow_score_from_standard_history(window: pd.DataFrame) -> float:
    if len(window) < 25:
        return 50.0
    data = window.copy()
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    if "amount" in data.columns:
        data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    score = 50.0
    vol5 = safe_float(data["volume"].tail(5).mean())
    vol20 = safe_float(data["volume"].tail(20).mean())
    if vol20:
        score += max(-18, min(18, (vol5 / vol20 - 1) * 22))
    if "amount" in data.columns:
        amount = safe_float(data["amount"].iloc[-1])
        amount20 = safe_float(data["amount"].tail(20).mean())
        if amount and amount20:
            score += max(-12, min(12, (amount / amount20 - 1) * 12))
    close = safe_float(data["close"].iloc[-1])
    prev_close = safe_float(data["close"].iloc[-2]) if len(data) >= 2 else None
    if close is not None and prev_close is not None and close > prev_close and vol20 and vol5 and vol5 / vol20 >= 1.2:
        score += 8
    return clamp_score(score)


def calculate_technical_score_from_standard_history(window: pd.DataFrame, variant: str = "legacy") -> float:
    """Calculate a legacy technical score or the research-only entry-timing score."""
    if variant == "entry_timing":
        return calculate_entry_timing_score_from_standard_history(window)
    if variant != "legacy":
        raise ValueError(f"Unsupported technical variant: {variant}")
    """Legacy technical score; retained unchanged for frozen research baselines."""
    closes = pd.to_numeric(window["close"], errors="coerce").dropna()
    if len(closes) < 35:
        return 50.0
    latest = safe_float(closes.iloc[-1])
    ma5 = safe_float(closes.tail(5).mean())
    ma20 = safe_float(closes.tail(20).mean())
    ma60 = safe_float(closes.tail(60).mean()) if len(closes) >= 60 else None
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    dif = safe_float((ema12 - ema26).iloc[-1])
    dea = safe_float((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1])
    delta = closes.diff()
    gains = delta.clip(lower=0).rolling(6).mean()
    losses = (-delta.clip(upper=0)).rolling(6).mean()
    latest_loss = safe_float(losses.iloc[-1])
    rsi = 100.0 if latest_loss == 0 else 100 - 100 / (1 + (safe_float(gains.iloc[-1]) or 0) / latest_loss) if latest_loss else 50.0
    score = 50.0
    if latest is not None and ma5 is not None and latest > ma5:
        score += 6
    if latest is not None and ma20 is not None and latest > ma20:
        score += 8
    if ma20 is not None and ma60 is not None and ma20 > ma60:
        score += 8
    if dif is not None and dea is not None and dif > dea:
        score += 8
    if 45 <= rsi <= 72:
        score += 6
    elif rsi > 82:
        score -= 12
    elif rsi < 30:
        score -= 6
    return clamp_score(score)


def calculate_entry_timing_score_from_standard_history(window: pd.DataFrame) -> float:
    """Score short-term entry quality without reusing the medium-term trend tests.

    Momentum owns 20/60-day returns and the MA20/MA60 relationship.  This
    research variant intentionally avoids those inputs and asks only whether a
    stock is extended, unstable, or making a measured short-term pullback.
    """
    if len(window) < 35:
        return 50.0
    return float(calculate_entry_timing_scores_from_standard_history(window).iloc[-1])


def calculate_entry_timing_scores_from_standard_history(history: pd.DataFrame) -> pd.Series:
    """Vectorised Entry Timing scores for a single stock's full history.

    Returning a date-aligned series lets research experiments reuse exactly the
    same formula at many signal dates without repeatedly recomputing rolling
    indicators. Values before the 35-observation warm-up are neutral.
    """
    if history is None or history.empty or "close" not in history.columns:
        return pd.Series(dtype=float)
    closes = pd.to_numeric(history["close"], errors="coerce")
    score = pd.Series(50.0, index=history.index, dtype=float)
    ma5 = closes.rolling(5).mean()
    deviation = (closes / ma5 - 1) * 100
    score.loc[deviation.between(-3, 3, inclusive="both")] += 8
    score.loc[deviation > 8] -= 12
    score.loc[deviation < -8] -= 8

    return3 = closes.pct_change(3) * 100
    score.loc[return3 > 10] -= 12
    score.loc[(return3 > 6) & (return3 <= 10)] -= 6
    pullback = (closes / closes.rolling(5).max() - 1) * 100
    score.loc[pullback.between(-5, -1, inclusive="both")] += 6
    score.loc[pullback < -8] -= 8

    macd_line = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    histogram_change = (macd_line - macd_line.ewm(span=9, adjust=False).mean()).diff()
    score.loc[histogram_change > 0] += 6
    score.loc[histogram_change < 0] -= 4

    delta = closes.diff()
    gains = delta.clip(lower=0).rolling(6).mean()
    losses = (-delta.clip(upper=0)).rolling(6).mean()
    rsi = 100 - 100 / (1 + gains / losses.replace(0, float("nan")))
    rsi = rsi.mask(losses.eq(0), 100.0).fillna(50.0)
    score.loc[rsi.between(45, 70, inclusive="both")] += 6
    score.loc[rsi > 82] -= 12
    score.loc[rsi < 25] -= 6

    volatility = closes.pct_change().rolling(10).std(ddof=1)
    score.loc[volatility > 0.06] -= 8
    if "open" in history.columns:
        opening = pd.to_numeric(history["open"], errors="coerce")
        gap = (opening / closes.shift(1) - 1).abs()
        score.loc[gap > 0.05] -= 8
    enough_history = closes.notna().cumsum() >= 35
    return score.where(enough_history, 50.0).clip(lower=0, upper=100)


def calculate_market_regime_score_from_history(history: pd.DataFrame, as_of_date: str) -> float:
    """Historical market score. It never reads dates after ``as_of_date``."""
    window = history_until(history, as_of_date)
    if len(window) < 60:
        return 50.0
    return calculate_momentum_score_from_standard_history(window)


def pct_return(closes: pd.Series, days: int) -> float | None:
    if len(closes) <= days:
        return None
    base = safe_float(closes.iloc[-days - 1])
    latest = safe_float(closes.iloc[-1])
    if base in (None, 0) or latest is None:
        return None
    return (latest - base) / base * 100
