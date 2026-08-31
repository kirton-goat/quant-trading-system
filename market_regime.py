from __future__ import annotations

import contextlib
import datetime as dt
import io
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import akshare as ak
import pandas as pd


@dataclass
class IndexTrend:
    name: str
    code: str
    pct_5d: float | None = None
    pct_20d: float | None = None
    pct_60d: float | None = None
    above_ma20: bool | None = None
    ma20_above_ma60: bool | None = None
    score: float = 50.0
    note: str = "数据不足"


@dataclass
class MarketRegimeReport:
    risk_level: str
    risk_score: float
    trend_score: float
    amount_score: float
    sentiment_score: float
    reason: str
    indexes: list[IndexTrend] = field(default_factory=list)
    market_amount_billion: float | None = None
    limit_up_count: int | None = None
    limit_down_count: int | None = None
    rising_ratio: float | None = None
    risk_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 2),
            "trend_score": round(self.trend_score, 2),
            "amount_score": round(self.amount_score, 2),
            "sentiment_score": round(self.sentiment_score, 2),
            "reason": self.reason,
            "market_amount_billion": self.market_amount_billion,
            "limit_up_count": self.limit_up_count,
            "limit_down_count": self.limit_down_count,
            "rising_ratio": self.rising_ratio,
            "risk_tags": self.risk_tags,
            "indexes": [trend.__dict__ for trend in self.indexes],
        }


INDEXES = {
    "上证指数": "000001",
    "沪深300": "000300",
    "中证500": "000905",
    "创业板指": "399006",
}


def evaluate_market_regime(timeout: int = 25) -> MarketRegimeReport:
    payload = _call_with_timeout(_evaluate_market_regime_direct, timeout=timeout)
    if isinstance(payload, MarketRegimeReport):
        return payload
    return MarketRegimeReport(
        risk_level="unknown",
        risk_score=50,
        trend_score=50,
        amount_score=50,
        sentiment_score=50,
        reason=f"市场环境接口不可用，默认中性：{payload or '无返回'}",
        risk_tags=["市场环境数据不足"],
    )


def _evaluate_market_regime_direct() -> MarketRegimeReport:
    index_trends = [analyze_index_trend(name, code) for name, code in INDEXES.items()]
    trend_score = average([item.score for item in index_trends]) or 50

    market_amount, rising_ratio = fetch_market_amount_and_breadth()
    amount_score = score_market_amount(market_amount)

    limit_up_count, limit_down_count = fetch_limit_counts()
    sentiment_score = score_sentiment(rising_ratio, limit_up_count, limit_down_count)

    risk_score = round(trend_score * 0.45 + amount_score * 0.25 + sentiment_score * 0.30, 2)
    risk_level, reason, tags = classify_risk(risk_score, trend_score, amount_score, sentiment_score)

    return MarketRegimeReport(
        risk_level=risk_level,
        risk_score=risk_score,
        trend_score=trend_score,
        amount_score=amount_score,
        sentiment_score=sentiment_score,
        reason=reason,
        indexes=index_trends,
        market_amount_billion=market_amount,
        limit_up_count=limit_up_count,
        limit_down_count=limit_down_count,
        rising_ratio=rising_ratio,
        risk_tags=tags,
    )


def analyze_index_trend(name: str, code: str) -> IndexTrend:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_zh_index_daily_em(symbol=code)
    except Exception as exc:
        return IndexTrend(name=name, code=code, note=f"指数接口失败：{exc}")

    if df is None or df.empty or len(df) < 60:
        return IndexTrend(name=name, code=code)

    df = df.tail(120).copy()
    close_col = "close" if "close" in df.columns else "收盘"
    close = pd.to_numeric(df[close_col], errors="coerce").dropna()
    if len(close) < 60:
        return IndexTrend(name=name, code=code)

    latest = float(close.iloc[-1])
    pct_5d = pct_change(close, 5)
    pct_20d = pct_change(close, 20)
    pct_60d = pct_change(close, 60)
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    above_ma20 = latest > ma20
    ma20_above_ma60 = ma20 > ma60

    score = 50.0
    if above_ma20:
        score += 12
    else:
        score -= 10
    if ma20_above_ma60:
        score += 12
    else:
        score -= 8
    if pct_20d is not None:
        score += max(-15, min(15, pct_20d * 1.8))
    if pct_60d is not None:
        score += max(-10, min(10, pct_60d * 0.8))

    note = "指数趋势偏强" if score >= 65 else "指数趋势中性" if score >= 45 else "指数趋势偏弱"
    return IndexTrend(
        name=name,
        code=code,
        pct_5d=pct_5d,
        pct_20d=pct_20d,
        pct_60d=pct_60d,
        above_ma20=above_ma20,
        ma20_above_ma60=ma20_above_ma60,
        score=clamp(score),
        note=note,
    )


def fetch_market_amount_and_breadth() -> tuple[float | None, float | None]:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_zh_a_spot_em()
    except Exception:
        return None, None
    if df is None or df.empty:
        return None, None

    amount = None
    if "成交额" in df.columns:
        amount = round(float(pd.to_numeric(df["成交额"], errors="coerce").sum()) / 100000000, 2)

    change_col = "涨跌幅" if "涨跌幅" in df.columns else None
    rising_ratio = None
    if change_col:
        changes = pd.to_numeric(df[change_col], errors="coerce").dropna()
        if len(changes) > 0:
            rising_ratio = round(float((changes > 0).sum()) / len(changes), 4)
    return amount, rising_ratio


def fetch_limit_counts() -> tuple[int | None, int | None]:
    today = dt.datetime.now().strftime("%Y%m%d")
    limit_up = None
    limit_down = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            up_df = ak.stock_zt_pool_em(date=today)
        limit_up = 0 if up_df is None else len(up_df)
    except Exception:
        limit_up = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            down_df = ak.stock_zt_pool_dtgc_em(date=today)
        limit_down = 0 if down_df is None else len(down_df)
    except Exception:
        limit_down = None
    return limit_up, limit_down


def score_market_amount(amount_billion: float | None) -> float:
    if amount_billion is None:
        return 50
    if amount_billion >= 11000:
        return 78
    if amount_billion >= 9000:
        return 68
    if amount_billion >= 7000:
        return 55
    if amount_billion >= 5500:
        return 45
    return 35


def score_sentiment(rising_ratio: float | None, limit_up: int | None, limit_down: int | None) -> float:
    score = 50.0
    if rising_ratio is not None:
        score += (rising_ratio - 0.5) * 70
    if limit_up is not None:
        score += min(14, limit_up / 8)
    if limit_down is not None:
        score -= min(18, limit_down / 5)
    return clamp(score)


def classify_risk(risk_score: float, trend_score: float, amount_score: float, sentiment_score: float) -> tuple[str, str, list[str]]:
    tags: list[str] = []
    if trend_score < 45:
        tags.append("大盘趋势偏弱")
    if amount_score < 45:
        tags.append("市场成交额不足")
    if sentiment_score < 45:
        tags.append("市场情绪偏弱")

    if risk_score >= 70:
        return "low", "市场环境偏强，可支持风险偏好", tags
    if risk_score >= 55:
        return "medium", "市场环境中性偏稳，适合控制仓位观察", tags
    if risk_score >= 42:
        return "high", "市场环境偏弱，需要降低信号权重", tags
    return "extreme", "市场风险较高，研究阶段应以防守为主", tags or ["市场综合风险较高"]


def pct_change(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    base = float(close.iloc[-days - 1])
    latest = float(close.iloc[-1])
    if base == 0:
        return None
    return round((latest - base) / base * 100, 2)


def average(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


def _call_with_timeout(fn, timeout: int) -> Any:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(queue, fn))
    process.daemon = True
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return f"超过 {timeout} 秒未响应"
    if queue.empty():
        return "无返回"
    status, payload = queue.get()
    return payload if status == "ok" else payload


def _worker(queue: mp.Queue, fn) -> None:
    try:
        queue.put(("ok", fn()))
    except Exception as exc:
        queue.put(("error", str(exc)))


if __name__ == "__main__":
    report = evaluate_market_regime()
    print(report.to_dict())
