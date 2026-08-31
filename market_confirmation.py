from __future__ import annotations

import datetime as dt
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import akshare as ak

from market import MarketSnapshot


@dataclass
class MarketConfirmation:
    stock_code: str
    pct_change_20d: Any = None
    pct_change_60d: Any = None
    volume_ratio_5_20: Any = None
    gap_pct: Any = None
    turnover_rate: Any = None
    market_already_priced: bool = False
    score: float = 0.5
    risk_tags: list[str] = field(default_factory=list)
    note: str = "市场反应中性"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "pct_change_20d": self.pct_change_20d,
            "pct_change_60d": self.pct_change_60d,
            "volume_ratio_5_20": self.volume_ratio_5_20,
            "gap_pct": self.gap_pct,
            "turnover_rate": self.turnover_rate,
            "market_already_priced": self.market_already_priced,
            "score": round(self.score, 2),
            "risk_tags": self.risk_tags,
            "note": self.note,
        }


def confirm_market_reaction(
    stock_code: str | None = None,
    snapshot: MarketSnapshot | None = None,
    timeout: int = 20,
) -> MarketConfirmation:
    code = stock_code or (snapshot.code if snapshot else "")
    confirmation = from_snapshot(snapshot) if snapshot else MarketConfirmation(stock_code=code)
    if code and confirmation.pct_change_60d is None:
        history = _fetch_history_metrics(code, timeout=timeout)
        if history:
            confirmation.pct_change_20d = confirmation.pct_change_20d if confirmation.pct_change_20d is not None else history.get("pct_change_20d")
            confirmation.pct_change_60d = history.get("pct_change_60d")
            confirmation.volume_ratio_5_20 = confirmation.volume_ratio_5_20 if confirmation.volume_ratio_5_20 is not None else history.get("volume_ratio_5_20")
            confirmation.gap_pct = confirmation.gap_pct if confirmation.gap_pct is not None else history.get("gap_pct")
            confirmation.turnover_rate = confirmation.turnover_rate if confirmation.turnover_rate is not None else history.get("turnover_rate")
    return score_confirmation(confirmation)


def from_snapshot(snapshot: MarketSnapshot | None) -> MarketConfirmation:
    if snapshot is None:
        return MarketConfirmation(stock_code="")
    return MarketConfirmation(
        stock_code=snapshot.code,
        pct_change_20d=getattr(snapshot, "pct_change_20d", None),
        pct_change_60d=getattr(snapshot, "pct_change_60d", None),
        volume_ratio_5_20=getattr(snapshot, "volume_ratio_5_20", None),
        gap_pct=getattr(snapshot, "gap_pct", None),
        turnover_rate=getattr(snapshot, "turnover_rate", None),
    )


def score_confirmation(confirmation: MarketConfirmation) -> MarketConfirmation:
    score = 0.56
    risks: list[str] = []
    notes: list[str] = []

    pct20 = safe_float(confirmation.pct_change_20d)
    pct60 = safe_float(confirmation.pct_change_60d)
    volume_ratio = safe_float(confirmation.volume_ratio_5_20)
    gap = safe_float(confirmation.gap_pct)
    turnover = safe_float(confirmation.turnover_rate)

    if pct20 is not None and pct20 >= 30:
        score -= 0.28
        risks.append("利好兑现风险")
        notes.append(f"近20日涨幅{pct20}%")
    elif pct20 is not None and pct20 >= 18:
        score -= 0.18
        risks.append("追涨风险")
        notes.append(f"近20日涨幅{pct20}%")

    if pct60 is not None and pct60 >= 50:
        score -= 0.18
        risks.append("中期高位风险")
        notes.append(f"近60日涨幅{pct60}%")
    elif pct60 is not None and pct60 <= -15:
        score -= 0.08
        risks.append("弱势反弹不确定")
        notes.append(f"近60日跌幅较大{pct60}%")

    if volume_ratio is not None and volume_ratio >= 2.5:
        score -= 0.1
        risks.append("异常放量风险")
        notes.append(f"5/20量能比{volume_ratio}")

    if gap is not None and gap >= 5:
        score -= 0.16
        risks.append("高开兑现风险")
        notes.append(f"跳空幅度{gap}%")

    if turnover is not None and turnover >= 18:
        score -= 0.08
        risks.append("高换手博弈风险")
        notes.append(f"换手率{turnover}%")

    confirmation.score = max(0.05, min(0.9, score))
    confirmation.risk_tags = dedupe(risks)
    confirmation.market_already_priced = confirmation.score < 0.45
    confirmation.note = "；".join(notes) if notes else "未发现明显提前炒作迹象"
    return confirmation


def _fetch_history_metrics(stock_code: str, timeout: int) -> dict[str, Any] | None:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_history_worker, args=(queue, stock_code))
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


def _history_worker(queue: mp.Queue, stock_code: str) -> None:
    try:
        end_date = dt.datetime.now().strftime("%Y%m%d")
        start_date = (dt.datetime.now() - dt.timedelta(days=180)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty or len(df) < 21:
            queue.put(("ok", {}))
            return
        latest = df.iloc[-1]
        base20 = df.iloc[-21]
        base60 = df.iloc[-61] if len(df) >= 61 else None
        payload = {
            "pct_change_20d": round((latest["收盘"] - base20["收盘"]) / base20["收盘"] * 100, 2) if base20["收盘"] else None,
            "pct_change_60d": round((latest["收盘"] - base60["收盘"]) / base60["收盘"] * 100, 2) if base60 is not None and base60["收盘"] else None,
            "volume_ratio_5_20": round(df["成交量"].tail(5).mean() / df["成交量"].tail(20).mean(), 2) if df["成交量"].tail(20).mean() else None,
            "gap_pct": round((latest["开盘"] - df.iloc[-2]["收盘"]) / df.iloc[-2]["收盘"] * 100, 2) if len(df) >= 2 and df.iloc[-2]["收盘"] else None,
            "turnover_rate": latest["换手率"] if "换手率" in df.columns else None,
        }
        queue.put(("ok", payload))
    except Exception as exc:
        queue.put(("error", str(exc)))


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, "", "-"):
            return None
        return float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result
