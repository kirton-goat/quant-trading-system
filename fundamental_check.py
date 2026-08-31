from __future__ import annotations

import datetime as dt
import multiprocessing as mp
import re
from dataclasses import dataclass, field
from typing import Any

import akshare as ak


@dataclass
class FundamentalCheckResult:
    stock_code: str
    revenue: Any = None
    net_profit: Any = None
    gross_margin: Any = None
    roe: Any = None
    business_relevance: str = "unknown"
    score: float = 0.5
    has_financial_impact: bool = False
    risk_tags: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "revenue": self.revenue,
            "net_profit": self.net_profit,
            "gross_margin": self.gross_margin,
            "roe": self.roe,
            "business_relevance": self.business_relevance,
            "score": round(self.score, 2),
            "has_financial_impact": self.has_financial_impact,
            "risk_tags": self.risk_tags,
            "note": self.note,
        }


STRONG_BUSINESS_KEYWORDS = ("订单", "合同", "中标", "收入", "营收", "净利润", "毛利率", "量产", "交付")
WEAK_BUSINESS_KEYWORDS = ("战略合作", "框架协议", "意向", "规划", "布局", "探索", "概念", "进入")
NEGATIVE_BUSINESS_PHRASES = (
    "未涉及订单",
    "未涉及收入",
    "未涉及利润",
    "未产生收入",
    "未产生利润",
    "尚未产生收入",
    "尚未产生利润",
    "暂无订单",
    "暂无收入",
    "暂无利润",
    "未形成订单",
    "未形成收入",
)


def check_fundamentals(
    stock_code: str,
    news_text: str = "",
    timeout: int = 20,
    *,
    formal_backtest: bool = False,
) -> FundamentalCheckResult:
    if formal_backtest:
        raise RuntimeError(
            "fundamental_check.py only supports current news research; formal backtests must use "
            "research.fundamentals.get_fundamentals(code, as_of_date)."
        )
    result = evaluate_business_relevance(news_text)
    if not stock_code:
        result.note = append_note(result.note, "未提供股票代码，仅做文本基本面判断")
        return result

    metrics = _fetch_financial_metrics(stock_code, timeout=timeout)
    result.stock_code = stock_code
    if not metrics:
        result.note = append_note(result.note, "财务接口不可用或无数据，保留文本判断")
        return result

    result.revenue = metrics.get("revenue")
    result.net_profit = metrics.get("net_profit")
    result.gross_margin = metrics.get("gross_margin")
    result.roe = metrics.get("roe")

    if safe_float(result.net_profit) is not None and safe_float(result.net_profit) <= 0:
        result.score = min(result.score, 0.42)
        result.risk_tags.append("盈利能力不足")
    if safe_float(result.roe) is not None and safe_float(result.roe) < 3:
        result.score = min(result.score, 0.5)
        result.risk_tags.append("ROE偏弱")
    return result


def evaluate_business_relevance(news_text: str) -> FundamentalCheckResult:
    text = news_text or ""
    strong_hits = sum(1 for keyword in STRONG_BUSINESS_KEYWORDS if keyword in text)
    weak_hits = sum(1 for keyword in WEAK_BUSINESS_KEYWORDS if keyword in text)
    result = FundamentalCheckResult(stock_code="")

    if any(phrase in text for phrase in NEGATIVE_BUSINESS_PHRASES) or has_negative_business_context(text):
        result.business_relevance = "weak"
        result.score = 0.32
        result.risk_tags.append("缺少财务兑现证据")
        result.note = "文本明确缺少订单、收入或利润证据"
        return result

    if strong_hits >= 2:
        result.business_relevance = "strong"
        result.score = 0.82
        result.has_financial_impact = True
        result.note = "新闻包含订单、收入、利润或交付等可验证业务要素"
    elif strong_hits == 1 and weak_hits == 0:
        result.business_relevance = "medium"
        result.score = 0.66
        result.has_financial_impact = True
        result.note = "新闻包含单一业务要素，需要继续验证规模"
    elif weak_hits > 0 and strong_hits == 0:
        result.business_relevance = "weak"
        result.score = 0.36
        result.risk_tags.append("概念业务缺少财务验证")
        result.note = "新闻偏概念、规划或战略合作，缺少收入利润证据"
    else:
        result.business_relevance = "unknown"
        result.score = 0.5
        result.note = "未发现明确财务影响"
    return result


def has_negative_business_context(text: str) -> bool:
    return bool(re.search(r"(未涉及|未形成|未产生|尚未|暂无).{0,12}(订单|收入|营收|利润|净利润)", text or ""))


def _fetch_financial_metrics(stock_code: str, timeout: int) -> dict[str, Any] | None:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_financial_worker, args=(queue, stock_code))
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


def _financial_worker(queue: mp.Queue, stock_code: str) -> None:
    try:
        year = str(dt.datetime.now().year - 1)
        df = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        if df.empty:
            queue.put(("ok", {}))
            return
        row = df.iloc[0]
        queue.put(
            (
                "ok",
                {
                    "year": year,
                    "revenue": first_existing(row, ("营业总收入", "营业收入")),
                    "net_profit": first_existing(row, ("净利润", "归母净利润")),
                    "gross_margin": first_existing(row, ("销售毛利率", "毛利率")),
                    "roe": first_existing(row, ("净资产收益率", "ROE")),
                },
            )
        )
    except Exception as exc:
        queue.put(("error", str(exc)))


def first_existing(row: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index:
            value = row[name]
            if value not in (None, "") and str(value).lower() != "nan":
                return value
    return None


def append_note(note: str, extra: str) -> str:
    return f"{note}；{extra}" if note else extra


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None
