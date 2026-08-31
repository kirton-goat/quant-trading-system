from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from market import MarketSnapshot, NewsItem


DEFAULT_HOLDING_DAYS = 3


@dataclass
class SimulatedTradePlan:
    signal_id: str
    sim_status: str
    sim_direction: str
    sim_entry_time: str
    sim_entry_price: Any
    sim_exit_time: str
    sim_exit_price: Any
    sim_holding_days: int
    sim_pnl_pct: Any
    sim_pnl_amount: Any
    sim_result: str
    sim_note: str

    def to_log_fields(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "sim_status": self.sim_status,
            "sim_direction": self.sim_direction,
            "sim_entry_time": self.sim_entry_time,
            "sim_entry_price": self.sim_entry_price,
            "sim_exit_time": self.sim_exit_time,
            "sim_exit_price": self.sim_exit_price,
            "sim_holding_days": self.sim_holding_days,
            "sim_pnl_pct": self.sim_pnl_pct,
            "sim_pnl_amount": self.sim_pnl_amount,
            "sim_result": self.sim_result,
            "sim_note": self.sim_note,
        }


def make_signal_id(news: NewsItem, mapped_target: str | None) -> str:
    raw = f"{news.published_at}|{news.source}|{news.title}|{mapped_target or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def build_simulated_trade_plan(
    news: NewsItem,
    mapped_target: str | None,
    snapshot: MarketSnapshot | None,
    final_action: str,
    holding_days: int = DEFAULT_HOLDING_DAYS,
) -> SimulatedTradePlan:
    signal_id = make_signal_id(news, mapped_target)
    direction = action_to_direction(final_action)

    if direction == "none":
        return SimulatedTradePlan(
            signal_id=signal_id,
            sim_status="research_only",
            sim_direction=direction,
            sim_entry_time="",
            sim_entry_price="",
            sim_exit_time="",
            sim_exit_price="",
            sim_holding_days=holding_days,
            sim_pnl_pct="",
            sim_pnl_amount="",
            sim_result="no_trade",
            sim_note="未通过多因子交易许可，仅作为研究观察，未生成模拟持仓",
        )

    if snapshot is None or snapshot.price in (None, "-"):
        return SimulatedTradePlan(
            signal_id=signal_id,
            sim_status="market_data_missing",
            sim_direction=direction,
            sim_entry_time="",
            sim_entry_price="",
            sim_exit_time="",
            sim_exit_price="",
            sim_holding_days=holding_days,
            sim_pnl_pct="",
            sim_pnl_amount="",
            sim_result="data_error",
            sim_note="已产生模拟交易意图，但缺少有效行情价格，未生成模拟持仓",
        )

    return SimulatedTradePlan(
        signal_id=signal_id,
        sim_status="pending_exit",
        sim_direction=direction,
        sim_entry_time=news.published_at,
        sim_entry_price=snapshot.price,
        sim_exit_time="",
        sim_exit_price="",
        sim_holding_days=holding_days,
        sim_pnl_pct="",
        sim_pnl_amount="",
        sim_result="pending",
        sim_note=f"模拟持仓计划：{holding_days}日后由回测模块计算盈亏",
    )


def action_to_direction(action: str | None) -> str:
    if action == "买入":
        return "long"
    if action == "卖出":
        return "short"
    return "none"
