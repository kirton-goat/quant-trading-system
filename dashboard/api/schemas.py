from __future__ import annotations

from pydantic import BaseModel, Field


class MarketStatusResponse(BaseModel):
    regime: str
    risk_score: float
    trend_score: float
    label: str
    as_of: str | None = None
    source: str
    note: str = ""


class RankingItem(BaseModel):
    stock_code: str
    stock_name: str
    as_of: str
    total_score: float
    momentum_score: float
    money_flow_score: float
    fundamental_score: float
    technical_score: float
    market_regime_score: float
    event_score: float
    source: str


class BacktestMetrics(BaseModel):
    model: str
    total_return_pct: float | None = None
    annualized_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    sharpe_ratio: float | None = None
    win_rate: float | None = None
    profit_loss_ratio: float | None = None
    trade_count: int = 0
    as_of: str | None = None
    note: str = ""


class EquityPoint(BaseModel):
    date: str
    equity: float
    return_pct: float
    benchmark_return_pct: float | None = None


class TradeHistoryItem(BaseModel):
    timestamp: str
    stock_code: str
    title: str
    signal_source: str
    signal_type: str
    entry_price: str | float | None = None
    status: str
    pnl_pct: str | float | None = None
    action: str
    note: str = ""


class EventItem(BaseModel):
    published_at: str
    event_type: str
    stock_code: str = ""
    industry: str = ""
    score: float = Field(ge=0, le=100)
    source: str
    publisher: str = ""
    source_url: str = ""
    source_kind: str = ""
    is_official: bool = False
    fetched_at: str = ""
    title: str
    note: str = ""
