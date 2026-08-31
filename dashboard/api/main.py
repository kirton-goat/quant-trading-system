from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

from . import repository
from research.v3.strategy_lab import StrategyLabError, StrategySpec, default_spec, list_strategy_lab_experiments, run_strategy_lab
from research.v4.strategy_lab import V4StrategyLabError, V4StrategySpec, default_spec as default_v4_strategy_spec, list_strategy_lab_experiments as list_v4_strategy_lab_experiments, run_strategy_lab as run_v4_strategy_lab
from research.v4.factor_recipes import active_recipes, create_recipe, recipe_history, set_active_recipe, set_experiment_approval
from research.v4.factor_validation import factor_experiment_list, market_predictive_audit, run_single_factor
from official_event_sources import sync_official_sources
from .schemas import EventItem, MarketStatusResponse, RankingItem, TradeHistoryItem


app = FastAPI(
    title="AI Quant Research Dashboard API",
    version="0.1.0",
    description="Read-only API for the AI quantitative research dashboard. No trading endpoints.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return repository.get_health()


@app.get("/api/market/status", response_model=MarketStatusResponse)
def market_status() -> dict:
    return repository.get_market_status()


@app.get("/api/stocks/ranking", response_model=list[RankingItem])
def stocks_ranking(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return repository.get_cached_ranking(limit)


@app.get("/api/backtest/results")
def backtest_results() -> dict:
    return repository.get_backtest_results()


@app.get("/api/backtest/versions")
def backtest_versions() -> dict:
    return repository.get_backtest_version_catalog()


@app.get("/api/strategy/versions")
def strategy_versions() -> dict:
    return repository.get_strategy_versions()


@app.get("/api/strategy/versions/{version}")
def strategy_version(version: str) -> dict:
    detail = repository.get_strategy_version(version)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Unknown strategy version: {version}")
    return detail


@app.get("/api/research/v3/status")
def v3_research_status() -> dict:
    return repository.get_v3_research_status()


@app.get("/api/research/v4/status")
def v4_research_status() -> dict:
    return repository.get_v4_research_status()


@app.get("/api/research/v4/price-gaps")
def v4_price_gaps() -> dict:
    return repository.get_v4_price_gaps()


class StrategyLabRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    hypothesis: str = Field(min_length=1, max_length=800)
    factor_weights: dict[str, float]
    top_n: int = 20
    market_regime_gate: bool = False
    market_min_score: float = 40.0


class FactorRecipeRequest(BaseModel):
    components: list[dict]
    parameters: dict = {}
    source_experiment_id: str | None = None


class FactorExperimentRequest(BaseModel):
    factor_name: str
    recipe_id: str
    hypothesis_note: str = Field(min_length=1, max_length=800)
    start_date: str | None = None
    end_date: str | None = None
    top_n_values: list[int] = [20]
    market_regime_gate: bool = False


class ApprovalRequest(BaseModel):
    approved: bool
    approval_note: str = Field(default="", max_length=800)


class ActivateRecipeRequest(BaseModel):
    experiment_id: str
    user_note: str = Field(default="", max_length=800)


@app.get("/api/research/v4/factor-recipes")
def v4_factor_recipes() -> dict:
    return {"active_recipes": active_recipes(), "factor_names": ["market_regime", "momentum", "money_flow", "fundamental", "technical"]}


@app.get("/api/research/v4/factor-recipes/{factor}")
def v4_factor_recipe_history(factor: str) -> list[dict]:
    return recipe_history(factor)


@app.post("/api/research/v4/factor-recipes/{factor}")
def v4_create_factor_recipe(factor: str, payload: FactorRecipeRequest) -> dict:
    try:
        return create_recipe(factor, payload.components, payload.parameters, payload.source_experiment_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/research/v4/factor-recipes/{factor}/{recipe_id}/activate")
def v4_activate_factor_recipe(factor: str, recipe_id: str, payload: ActivateRecipeRequest) -> dict:
    try:
        return set_active_recipe(factor, recipe_id, payload.experiment_id, payload.user_note)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/research/v4/factor-experiments")
def v4_factor_experiments(factor: str | None = None) -> list[dict]:
    return factor_experiment_list(factor)


@app.get("/api/research/v4/market-predictive-audit")
def v4_market_predictive_audit() -> dict:
    try:
        return market_predictive_audit()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/research/v4/factor-experiments")
def v4_factor_experiment(payload: FactorExperimentRequest) -> dict:
    recipe = next((item for item in recipe_history(payload.factor_name) if item["recipe_id"] == payload.recipe_id), None)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Unknown recipe.")
    try:
        return run_single_factor(payload.model_dump(), recipe)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/research/v4/factor-experiments/{experiment_id}/approval")
def v4_factor_experiment_approval(experiment_id: str, payload: ApprovalRequest) -> dict:
    return set_experiment_approval(experiment_id, payload.approved, payload.approval_note)


@app.get("/api/research/strategy-lab/default")
def strategy_lab_default() -> dict:
    return default_spec()


@app.get("/api/research/strategy-lab/experiments")
def strategy_lab_experiments(limit: int = Query(default=30, ge=1, le=100)) -> list[dict]:
    return list_strategy_lab_experiments(limit)


@app.post("/api/research/strategy-lab/experiments")
def strategy_lab_run(payload: StrategyLabRequest) -> dict:
    try:
        return run_strategy_lab(StrategySpec(**payload.model_dump()))
    except StrategyLabError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"实验运行失败：{error}") from error


@app.get("/api/research/v4/strategy-lab/default")
def v4_strategy_lab_default() -> dict:
    return default_v4_strategy_spec()


@app.get("/api/research/v4/strategy-lab/experiments")
def v4_strategy_lab_experiments(limit: int = Query(default=30, ge=1, le=100)) -> list[dict]:
    return list_v4_strategy_lab_experiments(limit)


@app.post("/api/research/v4/strategy-lab/experiments")
def v4_strategy_lab_run(payload: StrategyLabRequest) -> dict:
    try:
        return run_v4_strategy_lab(V4StrategySpec(**payload.model_dump()))
    except V4StrategyLabError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"V4 实验运行失败：{error}") from error


@app.get("/api/backtest/equity", response_model=list[dict])
def backtest_equity(model: str = Query(default="a", pattern="^[abAB]$")) -> list[dict]:
    return repository.get_equity_curve(model)


@app.get("/api/trades/history", response_model=list[TradeHistoryItem])
def trade_history(limit: int = Query(default=200, ge=1, le=500)) -> list[dict]:
    return repository.get_trade_history(limit)


@app.get("/api/trades/summary")
def trade_summary() -> dict:
    return repository.get_trade_summary()


@app.get("/api/events", response_model=list[EventItem])
def events(event_type: str = Query(default="all", alias="type", pattern="^(all|policy|announcement)$"), limit: int = Query(default=100, ge=1, le=500), official_only: bool = Query(default=True)) -> list[dict]:
    return repository.get_events(event_type, limit, official_only=official_only)


@app.post("/api/events/sync-official")
def sync_events_official(days: int = Query(default=1, ge=1, le=30), policy_limit: int = Query(default=40, ge=1, le=100)) -> dict:
    """Manually refresh research-only events from declared official sources."""
    try:
        return sync_official_sources(days=days, policy_limit=policy_limit)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"官方事件同步失败：{error}") from error
