# AI Quant Research Platform

[中文](README.md) | [English](README_EN.md)

> A local-first, research-only A-share quantitative research platform for testing portfolio hypotheses with historical point-in-time controls. It does **not** place orders or connect to a live brokerage.

![AI Quant Research logo](assets/ai-quant-logo.png)

## Project Overview

This project evolved from an AI-assisted news-analysis prototype into a modular quantitative research platform. Its focus is not “news-driven stock recommendations”, but building a more auditable workflow for:

1. constructing a historically valid A-share universe;
2. calculating market, momentum, money-flow, fundamental, and entry-timing signals;
3. applying point-in-time data rules and eligibility filters;
4. forming and replaying portfolios under fixed rebalancing rules; and
5. comparing strategy variants, factor experiments, risk metrics, and integrity checks in a web dashboard.

The implementation was developed iteratively with AI-assisted coding and review. The research design, constraints, validation goals, and interpretation remain human-directed. Results are research artifacts, not investment advice or evidence of future returns.

## Research Architecture

```text
Historical CSI300 / CSI500 constituents
        |
        v
Eligibility filters
  - listing age
  - ST / delisting / suspension
  - liquidity
        |
        v
Historical prices + point-in-time fundamentals
        |
        v
Factor panel
  Market | Momentum | Money Flow | Fundamental | Entry Timing
        |
        v
Cross-sectional ranking and Top N portfolio
        |
        v
Continuous-rebalance backtest + integrity checks
        |
        v
FastAPI -> React / ECharts research dashboard
```

## Historical Data Integrity

The project includes explicit controls intended to reduce common backtest biases:

- **Historical universe:** each rebalance uses the CSI300 / CSI500 constituents available on that historical date, instead of today’s index members.
- **Eligibility filtering:** filters new listings, ST securities, delisted securities, long suspensions, and insufficient-liquidity candidates.
- **Point-in-time fundamentals:** a financial report is visible only when `disclosure_date <= signal_date`; the latest currently known financial statement is never used to score an earlier date.
- **No future membership:** a security cannot be introduced before its historical membership or listing date.
- **Price lifecycle audit:** prices are expected only through a security’s actual trading life. Post-delisting non-trading days are not fabricated as prices or misclassified as ordinary data gaps.
- **Execution timing:** the continuous-rebalance engine generates a signal at `T` and executes on the next trading day using the configured convention.

These checks reduce, but do not eliminate, survivorship bias, data-revision risk, delisting-event handling complexity, vendor-data limitations, and all forms of model overfitting.

## Strategy Research

The platform separates a common market-state signal from cross-sectional stock-selection signals:

| Component | Role |
| --- | --- |
| Market regime | Time-series market state and optional exposure gate; it is common to all stocks on the same date. |
| Momentum | Medium-term trend strength. |
| Money flow | Volume / turnover confirmation. |
| Fundamental | Profitability, growth, cash-flow, and valuation components using PIT data. |
| Entry timing | Short-term entry-quality / overheat assessment; separated from medium-term momentum. |

Research modules include single-factor validation, Top5 / Top10 / Top20 comparisons, IC / Rank IC analysis, factor recipes, historical portfolio replay, Market Regime ablation, and V1/V2 timeline comparison.

## Backtest Baselines

### V2 Continuous Rebalance (validated historical baseline)

| Item | Value |
| --- | ---: |
| Period | 2020-01-01 to 2025-12-31 |
| Universe | Historical CSI300 + CSI500 constituents |
| Portfolio | Top 20, rebalance every 20 trading days |
| Initial capital | RMB 1,000,000 |
| Fee assumption | 0.15% |
| Total return | 76.66% |
| CAGR | 10.36% |
| Annualized volatility | 24.25% |
| Maximum drawdown | -41.21% |
| Sharpe ratio | 0.53 |
| Average exposure | 81.99% |

The metrics above describe one frozen historical configuration. They are **not** optimized claims, live performance, or a recommendation to trade. The research platform intentionally retains less successful candidate experiments as evidence of the validation process.

### Version Boundary

- **V1 Historical PIT:** frozen reference; retained with a documented rebalancing-timeline limitation.
- **V2 Continuous Rebalance:** current validated historical baseline.
- **V3 / V4:** long-sample and single-factor research environments; not promoted as production strategies.

## Web Dashboard

The Dashboard is a display and research layer, separated from Python research modules through FastAPI JSON endpoints.

- **Overview:** cached market status, version boundaries, portfolio metrics, and cross-sectional ranking.
- **Factor analysis:** current factor score display and factor explanations.
- **Backtest analysis:** curves and historical metrics for validated baselines.
- **Strategy Explorer:** code-audited metadata on universe, factor roles, weights, timing, risk filters, and known limitations.
- **V4 single-factor validation:** independent factor recipes and TopN / IC research records.

Screenshots and generated research artifacts are intentionally not committed because they can be large and local-data dependent. For a local preview, start the API and frontend, then open `http://127.0.0.1:5173/`.

## Technology Stack

- Python, Pandas, BaoStock, AKShare
- FastAPI / Uvicorn
- React, Vite, ECharts
- CSV-based local research cache with explicit cache provenance
- Pytest

## Repository Layout

```text
.
├── dashboard/                 # FastAPI API and React / ECharts dashboard
├── research/
│   ├── universe/              # Historical index members and eligibility filters
│   ├── fundamentals/          # Point-in-time financial statement access
│   ├── v3/                    # Long-sample research infrastructure
│   ├── v4/                    # Five-factor and single-factor research workspace
│   ├── experiments/           # Experiment runners (generated outputs ignored)
│   └── audits/                # Strategy/data logic audit documents
├── tests/                     # Automated integrity and research tests
├── assets/                    # Public logo assets
├── factor_engine.py           # Factor score orchestration
├── market_data_manager.py     # Historical market-data cache interface
├── universe_manager.py        # Universe management interface
├── backtest_config_v*.{yaml,json}
└── requirements.txt
```

## Local Setup

### Python API

```bash
pip install -r requirements.txt
python -m uvicorn dashboard.api.main:app --host 127.0.0.1 --port 8000
```

### Dashboard

```bash
cd dashboard/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/` after both services are running.

## Optional Local Integrations

The original prototype can optionally use an LLM for research-oriented news parsing and PushPlus for local notifications. These integrations are intentionally configured through environment variables and are not required for historical backtests or dashboard display.

```bash
# Optional: keep real values local. Do not commit a .env file.
DEEPSEEK_API_KEY=your_local_key
PUSHPLUS_TOKEN=your_local_token
```

## Tests

```bash
python -m pytest -q
cd dashboard/frontend && npm run build
```

## Limitations

- This is a research system, not a trading system. There is no order-routing or brokerage integration.
- Historical market and fundamental data depend on public/vendor sources and their coverage or revision behavior.
- A point-in-time rule improves historical realism but cannot guarantee that every corporate action or vendor correction is perfectly reconstructed.
- Backtest performance is sensitive to universe choice, data quality, cost assumptions, and model selection. It should not be interpreted as a forecast.

## License

No open-source license has been selected yet. Add an explicit license before treating the repository as reusable open-source software.
