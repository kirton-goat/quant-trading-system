# V3 Strict Historical Data Requirements

## What is already completed automatically

- Historical CSI300/CSI500 membership snapshots for all 131 planned 2015-2025 rebalance dates were obtained through BaoStock and cached locally.
- The V3 price builder was verified using `000001`: it writes a separate research cache with an explicit source, `adjustment=qfq`, and fetch timestamp.
- The legacy v1/v2 price cache remains untouched.

## Remaining data required for a validated V3 run

The frozen 2015-2025 configuration contains 1,522 unique historical CSI300/CSI500 constituents. A validated run requires all of the following.

| Dataset | Required fields | Required dates | Why |
| --- | --- | --- | --- |
| Strategy return prices | `date`, `stock_code`, `open`, `high`, `low`, `close`, `volume`, `amount`, `source`, `adjustment` | 2012-07-15 to 2025-12-31 | Price factors and continuous portfolio returns. `adjustment` must be consistently `qfq` or a documented total-return equivalent. |
| PIT statements | `stock_code`, `report_period`, `disclosure_date`, revenue, profit, operating cash flow, assets, liabilities, equity, shares | At least 2012-07-15 to 2025-12-31 disclosure history | Fundamental factor; records become visible only on `disclosure_date`. |
| Unadjusted valuation prices | `date`, `stock_code`, `close`, `source`, `adjustment=none` | Same period | Historical PE/PB with contemporaneous shares and PIT fundamentals. |
| Corporate-action evidence | stock code, event date, action type, cash/ratio detail | 2012-2025 | Test that return-price series remains economically continuous around dividends, rights and splits. |

## Accepted delivery options

1. A provider credential with documented historical constituents, adjusted prices and announcement dates. A Tushare Pro, Wind, Choice, iFinD, JoinQuant or RiceQuant subscription can work if it covers the required fields; the credential should be placed in an environment variable, never pasted into a project file.
2. CSV or Parquet exports from an existing terminal/subscription. Keep the original provider name, export date and adjustment setting with every file.
3. Continue with the existing resumable free-data builder. It is usable for exploratory collection but may take hours and cannot guarantee coverage for delisted names; its output stays `research_experiment / incomplete` until the preflight passes.

## Current commands

```powershell
# Resume the automatic historical constituent download (now already complete)
python -m research.v3.data_acquisition --stage universe

# Resume provenance-preserving qfq price collection. Omit --limit only when
# leaving the machine running; the process skips already-complete files.
python -m research.v3.data_acquisition --stage prices --limit 50

# Inspect invalid financial disclosure dates without changing source data.
python -m research.v3.fundamental_quality

# Re-run strict validation. It never produces a backtest when incomplete.
python -m research.v3.preflight
```

## Current blockers

- 1,521 of 1,522 required V3 adjusted-price files remain to be collected or imported.
- 74 cached fundamental rows have invalid disclosure dates and must be re-sourced or explicitly excluded.
- Corporate-action continuity tests are not yet possible until the return-price series is complete and provenance-tagged.

No imported or downloaded data is allowed to overwrite frozen v1/v2 outputs.
