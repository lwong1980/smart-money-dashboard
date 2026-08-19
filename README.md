# Smart Money Intelligence Dashboard

A Streamlit dashboard that pulls options flow, dark pool activity, congressional/insider trades,
fundamentals, technicals, 13F filings, and Polymarket data to spot smart-money vs. retail
divergence — plus an intraday **Day Prediction** engine with its own historical backtest and
self-calibration loop.

## Quick start

```bash
pip install -r requirements.txt
streamlit run dashboard.py

# optional, for live (paid) congressional data:
export QUIVER_API_KEY=...
# optional, for the on-demand AI briefing (TICKER DEEP-DIVE tab):
export ANTHROPIC_API_KEY=...
```

The dashboard reads/writes `smart_money.db` (SQLite, created on first run) and `watchlist.json`
(seeded from a 12-ticker default the first time it's created). Both are gitignored — they're
local runtime state, not source.

## Architecture

- **`data_engine.py`** — every fetcher (source-agnostic, dispatches on `DATA_SOURCE_CONFIG`),
  the DB-backed caching layer (`cached_*` wrappers, `fetch_log` freshness ledger), the
  divergence-scoring engine, the Earnings Simulator, and the Day Prediction / backtest /
  calibration system described below.
- **`dashboard.py`** — the Streamlit UI: 10 tabs, reads from `smart_money.db` and/or calls
  `data_engine.py`'s `cached_*` wrappers directly.

Every `cached_*` call returns `{"data": ..., "source": ..., "cache_hit": bool, "fetched_at": iso_str}`,
rendered next to its section as a `source_badge()` (e.g. `SOURCE: yfinance (FREE) · cached 3h ago`).

## Tabs

1. **DIVERGENCE MAP** — smart-money vs. retail signal per watchlist ticker
2. **OPTIONS FLOW** — unusual options activity (top-decile vol/OI within each ticker's own chain)
3. **CONGRESS & INSIDER** — congressional trades (QuiverQuant, paid, with a Senate EFD free
   fallback) + SEC Form 4 insider filings
4. **FUNDAMENTALS** — price/valuation/technicals across the watchlist
5. **POLYMARKET** — prediction-market data filtered by sector keywords
6. **EARNINGS SIMULATOR** — pre-earnings probability engine (raw pattern → Bayesian → trained
   model, honesty-tiered on sample size) with a scenario matrix, P/L simulation, and an AI
   Briefing synthesis
7. **TICKER DEEP-DIVE** — search any ticker for a full panel (price/technicals, AI briefing,
   fundamentals, earnings/analyst targets, smart money signals, options flow, LEAPS candidates,
   buybacks, dark pool, news) plus the manual Day Prediction backtest control (see below)
8. **NEWS** — watchlist-wide news feed
9. **RUNNERS** — real-time unusual-activity scan + the **Day Prediction** panel (see below)
10. **SETTINGS** — source health registry, watchlist configuration, track records, the
    calibration panel, and the Backtested Tickers registry

## Day Prediction (RUNNERS tab)

For each ticker showing on RUNNERS, a same-day price target is computed ~30 minutes after
market open (`compute_day_target`), using a blend of the ticker's realized daily volatility,
early-session momentum, technical structure, and (when available) the AI Briefing's qualitative
read — honesty-tiered exactly like the Earnings Simulator: **Mode 1** (raw pattern, <10 pooled
reconciled sessions) → **Mode 2** (Bayesian estimate) → **Mode 3** (trained model, only once it
demonstrably beats Mode 2 on held-out data).

The committed target is never a straight ruler-line to end-of-day — `simulate_intraday_path()`
generates a real drift-adjusted random walk (GBM-style, ~5-min steps) with genuine intraday
texture, generated once per session and cached, so the chart and the live predicted-vs-actual
comparison table always read the exact same path and can never disagree.

**Manual tickers.** Any watchlist ticker can be pinned to the Day Prediction panel even if it
never trips the "unusual" volume/move criteria (RUNNERS' `➕ Add` control) — pins persist for the
browser session and auto-drop the moment their ticker leaves the watchlist. A `✕ Remove` button
un-pins a ticker without hiding it if it's also a genuine runner that day.

**Last session.** When today's prediction isn't ready yet (before the model-start buffer, or
market closed), the panel doesn't go blank — it shows the most recent *live* session's full chart
and comparison table (or, for a ticker with no live session yet, the most recent *backtested*
session's metrics instead, clearly labeled as such), plus the Backtest Report below it. Neither
of those depends on today's session being ready.

### Historical backtest

`backtest_day_predictions(ticker, conn, lookback_days=330)` replays the same prediction pipeline
against ~330 days of real historical daily bars, day by day, using only data strictly *before*
each day (no lookahead) — only on days that would have genuinely qualified as a RUNNERS "unusual"
day (volume ≥2x 20-day avg or move ≥3%). Rows are tagged `source='backtest'` (vs. `'live'`),
reconciled immediately since the outcome is already history, and never overwrite a real live row
for the same session — including today's, which a backtest is never allowed to touch.

This runs automatically once per ticker (`ensure_ticker_data_ready`, the first time a ticker's
raw price history is fetched) and on demand from TICKER DEEP-DIVE's `🔬 Run Backtest` button for
*any* ticker, not just current watchlist/RUNNERS tickers. A **Backtest Report** (summary stats,
an error-distribution histogram, directional accuracy, best-5/worst-5 concrete examples) is
available anywhere a ticker's Day Prediction panel renders. SETTINGS' **Backtested Tickers**
registry lists every backtested ticker with a per-row `🔄 Re-run` to keep the trailing window
current.

Backtested rows count toward Mode 1/2/3 sample thresholds and the calibration pool — bootstrapping
a real, sizeable sample instantly instead of waiting weeks for live sessions to accumulate — but
every confidence caption states the composition explicitly (e.g. *"Mode 2: Bayesian estimate (18
backtested sessions + 2 live sessions)"*). Live sessions remain the higher-trust category as they
accumulate, since they capture real same-day intraday momentum and AI Briefing input that a
historical daily-OHLCV reconstruction structurally can't.

### Calibration loop

After every reconciliation (same-day, right after that session's closing snapshot, and once more
per day as a fallback), `calibrate_day_prediction_model()` checks the full reconciled history
(backtested + live) for a systematic bias and nudges two persisted, bounded scaling factors:

- **vol_scale** — up if actual moves are consistently larger than predicted (magnitude
  under-prediction), down if consistently smaller
- **drift_scale** — down (never up) if the directional call is reliably worse than a coin flip
  with enough samples to mean something; a confident wrong call is worse than a timid right one

Below an 8-sample floor, no adjustment is made and the UI says so plainly. Every run — adjusted or
not — is logged to `model_calibration_log`, visible in SETTINGS alongside the honest running
average: *"Current average error: X% over N reconciled sessions (target: 0-3%)"*, with an explicit
caveat that consistent 0-3% same-day error on a single name is a genuinely hard target even for
professional intraday models.

## Key constraints

- All data sources are free except congressional trades (QuiverQuant, paid, with a Senate EFD
  free fallback that's frequently blocked by Akamai bot protection from datacenter IPs).
- yfinance calls are rate-limited (0.3s between tickers) in `full_refresh`.
- Market hours are hardcoded as NYSE/NASDAQ 9:30–4:00 ET, weekday-only (no holiday calendar).

See `CLAUDE.md` for the full technical reference — schema details, every fetcher/provider
mapping, the divergence-scoring formula, and the conventions/gotchas this codebase has
accumulated (numpy→sqlite3 type coercion, mixed-timezone `pd.to_datetime`, WAL-mode DB safety,
etc.).

## License

Apache License 2.0 — see `LICENSE`.
