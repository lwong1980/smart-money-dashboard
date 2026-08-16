# Smart Money Intelligence Dashboard

## Purpose

Pulls options flow, dark pools, congressional trades, insider activity,
financials, technicals, 13Fs, and Polymarket data to spot smart money vs.
retail divergence.

## Architecture

- `data_engine.py` — source-agnostic fetchers + a DB-backed caching layer,
  writing to `smart_money.db` (SQLite).
- `dashboard.py` — Streamlit UI, 7 tabs, reads from `smart_money.db` and/or
  calls `data_engine.py`'s `cached_*` wrappers directly.

### Source-agnostic dispatch

`DATA_SOURCE_CONFIG` (top of `data_engine.py`) maps each fetcher to its
active provider (e.g. `"options_flow": "yfinance"`). The public fetcher
(`fetch_options_flow(ticker)`, `fetch_dark_pool(ticker)`,
`fetch_congressional_trades()`, `fetch_news(ticker)`, `fetch_13f_changes(ticker)`)
branches on that config and dispatches to a `_fetch_X_<provider>` impl;
swapping providers means adding a new branch, not touching any caller.
Every record carries a `"source"` field naming which provider actually
served it. An unimplemented provider raises `NotImplementedError` with a
comment showing the intended API call — except `congressional`, which has
its own two-tier fallback (see below) instead of a hard failure.

### Caching layer

Every fetcher has a matching `cached_*(conn, ticker, max_age_hours, force_refresh)`
wrapper (e.g. `cached_options_flow`, `cached_dark_pool`,
`cached_analyst_targets`, `cached_leaps_candidates`, ...). Each one:
1. Checks `should_refetch(conn, table, ticker, max_age_hours)` against the
   `fetch_log` table (the single freshness ledger for every fetcher).
2. On a cache hit, reads the persisted payload back out of that fetcher's
   table instead of hitting the network.
3. On a miss, calls the pure fetcher, persists the result, logs the
   attempt to `fetch_log` (success/failure/row count/error), and returns.

Every `cached_*` call returns the same envelope shape:
`{"data": ..., "source": ..., "cache_hit": bool, "fetched_at": iso_str}`.
The dashboard uses this directly to render `source_badge()` next to every
section ("SOURCE: yfinance (FREE) · cached 3h ago").

`full_refresh(tickers, force_refresh=False)` drives the watchlist-wide
refresh via these same `cached_*` wrappers, so the sidebar's refresh button
(which passes `force_refresh=True`) and normal scheduled runs share one
code path. `get_history(conn, table, ticker, days_back=90)` is the generic
time-indexed read path for any future ML model — table/date-column pairing
comes from a fixed internal allowlist, not caller input.

Cache windows: fundamentals/analyst targets/earnings ~12h, options flow
15min, dark pool 1h, news 2h, congressional 1h, 13F/buybacks 7 days, LEAPS
1h.

### Health checks

`check_source_health()` runs a lightweight connectivity probe per
configured source (not a full data pull) and returns per-source
status/source/latency/error, e.g. `{"congressional": {"status":
"not_configured", ...}}`. Status values: `up`, `down`, `degraded`,
`not_configured`. The dashboard probes once per session
(`st.session_state`, see `get_health()` in `dashboard.py`) and only
re-probes on the SYSTEM STATUS panel's "Recheck" button — never on every
rerun.

## Data sources (free unless noted)

| Fetcher | Provider | Notes |
|---|---|---|
| options_flow | yfinance | option chains |
| dark_pool | finra_proxy | tries the real FINRA ATS weekly-summary endpoint first (`source="finra_ats"`); falls back to a volume z-score heuristic (`source="finra_proxy"`) when that's unreachable/malformed |
| congressional | quiverquant (**paid**, needs `QUIVER_API_KEY`) → senate_efd_free fallback | see below |
| news | yfinance | `ticker.news`; handles both the old flat and newer `"content"`-nested schemas |
| 13f | sec_edgar_free | SEC full-text search (efts.sec.gov); v1 is "who recently filed", not a holdings diff |
| fundamentals / analyst targets / earnings / buybacks | yfinance | single-source, not in `DATA_SOURCE_CONFIG` |

### Congressional trades: the one non-free source, with a free fallback

`fetch_congressional_trades()` checks `QUIVER_API_KEY` first. If set, it
calls `quiverquant.quiver(key).congress_trading()` (**paid**, ~$30/mo per
their own README). If absent, it logs a warning and falls back to
`_fetch_congressional_senate_efd_free()` — a real 2-step scrape of
`efdsearch.senate.gov` (accept terms → search reports → parse each PTR's
transaction table).

**Gotcha:** `quiverquant`'s own `congress_trading()` never calls
`raise_for_status()`, so an auth/rate-limit error response (e.g.
`{"detail": "Invalid token."}`) comes back as a malformed 1-column
DataFrame instead of raising. Both `_probe_congressional()` (the health
check, which hits the API directly with `requests` instead of going
through the library) and `_normalize_quiver_congress_df()` (shared by the
global-feed and per-ticker fetchers, checks for a `"detail"`-only
DataFrame and raises) guard against this — without those checks, a
bad/expired key would silently report "up" and/or persist a single
garbage all-None record.

**Auth header:** QuiverQuant's real scheme is `Authorization: Token
<key>`, **not** `Bearer <key>` — confirmed against the live API (a
`Bearer` header 401s). `fetch_congressional_trades_by_ticker(ticker)`'s
REST fallback uses the verified-correct `Token` scheme.

`fetch_congressional_trades_by_ticker(ticker)` (paid key required) pulls
that ticker's *full* trade history via `quiver.congress_trading(ticker)`,
richer than filtering the market-wide recent-feed table for one symbol.
Tries the `quiverquant` package first, falls back to a raw REST call on
any failure. `cached_congressional_trades_by_ticker(conn, ticker,
max_age_hours=24, ...)` wraps it with a 24h cache window (vs. 1h for the
global feed, since it's a heavier per-ticker historical pull) and persists
into the same `congressional_trades` table via the shared
`_persist_congressional_trades`. Used by the TICKER DEEP-DIVE tab's
SMART MONEY SIGNALS section.

**Not independently verified against a real QuiverQuant key** — no
`QUIVER_API_KEY` has been available in the dev environment this was built
in. Every error path (missing key, invalid key → 401, malformed-DataFrame
guard, REST fallback) has been tested directly; the success path (a valid
key actually returning trade data) has not. Set `QUIVER_API_KEY` and run
`check_source_health()["congressional"]` to confirm `status="up"`.

**That free fallback returns [] in most hosted environments.**
`efdsearch.senate.gov` sits behind Akamai bot protection that 403s even a
plain `GET /search/` with a browser User-Agent from server/datacenter IPs
(confirmed live during development). It's kept as a genuine best-effort
path — a residential/browser connection can sometimes get through, and
there's no other free, no-key source for Senate stock trades — but expect
it to be empty. Both the SYSTEM STATUS panel and the CONGRESS & INSIDER /
TICKER DEEP-DIVE tabs surface *why* it's empty (`congressional_empty_reason()`
in `dashboard.py`, sourced from the same health probe) rather than looking
broken.

Capitol Trades (the source used before this) was dropped entirely: its
`?rss=1` param just returns normal site HTML, and its internal
`bff.capitoltrades.com` API returned 503 when checked — no working free
endpoint was ever found there.

### Polymarket

Sort with `order=volumeNum`, not `volume` — the latter is silently ignored
by the Gamma API and returns unsorted results.

## Database: `smart_money.db`

Core (populated by `full_refresh`): `options_flow`, `congressional_trades`,
`insider_trades`, `polymarket_events`, `divergence_scores`,
`dark_pool_signals`.

On-demand (populated by the deep-dive tab's `cached_*` calls):
`thirteenf_filings`, `fundamentals_info`, `analyst_targets`,
`earnings_calendar`, `buybacks`, `news_cache`, `leaps_candidates_cache`.

On-demand, paid, user-confirmed only: `ai_briefs` (ticker, brief_json,
fetched_at) — see "AI Briefing" below. Never populated by `full_refresh`.

Infra: `fetch_log` (ticker, data_type, fetched_at, success, rows_returned,
error_message — the freshness ledger every `cached_*` wrapper reads).

`init_db()` also runs `_run_migrations()`, which adds any columns defined
in `_COLUMN_MIGRATIONS` that are missing from an existing DB file (e.g.
`source` on `options_flow`/`dark_pool_signals`/`congressional_trades`,
the recommendation-breakdown columns on `analyst_targets`) — safe to
re-run against an older `smart_money.db`.

## Divergence scoring labels

- `SMART_BULLISH`
- `SMART_BEARISH_RETAIL_LONG`
- `RETAIL_FRENZY`
- `INSTITUTIONAL_ACTIVE`
- `NEUTRAL`

### Scoring formula (`compute_divergence`, weights documented in its docstring)

```
smart_signal = 0.30*insider_net + 0.15*congress_net + 0.55*options_smart_net
score = 100 * (0.30*|smart_signal| + 0.25*|retail_signal|
               + 0.20*institutional_magnitude + 0.25*earnings_proximity_magnitude)
```

`options_smart_net` is the call/put skew of "unusual" (top-decile vol/OI)
options contracts. It carries the majority weight in `smart_signal`
because insider/congress data is high-conviction *when present* but is
structurally sparse for any specific watchlist ticker — the SEC Form4 feed
samples only ~25 market-wide filings per refresh, and congressional data
needs a paid key — so a 50/50 insider/congress blend (the pre-2026-08-15
formula) computed to exactly `0.0` for every single ticker, every time,
whenever neither feed happened to cover it (i.e. almost always). Confirmed
live: 13/13 watchlist tickers had `smart_signal == 0.0` before this fix,
1/13 after (the remaining one had zero options contracts that day).

`earnings_proximity_magnitude` is new: unusual options activity in the 0-5
days before an earnings date pushes conviction (magnitude only, not
direction) higher, since that's a classic positioning-ahead-of-a-catalyst
pattern. It reads `days_to_earnings`/`atm_avg_iv_pct` from the
`earnings_signal` table — `compute_divergence` never fetches live itself,
so `full_refresh` and the deep-dive's "Fetch live signals" both call
`cached_earnings_signal` *before* `compute_divergence` runs. Getting that
ordering right matters: computing divergence before earnings data is fresh
was a real bug in the deep-dive's fetch-live flow, fixed alongside this.

The options `unusual` flag itself (`_fetch_options_flow_yfinance`) is also
percentile-based now, not a fixed cutoff: top decile of vol/OI ratio
*within that ticker's own chain* (plus a 100-contract volume floor to
filter noise), replacing a flat `volume>=500 and ratio>=1.5` rule that was
structurally too strict for lower-liquidity names — STX/WDC/LYFT were
seeing 1-3 flagged contracts out of 700-1000 vs. 20-65 for AAPL/AMD/TSLA.

### ticker_snapshots — append-only history

Every `compute_divergence()` call (and therefore every `full_refresh` and
every deep-dive "Fetch live signals") inserts a new row into
`ticker_snapshots` — never overwritten, so repeated queries over time
return different, comparable, time-stamped answers. `hours_to_earnings` is
signed (negative = before, positive = after; `(now - earnings_date)` in
hours). `get_earnings_snapshot_history(conn, ticker, earnings_date=None)`
is the read path for a future prediction model or a "conviction in the
72h before earnings" chart — the deep-dive's "Snapshot History" expander
uses a simpler unscoped last-10 query instead, since it wants "what
happened recently" rather than one specific earnings event.

### earnings_signal — ported from `~/trading/new_top.py`

`fetch_earnings_signal(ticker)` / `calculate_expected_move_iv(price, iv,
days=5)` preserve that script's calculation logic exactly: IV-based
expected move (`price * (iv/100) * sqrt(days/365)`), ATM straddle expected
move (`ATM call lastPrice + ATM put lastPrice`), ATM avg IV, and
close-to-close % price reaction around each historical earnings date. One
adaptation: `new_top.py` sourced historical earnings dates by scraping
marketbeat.com (fragile, no auth); this reads dates from
`fetch_earnings_calendar`'s yfinance-backed `eps_history` instead, already
in this file. **`new_top.py` itself has multiple hardcoded API
keys/credentials in plaintext (Anthropic, OpenAI, Reddit/praw) — none of
that was touched or ported; only the pure yfinance-based earnings math
was.**

## Running the project

```bash
pip install -r requirements.txt
streamlit run dashboard.py
# optional, for live (paid) congressional data:
export QUIVER_API_KEY=...
# optional, for the on-demand AI briefing (TICKER DEEP-DIVE tab):
export ANTHROPIC_API_KEY=...
```

The terminal logs one startup line confirming whether `QUIVER_API_KEY` was
detected (gated behind `st.cache_resource` so it fires once per server
process, not once per Streamlit rerun).

## Watchlist (`watchlist.json`)

The watchlist is no longer hardcoded. `data_engine.DEFAULT_STARTER_WATCHLIST`
is only a fallback seed (12 tickers) used the first time `watchlist.json`
is created or when the user clicks "Reset to default" — once the file
exists, it's the source of truth, not the constant.

- `load_watchlist(path="watchlist.json")` / `save_watchlist(tickers, path=...)`
  in `data_engine.py` — pure JSON read/write, uppercased and deduped.
- `dashboard.py`'s sidebar "⚙️ Configure Watchlist" expander is the only
  place the file is written from the UI: a comma-separated text area +
  Save button (writes `st.session_state.watchlist` and `watchlist.json`)
  and a Reset-to-default button.
- Every tab (DIVERGENCE MAP, OPTIONS FLOW, FUNDAMENTALS, the sidebar
  "⟳ REFRESH DATA" full refresh, etc.) reads the same `watchlist` variable,
  populated once per rerun from `st.session_state.watchlist` — there is a
  single list, not a per-tab selection.
- `full_refresh(tickers=None, ...)` and the `if __name__ == "__main__"`
  CLI entry point both default to `load_watchlist()` when no explicit
  ticker list is passed, so `python data_engine.py` picks up the same
  persisted list the dashboard uses.

## Key constraints

- All data sources must remain free, except congressional trades, which is
  explicitly paid-with-a-free-fallback (QuiverQuant vs. Senate EFD).
- Rate limit yfinance calls with a 0.3s sleep between tickers in
  `full_refresh`'s per-ticker loop.
- numpy int64 values from `ticker.recommendations` must be cast to plain
  `int()` before binding to sqlite3 — otherwise sqlite3 silently stores
  them as a raw BLOB (pickled bytes) instead of an INTEGER, since it has
  no adapter for numpy scalar types. See `_fetch_recommendation_breakdown`.
- `pd.to_datetime` on a column of mixed-offset ISO timestamps (e.g. earnings
  dates spanning a DST boundary) needs `utc=True` or raises
  `ValueError: Mixed timezones detected`.

## TICKER DEEP-DIVE tab (7th tab)

Search any ticker (no watchlist restriction) for a full panel: price/
technicals, an on-demand AI briefing (see below), fundamentals, earnings &
analyst targets (target range chart + Strong Buy→Strong Sell recommendation
breakdown), smart money signals (insider/congress), options flow, LEAPS
stock-replacement candidates, buybacks, dark pool signal, news (a single
full list — no hardcoded per-ticker thesis, every ticker gets real current
news, plus a "More sources" reference-only link to Financial Juice's site
and that ticker's page there), and related Polymarket markets. Every
section shows a `source_badge()`. "Fetch live signals" passes
`force_refresh=True` through every `cached_*` call for that ticker.

### AI Briefing (`generate_deep_analysis`)

A collapsible "🧠 AI BRIEFING" expander sits right after the price/RSI/MACD
charts, before any raw data table. It is **synthesis-only**: on click,
`generate_deep_analysis(ticker, conn)` in `data_engine.py` pulls everything
already cached for that ticker straight out of SQLite (fundamentals,
analyst targets, earnings calendar, insider trades, buybacks, unusual
options, news headlines, congressional trades, dark pool signal, and the
`ticker_snapshots`/`divergence_scores` history) and bundles it into one
JSON blob — it makes **zero** new external fetches. That bundle is sent to
`claude-opus-5` via the `anthropic` SDK (reads `ANTHROPIC_API_KEY` from the
environment) with `output_config.format` (structured outputs / JSON schema)
enforcing a fixed 5-field shape: `setup`, `institutional_positioning`,
`retail_vs_smart_money`, `catalysts` (a list of
`{catalyst, expected_timing, why_it_matters}` — specific named forward-
looking events pulled from the cached news headlines, not generic
statements), and `bottom_line`. The result is cached in `ai_briefs` for 4
hours (`AI_BRIEF_CACHE_HOURS`) and rendered as prose + a small card grid for
catalysts — the catalysts grid is the dynamic, per-ticker replacement for
any hand-written thesis text, so every ticker gets one, not just a few.

**This never runs automatically.** The only way to trigger a Claude API
call is the "🧠 Generate AI Briefing" button (separate from "Fetch live
signals", never wired to page load/rerun/refresh) or the "🔄 Re-run" button
inside an already-rendered briefing. Both funnel through the same
`@st.dialog`-based confirmation popup (`_confirm_ai_briefing_dialog` in
`dashboard.py`), which shows a rough cost estimate — from
`estimate_ai_briefing_cost(ticker, conn)`, a ~4-chars/token approximation
of the bundled context size against `claude-opus-5` list pricing, no API
call made to produce the estimate — and only calls `generate_deep_analysis`
after the user clicks "Continue". A failed call (e.g. no `ANTHROPIC_API_KEY`
set) is caught and shown as a plain `st.error` in the expander rather than
crashing the tab; `get_cached_ai_brief(conn, ticker)` is a pure DB read used
on every rerun to decide whether to show a cached brief or the empty state,
so revisiting a ticker within the cache window never re-triggers a call.

### Price target range chart (`render_target_range`)

Low/Mean/Median/High/Current markers sit on the true value line; labels
that would collide (any two values within ~8% of the chart's span) are
staggered onto separate vertical tiers with a thin dotted connector back
to the marker, so labels never overlap no matter how close two values
land — including the exact-equality case (current == mean == median).

### LEAPS candidates

`fetch_leaps_candidates` is ported directly from `~/trading/leap_picker.py`'s
methodology (Black-Scholes delta/theta, deep-ITM stock-replacement rules,
borderline-strike selection). Keep the two in sync if the rules change —
`Rules`, `build_itm_calls_table`, and `filter_pass_rules` are duplicated,
not imported, since `leap_picker.py` lives outside this project.

## SYSTEM STATUS panel

Rendered at the very top of the page (`render_system_status()` in
`dashboard.py`), above the title header: a compact one-row strip of
`● UP` / `● DEGRADED` / `● DOWN` / `○ NOT CONFIGURED` badges per fetcher,
a "🔄 Recheck" button, and a collapsible expander with per-source
latency/error/fix-hint detail (e.g. `export QUIVER_API_KEY=...`).
