"""
Data engine for the Smart Money Intelligence Dashboard.

Source-agnostic architecture: every fetcher dispatches on DATA_SOURCE_CONFIG
and returns a fixed, documented shape with a "source" field on every record,
regardless of which provider actually served it. Fetchers that hit a live
API are "pure" (no conn, no caching) -- a matching cached_* wrapper persists
results to SQLite, checks freshness via fetch_log before re-hitting the
network, and always returns an envelope: {"data", "source", "cache_hit",
"fetched_at"}.

Sources are best-effort: RSS/undocumented public endpoints (FINRA ATS,
Senate EFD) can change shape or go dark without notice, so every fetcher
fails soft (logs a warning, returns empty) instead of raising and killing
a refresh.
"""

import html
import json
import math
import os
import pickle
import random
import re
import sqlite3
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import feedparser
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from scipy.signal import argrelextrema

DEFAULT_DB_PATH = "smart_money.db"
WATCHLIST_CONFIG_PATH = "watchlist.json"

# Fallback starter list only -- used to seed watchlist.json the first time it's
# created, or when the user explicitly clicks "Reset to default". Not forced:
# once watchlist.json exists, it -- not this constant -- is the source of truth.
DEFAULT_STARTER_WATCHLIST = [
    "AAPL", "NVDA", "TSLA", "MSFT", "SPY", "AMD", "META", "AMZN",
    "STX", "WDC", "MU", "LYFT",
]


def load_watchlist(path=WATCHLIST_CONFIG_PATH):
    """Load the user's saved watchlist from a local JSON config file.

    Falls back to DEFAULT_STARTER_WATCHLIST (without writing it) if the file
    is missing, empty, or unreadable -- callers that want the fallback
    persisted should follow up with save_watchlist().
    """
    try:
        with open(path, "r") as f:
            tickers = json.load(f)
        if isinstance(tickers, list) and tickers:
            return [str(t).strip().upper() for t in tickers if str(t).strip()]
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    return list(DEFAULT_STARTER_WATCHLIST)


def save_watchlist(tickers, path=WATCHLIST_CONFIG_PATH):
    """Persist the watchlist to a local JSON config file so it survives restarts."""
    cleaned = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)
    return cleaned

DIVERGENCE_LABELS = [
    "SMART_BULLISH",
    "SMART_BEARISH_RETAIL_LONG",
    "RETAIL_FRENZY",
    "INSTITUTIONAL_ACTIVE",
    "NEUTRAL",
]

# --------------------------------------------------------------------------
# Source-agnostic architecture: which provider backs each fetcher. Swap a
# value here (and add a matching branch in the fetcher) to change providers
# without touching any caller -- every fetcher returns the same documented
# shape no matter which branch served it.
# --------------------------------------------------------------------------

DATA_SOURCE_CONFIG = {
    "options_flow": "yfinance",
    "dark_pool": "finra_proxy",
    "congressional": "quiverquant",   # falls back to senate_efd if no API key
    "13f": "sec_edgar_free",
    # "news" intentionally not here -- it's no longer a single provider
    # dispatch. News sources are a real, user-editable registry (the
    # news_sources table, managed from SETTINGS → News Feeds), since
    # multiple feeds can be enabled/added/removed independently. See
    # fetch_news()/add_news_source()/remove_news_source().
}

SEC_HEADERS = {"User-Agent": "SmartMoneyDashboard research@smartmoneydash.local"}
SEC_FORM4_FEED = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=4&company=&dateb=&owner=include&count=100&output=atom"
)

FINRA_ATS_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"

POLYMARKET_URL = "https://gamma-api.polymarket.com/markets"

# Default keyword set for "macro-relevant" Polymarket markets, used by
# fetch_polymarket(keywords=...) and by full_refresh to make sure these
# get pulled into the DB even when they're not in the raw top-volume slice.
MACRO_KEYWORDS = [
    "fed", "rate", "inflation", "china", "taiwan", "chip", "chips", "export",
    "tariff", "recession", "gdp", "unemployment", "treasury", "debt ceiling",
    "oil", "opec", "sanctions",
]

SEC_FULLTEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=13F-HR"

# Senate EFD (efdsearch.senate.gov) -- the free, no-key alternative to
# QuiverQuant for congressional trades. See _fetch_congressional_senate_efd_free.
SENATE_EFD_BASE = "https://efdsearch.senate.gov"
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
}

_MARKET_KEY = "_MARKET_"  # sentinel ticker for market-wide (non-per-ticker) fetches


# --------------------------------------------------------------------------
# Database schema
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS options_flow (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    fetch_date TEXT NOT NULL,
    expiration TEXT NOT NULL,
    strike REAL NOT NULL,
    option_type TEXT NOT NULL,
    volume INTEGER,
    open_interest INTEGER,
    volume_oi_ratio REAL,
    implied_volatility REAL,
    last_price REAL,
    underlying_price REAL,
    unusual INTEGER DEFAULT 0,
    source TEXT,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    rho REAL,
    bid REAL,
    ask REAL,
    created_at TEXT,
    UNIQUE(ticker, fetch_date, expiration, strike, option_type)
);

CREATE TABLE IF NOT EXISTS congressional_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    politician TEXT,
    ticker TEXT,
    transaction_type TEXT,
    amount_range TEXT,
    transaction_date TEXT,
    disclosure_date TEXT,
    chamber TEXT,
    party TEXT,
    source_url TEXT UNIQUE,
    source TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS insider_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    insider_name TEXT,
    title TEXT,
    transaction_type TEXT,
    shares REAL,
    price REAL,
    value REAL,
    transaction_date TEXT,
    filing_date TEXT,
    source_url TEXT,
    created_at TEXT,
    UNIQUE(ticker, insider_name, transaction_date, shares, source_url)
);

CREATE TABLE IF NOT EXISTS polymarket_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT UNIQUE,
    question TEXT,
    category TEXT,
    yes_price REAL,
    no_price REAL,
    volume REAL,
    liquidity REAL,
    end_date TEXT,
    active INTEGER,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS divergence_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    computed_date TEXT,
    score REAL,
    label TEXT,
    smart_signal REAL,
    retail_signal REAL,
    institutional_magnitude REAL,
    components_json TEXT,
    computed_at TEXT,
    UNIQUE(ticker, computed_date)
);

CREATE TABLE IF NOT EXISTS dark_pool_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    date TEXT,
    dark_pool_volume REAL,
    total_volume REAL,
    dark_pool_pct REAL,
    volume_zscore REAL,
    signal TEXT,
    is_proxy INTEGER DEFAULT 0,
    source TEXT,
    computed_at TEXT,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS thirteenf_filings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    institution TEXT,
    cik TEXT,
    filing_date TEXT,
    accession_no TEXT,
    form_type TEXT,
    source_url TEXT,
    source TEXT,
    created_at TEXT,
    UNIQUE(ticker, cik, accession_no)
);

CREATE TABLE IF NOT EXISTS marketbeat_institutional (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    ownership_pct TEXT,
    buyers INTEGER,
    sellers INTEGER,
    inflows TEXT,
    outflows TEXT,
    net_flow_bias_pct INTEGER,
    transactions_json TEXT,
    source TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    data_type TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    success INTEGER NOT NULL,
    rows_returned INTEGER DEFAULT 0,
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_lookup ON fetch_log(data_type, ticker, fetched_at);

CREATE TABLE IF NOT EXISTS fundamentals_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    source TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyst_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    current_price REAL,
    target_mean REAL,
    target_high REAL,
    target_low REAL,
    target_median REAL,
    recommendation_key TEXT,
    recommendation_mean REAL,
    num_analysts INTEGER,
    source TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    eps_estimate REAL,
    reported_eps REAL,
    surprise_pct REAL,
    source TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, earnings_date)
);

CREATE TABLE IF NOT EXISTS buybacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,
    buyback_value REAL,
    source TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, period)
);

CREATE TABLE IF NOT EXISTS news_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    publisher TEXT,
    link TEXT,
    published_at TEXT,
    source TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, link)
);

CREATE TABLE IF NOT EXISTS news_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    feed_url TEXT NOT NULL,
    url_type TEXT NOT NULL DEFAULT 'rss',
    enabled INTEGER NOT NULL DEFAULT 1,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leaps_candidates_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    contract_symbol TEXT,
    expiry TEXT,
    strike REAL,
    mid REAL,
    delta_est REAL,
    breakeven REAL,
    option_cost REAL,
    bear_price REAL, bear_roi REAL,
    base_price REAL, base_roi REAL,
    bull_price REAL, bull_roi REAL,
    source TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS earnings_signal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    current_price REAL,
    next_earnings_date TEXT,
    days_to_earnings REAL,
    atm_avg_iv_pct REAL,
    iv_expected_move_usd REAL,
    straddle_expected_move_usd REAL,
    historical_reactions_json TEXT,
    source TEXT,
    fetched_at TEXT NOT NULL
);

-- Append-only: never overwritten, so repeated queries over time return
-- different, comparable, time-stamped answers -- the point is to be able
-- to ask "how did conviction move in the 72h before earnings" later.
CREATE TABLE IF NOT EXISTS ticker_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    snapshot_type TEXT,        -- 'pre_earnings', 'post_earnings', 'routine'
    hours_to_earnings REAL,    -- signed: negative = before, positive = after
    conviction_score REAL,
    divergence_label TEXT,
    smart_call_signals INTEGER,
    smart_put_signals INTEGER,
    retail_heat REAL,
    iv_snapshot REAL,          -- current implied volatility at snapshot time
    price_snapshot REAL,
    earnings_date TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ticker_snapshots_lookup ON ticker_snapshots(ticker, earnings_date, fetched_at);

-- Cached output of generate_deep_analysis() -- a synthesis pass over data
-- already in this DB, sent to the Claude API on explicit user request only.
-- Re-used within a 4h window so revisiting a ticker doesn't re-trigger a
-- paid API call.
CREATE TABLE IF NOT EXISTS ai_briefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    brief_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ai_briefs_ticker ON ai_briefs(ticker, fetched_at);

-- Real (scraped MarketBeat + yfinance-matched) earnings history: one row per
-- reported quarter, with the real beat/miss AND the real next-day price
-- reaction computed from actual price history -- ported from new_top.py's
-- get_marketbeat_analyst_sentiment_structured + get_eps_and_move_summary_real_data,
-- never the ChatGPT-guessing equivalents.
CREATE TABLE IF NOT EXISTS earnings_history_real (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    earnings_date TEXT NOT NULL,
    quarter TEXT,
    consensus_eps TEXT,
    actual_eps TEXT,
    beat_miss TEXT,
    revenue_estimate TEXT,
    revenue_actual TEXT,
    eps_beat_miss_pct REAL,
    revenue_beat_miss_pct REAL,
    price_reaction_pct REAL,
    source TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, earnings_date)
);

-- Real retail sentiment posts (Reddit + StockTwits), each tagged with its
-- source and original timestamp -- ported from new_top.py's get_reddit_posts
-- / get_stocktwits_posts. Raw posts only; no sentiment-score guessing.
CREATE TABLE IF NOT EXISTS retail_sentiment_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    source TEXT NOT NULL,
    text TEXT,
    url TEXT,
    posted_at TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, source, text)
);

CREATE TABLE IF NOT EXISTS apewisdom_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    rank INTEGER,
    mentions INTEGER,
    upvotes INTEGER,
    rank_24h_ago INTEGER,
    mentions_24h_ago INTEGER,
    source TEXT,
    fetched_at TEXT NOT NULL
);

-- Daily OHLCV history, the store that makes delta loading possible for
-- price data: fetch_price_history_delta() pulls a full period="1y" only
-- the first time a ticker is seen, then just the days after MAX(date)
-- here on every call after that, merging the new rows into what's
-- already stored rather than re-pulling the whole trailing window.
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_price_history_lookup ON price_history(ticker, date);

-- EARNINGS SIMULATOR (Part 10) -- one row logged per simulator run, not
-- just per earnings event, so the timeline of how a prediction evolved
-- as new data came in is itself preserved. Only the last row before the
-- earnings date (is_final_prediction=1) is what reconcile_earnings_
-- predictions() grades against the actual outcome.
CREATE TABLE IF NOT EXISTS earnings_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    earnings_date TEXT,
    predicted_at TEXT,
    mode TEXT,                       -- 'raw_pattern' / 'bayesian' / 'trained_model'
    prob_up REAL,
    prob_down REAL,
    prob_flat REAL,
    magnitude_estimate_pct REAL,
    predicted_direction TEXT,
    recommended_strategy TEXT,
    source_briefing_id TEXT,         -- links to the ai_briefs row that informed this
    scenario_matrix_json TEXT,
    confidence_level TEXT,
    is_final_prediction INTEGER DEFAULT 0,
    actual_eps_result TEXT,
    actual_revenue_result TEXT,
    actual_price_move_pct REAL,
    actual_direction TEXT,
    prediction_correct INTEGER,
    reconciliation_notes TEXT,
    reconciled_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_earnings_predictions_lookup
    ON earnings_predictions(ticker, earnings_date, predicted_at);

-- Part 8 -- single-row cache of the last trained earnings-direction
-- model (pickled sklearn estimator as a BLOB) plus the held-out
-- comparison that justified switching to it. train_earnings_direction_
-- model() only retrains every 5 new reconciled predictions (checked via
-- n_samples_at_train here), not on every simulator run.
CREATE TABLE IF NOT EXISTS earnings_model_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TEXT,
    n_samples_at_train INTEGER,
    model_type TEXT,
    held_out_accuracy REAL,
    bayesian_baseline_accuracy REAL,
    beats_baseline INTEGER,
    feature_names_json TEXT,
    model_blob BLOB
);

-- Real per-firm analyst rating CHANGES (yfinance ticker.upgrades_downgrades)
-- -- breadth data (how many firms moved up/down recently) that the single
-- aggregate recommendationKey/mean-target in analyst_targets can't show.
-- Replaced wholesale on every fetch (not upserted per-row) since yfinance
-- itself only ever returns a bounded trailing window, so there's no
-- accumulation to preserve across fetches.
CREATE TABLE IF NOT EXISTS analyst_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    action_date TEXT NOT NULL,
    firm TEXT,
    from_grade TEXT,
    to_grade TEXT,
    action TEXT,
    source TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyst_actions_lookup ON analyst_actions(ticker, action_date);

-- 2-4 direct peers (curated map or same-sector watchlist fallback -- see
-- fetch_peer_comparison) with YTD return and trailing P/E, for the AI
-- Briefing's relative-positioning framing.
CREATE TABLE IF NOT EXISTS peer_comparison (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    peer_ticker TEXT NOT NULL,
    ytd_return_pct REAL,
    pe REAL,
    source TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, peer_ticker)
);

-- Ticker vs. benchmark (SPY/QQQ) total return over YTD/6mo/1y, plus the
-- 52-week high (and its date) computed from stored daily price_history --
-- see fetch_benchmark_relative_performance.
CREATE TABLE IF NOT EXISTS benchmark_relative_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    benchmark TEXT,
    windows_json TEXT,
    fifty_two_week_high REAL,
    fifty_two_week_high_date TEXT,
    pct_below_52wk_high REAL,
    source TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker)
);

-- Dedicated, cited web-research pass (Claude + web_search) built specifically
-- to surface deep, named product/technology/regulatory catalysts that the
-- shallow news_headlines RSS feed structurally misses -- see
-- fetch_catalyst_context. A real, paid Claude API call, so it's cached with
-- a long TTL (7 days) and only ever triggered from inside generate_deep_
-- analysis(), covered by that same user-confirmed "Generate AI Briefing"
-- click, never on its own.
CREATE TABLE IF NOT EXISTS catalyst_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    context_json TEXT NOT NULL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_catalyst_context_lookup ON catalyst_context(ticker, fetched_at);

-- RUNNERS tab's Day Prediction feature: one committed target per
-- (ticker, session_date) -- see compute_day_target/log_day_prediction.
-- The lean_pre_model/rsi14/macd_histogram/volume_vs_20d_avg_pct/
-- trend_score columns are the raw feature inputs behind that day's lean,
-- persisted at prediction time so a real training matrix can be built
-- later (train_day_direction_model) once enough rows reconcile --
-- mirrors earnings_predictions joining against earnings_history_real,
-- except there's no equivalent outcome-features table for daily
-- sessions, so the features are captured directly on this row instead.
CREATE TABLE IF NOT EXISTS day_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    session_date TEXT NOT NULL,
    model_start_time TEXT,
    model_start_price REAL,
    target_price REAL,
    predicted_direction TEXT,
    magnitude_estimate_pct REAL,
    lean_pre_model REAL,
    rsi14 REAL,
    macd_histogram REAL,
    volume_vs_20d_avg_pct REAL,
    trend_score REAL,
    mode TEXT,
    confidence_level TEXT,
    backtest_error_pct REAL,
    actual_close_price REAL,
    actual_direction TEXT,
    prediction_correct_direction INTEGER,
    error_pct REAL,
    reconciliation_notes TEXT,
    predicted_at TEXT,
    reconciled_at TEXT,
    UNIQUE(ticker, session_date)
);

-- One row per refresh (manual or auto) past model-start, for the RUNNERS
-- tab's live prediction-vs-actual table (Part 4) -- survives page
-- reloads within the same session_date, unlike a pure st.session_state
-- approach.
CREATE TABLE IF NOT EXISTS day_prediction_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    session_date TEXT NOT NULL,
    snapshot_time TEXT NOT NULL,
    actual_price REAL,
    UNIQUE(ticker, session_date, snapshot_time)
);

-- Mirrors earnings_model_cache exactly, for the Day Prediction feature's
-- own Mode 3 (train_day_direction_model) -- a separate cache since the
-- two models are trained on entirely different feature sets/tables.
CREATE TABLE IF NOT EXISTS day_model_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TEXT,
    n_samples_at_train INTEGER,
    model_type TEXT,
    held_out_accuracy REAL,
    baseline_accuracy REAL,
    beats_baseline INTEGER,
    feature_names_json TEXT,
    model_blob BLOB
);

-- Real, persisted 5-minute intraday bars -- yfinance's genuine free-tier
-- limit is ~60 days of history at this granularity, so this table never
-- holds more than that real window per ticker (see fetch_intraday_
-- history_delta). This is what lets Day Prediction's volatility estimate
-- eventually improve using real intraday patterns as days accumulate,
-- instead of always falling back to daily-vol-scaled estimates -- and
-- what ensure_ticker_data_ready backfills synchronously for a brand-new
-- ticker instead of leaving it to accumulate for weeks.
CREATE TABLE IF NOT EXISTS intraday_price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    datetime TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    fetched_at TEXT NOT NULL,
    UNIQUE(ticker, datetime)
);
CREATE INDEX IF NOT EXISTS idx_intraday_price_history_lookup ON intraday_price_history(ticker, datetime);

-- Append-only history of calibrate_day_prediction_model's own runs -- one
-- row per calibration pass (whether or not it actually adjusted anything),
-- so it's visible over time how (and whether) the model's drift/vol
-- assumptions have shifted as more sessions reconcile. The most recent row
-- is also the ACTIVE calibration read by compute_day_target.
CREATE TABLE IF NOT EXISTS model_calibration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    n_samples INTEGER,
    mean_abs_error_pct REAL,
    target_error_pct REAL,
    drift_scale REAL,
    vol_scale REAL,
    magnitude_ratio REAL,
    direction_accuracy REAL,
    note TEXT,
    calibrated_at TEXT
);

-- One row per backtest_day_predictions() run (Part 6 registry) -- a
-- ticker can be re-backtested as the trailing lookback window rolls
-- forward, so this is a log, not a single current-state row; the latest
-- row per ticker is what the "Backtested Tickers" registry table reads.
CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    run_at TEXT NOT NULL,
    lookback_days INTEGER,
    date_range_start TEXT,
    date_range_end TEXT,
    qualifying_sessions_found INTEGER,
    sessions_backtested INTEGER,
    mean_error_pct REAL,
    mean_abs_error_pct REAL,
    direction_accuracy REAL
);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_ticker ON backtest_runs(ticker, run_at);
"""

# Migrations for columns added after a table's original CREATE TABLE, so
# existing databases from earlier versions of this app pick them up too.
_COLUMN_MIGRATIONS = {
    "options_flow": [
        ("source", "TEXT"), ("delta", "REAL"), ("gamma", "REAL"), ("theta", "REAL"),
        ("vega", "REAL"), ("rho", "REAL"), ("bid", "REAL"), ("ask", "REAL"),
    ],
    "dark_pool_signals": [("source", "TEXT")],
    "congressional_trades": [("source", "TEXT")],
    "insider_trades": [("source", "TEXT")],
    "analyst_targets": [
        ("rec_period", "TEXT"), ("strong_buy", "INTEGER"), ("buy", "INTEGER"),
        ("hold", "INTEGER"), ("sell", "INTEGER"), ("strong_sell", "INTEGER"),
    ],
    "earnings_history_real": [
        ("revenue_estimate", "TEXT"), ("eps_beat_miss_pct", "REAL"), ("revenue_beat_miss_pct", "REAL"),
        # Part 9 -- deliberately NULL on backfilled rows (not available
        # historically); populated only going forward by the live
        # simulator/reconciliation pipeline once a row's earnings date
        # has actually happened with a real AI Briefing on record.
        ("catalyst_status", "TEXT"), ("buyback_signal", "TEXT"), ("pre_earnings_iv", "REAL"),
        ("pre_earnings_skew_pts", "REAL"),
    ],
    "apewisdom_sentiment": [("attention_change_pct", "REAL")],
    "retail_sentiment_posts": [("sentiment_tag", "TEXT")],
    # CapEx/OCF/FCF, added alongside the existing buyback-value column so
    # the AI Briefing can connect a FCF decline to a CapEx step-up
    # explicitly instead of listing them as unrelated numbers.
    "buybacks": [("capex", "REAL"), ("operating_cash_flow", "REAL"), ("fcf", "REAL")],
    # The cached simulated intraday path (see simulate_intraday_path) --
    # generated ONCE at first commit, never regenerated, so the chart and
    # the live comparison table always read the exact same random path.
    "day_predictions": [
        ("simulated_path_json", "TEXT"),
        # Part 5's path-shape reconciliation -- mean |error%| between the
        # cached simulated path and the real logged snapshots, separate
        # from error_pct (which only grades the final end-of-day target).
        ("path_mean_abs_error_pct", "REAL"),
        # "live" (real same-day prediction) vs "backtest" (historical
        # reconstruction from backtest_day_predictions) -- NULL on rows
        # written before this column existed, always treated as "live" in
        # every query that reads it (COALESCE(source, 'live') / `source or
        # "live"`), since every row before backtesting existed WAS live.
        ("source", "TEXT"),
    ],
}


def _run_migrations(conn):
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for col_name, col_type in columns:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
    conn.commit()


def init_db(db_path=DEFAULT_DB_PATH):
    """Create all tables (no-op if they already exist) and run migrations."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        _run_migrations(conn)
        _seed_news_sources(conn)
    finally:
        conn.close()


def _seed_news_sources(conn):
    """One-time seed of news_sources with whatever feed is actually live
    today -- confirmed by testing directly (feedparser.parse against the
    real URL returned real, current headlines) rather than assumed. This
    is the ONLY real news source this app has ever had: previously
    fetched via yf.Ticker(ticker).news (a Python library call, not a
    URL), now migrated to the equivalent real RSS URL so it fits the
    editable feed-registry model. No-op once the table has any row (the
    user may have since removed this default feed entirely -- that's a
    valid state, not something to re-seed back in)."""
    if conn.execute("SELECT COUNT(*) FROM news_sources").fetchone()[0] > 0:
        return
    conn.execute(
        """INSERT INTO news_sources (name, feed_url, url_type, enabled, added_at)
           VALUES (?,?,?,?,?)""",
        ("Yahoo Finance RSS", "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
         "rss", 1, datetime.utcnow().isoformat()),
    )
    conn.commit()


def get_connection(db_path=DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _safe_num(v, default=0.0):
    """Coerce a value (possibly NaN/None from yfinance) to a plain float,
    since `nan or default` returns nan -- NaN is truthy in Python."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(f) else f


def _none_if_nan(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


# Standard-normal CDF/PDF -- shared by every Black-Scholes calculation in
# this file (bs_greeks below, and the LEAPS scanner's _bs_call_delta_theta,
# now a thin wrapper over bs_greeks). Originally ported from
# ~/trading/leap_picker.py's norm_cdf/norm_pdf; kept here as the one
# definition both call sites use, rather than duplicated per section.
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


# --------------------------------------------------------------------------
# Caching infrastructure: fetch_log is the single freshness ledger every
# cached_* wrapper consults, so caching logic is identical across every
# fetcher regardless of what table its payload actually lives in.
# --------------------------------------------------------------------------

def _log_fetch(conn, data_type, ticker, success, rows_returned=0, error_message=None):
    conn.execute(
        """INSERT INTO fetch_log (ticker, data_type, fetched_at, success, rows_returned, error_message)
           VALUES (?,?,?,?,?,?)""",
        (ticker or _MARKET_KEY, data_type, datetime.utcnow().isoformat(),
         1 if success else 0, rows_returned, error_message),
    )
    conn.commit()


def should_refetch(conn, table, ticker, max_age_hours):
    """True if there is no successful fetch_log entry for (table, ticker)
    within max_age_hours -- i.e. it's time to hit the network again."""
    row = conn.execute(
        """SELECT fetched_at FROM fetch_log WHERE data_type=? AND ticker=? AND success=1
           ORDER BY fetched_at DESC LIMIT 1""",
        (table, ticker or _MARKET_KEY),
    ).fetchone()
    if not row or not row[0]:
        return True
    last = pd.Timestamp(row[0])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    age_hours = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600.0
    return age_hours >= max_age_hours


def _last_fetch_info(conn, table, ticker):
    """ISO timestamp of the most recent successful fetch, or None."""
    row = conn.execute(
        """SELECT fetched_at FROM fetch_log WHERE data_type=? AND ticker=? AND success=1
           ORDER BY fetched_at DESC LIMIT 1""",
        (table, ticker or _MARKET_KEY),
    ).fetchone()
    return row[0] if row else None


# (table, date_column) pairs get_last_timestamp() is allowed to query --
# both are interpolated into raw SQL (sqlite3 can't parametrize identifiers),
# so this allowlist is validated before every call, same pattern as
# _TABLE_DATE_COLUMNS/get_history() above.
_LAST_TIMESTAMP_ALLOWED = {
    ("options_flow", "fetch_date"),
    ("news_cache", "published_at"),
    ("price_history", "date"),
    ("congressional_trades", "disclosure_date"),
    ("congressional_trades", "transaction_date"),
    ("insider_trades", "filing_date"),
    ("insider_trades", "transaction_date"),
}


def get_last_timestamp(conn, table, ticker, date_column="fetched_at"):
    """Returns the most recent timestamp/date we have stored for this
    ticker in the given table, or None if we have no data at all
    (meaning a full historical pull is needed). The delta-loading
    counterpart to should_refetch()'s binary fresh/stale gate: instead of
    "is it time to refetch", this answers "refetch starting from where."
    """
    if (table, date_column) not in _LAST_TIMESTAMP_ALLOWED:
        raise ValueError(
            f"get_last_timestamp: unsupported (table, date_column) pair: ({table!r}, {date_column!r})"
        )
    row = conn.execute(f"SELECT MAX({date_column}) FROM {table} WHERE ticker=?", (ticker,)).fetchone()
    return row[0] if row and row[0] else None


# Table -> the column that carries its "when did this happen" date, used by
# get_history for a uniform time-indexed query across otherwise-different
# schemas. `table` is validated against this fixed allowlist before being
# interpolated into SQL, so only these exact literal names are ever used.
_TABLE_DATE_COLUMNS = {
    "options_flow": "fetch_date",
    "dark_pool_signals": "date",
    "divergence_scores": "computed_date",
    "insider_trades": "transaction_date",
    "congressional_trades": "transaction_date",
    "thirteenf_filings": "filing_date",
    "analyst_targets": "fetched_at",
    "earnings_calendar": "earnings_date",
    "buybacks": "period",
    "news_cache": "published_at",
    "leaps_candidates_cache": "fetched_at",
    "polymarket_events": "last_updated",
    "fetch_log": "fetched_at",
}


def get_history(conn, table, ticker, days_back=90):
    """Clean time-indexed historical query: every row for `ticker` in
    `table` from the last `days_back` days, ordered oldest -> newest. Meant
    as the read path for any future ML model -- one call works the same way
    regardless of which table backs it."""
    if table not in _TABLE_DATE_COLUMNS:
        raise ValueError(f"get_history: unknown table '{table}'. Known tables: {sorted(_TABLE_DATE_COLUMNS)}")
    date_col = _TABLE_DATE_COLUMNS[table]
    query = f"""
        SELECT * FROM {table}
        WHERE ticker = ? AND {date_col} >= date('now', ?)
        ORDER BY {date_col} ASC
    """
    return pd.read_sql_query(query, conn, params=(ticker, f"-{int(days_back)} day"))


# --------------------------------------------------------------------------
# Technical indicator helpers
# --------------------------------------------------------------------------

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# --------------------------------------------------------------------------
# fetch_fundamentals -- price history (cheap, always fetched fresh) + info
# (expensive .get_info() call, DB-cached via cached_fundamentals)
# --------------------------------------------------------------------------

def _add_technicals(hist):
    hist = hist.copy()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    hist["SMA200"] = hist["Close"].rolling(200).mean()
    hist["RSI14"] = _rsi(hist["Close"])
    macd_line, signal_line, hist_bar = _macd(hist["Close"])
    hist["MACD"] = macd_line
    hist["MACD_signal"] = signal_line
    hist["MACD_hist"] = hist_bar
    return hist


def _build_snapshot(hist):
    latest = hist.iloc[-1]
    sma50, sma200 = latest.get("SMA50"), latest.get("SMA200")
    if pd.notna(sma50) and pd.notna(sma200):
        trend = "BULLISH" if sma50 > sma200 else "BEARISH"
    else:
        trend = "N/A"
    return {
        "last_price": float(latest["Close"]),
        "change_pct": float((hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100)
        if len(hist) > 1 else 0.0,
        "sma50": float(sma50) if pd.notna(sma50) else None,
        "sma200": float(sma200) if pd.notna(sma200) else None,
        "rsi14": float(latest["RSI14"]) if pd.notna(latest["RSI14"]) else None,
        "macd": float(latest["MACD"]) if pd.notna(latest["MACD"]) else None,
        "macd_signal": float(latest["MACD_signal"]) if pd.notna(latest["MACD_signal"]) else None,
        "trend": trend,
    }


# --------------------------------------------------------------------------
# fetch_price_history -- interval/lookback-aware price history for the
# FUNDAMENTALS and TICKER DEEP-DIVE price charts (distinct from
# cached_fundamentals' fixed 1y/1d history, which backs the header
# snapshot/RSI/MACD elsewhere).
#
# Candle granularity (interval) and visible range (lookback) are two
# independent axes -- exactly how TradingView's own controls work -- not
# one dropdown conflating both. The earlier single-preset design (e.g. a
# "5m" preset meaning "5m candles over a fixed 5-day window") baked a
# lookback choice into a granularity choice, so picking "5m" (clearly a
# request for fine detail) unexpectedly meant "show 5 days," not "show
# today." Splitting them lets the UI default 5m to a single session (1d
# lookback) while still letting a user explicitly widen it.
# --------------------------------------------------------------------------

INTERVAL_OPTIONS = ["5m", "15m", "1h", "1d"]
DEFAULT_INTERVAL = "1d"

LOOKBACK_OPTIONS = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
DEFAULT_LOOKBACK = "6mo"

# level_tolerance_pct is the % band _cluster_levels() uses to merge nearby
# swing points into one support/resistance level, and sma_windows/horizon
# frame the RSI/MACD narrative labels -- all three are properties of candle
# *granularity* (how noisy/fine-grained each bar is), not of how far back
# the chart looks, which is why they key off interval alone now rather
# than the old combined timeframe preset.
_INTERVAL_CONFIG = {
    "5m":  {"yf_interval": "5m",  "sma_windows": (10, 30),  "horizon": "intraday", "level_tolerance_pct": 0.6},
    "15m": {"yf_interval": "15m", "sma_windows": (10, 30),  "horizon": "intraday", "level_tolerance_pct": 0.6},
    "1h":  {"yf_interval": "60m", "sma_windows": (20, 50),  "horizon": "intraday", "level_tolerance_pct": 0.8},
    "1d":  {"yf_interval": "1d",  "sma_windows": (50, 200), "horizon": "swing",    "level_tolerance_pct": 2.0},
}

# yfinance/Yahoo's real intraday lookback ceilings (well documented, not
# arbitrary): 5m/15m data is only available for the trailing ~60 days;
# 60m/1h for ~730 days (~2 years); 1d has no meaningful ceiling.
_INTERVAL_MAX_LOOKBACK_DAYS = {"5m": 60, "15m": 60, "1h": 730, "1d": None}

_LOOKBACK_DAYS = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 182, "1y": 365}
# Padded past the raw calendar-day count above so a delta-cached daily
# pull has enough buffer for weekends/holidays to still return the full
# requested lookback in trading days.
_LOOKBACK_DAYS_BACK_PADDED = {"1d": 10, "5d": 15, "1mo": 45, "3mo": 110, "6mo": 200, "1y": 400}


def _resolve_lookback(interval, lookback):
    """Clamps `lookback` down to the largest LOOKBACK_OPTIONS value that
    still fits within `interval`'s real yfinance ceiling (see
    _INTERVAL_MAX_LOOKBACK_DAYS), returning (resolved_lookback, note).
    note is None when no clamping was needed."""
    max_days = _INTERVAL_MAX_LOOKBACK_DAYS.get(interval)
    if max_days is None or _LOOKBACK_DAYS[lookback] <= max_days:
        return lookback, None
    fitting = [lb for lb in LOOKBACK_OPTIONS if _LOOKBACK_DAYS[lb] <= max_days]
    resolved = fitting[-1] if fitting else "1d"
    return resolved, (f"Lookback reduced to {resolved} — {interval} data is only available "
                       f"for the trailing ~{max_days} days via yfinance.")


def fetch_price_history(ticker, interval=DEFAULT_INTERVAL, lookback=DEFAULT_LOOKBACK, conn=None):
    """Real price history for `ticker` at the given candle interval and
    visible-range lookback (independent axes -- see the module docstring
    above). Returns (DataFrame, cfg) where cfg carries the resolved
    sma_windows, horizon ('intraday' vs 'swing'), and -- when the
    requested lookback exceeded what yfinance supports at this interval
    (e.g. 1y at 5m) -- a `lookback_note` string for the UI to surface.
    Falls back to the (DEFAULT_INTERVAL, DEFAULT_LOOKBACK) combo if
    nothing comes back at all -- the "no data came back" safety net,
    distinct from lookback_note's "range was reduced" case.

    When `conn` is provided and interval is "1d" (the only interval that
    can share the delta-cached daily price_history table regardless of
    which lookback is chosen), price data is delta-loaded via
    fetch_price_history_delta() instead of an unconditional live pull.
    Every other interval (5m/15m/1h) always fetches live regardless of
    `conn` -- intraday bars can't be stored in that one-row-per-day
    table."""
    interval = interval if interval in _INTERVAL_CONFIG else DEFAULT_INTERVAL
    lookback = lookback if lookback in _LOOKBACK_DAYS else DEFAULT_LOOKBACK
    resolved_lookback, lookback_note = _resolve_lookback(interval, lookback)

    cfg = dict(_INTERVAL_CONFIG[interval])
    cfg["interval"] = interval
    cfg["lookback"] = resolved_lookback
    cfg["requested_lookback"] = lookback
    if lookback_note:
        cfg["lookback_note"] = lookback_note

    def _fetch_for(resolved_cfg):
        if conn is not None and resolved_cfg["interval"] == "1d":
            h, request_desc = fetch_price_history_delta(
                conn, ticker, days_back=_LOOKBACK_DAYS_BACK_PADDED[resolved_cfg["lookback"]]
            )
            resolved_cfg["price_request"] = request_desc
            return h
        try:
            return yf.Ticker(ticker).history(period=resolved_cfg["lookback"], interval=resolved_cfg["yf_interval"])
        except Exception:
            return pd.DataFrame()

    hist = _fetch_for(cfg)

    if hist.empty and (interval, resolved_lookback) != (DEFAULT_INTERVAL, DEFAULT_LOOKBACK):
        fallback = dict(_INTERVAL_CONFIG[DEFAULT_INTERVAL])
        fallback["interval"] = DEFAULT_INTERVAL
        fallback["lookback"] = DEFAULT_LOOKBACK
        fallback["fallback_from"] = f"{interval}/{lookback}"
        hist = _fetch_for(fallback)
        cfg = fallback

    if hist.empty:
        return hist, cfg

    short_w, long_w = cfg["sma_windows"]
    hist = hist.copy()
    hist["SMA_short"] = hist["Close"].rolling(short_w).mean()
    hist["SMA_long"] = hist["Close"].rolling(long_w).mean()
    hist["RSI14"] = _rsi(hist["Close"])
    macd_line, signal_line, macd_hist = _macd(hist["Close"])
    hist["MACD"] = macd_line
    hist["MACD_signal"] = signal_line
    hist["MACD_hist"] = macd_hist
    return hist, cfg


def fetch_session_price_summary(ticker):
    """Real per-session price stats for `ticker`, independent of whatever
    chart interval/lookback is currently selected: the most recent trade
    price (and its % change vs. the prior session's close), the current
    session's Open and running VWAP (cumulative price*volume / cumulative
    volume, not a simple average), plus the prior session's Open and
    Close. Always pulls
    its own short 5m/5d intraday window rather than reusing whatever
    fetch_price_history() returned for the chart -- if the chart is
    showing 1y of daily bars there's no intraday detail in that data to
    compute a same-day VWAP from, so this fetches independently.

    `is_today` is False when the most recent session in the data isn't
    today (market closed, weekend, or a stale/late feed) -- callers should
    label the row 'Most recent session' rather than 'Today' in that case.
    Fails soft -- returns None on any error or empty response, same
    fail-soft convention as every other real-data fetcher in this file."""
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="5m")
    except Exception:
        return None
    if hist is None or hist.empty:
        return None

    hist = hist.copy()
    # hist.index is tz-aware in the exchange's local timezone (e.g.
    # America/New_York for US tickers) -- comparing "today" in that same
    # timezone avoids an off-by-one-day mistake from comparing against a
    # naive UTC date near the exchange's midnight boundary.
    tz = hist.index.tz
    hist["session_date"] = hist.index.tz_convert(tz).date if tz is not None else hist.index.date
    session_dates = sorted(hist["session_date"].unique())
    if not session_dates:
        return None

    latest_date = session_dates[-1]
    latest = hist[hist["session_date"] == latest_date]
    prior_date = session_dates[-2] if len(session_dates) >= 2 else None
    prior = hist[hist["session_date"] == prior_date] if prior_date is not None else pd.DataFrame()

    session_open = float(latest["Open"].iloc[0]) if not latest.empty else None
    pv_cum = (latest["Close"] * latest["Volume"]).cumsum()
    vol_cum = latest["Volume"].cumsum()
    session_vwap = (
        float(pv_cum.iloc[-1] / vol_cum.iloc[-1])
        if not latest.empty and vol_cum.iloc[-1] > 0 else None
    )

    prior_open = float(prior["Open"].iloc[0]) if not prior.empty else None
    prior_close = float(prior["Close"].iloc[-1]) if not prior.empty else None

    # The most recent trade price -- the last bar's Close in the latest
    # session, live-updating during market hours (each new 5m bar shifts
    # this) and simply the last close once the market's shut for the day.
    current_price = float(latest["Close"].iloc[-1]) if not latest.empty else None
    current_price_change_pct = (
        (current_price - prior_close) / prior_close * 100
        if current_price is not None and prior_close else None
    )

    now_local = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.utcnow()
    is_today = latest_date == now_local.date()

    return {
        "ticker": ticker,
        "session_date": str(latest_date),
        "is_today": is_today,
        "current_price": current_price,
        "current_price_change_pct": current_price_change_pct,
        "session_open": session_open,
        "session_vwap": session_vwap,
        "prior_session_date": str(prior_date) if prior_date is not None else None,
        "prior_session_open": prior_open,
        "prior_session_close": prior_close,
        "source": "yfinance",
    }


def _swing_extrema_prices(hist, order):
    """Local swing highs/lows via scipy's argrelextrema (strict comparator,
    so flat plateaus don't produce duplicate-adjacent matches). `order` is
    the number of bars on each side that must be lower/higher -- scaled to
    the series length by the caller, never a fixed per-ticker threshold."""
    highs = hist["High"].to_numpy()
    lows = hist["Low"].to_numpy()
    high_idx = argrelextrema(highs, np.greater, order=order)[0]
    low_idx = argrelextrema(lows, np.less, order=order)[0]
    return highs[high_idx].tolist(), lows[low_idx].tolist(), high_idx, low_idx


def _cluster_levels(prices, tolerance_pct=1.5, min_touches=2):
    """Greedily groups swing prices into support/resistance clusters: a
    price joins the nearest open cluster if it's within tolerance_pct% of
    that cluster's running mean, else it starts a new cluster. Clusters
    with fewer than min_touches members aren't real support/resistance --
    a single unconfirmed swing point doesn't count as a 'level'. Returns
    levels sorted by touch count (strongest first)."""
    if not prices:
        return []
    clusters = []
    for p in sorted(prices):
        placed = False
        for c in clusters:
            center = c["sum"] / c["count"]
            if center > 0 and abs(p - center) / center * 100 <= tolerance_pct:
                c["sum"] += p
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"sum": p, "count": 1})
    levels = [
        {"level": round(c["sum"] / c["count"], 2), "touches": c["count"]}
        for c in clusters if c["count"] >= min_touches
    ]
    levels.sort(key=lambda x: x["touches"], reverse=True)
    return levels


def _volume_profile_nodes(hist, num_bins=24, top_n=3):
    """Buckets each bar's typical price ((H+L+C)/3) into `num_bins` equal
    price bands across the window's full range and sums that bar's volume
    into its band -- the standard volume-profile approximation. The
    highest-volume bands are the price levels the market has spent the
    most volume agreeing on, which tend to act as support/resistance."""
    if hist.empty:
        return []
    lo, hi = float(hist["Low"].min()), float(hist["High"].max())
    if not (hi > lo):
        return []
    typical = ((hist["High"] + hist["Low"] + hist["Close"]) / 3).to_numpy()
    bins = np.linspace(lo, hi, num_bins + 1)
    bin_idx = np.clip(np.digitize(typical, bins) - 1, 0, num_bins - 1)
    vol_by_bin = {}
    for idx, vol in zip(bin_idx, hist["Volume"].to_numpy()):
        if pd.notna(vol):
            vol_by_bin[int(idx)] = vol_by_bin.get(int(idx), 0.0) + float(vol)
    nodes = [
        {"level": round(float((bins[i] + bins[i + 1]) / 2), 2), "volume": v}
        for i, v in vol_by_bin.items()
    ]
    nodes.sort(key=lambda x: x["volume"], reverse=True)
    return nodes[:top_n]


def _trend_structure(hist, swing_high_prices, swing_low_prices, sma_short_w, sma_long_w,
                      key_level_below, key_level_above, break_window=3):
    """Mechanically labels trend structure -- never an LLM call, purely
    price/SMA/swing-sequence rules -- so the label and its one-line
    explanation are exactly reproducible from the same input data."""
    close = float(hist["Close"].iloc[-1])
    sma_short = hist["SMA_short"].iloc[-1] if "SMA_short" in hist.columns else None
    sma_long = hist["SMA_long"].iloc[-1] if "SMA_long" in hist.columns else None
    sma_short = float(sma_short) if pd.notna(sma_short) else None
    sma_long = float(sma_long) if pd.notna(sma_long) else None

    above_both = sma_short is not None and sma_long is not None and close > sma_short and close > sma_long
    below_both = sma_short is not None and sma_long is not None and close < sma_short and close < sma_long

    hh = len(swing_high_prices) >= 2 and swing_high_prices[-1] > swing_high_prices[-2]
    hl = len(swing_low_prices) >= 2 and swing_low_prices[-1] > swing_low_prices[-2]
    lh = len(swing_high_prices) >= 2 and swing_high_prices[-1] < swing_high_prices[-2]
    ll = len(swing_low_prices) >= 2 and swing_low_prices[-1] < swing_low_prices[-2]

    # A break in the last `break_window` bars beats any slower-forming
    # trend label -- it's the most decision-relevant thing that just
    # happened, per the reference "watch X, if it breaks..." framing.
    lookback = hist["Close"].iloc[-(break_window + 1):-1] if len(hist) > break_window else hist["Close"].iloc[:0]
    if key_level_below is not None and not lookback.empty and (lookback >= key_level_below).any() \
            and close < key_level_below:
        return ("breaking down",
                f"Breaking down — price closed below the {key_level_below:.2f} support level "
                f"within the last {break_window} bars.")
    if key_level_above is not None and not lookback.empty and (lookback <= key_level_above).any() \
            and close > key_level_above:
        return ("breaking out",
                f"Breaking out — price closed above the {key_level_above:.2f} resistance level "
                f"within the last {break_window} bars.")

    if above_both and (hh or hl) and not (lh and ll):
        structure = "higher highs and higher lows" if (hh and hl) else ("higher lows" if hl else "higher highs")
        return ("uptrend", f"Uptrend — price above both SMA{sma_short_w} and SMA{sma_long_w}, {structure} "
                            f"over the lookback window.")
    if below_both and (lh or ll) and not (hh and hl):
        structure = "lower highs and lower lows" if (lh and ll) else ("lower lows" if ll else "lower highs")
        return ("downtrend", f"Downtrend — price below both SMA{sma_short_w} and SMA{sma_long_w}, {structure} "
                              f"over the lookback window.")
    return ("consolidating",
            "Consolidating — price structure/SMA positioning show no clear directional bias "
            "over the lookback window.")


def _compute_technical_levels(hist, cfg, ticker):
    """Pure computation half of detect_technical_levels() -- takes an
    already-fetched (hist, cfg) pair (the same shape fetch_price_history()
    returns) instead of fetching again. Split out so callers that already
    have the history in hand (e.g. the chart-rendering section, which
    fetches it once for the candlestick itself) don't trigger a second
    fetch_price_history() call -- for intraday intervals that always hit
    yfinance live regardless of `conn`, a second call would double every
    live request for no reason."""
    result = {
        "ticker": ticker, "interval": cfg.get("interval"), "lookback": cfg.get("lookback"),
        "bars_analyzed": 0 if hist.empty else len(hist),
        "current_price": None, "support_levels": [], "resistance_levels": [],
        "current_zone": None, "trend_structure": None, "trend_explanation": None,
        "key_level_below": None, "key_level_above": None,
        "breakdown_target": None, "breakout_target": None,
        "volume_profile_nodes": [],
        "method": (
            f"scipy.signal.argrelextrema swing points, clustered within "
            f"{cfg.get('level_tolerance_pct', 1.5):.1f}% tolerance for this interval (>=2 touches), "
            f"volume-profile nodes over 24 equal price bins"
        ),
    }
    if hist.empty or len(hist) < 10:
        result["current_zone"] = "insufficient price history for this interval/lookback"
        result["trend_structure"] = "unknown"
        result["trend_explanation"] = "Not enough bars in this window to determine structure."
        return result

    # No upper clamp on `order`: it used to be capped at 10 bars, tuned
    # back when only daily-interval presets (~125-250 bars) existed. Now
    # that most presets use intraday intervals with 250-900+ bars (Part 2),
    # a fixed ceiling meant almost every timeframe hit the same cap and
    # `order` stopped being adaptive at all -- letting it scale linearly
    # with bar count keeps "how many bars must be lower/higher to count as
    # a swing" meaningful across both a 390-bar 1-day chart and an 875-bar
    # 6-month chart.
    order = max(2, len(hist) // 25)
    swing_high_prices, swing_low_prices, _, _ = _swing_extrema_prices(hist, order)

    tolerance_pct = cfg.get("level_tolerance_pct", 1.5)
    support_levels = _cluster_levels(swing_low_prices, tolerance_pct=tolerance_pct)
    resistance_levels = _cluster_levels(swing_high_prices, tolerance_pct=tolerance_pct)
    result["support_levels"] = support_levels
    result["resistance_levels"] = resistance_levels

    current_price = float(hist["Close"].iloc[-1])
    result["current_price"] = round(current_price, 2)

    supports_below = [lv["level"] for lv in support_levels if lv["level"] < current_price]
    resistances_above = [lv["level"] for lv in resistance_levels if lv["level"] > current_price]
    key_level_below = max(supports_below) if supports_below else None
    key_level_above = min(resistances_above) if resistances_above else None
    result["key_level_below"] = key_level_below
    result["key_level_above"] = key_level_above

    if key_level_below is not None and key_level_above is not None:
        range_width = key_level_above - key_level_below
        result["breakdown_target"] = round(key_level_below - range_width, 2)
        result["breakout_target"] = round(key_level_above + range_width, 2)
        near_support = (current_price - key_level_below) / key_level_below * 100 <= 1.5
        near_resistance = (key_level_above - current_price) / key_level_above * 100 <= 1.5
        if near_support:
            result["current_zone"] = f"At support (~{key_level_below:.2f})"
        elif near_resistance:
            result["current_zone"] = f"At resistance (~{key_level_above:.2f})"
        else:
            result["current_zone"] = f"Mid-range between {key_level_below:.2f} support and {key_level_above:.2f} resistance"
    elif key_level_below is not None:
        result["current_zone"] = f"Above all detected resistance, nearest support ~{key_level_below:.2f}"
    elif key_level_above is not None:
        result["current_zone"] = f"Below all detected support, nearest resistance ~{key_level_above:.2f}"
    else:
        result["current_zone"] = "No confirmed (>=2-touch) support/resistance levels detected in this window"

    short_w, long_w = cfg["sma_windows"]
    trend_structure, trend_explanation = _trend_structure(
        hist, swing_high_prices, swing_low_prices, short_w, long_w, key_level_below, key_level_above,
    )
    result["trend_structure"] = trend_structure
    result["trend_explanation"] = trend_explanation
    result["volume_profile_nodes"] = _volume_profile_nodes(hist)
    return result


def detect_technical_levels(ticker, interval=DEFAULT_INTERVAL, lookback=DEFAULT_LOOKBACK, conn=None):
    """Algorithmic (non-LLM) support/resistance + trend-structure read for
    `ticker` at the given candle interval and lookback range (independent
    axes -- see the fetch_price_history module docstring). Every number
    here is directly computed from real price/volume data -- nothing here
    is an LLM guess, which is what makes it safe to feed into the AI
    Briefing as ground truth (see generate_deep_analysis) rather than
    something the model could hallucinate a plausible-sounding but wrong
    number for.

    Swing highs/lows come from scipy.signal.argrelextrema with an `order`
    (bars-on-each-side) that scales with the series length, not a fixed
    per-ticker threshold, so the same logic applies whether the window is
    a 90-bar 3mo chart or a 260-bar 1y chart. Levels are clusters of >=2
    swing points within a tolerance band that widens on coarser intervals
    (see _INTERVAL_CONFIG's level_tolerance_pct) -- a single unconfirmed
    swing doesn't count as a real level.

    Fetches price history itself via fetch_price_history(). If you already
    have a (hist, cfg) pair in hand (e.g. rendering a chart that just
    fetched the same data), call _compute_technical_levels(hist, cfg,
    ticker) directly instead to avoid fetching twice."""
    hist, cfg = fetch_price_history(ticker, interval, lookback, conn=conn)
    return _compute_technical_levels(hist, cfg, ticker)


def _fetch_fundamentals_info_yfinance(ticker):
    tk = yf.Ticker(ticker)
    try:
        raw_info = tk.get_info()
    except Exception:
        raw_info = {}
    return {
        "shortName": raw_info.get("shortName") or ticker,
        "sector": raw_info.get("sector"),
        "industry": raw_info.get("industry"),
        "marketCap": raw_info.get("marketCap"),
        "trailingPE": raw_info.get("trailingPE"),
        "forwardPE": raw_info.get("forwardPE"),
        "pegRatio": raw_info.get("trailingPegRatio") or raw_info.get("pegRatio"),
        "priceToBook": raw_info.get("priceToBook"),
        "profitMargins": raw_info.get("profitMargins"),
        "revenueGrowth": raw_info.get("revenueGrowth"),
        "debtToEquity": raw_info.get("debtToEquity"),
        "returnOnEquity": raw_info.get("returnOnEquity"),
        "beta": raw_info.get("beta"),
        "fiftyTwoWeekHigh": raw_info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": raw_info.get("fiftyTwoWeekLow"),
        "shortPercentOfFloat": raw_info.get("shortPercentOfFloat"),
        "targetMeanPrice": raw_info.get("targetMeanPrice"),
        "recommendationKey": raw_info.get("recommendationKey"),
        "source": "yfinance",
    }


def fetch_fundamentals(ticker):
    """Pure, uncached fetch: price_df (OHLCV + SMA50/200 + RSI14 + MACD) +
    info + snapshot. Prefer cached_fundamentals(conn, ticker) from the
    dashboard so the expensive .get_info() lookup is DB-cached."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y", interval="1d")
    if hist.empty:
        return {"ticker": ticker, "price_df": pd.DataFrame(), "info": {}, "snapshot": {}}
    hist = _add_technicals(hist)
    info = _fetch_fundamentals_info_yfinance(ticker)
    snapshot = _build_snapshot(hist)
    return {"ticker": ticker, "price_df": hist, "info": info, "snapshot": snapshot}


def _persist_fundamentals_info(conn, ticker, info):
    conn.execute(
        "INSERT INTO fundamentals_info (ticker, payload_json, source, fetched_at) VALUES (?,?,?,?)",
        (ticker, json.dumps(info, default=str), info.get("source"), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_fundamentals_info_cache(conn, ticker):
    row = conn.execute(
        "SELECT payload_json FROM fundamentals_info WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    return json.loads(row[0]) if row else None


# --------------------------------------------------------------------------
# Delta-loading daily price history -- backs both cached_fundamentals'
# price_df and fetch_price_history()'s daily-interval timeframes (5d/1mo/
# 3mo/6mo). Replaces an unconditional tk.history(period="1y", interval="1d")
# on every call (the exact call that hit the yfinance rate limit) with:
# full pull only the first time a ticker is seen, delta-only (start=) after
# that.
# --------------------------------------------------------------------------

def _persist_price_history(conn, ticker, hist):
    """Upserts daily OHLCV rows. ON CONFLICT UPDATE (not INSERT OR IGNORE)
    because yfinance can revise the most recent day's bar intraday (e.g. a
    force_refresh same-day re-pull), so a re-fetched row for a date we
    already have should overwrite, not be silently dropped."""
    if hist is None or hist.empty:
        return
    cur = conn.cursor()
    fetched_at = datetime.utcnow().isoformat()
    for idx, row in hist.iterrows():
        date_str = pd.Timestamp(idx).date().isoformat()
        cur.execute(
            """INSERT INTO price_history (ticker, date, open, high, low, close, volume, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker, date) DO UPDATE SET
                   open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                   volume=excluded.volume, fetched_at=excluded.fetched_at""",
            (ticker, date_str, _safe_num(row.get("Open"), None), _safe_num(row.get("High"), None),
             _safe_num(row.get("Low"), None), _safe_num(row.get("Close"), None),
             int(row["Volume"]) if pd.notna(row.get("Volume")) else None, fetched_at),
        )
    conn.commit()


def _read_price_history(conn, ticker, days_back=400):
    """Reads price_history back into the same shape yfinance's own
    .history() returns (DatetimeIndex named 'Date', Open/High/Low/Close/
    Volume columns) so _add_technicals/_rsi/_macd/_build_snapshot all work
    unchanged regardless of whether the DataFrame came from a live call or
    storage."""
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM price_history
           WHERE ticker=? AND date >= date('now', ?) ORDER BY date ASC""",
        conn, params=(ticker, f"-{int(days_back)} day"),
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })
    df.index.name = "Date"
    return df


def fetch_price_history_delta(conn, ticker, full_period="1y", days_back=400, max_age_hours=4):
    """Delta-aware daily price history: a full yf.Ticker(ticker).history(
    period=full_period, interval="1d") only the first time this ticker has
    no stored price_history rows; after that, only `start=<day after our
    last stored date>`.

    Two independent things gate a network call, and either one alone can
    make this return purely from storage:
    1. should_refetch() TTL gate (max_age_hours, default 4h) -- don't
       re-check for a new trading day more than a few times a day. This is
       what makes back-to-back calls in the same session genuinely free
       (0 network calls), not just "asks and gets told nothing's new" --
       confirmed live: on a non-trading day (weekend/holiday) the delta
       window math alone would still issue a `start=` request every call
       since "the day after our last stored date" is always in the past
       relative to a calendar day with no new bar, and get a real but
       empty response back. The TTL gate is what actually stops that.
    2. The delta-window check itself -- if "the day after our last stored
       date" is in the future relative to today's calendar date (i.e. we
       already have a bar for today), skip regardless of the TTL.

    Every check -- including the two skip paths -- logs to fetch_log via
    _log_fetch so should_refetch()'s clock actually advances; only the
    logging itself never happens if the TTL gate below made this whole
    function return before any real work.

    Returns (hist, request_desc) -- hist is the FULL merged window read
    back from storage (indicators need the whole trailing window, not just
    the delta slice), request_desc is a short human-readable string
    describing exactly what request (if any) was made, for verification/
    logging."""
    table = "price_history"
    last_date = get_last_timestamp(conn, table, ticker, "date")

    if last_date is not None and not should_refetch(conn, table, ticker, max_age_hours):
        merged = _read_price_history(conn, ticker, days_back=days_back)
        return merged, (
            f"SKIPPED (checked within the last {max_age_hours}h; last stored date {last_date}) "
            f"-- 0 network calls"
        )

    if last_date is None:
        try:
            hist = yf.Ticker(ticker).history(period=full_period, interval="1d")
        except Exception:
            hist = pd.DataFrame()
        request_desc = f'FULL PULL: history(period="{full_period}", interval="1d")'
    else:
        start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
        today = pd.Timestamp.now().normalize()
        if start > today:
            hist = pd.DataFrame()
            request_desc = f"SKIPPED (already up to date as of {last_date}) -- 0 network calls"
        else:
            start_str = start.strftime("%Y-%m-%d")
            try:
                hist = yf.Ticker(ticker).history(start=start_str, interval="1d")
            except Exception:
                hist = pd.DataFrame()
            request_desc = f'DELTA PULL: history(start="{start_str}", interval="1d")'

    if not hist.empty:
        _persist_price_history(conn, ticker, hist)
    _log_fetch(conn, table, ticker, True, len(hist))

    merged = _read_price_history(conn, ticker, days_back=days_back)
    return (merged if not merged.empty else hist), request_desc


def cached_fundamentals(conn, ticker, max_age_hours=12, force_refresh=False):
    """price_df is delta-loaded via fetch_price_history_delta() -- a full
    tk.history(period="1y") only the first time this ticker is seen, just
    the new days after that (see that function's docstring). Only the
    expensive .get_info()-derived info dict is separately DB-cached on its
    own max_age_hours/should_refetch gate. Returns an envelope:
    {"data": {"ticker","price_df","info","snapshot"}, "source",
    "cache_hit", "fetched_at"}."""
    table = "fundamentals_info"
    hist, _price_request_desc = fetch_price_history_delta(conn, ticker)

    if hist.empty:
        return {"data": {"ticker": ticker, "price_df": pd.DataFrame(), "info": {}, "snapshot": {}},
                "source": None, "cache_hit": False, "fetched_at": None}

    hist = _add_technicals(hist)
    snapshot = _build_snapshot(hist)

    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        info = _read_fundamentals_info_cache(conn, ticker)
        if info:
            return {"data": {"ticker": ticker, "price_df": hist, "info": info, "snapshot": snapshot},
                    "source": info.get("source"), "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}

    try:
        info = _fetch_fundamentals_info_yfinance(ticker)
        _persist_fundamentals_info(conn, ticker, info)
        _log_fetch(conn, table, ticker, True, 1)
        return {"data": {"ticker": ticker, "price_df": hist, "info": info, "snapshot": snapshot},
                "source": info.get("source"), "cache_hit": False, "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        info = _read_fundamentals_info_cache(conn, ticker) or {}
        return {"data": {"ticker": ticker, "price_df": hist, "info": info, "snapshot": snapshot},
                "source": info.get("source"), "cache_hit": bool(info),
                "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_options_flow(ticker) -> DataFrame[ticker, expiry, strike, type,
# volume, open_interest, vol_oi_ratio, implied_volatility, last_price,
# unusual, underlying_price, source]. Source-agnostic: dispatches on
# DATA_SOURCE_CONFIG["options_flow"].
# --------------------------------------------------------------------------

_OPTIONS_FLOW_COLUMNS = [
    "ticker", "expiry", "strike", "type", "volume", "open_interest", "vol_oi_ratio",
    "implied_volatility", "last_price", "unusual", "underlying_price", "source",
    "delta", "gamma", "theta", "vega", "rho", "bid", "ask",
]

RISK_FREE_RATE_DEFAULT = 0.04  # matches fetch_leaps_candidates' own default -- one assumed rate app-wide


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Black-Scholes theoretical option price (per share) -- the pricing
    counterpart to bs_greeks below, added for the EARNINGS SIMULATOR's
    P/L scenario simulation (Part 6), which needs a real post-scenario
    option VALUE (spot moved, IV crushed, some time decayed), not just
    the Greeks at today's price/IV. Same degenerate-input guard as
    bs_greeks (returns NaN for T<=0/sigma<=0/S<=0/K<=0) -- for an
    expired/zero-IV contract, use intrinsic value directly instead."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "put":
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)
    return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Full Black-Scholes Greeks for a single contract -- delta, gamma,
    theta (per day, per share), vega (per 1-point/1% IV move), and rho
    (per 1% rate move). Extends leap_picker.py's call-only
    bs_call_delta_theta (ported as _bs_call_delta_theta in the LEAPS
    section below, now a thin wrapper over this) to both option types and
    the full Greek set, reusing the same _norm_cdf/_norm_pdf helpers so
    the call-side math is byte-for-byte consistent with the LEAPS scanner.

    Gamma and vega are identical for calls and puts (put-call parity);
    delta and theta differ by option_type, using the standard put
    formulas (put delta = call delta - 1, dividend-adjusted; put theta
    has its risk-free/dividend terms sign-flipped vs. the call).

    Returns an all-NaN dict for degenerate inputs (T<=0, sigma<=0, S<=0,
    K<=0 -- e.g. an expired or zero-IV contract) rather than raising, same
    guard as the existing call-only version."""
    nan = float("nan")
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": nan, "gamma": nan, "theta": nan, "vega": nan, "rho": nan}

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    gamma = math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * sqrtT)
    vega = S * math.exp(-q * T) * _norm_pdf(d1) * sqrtT / 100.0

    if option_type == "put":
        delta = math.exp(-q * T) * (_norm_cdf(d1) - 1.0)
        theta_per_year = (
            -(S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2.0 * sqrtT)
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
            - q * S * math.exp(-q * T) * _norm_cdf(-d1)
        )
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100.0
    else:
        delta = math.exp(-q * T) * _norm_cdf(d1)
        theta_per_year = (
            -(S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2.0 * sqrtT)
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
            + q * S * math.exp(-q * T) * _norm_cdf(d1)
        )
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100.0

    return {"delta": delta, "gamma": gamma, "theta": theta_per_year / 365.0, "vega": vega, "rho": rho}


# Was 4 -- confirmed too thin for names with dense near-term expiries
# (PYPL's first 4 only reached 26 days out of 14 total, silently dropping
# everything from ~1mo to its 2028 LEAPS dates). 8 is still a deliberate
# cap, not "all" -- covers roughly 2-3 months out for most chains without
# tripling every routine full_refresh's yfinance call count per ticker.
# The OPTIONS FLOW tab's "Load ALL expirations" button (dashboard.py)
# passes an explicit high max_expirations for genuine full-chain coverage
# on demand, rather than making every background refresh pay that cost.
DEFAULT_OPTIONS_MAX_EXPIRATIONS = 8


def fetch_options_flow(ticker, max_expirations=DEFAULT_OPTIONS_MAX_EXPIRATIONS):
    source = DATA_SOURCE_CONFIG.get("options_flow", "yfinance")
    if source == "yfinance":
        return _fetch_options_flow_yfinance(ticker, max_expirations=max_expirations)
    raise NotImplementedError(
        f"options_flow source '{source}' is not implemented. Intended: e.g. "
        "Polygon.io GET /v3/snapshot/options/{ticker} or Tradier GET /v1/markets/options/chains"
    )


def _fetch_options_flow_yfinance(ticker, max_expirations=DEFAULT_OPTIONS_MAX_EXPIRATIONS):
    tk = yf.Ticker(ticker)

    try:
        underlying_price = float(tk.fast_info["lastPrice"])
    except Exception:
        h = tk.history(period="1d")
        underlying_price = float(h["Close"].iloc[-1]) if not h.empty else None

    try:
        expirations = list(tk.options)[:max_expirations]
    except Exception:
        expirations = []

    rows = []
    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            continue
        # Time-to-expiry is per-expiration, not per-contract -- computed
        # once here and reused for every strike/type in this chain.
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            T = max((exp_dt - datetime.now(timezone.utc)).total_seconds(), 0.0) / (365.0 * 24 * 3600)
        except ValueError:
            T = None
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            for _, row in df.iterrows():
                volume = int(_safe_num(row.get("volume")))
                oi = int(_safe_num(row.get("openInterest")))
                ratio = (volume / oi) if oi else None
                strike = float(row["strike"])
                # That SPECIFIC contract's own impliedVolatility, never a
                # chain-wide average -- IV varies by strike/expiry (skew/
                # smile), so a per-contract Greek needs a per-contract IV.
                iv = _safe_num(row.get("impliedVolatility"))
                greeks = (
                    bs_greeks(underlying_price, strike, T, RISK_FREE_RATE_DEFAULT, iv, option_type=option_type)
                    if T is not None and underlying_price is not None
                    else {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
                )
                rows.append({
                    "ticker": ticker, "expiry": exp, "strike": strike,
                    "type": option_type, "volume": volume, "open_interest": oi,
                    "vol_oi_ratio": ratio, "implied_volatility": iv,
                    "last_price": _safe_num(row.get("lastPrice")), "unusual": False,
                    "underlying_price": underlying_price, "source": "yfinance",
                    "delta": _none_if_nan(greeks["delta"]), "gamma": _none_if_nan(greeks["gamma"]),
                    "theta": _none_if_nan(greeks["theta"]), "vega": _none_if_nan(greeks["vega"]),
                    "rho": _none_if_nan(greeks["rho"]),
                    "bid": _none_if_nan(_safe_num(row.get("bid"), default=float("nan"))),
                    "ask": _none_if_nan(_safe_num(row.get("ask"), default=float("nan"))),
                })

    result = pd.DataFrame(rows, columns=_OPTIONS_FLOW_COLUMNS)
    if result.empty:
        return result

    # "Unusual" flags the top decile of vol/OI ratio *within this ticker's own
    # chain* rather than a fixed cutoff. A fixed >=1.5 ratio + >=500 volume
    # floor was structurally too strict for lower-liquidity names (STX/WDC/LYFT
    # were seeing 1-3 flagged contracts out of 700-1000, vs. 20-65 for
    # higher-volume names like AAPL/AMD/TSLA) -- starving the smart-money
    # signal of any options-based data for exactly the tickers where a big
    # relative bet is most notable. A volume floor is kept (lowered to 100)
    # purely to filter out near-zero-liquidity noise, not to gate "unusual".
    valid_ratios = result["vol_oi_ratio"].dropna()
    ratio_threshold = valid_ratios.quantile(0.90) if len(valid_ratios) >= 10 else 1.5
    min_volume_floor = 100
    result["unusual"] = (
        (result["volume"] >= min_volume_floor)
        & ((result["open_interest"] == 0) | (result["vol_oi_ratio"] >= ratio_threshold))
    )
    return result


_OPTIONS_FLOW_DB_TO_SPEC = {"expiration": "expiry", "option_type": "type", "volume_oi_ratio": "vol_oi_ratio"}


def _persist_options_flow(conn, ticker, df):
    if df.empty:
        return
    fetch_date = date.today().isoformat()
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO options_flow
                (ticker, fetch_date, expiration, strike, option_type, volume, open_interest,
                 volume_oi_ratio, implied_volatility, last_price, underlying_price, unusual, source,
                 delta, gamma, theta, vega, rho, bid, ask, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker, fetch_date, expiration, strike, option_type) DO UPDATE SET
                volume=excluded.volume, open_interest=excluded.open_interest,
                volume_oi_ratio=excluded.volume_oi_ratio, implied_volatility=excluded.implied_volatility,
                last_price=excluded.last_price, underlying_price=excluded.underlying_price,
                unusual=excluded.unusual, source=excluded.source,
                delta=excluded.delta, gamma=excluded.gamma, theta=excluded.theta,
                vega=excluded.vega, rho=excluded.rho, bid=excluded.bid, ask=excluded.ask
            """,
            (ticker, fetch_date, row["expiry"], float(row["strike"]), row["type"], int(row["volume"]),
             int(row["open_interest"]), row["vol_oi_ratio"], row["implied_volatility"], row["last_price"],
             row.get("underlying_price"), int(bool(row["unusual"])), row.get("source"),
             _none_if_nan(row.get("delta")), _none_if_nan(row.get("gamma")), _none_if_nan(row.get("theta")),
             _none_if_nan(row.get("vega")), _none_if_nan(row.get("rho")),
             _none_if_nan(row.get("bid")), _none_if_nan(row.get("ask")),
             datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_options_flow_cache(conn, ticker):
    fetch_date = date.today().isoformat()
    df = pd.read_sql_query(
        """SELECT ticker, expiration, strike, option_type, volume, open_interest, volume_oi_ratio,
                  implied_volatility, last_price, underlying_price, unusual, source,
                  delta, gamma, theta, vega, rho, bid, ask
           FROM options_flow WHERE ticker=? AND fetch_date=?""",
        conn, params=(ticker, fetch_date),
    )
    return df.rename(columns=_OPTIONS_FLOW_DB_TO_SPEC)


def cached_options_flow(conn, ticker, max_age_hours=0.25, force_refresh=False,
                         max_expirations=DEFAULT_OPTIONS_MAX_EXPIRATIONS):
    """Options chains are a full-state snapshot -- yfinance has no delta/
    incremental fetch for a chain, you get the whole current chain or
    nothing -- so the meaningful cache boundary is "once per trading day,"
    not a fine-grained TTL. If we already have today's fetch_date stored
    for this ticker, skip the network call entirely regardless of how many
    hours old that snapshot is. `max_age_hours` is accepted for signature
    compatibility with every other cached_* wrapper but no longer gates
    this one -- the day-based check below does.

    `max_expirations` (default DEFAULT_OPTIONS_MAX_EXPIRATIONS, matching fetch_options_flow's own
    default) only affects a real network fetch -- it can't widen an
    already-cached same-day snapshot fetched with a smaller value. The
    Option Picker (find_option_candidates) needs expirations reaching
    further out than the default 4 nearest ones cover for its longer
    horizons (see TIME_HORIZON_MAX_EXPIRATIONS), so its own call site
    always passes force_refresh=True alongside a wider max_expirations
    rather than relying on whatever happened to be cached from an
    unrelated tab's narrower fetch."""
    table = "options_flow"
    last_fetch_date = get_last_timestamp(conn, table, ticker, "fetch_date")
    have_today = last_fetch_date == date.today().isoformat()

    if not force_refresh and have_today:
        cached_df = _read_options_flow_cache(conn, ticker)
        if not cached_df.empty:
            return {"data": cached_df, "source": cached_df["source"].iloc[0], "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        df = fetch_options_flow(ticker, max_expirations=max_expirations)
        _persist_options_flow(conn, ticker, df)
        _log_fetch(conn, table, ticker, True, len(df))
        source = df["source"].iloc[0] if not df.empty else DATA_SOURCE_CONFIG.get("options_flow")
        return {"data": df, "source": source, "cache_hit": False, "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached_df = _read_options_flow_cache(conn, ticker)
        return {"data": cached_df,
                "source": cached_df["source"].iloc[0] if not cached_df.empty else None,
                "cache_hit": not cached_df.empty, "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_dark_pool(ticker) -> dict[ticker, dark_pool_score, volume_z_score,
# signal, source, total_volume]. Source-agnostic: dispatches on
# DATA_SOURCE_CONFIG["dark_pool"]. The "finra_proxy" branch tries FINRA's
# public ATS weekly-summary endpoint first (source="finra_ats" on success)
# and falls back to a volume z-score heuristic (source="finra_proxy") when
# that endpoint's availability/shape can't be relied on.
# --------------------------------------------------------------------------

def fetch_dark_pool(ticker):
    source = DATA_SOURCE_CONFIG.get("dark_pool", "finra_proxy")
    if source == "finra_proxy":
        return _fetch_dark_pool_finra_proxy(ticker)
    raise NotImplementedError(
        f"dark_pool source '{source}' is not implemented. Intended: e.g. a paid FINRA "
        "OTC Transparency subscription or a consolidated-tape off-exchange volume feed."
    )


def _fetch_dark_pool_finra_proxy(ticker):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3mo")
    if hist.empty or len(hist) < 5:
        return {"ticker": ticker, "dark_pool_score": None, "volume_z_score": None,
                "signal": None, "source": None, "total_volume": None}

    volumes = hist["Volume"]
    today_volume = float(volumes.iloc[-1])
    baseline = volumes.iloc[:-1]
    mean_vol = float(baseline.mean())
    std_vol = float(baseline.std()) or 1.0
    volume_z_score = (today_volume - mean_vol) / std_vol

    dark_pool_score = None
    source = "finra_proxy"

    try:
        resp = requests.post(
            FINRA_ATS_URL,
            json={
                "compareFilters": [
                    {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": ticker}
                ],
                "sortFields": ["-weekStartDate"],
                "limit": 1,
            },
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.ok:
            data = resp.json()
            if data:
                rec = data[0]
                ats_shares = next(
                    (v for k, v in rec.items() if "ats" in k.lower() and "share" in k.lower()), None
                )
                total_shares = next(
                    (v for k, v in rec.items() if "total" in k.lower() and "share" in k.lower()), None
                )
                if ats_shares and total_shares:
                    dark_pool_score = (float(ats_shares) / float(total_shares)) * 100
                    source = "finra_ats"
    except Exception:
        pass

    if dark_pool_score is None:
        # Proxy: FINRA reports off-exchange (ATS + non-ATS OTC) volume runs
        # roughly 35-45% of consolidated tape volume market-wide. Nudge that
        # baseline with the volume z-score as a rough proxy for elevated
        # off-exchange interest -- this is a heuristic, not a measured figure.
        dark_pool_score = min(65.0, max(20.0, 38.0 + volume_z_score * 4))
        source = "finra_proxy"

    signal = "ELEVATED" if (dark_pool_score > 45 or volume_z_score > 1.5) else "NORMAL"

    return {"ticker": ticker, "dark_pool_score": round(dark_pool_score, 2),
            "volume_z_score": round(volume_z_score, 3), "signal": signal, "source": source,
            "total_volume": today_volume}


def _persist_dark_pool(conn, ticker, result):
    today = date.today().isoformat()
    total_volume = result.get("total_volume")
    score = result.get("dark_pool_score")
    dark_pool_volume = (total_volume * score / 100) if (total_volume is not None and score is not None) else None
    is_proxy = 0 if result.get("source") == "finra_ats" else 1
    conn.execute(
        """
        INSERT INTO dark_pool_signals
            (ticker, date, dark_pool_volume, total_volume, dark_pool_pct, volume_zscore,
             signal, is_proxy, source, computed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            dark_pool_volume=excluded.dark_pool_volume, total_volume=excluded.total_volume,
            dark_pool_pct=excluded.dark_pool_pct, volume_zscore=excluded.volume_zscore,
            signal=excluded.signal, is_proxy=excluded.is_proxy, source=excluded.source,
            computed_at=excluded.computed_at
        """,
        (ticker, today, dark_pool_volume, total_volume, score, result.get("volume_z_score"),
         result.get("signal"), is_proxy, result.get("source"), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_dark_pool_cache(conn, ticker):
    row = conn.execute(
        """SELECT dark_pool_pct, volume_zscore, signal, source, total_volume
           FROM dark_pool_signals WHERE ticker=? ORDER BY date DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    pct, zscore, signal, source, total_volume = row
    return {"ticker": ticker, "dark_pool_score": pct, "volume_z_score": zscore, "signal": signal,
            "source": source, "total_volume": total_volume}


def cached_dark_pool(conn, ticker, max_age_hours=1, force_refresh=False):
    table = "dark_pool_signals"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_dark_pool_cache(conn, ticker)
        if cached is not None:
            return {"data": cached, "source": cached.get("source"), "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_dark_pool(ticker)
        _persist_dark_pool(conn, ticker, result)
        _log_fetch(conn, table, ticker, True, 1)
        return {"data": result, "source": result.get("source"), "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_dark_pool_cache(conn, ticker)
        return {"data": cached, "source": cached.get("source") if cached else None,
                "cache_hit": cached is not None, "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_congressional_trades() -> list[dict(senator, ticker, trade_type,
# amount, trade_date, disclosure_date, source)]. Market-wide, not per-ticker.
#
# Checks QUIVER_API_KEY first (paid, source="quiverquant"); falls back to a
# best-effort free Senate EFD scrape (source="senate_efd_free") if the key
# is absent or the Quiver call fails. Never raises -- always returns a list,
# empty if both paths fail, with the reason logged.
# --------------------------------------------------------------------------

def fetch_congressional_trades():
    api_key = os.environ.get("QUIVER_API_KEY")
    if api_key:
        try:
            return _fetch_congressional_quiverquant(api_key)
        except Exception as e:
            print(f"[congressional] quiverquant failed ({e}); falling back to free source")
    else:
        print("[congressional] QUIVER_API_KEY not set; using free-tier fallback (senate_efd) "
              "-- may be empty or unreliable. Add QUIVER_API_KEY for live data.")

    try:
        return _fetch_congressional_senate_efd_free()
    except Exception as e:
        print(f"[congressional] senate_efd fallback failed too: {e}")
        return []


def _quiver_row_get(row, *names, default=None):
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return default


def _normalize_quiver_congress_df(df, source="quiverquant"):
    """Shared by the global-feed and per-ticker fetchers: normalizes a
    quiverquant congress_trading() DataFrame (or an equivalent raw-REST
    JSON response) into our standard schema: senator, ticker, trade_type,
    amount, trade_date, disclosure_date, source."""
    if df is None or df.empty:
        return []
    if "detail" in df.columns and len(df.columns) <= 2:
        # quiverquant's congress_trading() never calls raise_for_status(), so an
        # auth/rate-limit error response (e.g. {"detail": "Invalid token."}) comes
        # back as a malformed 1-column DataFrame instead of raising -- without this
        # check it would silently persist a single garbage all-None record.
        raise RuntimeError(f"QuiverQuant API error: {df['detail'].iloc[0]}")

    records = []
    for _, row in df.iterrows():
        raw_type = str(_quiver_row_get(row, "Transaction", "Type", default="")).upper()
        trade_type = (
            "BUY" if "PURCHASE" in raw_type or "BUY" in raw_type else
            "SELL" if "SALE" in raw_type or "SELL" in raw_type else
            (raw_type or None)
        )
        trade_date = _quiver_row_get(row, "TransactionDate", "Traded")
        disclosure_date = _quiver_row_get(row, "ReportDate", "Filed", "last_modified")

        records.append({
            "senator": _quiver_row_get(row, "Representative", "Senator", "Name"),
            "ticker": _quiver_row_get(row, "Ticker"),
            "trade_type": trade_type,
            "amount": _quiver_row_get(row, "Range", "Amount"),
            "trade_date": str(trade_date) if trade_date is not None else None,
            "disclosure_date": str(disclosure_date) if disclosure_date is not None else None,
            "source": source,
        })
    return records


def _fetch_congressional_quiverquant(api_key):
    import quiverquant  # optional dependency; only required when QUIVER_API_KEY is set

    quiver = quiverquant.quiver(api_key)
    df = quiver.congress_trading()
    return _normalize_quiver_congress_df(df)


def fetch_congressional_trades_by_ticker(ticker):
    """Per-ticker historical congressional activity for the TICKER
    DEEP-DIVE tab -- richer than filtering the market-wide recent-feed
    table (which only has whatever happened to be in the last "recent"
    pull), since quiver.congress_trading(ticker) returns that ticker's
    full trade history. Falls back to a raw REST call if the installed
    quiverquant package's per-ticker path fails for any reason.

    NOTE on auth header: quiverquant's own client sends
    `Authorization: Token <key>`, not `Bearer <key>` -- confirmed against
    the live API (a `Bearer` header 401s). The REST fallback below uses
    the verified-correct `Token` scheme so it actually works.
    """
    api_key = os.environ.get("QUIVER_API_KEY")
    if not api_key:
        print(f"[congressional] QUIVER_API_KEY not set; no per-ticker data for {ticker}.")
        return []

    try:
        import quiverquant
        quiver = quiverquant.quiver(api_key)
        df = quiver.congress_trading(ticker)
        return _normalize_quiver_congress_df(df)
    except Exception as e:
        print(f"[congressional] quiverquant per-ticker lookup failed for {ticker} ({e}); "
              f"falling back to raw REST call")

    try:
        resp = requests.get(
            f"https://api.quiverquant.com/beta/historical/congresstrading/{ticker}",
            headers={"accept": "application/json", "Authorization": f"Token {api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        return _normalize_quiver_congress_df(pd.DataFrame(resp.json()))
    except Exception as e:
        print(f"[congressional] REST fallback also failed for {ticker}: {e}")
        return []


def _fetch_congressional_senate_efd_free(max_reports=40):
    """Best-effort scrape of the Senate's official electronic financial
    disclosure search (efdsearch.senate.gov) -- the free, no-key alternative
    to QuiverQuant. NOTE: this endpoint sits behind Akamai bot protection
    that returns a blanket 403 to non-browser traffic from many server/
    datacenter IPs (confirmed during development -- `requests` gets 403 even
    on the plain search page with a browser User-Agent). It may simply
    return [] in most hosted environments. Kept as a genuine best-effort
    path since a residential/browser-like connection can sometimes get
    through, and because there is no other free, no-key source for Senate
    stock trades.
    """
    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)

    home = session.get(f"{SENATE_EFD_BASE}/search/", timeout=15)
    home.raise_for_status()
    csrf_match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', home.text)
    if not csrf_match:
        raise RuntimeError("could not find CSRF token on efdsearch.senate.gov (likely blocked)")
    csrf_token = csrf_match.group(1)

    session.post(
        f"{SENATE_EFD_BASE}/search/home/",
        data={"csrfmiddlewaretoken": csrf_token, "prohibition_agreement": "1"},
        headers={"Referer": f"{SENATE_EFD_BASE}/search/"},
        timeout=15,
    )

    resp = session.post(
        f"{SENATE_EFD_BASE}/search/report/data/",
        data={
            "start": "0", "length": str(max_reports), "report_types": "[11]",  # 11 = Periodic Transaction Report
            "filer_types": "[]", "submitted_start_date": "", "submitted_end_date": "",
            "candidate_state": "", "senator_state": "", "office_id": "", "first_name": "",
            "last_name": "", "csrfmiddlewaretoken": csrf_token,
        },
        headers={"Referer": f"{SENATE_EFD_BASE}/search/", "X-Requested-With": "XMLHttpRequest"},
        timeout=15,
    )
    resp.raise_for_status()
    report_rows = (resp.json() or {}).get("data", [])

    records = []
    for row in report_rows[:max_reports]:
        # Each row is typically [first_name, last_name, office, report_type_html, date_filed]
        # where report_type_html contains an <a href="/search/view/ptr/<id>/"> link.
        try:
            first_name, last_name = row[0], row[1]
            link_match = re.search(r'href="([^"]+)"', row[3])
            if not link_match:
                continue
            ptr_url = SENATE_EFD_BASE + link_match.group(1)
            disclosure_date = row[4] if len(row) > 4 else None
            senator = re.sub("<[^>]+>", "", f"{first_name} {last_name}").strip()
        except (IndexError, TypeError):
            continue

        try:
            detail = session.get(ptr_url, timeout=15)
            detail.raise_for_status()
        except Exception:
            continue

        # Transaction table rows: ticker, asset name, transaction type, date, amount range.
        for txn_match in re.finditer(r"<tr[^>]*>.*?</tr>", detail.text, re.DOTALL):
            txn_html = txn_match.group(0)
            ticker_m = re.search(r">([A-Z]{1,5})<", txn_html)
            type_m = re.search(r"\b(Purchase|Sale|Exchange)\b", txn_html, re.IGNORECASE)
            amount_m = re.search(r"\$[\d,]+\s*-\s*\$[\d,]+", txn_html)
            date_m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", txn_html)
            if not (ticker_m and type_m):
                continue
            raw_type = type_m.group(1).upper()
            trade_type = "BUY" if raw_type == "PURCHASE" else "SELL" if raw_type == "SALE" else raw_type
            records.append({
                "senator": senator, "ticker": ticker_m.group(1), "trade_type": trade_type,
                "amount": amount_m.group(0) if amount_m else None,
                "trade_date": date_m.group(1) if date_m else None,
                "disclosure_date": disclosure_date, "source": "senate_efd_free",
            })

    return records


def _persist_congressional_trades(conn, records):
    """No natural unique URL for quiver/senate_efd rows, so dedupe on the
    natural key (who/what/when/how-much) via an existence check rather than
    a DB-level UNIQUE constraint."""
    cur = conn.cursor()
    inserted = 0
    for r in records:
        exists = cur.execute(
            """SELECT 1 FROM congressional_trades WHERE politician IS ? AND ticker IS ? AND
               transaction_type IS ? AND transaction_date IS ? AND amount_range IS ? LIMIT 1""",
            (r.get("senator"), r.get("ticker"), r.get("trade_type"), r.get("trade_date"), r.get("amount")),
        ).fetchone()
        if exists:
            continue
        cur.execute(
            """INSERT INTO congressional_trades
                (politician, ticker, transaction_type, amount_range, transaction_date, disclosure_date,
                 chamber, party, source_url, source, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (r.get("senator"), r.get("ticker"), r.get("trade_type"), r.get("amount"), r.get("trade_date"),
             r.get("disclosure_date"), None, None, None, r.get("source"), datetime.utcnow().isoformat()),
        )
        inserted += 1
    conn.commit()
    return inserted


def _read_congressional_cache(conn, limit=200):
    rows = conn.execute(
        """SELECT politician, ticker, transaction_type, amount_range, transaction_date, disclosure_date, source
           FROM congressional_trades ORDER BY transaction_date DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [
        {"senator": r[0], "ticker": r[1], "trade_type": r[2], "amount": r[3],
         "trade_date": r[4], "disclosure_date": r[5], "source": r[6]}
        for r in rows
    ]


def cached_congressional_trades(conn, max_age_hours=1, force_refresh=False):
    table = "congressional_trades"
    if not force_refresh and not should_refetch(conn, table, None, max_age_hours):
        cached = _read_congressional_cache(conn)
        return {"data": cached, "source": (cached[0]["source"] if cached else None), "cache_hit": True,
                "fetched_at": _last_fetch_info(conn, table, None)}

    try:
        records = fetch_congressional_trades()
        # Neither QuiverQuant's congress_trading() nor the Senate EFD scrape
        # support a since= filter -- both always return their current
        # "recent" window, so the request can't be narrowed. The delta
        # discipline is at the persist layer: _persist_congressional_trades
        # already skips records it has seen before (existence check on the
        # natural key) and reports how many were genuinely new; log that
        # instead of the full recent-window length so fetch_log reflects
        # real deltas, not "how big is the feed's window."
        new_count = _persist_congressional_trades(conn, records)
        _log_fetch(conn, table, None, True, new_count)
        if records:
            return {"data": records, "source": records[0]["source"], "cache_hit": False,
                    "fetched_at": datetime.utcnow().isoformat()}
        # Both providers came back empty this round (e.g. no key + Senate EFD
        # blocked) -- surface any previously-cached rows instead of claiming
        # a source that didn't actually deliver anything this time.
        cached = _read_congressional_cache(conn)
        return {"data": cached, "source": (cached[0]["source"] if cached else None),
                "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, None)}
    except Exception as e:
        _log_fetch(conn, table, None, False, 0, str(e))
        cached = _read_congressional_cache(conn)
        return {"data": cached, "source": (cached[0]["source"] if cached else None),
                "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, None)}


def _read_congressional_cache_by_ticker(conn, ticker, limit=50):
    rows = conn.execute(
        """SELECT politician, ticker, transaction_type, amount_range, transaction_date, disclosure_date, source
           FROM congressional_trades WHERE ticker=? ORDER BY transaction_date DESC LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    return [
        {"senator": r[0], "ticker": r[1], "trade_type": r[2], "amount": r[3],
         "trade_date": r[4], "disclosure_date": r[5], "source": r[6]}
        for r in rows
    ]


def cached_congressional_trades_by_ticker(conn, ticker, max_age_hours=24, force_refresh=False):
    """Per-ticker variant of cached_congressional_trades, backed by
    fetch_congressional_trades_by_ticker -- a longer cache window (24h vs.
    1h for the global feed) since it's a paid-API, per-ticker historical
    lookup rather than a cheap "what's new today" sweep. Persists into the
    same congressional_trades table so both the global feed and per-ticker
    lookups accumulate in one place."""
    table = "congressional_trades_by_ticker"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_congressional_cache_by_ticker(conn, ticker)
        if cached:
            return {"data": cached, "source": cached[0]["source"], "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}

    try:
        records = fetch_congressional_trades_by_ticker(ticker)
        # Same delta discipline as cached_congressional_trades: the API call
        # itself can't be narrowed (quiver.congress_trading(ticker) returns
        # this ticker's full history every time, no since= param), so log
        # the persist layer's genuinely-new count, not the full response size.
        new_count = _persist_congressional_trades(conn, records)
        _log_fetch(conn, table, ticker, True, new_count)
        if records:
            return {"data": records, "source": records[0]["source"], "cache_hit": False,
                    "fetched_at": datetime.utcnow().isoformat()}
        cached = _read_congressional_cache_by_ticker(conn, ticker)
        return {"data": cached, "source": (cached[0]["source"] if cached else None),
                "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, ticker)}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_congressional_cache_by_ticker(conn, ticker)
        return {"data": cached, "source": (cached[0]["source"] if cached else None),
                "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_insider_trades -- SEC EDGAR Form 4 Atom feed (market-wide "current").
# Not part of DATA_SOURCE_CONFIG / not cached per-ticker: it's a single
# market-wide feed pull, already deduped via its own UNIQUE constraint, and
# out of scope for the Part 3 caching list.
# --------------------------------------------------------------------------

def _xml_text(elem, path):
    found = elem.find(path)
    return found.text.strip() if found is not None and found.text else None


def _xml_float(elem, path):
    txt = _xml_text(elem, path)
    try:
        return float(txt) if txt is not None else None
    except ValueError:
        return None


def fetch_insider_trades(conn, max_filings=25):
    """The SEC Form 4 feed is a fixed-size "most recent N filings" window
    with no since= parameter, so the request itself can't be narrowed --
    but each filing costs two network round-trips (its index page, then
    the ownership XML doc), and most of them are ones we've already
    parsed on the last call. Delta discipline here means skipping that
    parse work for filings at/before our stored high-water mark instead
    of paying for it again on every call."""
    feed = feedparser.parse(SEC_FORM4_FEED, request_headers=SEC_HEADERS)
    if not feed.entries:
        print(f"[insider_trades] no entries (bozo={feed.get('bozo')}); feed may be unavailable")
        return 0

    last_seen = conn.execute("SELECT MAX(filing_date) FROM insider_trades").fetchone()[0]

    cur = conn.cursor()
    inserted = 0
    skipped_already_seen = 0

    for entry in feed.entries[:max_filings]:
        entry_updated = entry.get("updated")
        if last_seen and entry_updated and entry_updated <= last_seen:
            skipped_already_seen += 1
            continue
        index_url = entry.get("link")
        if not index_url:
            continue
        try:
            resp = requests.get(index_url, headers=SEC_HEADERS, timeout=10)
            resp.raise_for_status()
            xml_links = re.findall(r'href="([^"]+\.xml)"', resp.text)
            # The /xslF345X06/*.xml link is the browser-rendered HTML view;
            # the raw ownership XML lives at the top-level path alongside it.
            raw_candidates = [l for l in xml_links if "/xsl" not in l.lower()]
            doc_url = None
            for l in raw_candidates:
                if "primary_doc" in l.lower():
                    doc_url = l
                    break
            if not doc_url and raw_candidates:
                doc_url = raw_candidates[0]
            if not doc_url:
                continue
            if not doc_url.startswith("http"):
                doc_url = f"https://www.sec.gov{doc_url}"

            xml_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=10)
            xml_resp.raise_for_status()
            root = ET.fromstring(xml_resp.content)

            ticker = _xml_text(root, ".//issuer/issuerTradingSymbol")
            insider_name = _xml_text(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
            title = (
                _xml_text(root, ".//reportingOwner/reportingOwnerRelationship/officerTitle")
                or ("Director" if _xml_text(root, ".//reportingOwner/reportingOwnerRelationship/isDirector") == "1"
                    else "Insider")
            )

            for txn in root.findall(".//nonDerivativeTransaction"):
                shares = _xml_float(txn, ".//transactionAmounts/transactionShares/value")
                price = _xml_float(txn, ".//transactionAmounts/transactionPricePerShare/value")
                code = _xml_text(txn, ".//transactionCoding/transactionCode")
                txn_date = _xml_text(txn, ".//transactionDate/value")
                transaction_type = {"P": "BUY", "S": "SELL"}.get(code, code)
                value = (shares or 0) * (price or 0)

                cur.execute(
                    """
                    INSERT OR IGNORE INTO insider_trades
                        (ticker, insider_name, title, transaction_type, shares, price, value,
                         transaction_date, filing_date, source_url, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (ticker, insider_name, title, transaction_type, shares, price, value,
                     txn_date, entry.get("updated"), doc_url, datetime.utcnow().isoformat()),
                )
                if cur.rowcount:
                    inserted += 1

            time.sleep(0.2)
        except Exception:
            continue

    conn.commit()
    print(f"[insider_trades] {inserted} new transactions inserted, "
          f"{skipped_already_seen} filings already seen (skipped re-parse)")
    return inserted


# --------------------------------------------------------------------------
# fetch_polymarket -- Polymarket Gamma API (no auth). Not part of
# DATA_SOURCE_CONFIG: single free source, market-wide, out of scope for the
# Part 3 caching list.
# --------------------------------------------------------------------------

def _fetch_polymarket_pages(max_pages=6, page_size=100):
    """The Gamma API caps each response at ~100 rows regardless of the
    requested `limit`, so a keyword search over a meaningful candidate pool
    needs pagination via `offset` (confirmed working against the live API)."""
    all_markets = []
    for page in range(max_pages):
        try:
            resp = requests.get(
                POLYMARKET_URL,
                params={"closed": "false", "limit": page_size, "offset": page * page_size,
                        "order": "volumeNum", "ascending": "false"},
                timeout=15,
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception:
            break
        if not batch:
            break
        all_markets.extend(batch)
        if len(batch) < page_size:
            break
    return all_markets


def fetch_polymarket(conn, limit=75, keywords=None):
    """keywords: optional list of terms to filter market questions by
    (case-insensitive substring match). When given, pages through a larger
    candidate pool first (see _fetch_polymarket_pages) since relevant but
    lower-volume markets -- e.g. a niche chip-export-controls market --
    often aren't in the raw top-`limit`-by-volume slice at all."""
    try:
        if keywords:
            markets = _fetch_polymarket_pages()
            kw_lower = [k.lower() for k in keywords]
            markets = [m for m in markets if any(kw in (m.get("question") or "").lower() for kw in kw_lower)]
            markets.sort(key=lambda m: _safe_num(m.get("volumeNum") or m.get("volume")), reverse=True)
            markets = markets[:limit]
        else:
            resp = requests.get(
                POLYMARKET_URL,
                params={"closed": "false", "limit": limit, "order": "volumeNum", "ascending": "false"},
                timeout=15,
            )
            resp.raise_for_status()
            markets = resp.json()
    except Exception as e:
        print(f"[polymarket] fetch failed: {e}")
        return 0

    cur = conn.cursor()
    inserted = 0

    for m in markets:
        try:
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
        except (json.JSONDecodeError, TypeError):
            outcomes, prices = [], []
        price_map = dict(zip(outcomes, prices))

        def _to_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        yes_price = _to_float(price_map.get("Yes"))
        no_price = _to_float(price_map.get("No"))

        cur.execute(
            """
            INSERT INTO polymarket_events
                (market_id, question, category, yes_price, no_price, volume, liquidity,
                 end_date, active, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(market_id) DO UPDATE SET
                question=excluded.question, category=excluded.category,
                yes_price=excluded.yes_price, no_price=excluded.no_price,
                volume=excluded.volume, liquidity=excluded.liquidity,
                end_date=excluded.end_date, active=excluded.active,
                last_updated=excluded.last_updated
            """,
            (
                str(m.get("id")), m.get("question"), m.get("category") or m.get("groupItemTitle"),
                yes_price, no_price, _safe_num(m.get("volumeNum") or m.get("volume")),
                _safe_num(m.get("liquidityNum") or m.get("liquidity")),
                m.get("endDate"), 1 if m.get("active") else 0, datetime.utcnow().isoformat(),
            ),
        )
        inserted += 1

    conn.commit()
    return inserted


# --------------------------------------------------------------------------
# compute_divergence — smart money vs retail scoring. Unchanged from prior
# version: reads directly from options_flow / dark_pool_signals /
# insider_trades / congressional_trades, all of which are still populated
# with the same column names by the new cached_* wrappers above.
# --------------------------------------------------------------------------

_AMOUNT_RANGE_RE = re.compile(r"\$?([\d,]+)\s*-\s*\$?([\d,]+)")


def _parse_amount_range(amount_range):
    if not amount_range:
        return 1.0
    match = _AMOUNT_RANGE_RE.search(amount_range)
    if not match:
        return 1.0
    low = float(match.group(1).replace(",", ""))
    high = float(match.group(2).replace(",", ""))
    return (low + high) / 2


def _net_buy_score(pairs):
    """pairs: iterable of (transaction_type, weight). Returns value-weighted
    net-buy ratio clipped to [-1, 1], 0 if no signal."""
    total_weight = 0.0
    signed_weight = 0.0
    for transaction_type, weight in pairs:
        weight = abs(weight or 0)
        if weight == 0:
            continue
        if transaction_type == "BUY":
            signed_weight += weight
        elif transaction_type == "SELL":
            signed_weight -= weight
        else:
            continue
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return max(-1.0, min(1.0, signed_weight / total_weight))


def compute_divergence(ticker, conn):
    """Scoring formula (weights sum to 1.0 in each blend, documented here so
    the math stays legible as it evolves):

      smart_signal = 0.30*insider_net + 0.15*congress_net + 0.55*options_smart_net

        insider_net/congress_net come from Form4/congressional trade data,
        which is high-conviction when present but sparse to the point of
        being ~always empty for any *specific* watchlist ticker -- the SEC
        Form4 feed samples only ~25 market-wide filings per refresh, and
        congressional data needs a paid QuiverQuant key. options_smart_net
        (the call/put skew of top-decile-vol/OI "unusual" contracts) is
        populated for essentially every ticker every refresh, so it carries
        the majority weight -- this is the fix for smart_signal computing
        to exactly 0.0 for every ticker, every time, which is what a 50/50
        insider/congress-only blend does whenever neither feed happens to
        cover that ticker (the common case).

      score = 100 * (0.30*|smart_signal| + 0.25*|retail_signal|
                      + 0.20*institutional_magnitude + 0.25*earnings_proximity_magnitude)

        earnings_proximity_magnitude is new: unusual options activity in the
        days just before an earnings date is a classic smart-money-
        positioning-ahead-of-catalyst pattern, so it pushes conviction
        (magnitude, not direction) higher independent of the other three
        components. Reweighted down from the prior 0.40/0.35/0.25 split to
        make room for it without diluting the others below relevance.
    """
    cur = conn.cursor()
    today = date.today().isoformat()

    # --- insider component ---
    insider_rows = cur.execute(
        """SELECT transaction_type, value FROM insider_trades
           WHERE ticker=? AND transaction_date >= date('now','-90 day')""",
        (ticker,),
    ).fetchall()
    insider_net = _net_buy_score(insider_rows)

    # --- congressional component ---
    congress_rows = cur.execute(
        """SELECT transaction_type, amount_range FROM congressional_trades
           WHERE ticker=? AND transaction_date >= date('now','-90 day')""",
        (ticker,),
    ).fetchall()
    congress_pairs = [(t, _parse_amount_range(a)) for t, a in congress_rows]
    congress_net = _net_buy_score(congress_pairs)

    # --- options flow: retail-heat AND an options-based smart-money proxy ---
    total_contracts = (
        cur.execute("SELECT COUNT(*) FROM options_flow WHERE ticker=? AND fetch_date=?", (ticker, today))
        .fetchone()[0]
    )
    opt_rows = cur.execute(
        """SELECT option_type, SUM(volume), SUM(CASE WHEN unusual=1 THEN volume ELSE 0 END),
                  SUM(CASE WHEN unusual=1 THEN 1 ELSE 0 END)
           FROM options_flow WHERE ticker=? AND fetch_date=? GROUP BY option_type""",
        (ticker, today),
    ).fetchall()
    call_vol = put_vol = unusual_call_vol = unusual_put_vol = 0
    smart_call_signals = smart_put_signals = 0
    for option_type, vol, unusual_vol, unusual_count in opt_rows:
        if option_type == "call":
            call_vol, unusual_call_vol, smart_call_signals = vol or 0, unusual_vol or 0, unusual_count or 0
        else:
            put_vol, unusual_put_vol, smart_put_signals = vol or 0, unusual_vol or 0, unusual_count or 0

    total_opt_vol = call_vol + put_vol
    total_unusual_contracts = smart_call_signals + smart_put_signals
    call_put_skew = ((call_vol - put_vol) / total_opt_vol) if total_opt_vol else 0.0
    unusual_ratio = ((unusual_call_vol + unusual_put_vol) / total_opt_vol) if total_opt_vol else 0.0
    retail_signal = max(-1.0, min(1.0, call_put_skew * (0.5 + 0.5 * unusual_ratio)))

    options_smart_net = (
        (smart_call_signals - smart_put_signals) / total_unusual_contracts if total_unusual_contracts else 0.0
    )

    print(
        f"[compute_divergence] {ticker}: {total_contracts} option contracts "
        f"(call_vol={call_vol} put_vol={put_vol}), unusual flagged: "
        f"{smart_call_signals} calls / {smart_put_signals} puts, "
        f"insider_rows={len(insider_rows)} congress_rows={len(congress_rows)}"
    )

    has_smart_data = bool(insider_rows) or bool(congress_rows) or total_unusual_contracts > 0

    W_INSIDER, W_CONGRESS, W_OPTIONS_SMART = 0.30, 0.15, 0.55
    smart_signal = W_INSIDER * insider_net + W_CONGRESS * congress_net + W_OPTIONS_SMART * options_smart_net

    # --- dark pool / institutional component ---
    dp_row = cur.execute(
        "SELECT dark_pool_pct, volume_zscore FROM dark_pool_signals WHERE ticker=? ORDER BY date DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    dark_pool_pct, volume_zscore = dp_row if dp_row else (0.0, 0.0)
    pct_component = min(1.0, max(0.0, (dark_pool_pct - 35) / 30))
    zscore_component = min(1.0, max(0.0, volume_zscore / 3))
    institutional_magnitude = max(pct_component, zscore_component) if dp_row else 0.0

    # --- earnings-proximity component (reads the cached earnings_signal
    # table -- compute_divergence never fetches live itself, same as every
    # other component here) ---
    earn_row = cur.execute(
        """SELECT days_to_earnings, atm_avg_iv_pct FROM earnings_signal
           WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    days_to_earnings, atm_avg_iv_pct = earn_row if earn_row else (None, None)
    earnings_proximity_magnitude = 0.0
    if days_to_earnings is not None and 0 <= days_to_earnings <= 5:
        proximity_component = 1.0 - (days_to_earnings / 5.0)   # 1.0 at day-of, 0.0 five days out
        activity_component = min(1.0, unusual_ratio / 0.15)     # saturates at 15% unusual-by-volume
        earnings_proximity_magnitude = round(proximity_component * activity_component, 3)

    score = round(
        100 * (
            0.30 * abs(smart_signal) + 0.25 * abs(retail_signal)
            + 0.20 * institutional_magnitude + 0.25 * earnings_proximity_magnitude
        ), 1,
    )

    if institutional_magnitude > 0.6 and abs(smart_signal) < 0.25 and abs(retail_signal) < 0.25:
        label = "INSTITUTIONAL_ACTIVE"
    elif abs(retail_signal) > 0.5 and abs(smart_signal) < 0.2:
        label = "RETAIL_FRENZY"
    elif has_smart_data and smart_signal < -0.15 and retail_signal > 0.25:
        label = "SMART_BEARISH_RETAIL_LONG"
    elif has_smart_data and smart_signal > 0.2 and retail_signal <= 0.15:
        label = "SMART_BULLISH"
    else:
        label = "NEUTRAL"

    components = {
        "insider_net": round(insider_net, 3),
        "congress_net": round(congress_net, 3),
        "options_smart_net": round(options_smart_net, 3),
        "smart_call_signals": smart_call_signals,
        "smart_put_signals": smart_put_signals,
        "smart_signal": round(smart_signal, 3),
        "call_vol": call_vol,
        "put_vol": put_vol,
        "call_put_skew": round(call_put_skew, 3),
        "unusual_ratio": round(unusual_ratio, 3),
        "retail_signal": round(retail_signal, 3),
        "dark_pool_pct": round(dark_pool_pct, 2) if dp_row else None,
        "volume_zscore": round(volume_zscore, 2) if dp_row else None,
        "institutional_magnitude": round(institutional_magnitude, 3),
        "days_to_earnings": days_to_earnings,
        "atm_avg_iv_pct": atm_avg_iv_pct,
        "earnings_proximity_magnitude": earnings_proximity_magnitude,
        "weights": {
            "smart": 0.30, "retail": 0.25, "institutional": 0.20, "earnings_proximity": 0.25,
            "smart_insider": W_INSIDER, "smart_congress": W_CONGRESS, "smart_options": W_OPTIONS_SMART,
        },
    }

    cur.execute(
        """
        INSERT INTO divergence_scores
            (ticker, computed_date, score, label, smart_signal, retail_signal,
             institutional_magnitude, components_json, computed_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(ticker, computed_date) DO UPDATE SET
            score=excluded.score, label=excluded.label, smart_signal=excluded.smart_signal,
            retail_signal=excluded.retail_signal, institutional_magnitude=excluded.institutional_magnitude,
            components_json=excluded.components_json, computed_at=excluded.computed_at
        """,
        (ticker, today, score, label, smart_signal, retail_signal, institutional_magnitude,
         json.dumps(components), datetime.utcnow().isoformat()),
    )
    conn.commit()

    price_row = cur.execute(
        "SELECT underlying_price FROM options_flow WHERE ticker=? AND fetch_date=? LIMIT 1", (ticker, today)
    ).fetchone()
    _persist_ticker_snapshot(
        conn, ticker, score=score, label=label,
        smart_call_signals=smart_call_signals, smart_put_signals=smart_put_signals,
        retail_heat=retail_signal, iv_snapshot=atm_avg_iv_pct,
        price_snapshot=price_row[0] if price_row else None,
    )

    return {"ticker": ticker, "score": score, "label": label, "components": components}


# --------------------------------------------------------------------------
# ticker_snapshots -- append-only historical record, written every time
# compute_divergence() runs (and therefore every full_refresh()), so
# repeated queries over time return different, comparable, time-stamped
# answers instead of always reading "now".
# --------------------------------------------------------------------------

def _persist_ticker_snapshot(conn, ticker, score, label, smart_call_signals, smart_put_signals,
                              retail_heat, iv_snapshot, price_snapshot):
    earn_row = conn.execute(
        "SELECT next_earnings_date FROM earnings_signal WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    earnings_date = earn_row[0] if earn_row else None

    hours_to_earnings = None
    if earnings_date:
        try:
            ed = pd.Timestamp(earnings_date)
            if ed.tzinfo is None:
                ed = ed.tz_localize("UTC")
            # Signed so negative = before, positive = after, per the schema's
            # own convention: (now - earnings_date) is negative while the
            # earnings date is still in the future.
            hours_to_earnings = (pd.Timestamp.now(tz="UTC") - ed).total_seconds() / 3600.0
        except (TypeError, ValueError):
            hours_to_earnings = None

    if hours_to_earnings is None:
        snapshot_type = "routine"
    elif -48 <= hours_to_earnings < 0:
        snapshot_type = "pre_earnings"
    elif 0 <= hours_to_earnings <= 48:
        snapshot_type = "post_earnings"
    else:
        snapshot_type = "routine"

    conn.execute(
        """INSERT INTO ticker_snapshots
            (ticker, snapshot_type, hours_to_earnings, conviction_score, divergence_label,
             smart_call_signals, smart_put_signals, retail_heat, iv_snapshot, price_snapshot,
             earnings_date, fetched_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ticker, snapshot_type, hours_to_earnings, score, label, smart_call_signals, smart_put_signals,
         retail_heat, iv_snapshot, price_snapshot, earnings_date, datetime.utcnow().isoformat()),
    )
    conn.commit()


def get_earnings_snapshot_history(conn, ticker, earnings_date=None):
    """All snapshots tied to a specific earnings event (or the most recent
    one on file for this ticker, if not given), sorted by fetched_at --
    the read path for a future prediction model, or a "how did conviction
    move in the 72h before earnings" chart."""
    if earnings_date is None:
        row = conn.execute(
            """SELECT earnings_date FROM ticker_snapshots
               WHERE ticker=? AND earnings_date IS NOT NULL
               ORDER BY fetched_at DESC LIMIT 1""",
            (ticker,),
        ).fetchone()
        earnings_date = row[0] if row else None
    if earnings_date is None:
        return pd.DataFrame()
    return pd.read_sql_query(
        "SELECT * FROM ticker_snapshots WHERE ticker=? AND earnings_date=? ORDER BY fetched_at ASC",
        conn, params=(ticker, earnings_date),
    )


# --------------------------------------------------------------------------
# fetch_earnings_calendar — next earnings date + recent EPS estimate vs
# actual, via yfinance. Single source (not in DATA_SOURCE_CONFIG).
# --------------------------------------------------------------------------

def fetch_earnings_calendar(ticker, limit=4):
    t = yf.Ticker(ticker)
    next_earnings_date = None

    try:
        cal = t.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, (list, tuple)) and ed:
                next_earnings_date = ed[0]
            elif ed is not None:
                next_earnings_date = ed
        elif cal is not None and hasattr(cal, "empty") and not cal.empty and "Earnings Date" in cal.index:
            val = cal.loc["Earnings Date"]
            next_earnings_date = val.iloc[0] if hasattr(val, "iloc") else val
    except Exception:
        pass

    eps_history = pd.DataFrame()
    try:
        dates_df = t.get_earnings_dates(limit=max(limit, 4) + 4)
    except Exception:
        dates_df = None

    if dates_df is not None and not dates_df.empty:
        eps_history = dates_df.reset_index().rename(columns={"index": "Earnings Date"})

        if next_earnings_date is None:
            ed_col = pd.to_datetime(eps_history["Earnings Date"], utc=True, errors="coerce")
            now = pd.Timestamp.now(tz="UTC")
            future = eps_history[ed_col >= now]
            if not future.empty:
                next_earnings_date = future.iloc[0]["Earnings Date"]

        eps_history = eps_history.head(limit)

    return {"ticker": ticker, "next_earnings_date": next_earnings_date, "eps_history": eps_history,
            "source": "yfinance"}


def _persist_earnings_calendar(conn, ticker, result):
    eps_history = result.get("eps_history")
    if eps_history is None or eps_history.empty:
        return
    cur = conn.cursor()
    for _, row in eps_history.iterrows():
        ed = row.get("Earnings Date")
        if pd.isna(ed):
            continue
        ed_str = pd.Timestamp(ed).isoformat()
        cur.execute(
            """INSERT INTO earnings_calendar
                (ticker, earnings_date, eps_estimate, reported_eps, surprise_pct, source, fetched_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                   eps_estimate=excluded.eps_estimate, reported_eps=excluded.reported_eps,
                   surprise_pct=excluded.surprise_pct, source=excluded.source, fetched_at=excluded.fetched_at""",
            (ticker, ed_str, _none_if_nan(row.get("EPS Estimate")), _none_if_nan(row.get("Reported EPS")),
             _none_if_nan(row.get("Surprise(%)")), result.get("source"), datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_earnings_calendar_cache(conn, ticker, limit=8):
    df = pd.read_sql_query(
        """SELECT earnings_date AS "Earnings Date", eps_estimate AS "EPS Estimate",
                  reported_eps AS "Reported EPS", surprise_pct AS "Surprise(%)"
           FROM earnings_calendar WHERE ticker=? ORDER BY earnings_date DESC LIMIT ?""",
        conn, params=(ticker, limit),
    )
    if not df.empty:
        df["Earnings Date"] = pd.to_datetime(df["Earnings Date"], utc=True, errors="coerce")
    return df


def _next_future_date(df, col):
    now = pd.Timestamp.now(tz="UTC")
    dates = pd.to_datetime(df[col], utc=True, errors="coerce")
    mask = dates >= now
    if not mask.any():
        return None
    return df.loc[dates[mask].idxmin(), col]


def cached_earnings_calendar(conn, ticker, max_age_hours=12, force_refresh=False):
    table = "earnings_calendar"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        eps_hist = _read_earnings_calendar_cache(conn, ticker)
        if not eps_hist.empty:
            return {"data": {"ticker": ticker, "next_earnings_date": _next_future_date(eps_hist, "Earnings Date"),
                              "eps_history": eps_hist},
                    "source": "yfinance", "cache_hit": True, "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_earnings_calendar(ticker)
        _persist_earnings_calendar(conn, ticker, result)
        _log_fetch(conn, table, ticker, True, len(result.get("eps_history", [])))
        return {"data": result, "source": result.get("source"), "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        eps_hist = _read_earnings_calendar_cache(conn, ticker)
        return {"data": {"ticker": ticker,
                          "next_earnings_date": _next_future_date(eps_hist, "Earnings Date") if not eps_hist.empty else None,
                          "eps_history": eps_hist},
                "source": "yfinance" if not eps_hist.empty else None, "cache_hit": not eps_hist.empty,
                "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_analyst_targets — yfinance info: price targets & recommendation.
# Single source (not in DATA_SOURCE_CONFIG).
# --------------------------------------------------------------------------

def _fetch_recommendation_breakdown(t):
    """Actual Strong Buy/Buy/Hold/Sell/Strong Sell analyst counts for the
    most recent period, from ticker.recommendations -- info.recommendationKey
    alone only gives a single overall label, not the underlying distribution."""
    try:
        rec = t.recommendations
    except Exception:
        rec = None
    if rec is None or rec.empty:
        return {"rec_period": None, "strongBuy": None, "buy": None, "hold": None,
                "sell": None, "strongSell": None}

    row = rec[rec["period"] == "0m"]
    row = row.iloc[0] if not row.empty else rec.iloc[0]

    def _int_or_none(v):
        # sqlite3 silently stores numpy int64 as a raw BLOB instead of an
        # INTEGER (it has no adapter for it) -- cast to plain Python int.
        v = _none_if_nan(v)
        return int(v) if v is not None else None

    return {
        "rec_period": str(row.get("period")),
        "strongBuy": _int_or_none(row.get("strongBuy")),
        "buy": _int_or_none(row.get("buy")),
        "hold": _int_or_none(row.get("hold")),
        "sell": _int_or_none(row.get("sell")),
        "strongSell": _int_or_none(row.get("strongSell")),
    }


def fetch_analyst_targets(ticker):
    t = yf.Ticker(ticker)
    try:
        info = t.get_info()
    except Exception:
        info = {}

    breakdown = _fetch_recommendation_breakdown(t)

    return {
        "ticker": ticker,
        "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
        "targetMeanPrice": info.get("targetMeanPrice"),
        "targetHighPrice": info.get("targetHighPrice"),
        "targetLowPrice": info.get("targetLowPrice"),
        "targetMedianPrice": info.get("targetMedianPrice"),
        "recommendationKey": info.get("recommendationKey"),
        "recommendationMean": info.get("recommendationMean"),
        "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
        "source": "yfinance",
        **breakdown,
    }


def fetch_analyst_price_target_breakdown(ticker, limit=15):
    """Real per-firm analyst price-target actions from yfinance's
    upgrades_downgrades feed (firm name, action, current price target,
    date) -- genuinely available on the free tier via that endpoint,
    unlike targetMeanPrice/etc. which are aggregate-only. Not every row
    carries a price target (many are pure rating changes with no target
    attached), so this filters to rows that do. Not DB-cached, same
    "cheap, always fetch fresh" treatment as price history elsewhere in
    this file. Fails soft -- empty DataFrame on any error."""
    try:
        ud = yf.Ticker(ticker).upgrades_downgrades
    except Exception:
        return pd.DataFrame()
    if ud is None or ud.empty or "currentPriceTarget" not in ud.columns:
        return pd.DataFrame()
    df = ud.reset_index()
    df = df[df["currentPriceTarget"].fillna(0) > 0]
    if df.empty:
        return df
    df = df.sort_values("GradeDate", ascending=False).head(limit)
    return df[["GradeDate", "Firm", "Action", "ToGrade", "currentPriceTarget"]].reset_index(drop=True)


def _persist_analyst_targets(conn, ticker, result):
    conn.execute(
        """INSERT INTO analyst_targets
            (ticker, current_price, target_mean, target_high, target_low, target_median,
             recommendation_key, recommendation_mean, num_analysts, rec_period,
             strong_buy, buy, hold, sell, strong_sell, source, fetched_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ticker, result.get("currentPrice"), result.get("targetMeanPrice"), result.get("targetHighPrice"),
         result.get("targetLowPrice"), result.get("targetMedianPrice"), result.get("recommendationKey"),
         result.get("recommendationMean"), result.get("numberOfAnalystOpinions"), result.get("rec_period"),
         result.get("strongBuy"), result.get("buy"), result.get("hold"), result.get("sell"),
         result.get("strongSell"), result.get("source"), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_analyst_targets_cache(conn, ticker):
    row = conn.execute(
        """SELECT current_price, target_mean, target_high, target_low, target_median,
                  recommendation_key, recommendation_mean, num_analysts, rec_period,
                  strong_buy, buy, hold, sell, strong_sell, source
           FROM analyst_targets WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    keys = ["currentPrice", "targetMeanPrice", "targetHighPrice", "targetLowPrice", "targetMedianPrice",
            "recommendationKey", "recommendationMean", "numberOfAnalystOpinions", "rec_period",
            "strongBuy", "buy", "hold", "sell", "strongSell", "source"]
    result = dict(zip(keys, row))
    result["ticker"] = ticker
    return result


def cached_analyst_targets(conn, ticker, max_age_hours=12, force_refresh=False):
    table = "analyst_targets"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_analyst_targets_cache(conn, ticker)
        if cached:
            return {"data": cached, "source": cached.get("source"), "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_analyst_targets(ticker)
        _persist_analyst_targets(conn, ticker, result)
        _log_fetch(conn, table, ticker, True, 1)
        return {"data": result, "source": result.get("source"), "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_analyst_targets_cache(conn, ticker)
        return {"data": cached or {}, "source": (cached or {}).get("source"), "cache_hit": bool(cached),
                "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_buybacks — quarterly "Repurchase Of Capital Stock" trend via
# yfinance ticker.cashflow. Single source (not in DATA_SOURCE_CONFIG).
# --------------------------------------------------------------------------

_BUYBACK_ROW_CANDIDATES = [
    "Repurchase Of Capital Stock",
    "Repurchase Of CapitalStock",
    "Common Stock Repurchased",
    "Repurchase Of Stock",
]


_CAPEX_ROW_CANDIDATES = ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"]
_OCF_ROW_CANDIDATES = [
    "Operating Cash Flow", "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
]


def _find_cashflow_row(cf, candidates, must_contain):
    row_name = next((c for c in candidates if c in cf.index), None)
    if row_name is not None:
        return row_name
    contains_all = [w.lower() for w in must_contain]
    return next(
        (idx for idx in cf.index if all(w in str(idx).lower() for w in contains_all)),
        None,
    )


def fetch_buybacks(ticker):
    """Buyback spend, CapEx, and Operating Cash Flow together (all from the
    same yfinance ticker.cashflow statement) so Free Cash Flow (OCF - CapEx)
    can be computed and its trend connected to CapEx explicitly in the AI
    Briefing (Part 2), rather than the buyback figure standing alone."""
    t = yf.Ticker(ticker)
    try:
        cf = t.cashflow
    except Exception:
        cf = None

    empty = {
        "ticker": ticker, "history": pd.DataFrame(), "trend": "N/A", "fcf_trend": "N/A", "source": "yfinance",
    }
    if cf is None or cf.empty:
        return empty

    buyback_row = _find_cashflow_row(cf, _BUYBACK_ROW_CANDIDATES, ["repurchase", "stock"])
    capex_row = _find_cashflow_row(cf, _CAPEX_ROW_CANDIDATES, ["capital expenditure"])
    ocf_row = _find_cashflow_row(cf, _OCF_ROW_CANDIDATES, ["operating", "cash"])

    if buyback_row is None and capex_row is None and ocf_row is None:
        return empty

    frame = pd.DataFrame(index=cf.columns)
    if buyback_row is not None:
        frame["buyback_value"] = cf.loc[buyback_row]
    if capex_row is not None:
        # yfinance reports CapEx as a cash outflow (negative); store the
        # magnitude so "elevated CapEx" reads as a positive, growing number.
        frame["capex"] = cf.loc[capex_row].abs()
    if ocf_row is not None:
        frame["operating_cash_flow"] = cf.loc[ocf_row]
    if "operating_cash_flow" in frame.columns and "capex" in frame.columns:
        frame["fcf"] = frame["operating_cash_flow"] - frame["capex"]

    frame = frame.dropna(how="all")
    if frame.empty:
        return empty

    history = frame.reset_index().rename(columns={"index": "period"})
    for col in ("buyback_value", "capex", "operating_cash_flow", "fcf"):
        if col not in history.columns:
            history[col] = None
    history["buyback_magnitude"] = history["buyback_value"].abs()
    history = history.sort_values("period").reset_index(drop=True)
    history["ticker"] = ticker

    trend = _buyback_trend(history)
    fcf_trend = _fcf_trend(history)
    return {
        "ticker": ticker, "history": history.tail(4), "trend": trend, "fcf_trend": fcf_trend,
        "source": "yfinance",
    }


def _buyback_trend(history):
    if history is None or len(history) < 2 or history["buyback_magnitude"].isna().all():
        return "N/A"
    recent, prior = history["buyback_magnitude"].iloc[-1], history["buyback_magnitude"].iloc[-2]
    if pd.isna(recent) or pd.isna(prior):
        return "N/A"
    if prior == 0:
        return "NEW" if recent > 0 else "FLAT"
    if recent > prior * 1.05:
        return "INCREASING"
    if recent < prior * 0.95:
        return "DECREASING"
    return "FLAT"


def _fcf_trend(history):
    """DECLINING/IMPROVING/FLAT on the most recent two periods' Free Cash
    Flow -- the counterpart to _buyback_trend, read alongside the raw
    capex/fcf figures so the AI Briefing can state (per Part 2's spec)
    whether a FCF decline coincided with a CapEx step-up, not just that
    FCF moved."""
    if history is None or "fcf" not in history.columns or len(history) < 2 or history["fcf"].isna().all():
        return "N/A"
    recent, prior = history["fcf"].iloc[-1], history["fcf"].iloc[-2]
    if pd.isna(recent) or pd.isna(prior):
        return "N/A"
    if prior == 0:
        return "FLAT"
    change_pct = (recent - prior) / abs(prior)
    if change_pct <= -0.05:
        return "DECLINING"
    if change_pct >= 0.05:
        return "IMPROVING"
    return "FLAT"


def _persist_buybacks(conn, ticker, result):
    history = result.get("history")
    if history is None or history.empty:
        return
    cur = conn.cursor()
    for _, row in history.iterrows():
        cur.execute(
            """INSERT INTO buybacks (ticker, period, buyback_value, capex, operating_cash_flow, fcf,
                                      source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker, period) DO UPDATE SET
                   buyback_value=excluded.buyback_value, capex=excluded.capex,
                   operating_cash_flow=excluded.operating_cash_flow, fcf=excluded.fcf,
                   source=excluded.source, fetched_at=excluded.fetched_at""",
            (ticker, str(row["period"]), _none_if_nan(row.get("buyback_value")),
             _none_if_nan(row.get("capex")), _none_if_nan(row.get("operating_cash_flow")),
             _none_if_nan(row.get("fcf")), result.get("source"), datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_buybacks_cache(conn, ticker, limit=4):
    df = pd.read_sql_query(
        "SELECT period, buyback_value, capex, operating_cash_flow, fcf FROM buybacks "
        "WHERE ticker=? ORDER BY period DESC LIMIT ?",
        conn, params=(ticker, limit),
    )
    if df.empty:
        return df
    df = df.sort_values("period").reset_index(drop=True)
    df["buyback_magnitude"] = df["buyback_value"].abs()
    df["ticker"] = ticker
    return df


def cached_buybacks(conn, ticker, max_age_hours=168, force_refresh=False):
    table = "buybacks"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        history = _read_buybacks_cache(conn, ticker)
        if not history.empty:
            return {"data": {"ticker": ticker, "history": history, "trend": _buyback_trend(history),
                              "fcf_trend": _fcf_trend(history)},
                    "source": "yfinance", "cache_hit": True, "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_buybacks(ticker)
        _persist_buybacks(conn, ticker, result)
        _log_fetch(conn, table, ticker, True, len(result.get("history", [])))
        return {"data": result, "source": result.get("source"), "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        history = _read_buybacks_cache(conn, ticker)
        return {"data": {"ticker": ticker, "history": history, "trend": _buyback_trend(history),
                          "fcf_trend": _fcf_trend(history)},
                "source": "yfinance" if not history.empty else None, "cache_hit": not history.empty,
                "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# News source management -- news_sources is a real, user-editable registry
# of feed URLs (SETTINGS tab's "News Feeds" section), not a hardcoded
# single dispatch like every other DATA_SOURCE_CONFIG-driven fetcher. This
# replaced the previous single yfinance-library-based implementation
# (yf.Ticker(ticker).news) entirely -- that was a Python library call, not
# a URL, so it couldn't be represented, tested, or removed the way a
# registry row can be. _seed_news_sources() migrates that to the
# equivalent real Yahoo RSS URL (confirmed live) as the one default row.
# --------------------------------------------------------------------------

def _read_news_sources(conn, enabled_only=False):
    query = "SELECT id, name, feed_url, url_type, enabled, added_at FROM news_sources"
    if enabled_only:
        query += " WHERE enabled=1"
    query += " ORDER BY id ASC"
    rows = conn.execute(query).fetchall()
    keys = ["id", "name", "feed_url", "url_type", "enabled", "added_at"]
    return [dict(zip(keys, r)) for r in rows]


def add_news_source(conn, name, feed_url, url_type="rss"):
    conn.execute(
        "INSERT INTO news_sources (name, feed_url, url_type, enabled, added_at) VALUES (?,?,?,1,?)",
        (name.strip(), feed_url.strip(), url_type, datetime.utcnow().isoformat()),
    )
    conn.commit()


def remove_news_source(conn, source_id):
    """Deletes the row entirely -- per Part 2's spec, removal is real
    removal (stops being fetched immediately), not a soft hide."""
    conn.execute("DELETE FROM news_sources WHERE id=?", (source_id,))
    conn.commit()


def set_news_source_enabled(conn, source_id, enabled):
    conn.execute("UPDATE news_sources SET enabled=? WHERE id=?", (1 if enabled else 0, source_id))
    conn.commit()


def test_news_feed(url):
    """Real feedparser.parse(url) test for the SETTINGS 'Test Feed' button
    -- called on a raw candidate URL (not yet saved), so a {ticker}
    placeholder is substituted with a real, liquid ticker (SPY) purely so
    the test fetch has something concrete to request; the placeholder
    itself is preserved in the saved feed_url. Returns {"ok", "entry_count",
    "preview" (first 3 titles), "error"} -- ok is False on zero entries or
    a parse error, never silently treated as success."""
    test_url = url.replace("{ticker}", "SPY")
    try:
        parsed = feedparser.parse(test_url)
    except Exception as e:
        return {"ok": False, "entry_count": 0, "preview": [], "error": str(e)}
    if parsed.bozo and not parsed.entries:
        err = str(parsed.get("bozo_exception") or "feed did not parse (malformed XML or non-feed response)")
        return {"ok": False, "entry_count": 0, "preview": [], "error": err}
    if not parsed.entries:
        return {"ok": False, "entry_count": 0, "preview": [], "error": "feed parsed but returned zero entries"}
    preview = [e.get("title") for e in parsed.entries[:3] if e.get("title")]
    return {"ok": True, "entry_count": len(parsed.entries), "preview": preview, "error": None}


def _derive_publisher_from_link(link):
    """The per-article 'who actually wrote this' attribution -- RSS
    entries from this feed carry no clean publisher field of their own
    (confirmed: Yahoo's feed entries have no author/source field), but the
    link's domain is a real, derived (not fabricated) signal for it."""
    if not link:
        return None
    try:
        netloc = urllib.parse.urlparse(link).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc or None
    except Exception:
        return None


def _ticker_relevance_pattern(symbol):
    """Whole-word ticker match, optional leading '$', bounded by non-
    alphanumeric characters or string start/end -- so a short symbol like
    'MU' doesn't match as a bare substring inside ordinary words ('Promising
    MUsic Stocks', 'Community Bancorp'), confirmed as a real false-positive
    against a live general feed (MarketBeat's headlines RSS). Same
    convention already used for Reddit ticker matching in
    smartmoneydashboard/src/server/sentiment.ts's tickerPattern()."""
    return re.compile(rf"(?:^|[^A-Za-z0-9])\$?{re.escape(symbol)}(?:[^A-Za-z0-9]|$)", re.IGNORECASE)


def _fetch_news_from_feed_row(feed, ticker, limit):
    """Fetches one news_sources row's feed and returns a list of item
    dicts. A '{ticker}' placeholder in feed_url makes this feed
    ticker-parameterizable (a fresh, ticker-scoped fetch every call);
    without one, it's a general feed fetched as-is (see fetch_news for how
    ticker-relevance is applied to those). url_type='scrape' isn't
    implemented yet -- no real scrape-based news source exists in this
    app currently -- so it's logged and skipped rather than guessed at."""
    is_ticker_feed = "{ticker}" in feed["feed_url"]
    url = feed["feed_url"].format(ticker=ticker) if is_ticker_feed else feed["feed_url"]

    if feed["url_type"] == "scrape":
        print(f"[news] '{feed['name']}' is url_type=scrape, not implemented -- skipped")
        return [], is_ticker_feed
    if feed["url_type"] != "rss":
        print(f"[news] '{feed['name']}' has unknown url_type={feed['url_type']!r} -- skipped")
        return [], is_ticker_feed

    try:
        parsed = feedparser.parse(url)
    except Exception as e:
        print(f"[news] '{feed['name']}' fetch failed: {e}")
        return [], is_ticker_feed

    # `limit` truncates BEFORE ticker-relevance filtering is applied (see
    # fetch_news), which is only correct for a ticker-parameterized feed --
    # every entry there is already about that ticker, so keeping just the
    # newest `limit` is right. For a general feed, the ticker-relevant
    # entries are sparse and scattered through the whole feed (confirmed
    # live: a real DUOL headline sat at index 74 of 250 on MarketBeat's
    # feed) -- truncating to the first 8 before filtering meant a genuinely
    # matching headline almost never got the chance to be seen at all.
    # General feeds scan every entry; fetch_news truncates the final
    # ticker-relevant results to `limit` afterward instead.
    candidate_entries = parsed.entries[:limit] if is_ticker_feed else parsed.entries

    items = []
    for entry in candidate_entries:
        title = entry.get("title")
        if not title:
            continue
        link = entry.get("link")
        pub_struct = entry.get("published_parsed")
        published_at = (
            pd.Timestamp(*pub_struct[:6], tz="UTC") if pub_struct else
            pd.to_datetime(entry.get("published"), utc=True, errors="coerce")
        )
        items.append({
            "title": title, "publisher": _derive_publisher_from_link(link),
            "link": link, "published_at": published_at, "source": feed["name"],
        })
    return items, is_ticker_feed


def fetch_news(ticker, conn, limit=8):
    """Pulls from every enabled news_sources feed (SETTINGS-managed
    registry), combines and dedupes by normalized title, and returns a
    DataFrame[ticker, title, publisher, link, published_at, source] --
    same shape callers already expect. Ticker-parameterizable feeds
    (feed_url contains '{ticker}') are always included; general feeds
    (no placeholder) are included only for headlines whose title actually
    mentions the ticker symbol, since a general market feed fetched for
    every watchlist ticker would otherwise flood each ticker's per-ticker
    view with irrelevant headlines. Fails soft per-feed (one broken feed
    doesn't drop the others)."""
    feeds = _read_news_sources(conn, enabled_only=True)
    all_items = []
    for feed in feeds:
        items, is_ticker_feed = _fetch_news_from_feed_row(feed, ticker, limit)
        if not is_ticker_feed:
            pattern = _ticker_relevance_pattern(ticker)
            items = [it for it in items if pattern.search(it["title"])]
        all_items.extend(items)

    seen_titles = set()
    deduped = []
    for it in all_items:
        key = re.sub(r"\s+", " ", it["title"].strip().lower())
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append({**it, "ticker": ticker})

    df = pd.DataFrame(deduped)
    if not df.empty:
        df = df.sort_values("published_at", ascending=False, na_position="last").head(limit).reset_index(drop=True)
    return df


def _persist_news(conn, ticker, df):
    """UPSERT, not INSERT OR IGNORE: a (ticker, link) pair that already
    exists gets its source/title/publisher refreshed instead of silently
    staying frozen forever. INSERT OR IGNORE was the bug behind a real
    incident -- migrating the news source-tagging convention (source=
    "yfinance" -> the feed's real registry name, e.g. "Yahoo Finance RSS")
    left every already-cached article stuck under the old tag, since the
    exact same article link recurring on a later fetch just got silently
    skipped rather than updated. Since news_sources.name is user-editable
    (rename a feed, or two different rows briefly resolve to the same
    URL), the tag genuinely can change out from under an existing
    cached link, and it should stick."""
    if df is None or df.empty:
        return
    cur = conn.cursor()
    for _, row in df.iterrows():
        link = row.get("link")
        if not link:
            continue
        published_at = row.get("published_at")
        published_iso = published_at.isoformat() if pd.notna(published_at) else None
        cur.execute(
            """INSERT INTO news_cache (ticker, title, publisher, link, published_at, source, fetched_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(ticker, link) DO UPDATE SET
                   title=excluded.title, publisher=excluded.publisher,
                   published_at=excluded.published_at, source=excluded.source,
                   fetched_at=excluded.fetched_at""",
            (ticker, row.get("title"), row.get("publisher"), link, published_iso, row.get("source"),
             datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_news_cache(conn, ticker, limit=8):
    df = pd.read_sql_query(
        """SELECT title, publisher, link, published_at, source FROM news_cache
           WHERE ticker=? ORDER BY published_at DESC LIMIT ?""",
        conn, params=(ticker, limit),
    )
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df


def cached_news(conn, ticker, limit=8, max_age_hours=2, force_refresh=False):
    """yfinance's news endpoint has no "since" parameter -- it always
    returns its current recent list, so the network call itself can't be
    narrowed to a delta window. The delta discipline is applied at the
    storage layer instead: only articles published after our stored
    high-water mark (get_last_timestamp) are counted as genuinely new in
    fetch_log, so repeat calls don't inflate rows_returned with headlines
    we already have (INSERT OR IGNORE was already silently no-op'ing the
    duplicate writes; this just makes the accounting honest)."""
    table = "news_cache"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_news_cache(conn, ticker, limit=limit)
        if not cached.empty:
            return {"data": cached, "source": cached["source"].iloc[0], "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        last_ts = get_last_timestamp(conn, table, ticker, "published_at")
        df = fetch_news(ticker, conn, limit=limit)
        new_count = len(df)
        if last_ts and not df.empty:
            new_count = int((df["published_at"] > pd.Timestamp(last_ts)).sum())
        _persist_news(conn, ticker, df)
        _log_fetch(conn, table, ticker, True, new_count)
        # Re-read so newly-fetched items merge with any previously cached
        # headlines still within the requested limit, sorted by publish date.
        merged = _read_news_cache(conn, ticker, limit=limit)
        result_df = merged if not merged.empty else df
        source = df["source"].iloc[0] if not df.empty else None
        return {"data": result_df, "source": source, "cache_hit": False, "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_news_cache(conn, ticker, limit=limit)
        return {"data": cached, "source": cached["source"].iloc[0] if not cached.empty else None,
                "cache_hit": not cached.empty, "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_13f_changes(ticker) -> list[dict]. SEC EDGAR full text search for
# recent 13F-HR filings mentioning this ticker. Source-agnostic: dispatches
# on DATA_SOURCE_CONFIG["13f"]. v1: "who recently filed a 13F involving this
# name" -- a real holdings diff would require parsing each institution's
# 13F XML line items.
# --------------------------------------------------------------------------

def fetch_13f_changes(ticker, limit=20):
    source = DATA_SOURCE_CONFIG.get("13f", "sec_edgar_free")
    if source == "sec_edgar_free":
        return _fetch_13f_changes_sec_edgar(ticker, limit=limit)
    raise NotImplementedError(
        f"13f source '{source}' is not implemented. Intended: e.g. a paid WhaleWisdom/13F-aggregator API."
    )


def _fetch_13f_changes_sec_edgar(ticker, limit=20):
    try:
        resp = requests.get(SEC_FULLTEXT_SEARCH_URL.format(ticker=ticker), headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[13f_changes] fetch failed for {ticker}: {e}")
        return []

    hits = (data.get("hits") or {}).get("hits") or []
    # The search API ranks by text relevance, not recency -- re-sort so
    # "recent 13F filings" actually means recent.
    hits.sort(key=lambda h: h.get("_source", {}).get("file_date") or "", reverse=True)

    records = []
    for hit in hits[:limit]:
        src = hit.get("_source", {})

        display_names = src.get("display_names")
        institution = display_names[0] if isinstance(display_names, list) and display_names else None
        if institution:
            institution = re.sub(r"\s*\(CIK\s+\d+\)\s*$", "", institution).strip()

        ciks = src.get("ciks")
        cik = ciks[0] if isinstance(ciks, list) and ciks else None

        filing_date = src.get("file_date")
        form_type = src.get("form") or (src.get("root_forms") or [None])[0]
        adsh = src.get("adsh") or (hit.get("_id") or "").split(":")[0]

        source_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{adsh}-index.htm"
            if cik and adsh else None
        )

        records.append({
            "ticker": ticker, "institution": institution, "cik": cik, "filing_date": filing_date,
            "accession_no": adsh, "form_type": form_type, "source_url": source_url, "source": "sec_edgar_free",
        })
    return records


def _persist_13f(conn, records):
    cur = conn.cursor()
    inserted = 0
    for r in records:
        cur.execute(
            """INSERT OR IGNORE INTO thirteenf_filings
                (ticker, institution, cik, filing_date, accession_no, form_type, source_url, source, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (r["ticker"], r["institution"], r["cik"], r["filing_date"], r["accession_no"], r["form_type"],
             r["source_url"], r["source"], datetime.utcnow().isoformat()),
        )
        if cur.rowcount:
            inserted += 1
    conn.commit()
    return inserted


def _read_13f_cache(conn, ticker, limit=20):
    rows = conn.execute(
        """SELECT institution, cik, filing_date, accession_no, form_type, source_url, source
           FROM thirteenf_filings WHERE ticker=? ORDER BY filing_date DESC LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    keys = ["institution", "cik", "filing_date", "accession_no", "form_type", "source_url", "source"]
    return [dict(zip(keys, r), ticker=ticker) for r in rows]


def cached_13f_changes(conn, ticker, max_age_hours=168, force_refresh=False):
    table = "thirteenf_filings"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_13f_cache(conn, ticker)
        if cached:
            return {"data": cached, "source": cached[0]["source"], "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        records = fetch_13f_changes(ticker)
        _persist_13f(conn, records)
        _log_fetch(conn, table, ticker, True, len(records))
        source = records[0]["source"] if records else DATA_SOURCE_CONFIG.get("13f")
        return {"data": records or _read_13f_cache(conn, ticker), "source": source, "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_13f_cache(conn, ticker)
        return {"data": cached, "source": cached[0]["source"] if cached else None,
                "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_leaps_candidates — deep-ITM LEAPS stock-replacement scanner, ported
# directly from leap_picker.py's Black-Scholes delta/theta filter + borderline
# candidate selection methodology. Single source (not in DATA_SOURCE_CONFIG).
# --------------------------------------------------------------------------

def _bs_call_delta_theta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes call delta and theta (per day, per share) -- thin
    wrapper over bs_greeks() (see the options-flow Greeks section above
    fetch_options_flow) so the LEAPS scanner's math is byte-for-byte the
    same call it always was, just no longer duplicated."""
    g = bs_greeks(S, K, T, r, sigma, option_type="call", q=q)
    return g["delta"], g["theta"]


def _parse_yf_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _pick_leaps_expiration(expirations, min_months_out):
    now = datetime.now(timezone.utc)
    min_days = int(min_months_out * 30.44)
    candidates = []
    for e in expirations:
        days_out = (_parse_yf_date(e) - now).days
        if days_out >= min_days:
            candidates.append((days_out, e))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _leaps_safe_float(x):
    try:
        return float("nan") if x is None else float(x)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class Rules:
    min_months_out: int = 18
    min_delta: float = 0.85
    min_intrinsic_pct: float = 0.85
    max_theta_abs_per_day: float = 0.05  # per share/day
    min_open_interest: int = 50
    max_spread_pct: float = 0.05  # (ask-bid)/mid


def build_itm_calls_table(ticker, expiry, risk_free_rate=0.04, dividend_yield=0.0):
    t = yf.Ticker(ticker)

    hist = t.history(period="5d")
    if hist.empty:
        raise RuntimeError(f"Could not fetch price for {ticker}")
    spot = float(hist["Close"].iloc[-1])

    now = datetime.now(timezone.utc)
    exp_dt = _parse_yf_date(expiry)
    T = max((exp_dt - now).total_seconds(), 0.0) / (365.0 * 24.0 * 3600.0)
    days_to_exp = (exp_dt - now).days

    chain = t.option_chain(expiry)
    calls = chain.calls.copy()

    for col in ["bid", "ask", "lastPrice", "strike", "impliedVolatility"]:
        calls[col] = calls[col].apply(_leaps_safe_float)

    calls["mid"] = (calls["bid"] + calls["ask"]) / 2.0
    calls.loc[calls["mid"].isna(), "mid"] = calls["lastPrice"]

    calls["spot"] = spot
    calls["expiry"] = expiry
    calls["days_to_exp"] = days_to_exp

    calls["intrinsic"] = (spot - calls["strike"]).clip(lower=0.0)
    calls["time_value"] = (calls["mid"] - calls["intrinsic"]).clip(lower=0.0)
    calls["intrinsic_pct"] = calls["intrinsic"] / calls["mid"]

    calls["spread"] = (calls["ask"] - calls["bid"]).clip(lower=0.0)
    calls["spread_pct"] = calls["spread"] / calls["mid"]

    deltas, thetas = [], []
    for _, row in calls.iterrows():
        d, th = _bs_call_delta_theta(
            spot, float(row["strike"]), T, risk_free_rate, float(row["impliedVolatility"]), q=dividend_yield
        )
        deltas.append(d)
        thetas.append(th)

    calls["delta_est"] = deltas
    calls["theta_est_per_day"] = thetas

    itm = calls[calls["intrinsic"] > 0].copy()
    itm.sort_values("strike", inplace=True)

    cols = [
        "contractSymbol", "expiry", "days_to_exp", "strike", "spot", "bid", "ask", "mid",
        "intrinsic", "time_value", "intrinsic_pct", "delta_est", "theta_est_per_day",
        "impliedVolatility", "volume", "openInterest", "spread_pct",
    ]
    return itm[cols], spot


def filter_pass_rules(itm, rules):
    df = itm.copy()
    df["openInterest"] = df["openInterest"].fillna(0).astype(int)
    df["spread_pct"] = df["spread_pct"].fillna(1.0)

    passed = df[
        (df["delta_est"] >= rules.min_delta)
        & (df["intrinsic_pct"] >= rules.min_intrinsic_pct)
        & (df["theta_est_per_day"].abs() <= rules.max_theta_abs_per_day)
        & (df["openInterest"] >= rules.min_open_interest)
        & (df["spread_pct"] <= rules.max_spread_pct)
    ].copy()

    passed["delta_margin"] = passed["delta_est"] - rules.min_delta
    passed["intrinsic_margin"] = passed["intrinsic_pct"] - rules.min_intrinsic_pct
    passed["theta_margin"] = rules.max_theta_abs_per_day - passed["theta_est_per_day"].abs()
    passed["spread_margin"] = rules.max_spread_pct - passed["spread_pct"]

    # Distance score: lower = closer to the boundary (barely passes).
    passed["border_score"] = (
        passed["delta_margin"].clip(lower=0) / max(1e-6, (1.0 - rules.min_delta))
        + passed["intrinsic_margin"].clip(lower=0) / max(1e-6, (1.0 - rules.min_intrinsic_pct))
        + passed["theta_margin"].clip(lower=0) / max(1e-6, rules.max_theta_abs_per_day)
        + passed["spread_margin"].clip(lower=0) / max(1e-6, rules.max_spread_pct)
    )

    # "Borderline" = minimal border_score; ties broken by highest strike (closest to ATM).
    passed.sort_values(["border_score", "strike"], ascending=[True, False], inplace=True)
    return passed


def _terminal_grid_from_spot(spot):
    return [round(spot * 0.6, 2), round(spot, 2), round(spot * 1.4, 2)]


def _option_pl_roi_at_expiry(strike, mid, terminal):
    cost = mid * 100.0
    value = max(terminal - strike, 0.0) * 100.0
    pl = value - cost
    roi = (pl / cost) * 100.0 if cost > 0 else float("nan")
    return pl, roi


def fetch_leaps_candidates(ticker, rules=None, risk_free_rate=0.04, dividend_yield=0.0,
                            terminal_prices=None, top_n=5):
    """Deep-ITM stock-replacement LEAPS scanner. Auto-selects the nearest
    expiry >= rules.min_months_out, filters to contracts passing the
    stock-replacement rules, and returns the `top_n` candidates starting at
    the borderline (barely-passing) strike and going deeper ITM."""
    rules = rules or Rules()

    t = yf.Ticker(ticker)
    try:
        expirations = list(t.options)
    except Exception:
        expirations = []
    if not expirations:
        return pd.DataFrame()

    expiry = _pick_leaps_expiration(expirations, rules.min_months_out)
    if not expiry:
        return pd.DataFrame()

    try:
        itm_calls, spot = build_itm_calls_table(
            ticker, expiry, risk_free_rate=risk_free_rate, dividend_yield=dividend_yield
        )
    except Exception:
        return pd.DataFrame()

    passed = filter_pass_rules(itm_calls, rules)
    if passed.empty:
        return pd.DataFrame()

    borderline_strike = float(passed.iloc[0]["strike"])
    deeper = passed[passed["strike"] <= borderline_strike].copy()
    deeper.sort_values("strike", ascending=False, inplace=True)
    selected = deeper.head(top_n).copy()

    if terminal_prices is None:
        bear_t, base_t, bull_t = _terminal_grid_from_spot(spot)
    else:
        bear_t, base_t, bull_t = terminal_prices[:3]

    rows = []
    for _, r in selected.iterrows():
        strike, mid = float(r["strike"]), float(r["mid"])
        option_cost = mid * 100.0
        leverage = (spot * 100.0 / option_cost) if option_cost > 0 else float("nan")
        breakeven = strike + mid

        bear_pl, bear_roi = _option_pl_roi_at_expiry(strike, mid, bear_t)
        base_pl, base_roi = _option_pl_roi_at_expiry(strike, mid, base_t)
        bull_pl, bull_roi = _option_pl_roi_at_expiry(strike, mid, bull_t)

        rows.append({
            "ticker": ticker, "contractSymbol": r["contractSymbol"], "expiry": expiry,
            "days_to_exp": int(r["days_to_exp"]), "strike": strike, "spot": spot, "mid": mid,
            "delta_est": float(r["delta_est"]), "intrinsic_pct": float(r["intrinsic_pct"]),
            "theta_est_per_day": float(r["theta_est_per_day"]), "spread_pct": float(r["spread_pct"]),
            "open_interest": int(r["openInterest"]), "option_cost": option_cost, "leverage": leverage,
            "breakeven": breakeven,
            "bear_price": bear_t, "bear_pl": bear_pl, "bear_roi": bear_roi,
            "base_price": base_t, "base_pl": base_pl, "base_roi": base_roi,
            "bull_price": bull_t, "bull_pl": bull_pl, "bull_roi": bull_roi,
        })

    return pd.DataFrame(rows)


def _persist_leaps(conn, ticker, df, source):
    if df is None or df.empty:
        return None
    fetched_at = datetime.utcnow().isoformat()
    cur = conn.cursor()
    for _, row in df.iterrows():
        cur.execute(
            """INSERT INTO leaps_candidates_cache
                (ticker, contract_symbol, expiry, strike, mid, delta_est, breakeven, option_cost,
                 bear_price, bear_roi, base_price, base_roi, bull_price, bull_roi, source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker, row["contractSymbol"], row["expiry"], row["strike"], row["mid"], row["delta_est"],
             row["breakeven"], row["option_cost"], row["bear_price"], row["bear_roi"], row["base_price"],
             row["base_roi"], row["bull_price"], row["bull_roi"], source, fetched_at),
        )
    conn.commit()
    return fetched_at


def _read_leaps_cache(conn, ticker):
    latest = conn.execute(
        "SELECT MAX(fetched_at) FROM leaps_candidates_cache WHERE ticker=?", (ticker,)
    ).fetchone()
    if not latest or not latest[0]:
        return pd.DataFrame(), None
    fetched_at = latest[0]
    df = pd.read_sql_query(
        """SELECT contract_symbol AS contractSymbol, expiry, strike, mid, delta_est, breakeven, option_cost,
                  bear_price, bear_roi, base_price, base_roi, bull_price, bull_roi, source
           FROM leaps_candidates_cache WHERE ticker=? AND fetched_at=? ORDER BY strike DESC""",
        conn, params=(ticker, fetched_at),
    )
    return df, fetched_at


def cached_leaps_candidates(conn, ticker, max_age_hours=1, force_refresh=False):
    table = "leaps_candidates_cache"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        df, fetched_at = _read_leaps_cache(conn, ticker)
        if not df.empty:
            return {"data": df, "source": df["source"].iloc[0], "cache_hit": True, "fetched_at": fetched_at}
    try:
        df = fetch_leaps_candidates(ticker, top_n=5)
        source = "yfinance"
        fetched_at = _persist_leaps(conn, ticker, df, source)
        _log_fetch(conn, table, ticker, True, len(df))
        return {"data": df, "source": source, "cache_hit": False,
                "fetched_at": fetched_at or datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        df, fetched_at = _read_leaps_cache(conn, ticker)
        return {"data": df, "source": df["source"].iloc[0] if not df.empty else None,
                "cache_hit": not df.empty, "fetched_at": fetched_at}


# --------------------------------------------------------------------------
# fetch_earnings_signal — ported from ~/trading/new_top.py's
# analyze_earnings_template / calculate_expected_move_iv /
# get_next_day_price_move. The calculation logic is preserved exactly:
#   - IV-based expected move:  price * (iv/100) * sqrt(days/365)
#   - straddle expected move:  ATM call lastPrice + ATM put lastPrice
#   - ATM avg IV:              mean of ATM call IV and ATM put IV
#   - historical earnings reaction: % close-to-close move from the trading
#     day at/before an earnings date to the next trading day after it
#
# One adaptation: new_top.py sourced historical earnings dates by scraping
# marketbeat.com (hardcoded fake headers, no auth, fragile -- and that repo
# also has multiple hardcoded API keys/credentials in plaintext, which are
# not touched or ported here). This instead reads dates from
# fetch_earnings_calendar's yfinance-backed eps_history, already in this
# file and carrying real Reported EPS/Surprise data -- the exact thing
# MarketBeat's scrape was trying to approximate, from a source that's
# already integrated and doesn't require a new fragile dependency.
# --------------------------------------------------------------------------

def calculate_expected_move_iv(price, iv, days=5):
    """Ported verbatim from new_top.py. price: underlying price. iv: IV as
    a percent (e.g. 42.5, not 0.425). days: horizon in calendar days.

    Audited (2026-08-16): the `iv` this receives is never a chain-wide
    average. Its one caller, fetch_earnings_signal, derives it via
    `atm_strike = min(calls["strike"], key=lambda x: abs(x - current_price))`
    then averages exactly that one strike's call+put IV -- the true ATM
    pair, not the whole chain. Confirmed this survived the new_top.py port
    unchanged."""
    return round(price * (iv / 100) * math.sqrt(days / 365), 2)


def fetch_earnings_signal(ticker):
    tk = yf.Ticker(ticker)

    try:
        hist_1d = tk.history(period="1d")
        current_price = float(hist_1d["Close"].iloc[-1]) if not hist_1d.empty else None
    except Exception:
        current_price = None

    if current_price is None:
        return {"ticker": ticker, "current_price": None, "next_earnings_date": None,
                "days_to_earnings": None, "atm_avg_iv_pct": None, "iv_expected_move_usd": None,
                "straddle_expected_move_usd": None, "historical_reactions": [], "source": "yfinance"}

    earnings = fetch_earnings_calendar(ticker)
    next_earnings_date = earnings.get("next_earnings_date")
    days_to_earnings = None
    if next_earnings_date is not None:
        try:
            ed = pd.Timestamp(next_earnings_date)
            if ed.tzinfo is None:
                ed = ed.tz_localize("UTC")
            days_to_earnings = (ed - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400.0
        except (TypeError, ValueError):
            days_to_earnings = None

    # --- ATM straddle at the nearest expiration: avg IV + both expected-move flavors ---
    avg_iv = None
    straddle_expected_move = None
    iv_expected_move = None
    try:
        expirations = list(tk.options)
        if expirations:
            chain = tk.option_chain(expirations[0])
            calls, puts = chain.calls, chain.puts
            if not calls.empty:
                atm_strike = min(calls["strike"], key=lambda x: abs(x - current_price))
                atm_call = calls[calls["strike"] == atm_strike]
                atm_put = puts[puts["strike"] == atm_strike]
                if not atm_call.empty and not atm_put.empty:
                    avg_iv = ((float(atm_call["impliedVolatility"].iloc[0]) +
                               float(atm_put["impliedVolatility"].iloc[0])) / 2) * 100
                    straddle_expected_move = round(
                        float(atm_call["lastPrice"].iloc[0]) + float(atm_put["lastPrice"].iloc[0]), 2
                    )
                if avg_iv is not None:
                    iv_expected_move = calculate_expected_move_iv(current_price, avg_iv, days=5)
    except Exception:
        pass

    # --- historical post-earnings 1-day price reaction, keyed off eps_history ---
    reactions = []
    eps_history = earnings.get("eps_history")
    if eps_history is not None and not eps_history.empty:
        try:
            price_hist = tk.history(period="2y")
            price_hist.index = pd.to_datetime(price_hist.index.date)
            for _, row in eps_history.iterrows():
                ed_raw = row.get("Earnings Date")
                reported = row.get("Reported EPS")
                estimate = row.get("EPS Estimate")
                if pd.isna(ed_raw) or pd.isna(reported):
                    continue  # future/unreported earnings -- no reaction to measure yet
                edate = pd.Timestamp(ed_raw).tz_localize(None).normalize()

                close_before = close_after = None
                for offset in range(0, 4):
                    day = edate + pd.Timedelta(days=offset)
                    if day in price_hist.index:
                        close_before = float(price_hist.loc[day, "Close"])
                        break
                if close_before is not None:
                    for offset in range(1, 5):
                        day = edate + pd.Timedelta(days=offset)
                        if day in price_hist.index:
                            close_after = float(price_hist.loc[day, "Close"])
                            break
                if close_before is None or close_after is None:
                    continue

                pct_move = (close_after - close_before) / close_before * 100
                beat_miss = None
                if pd.notna(reported) and pd.notna(estimate):
                    beat_miss = "BEAT" if reported > estimate else "MISS" if reported < estimate else "INLINE"
                reactions.append({
                    "earnings_date": edate.date().isoformat(),
                    "pct_move": round(float(pct_move), 2),
                    "beat_miss": beat_miss,
                })
        except Exception:
            pass

    return {
        "ticker": ticker,
        "current_price": current_price,
        "next_earnings_date": next_earnings_date,
        "days_to_earnings": round(days_to_earnings, 2) if days_to_earnings is not None else None,
        "atm_avg_iv_pct": round(avg_iv, 2) if avg_iv is not None else None,
        "iv_expected_move_usd": iv_expected_move,
        "straddle_expected_move_usd": straddle_expected_move,
        "historical_reactions": reactions,
        "source": "yfinance",
    }


def _persist_earnings_signal(conn, ticker, result):
    conn.execute(
        """INSERT INTO earnings_signal
            (ticker, current_price, next_earnings_date, days_to_earnings, atm_avg_iv_pct,
             iv_expected_move_usd, straddle_expected_move_usd, historical_reactions_json,
             source, fetched_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (ticker, result.get("current_price"),
         str(result.get("next_earnings_date")) if result.get("next_earnings_date") is not None else None,
         result.get("days_to_earnings"), result.get("atm_avg_iv_pct"), result.get("iv_expected_move_usd"),
         result.get("straddle_expected_move_usd"), json.dumps(result.get("historical_reactions") or []),
         result.get("source"), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_earnings_signal_cache(conn, ticker):
    row = conn.execute(
        """SELECT current_price, next_earnings_date, days_to_earnings, atm_avg_iv_pct,
                  iv_expected_move_usd, straddle_expected_move_usd, historical_reactions_json, source
           FROM earnings_signal WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    (current_price, next_earnings_date, days_to_earnings, atm_avg_iv_pct, iv_expected_move_usd,
     straddle_expected_move_usd, reactions_json, source) = row
    return {
        "ticker": ticker, "current_price": current_price, "next_earnings_date": next_earnings_date,
        "days_to_earnings": days_to_earnings, "atm_avg_iv_pct": atm_avg_iv_pct,
        "iv_expected_move_usd": iv_expected_move_usd, "straddle_expected_move_usd": straddle_expected_move_usd,
        "historical_reactions": json.loads(reactions_json) if reactions_json else [],
        "source": source,
    }


def cached_earnings_signal(conn, ticker, max_age_hours=12, force_refresh=False):
    table = "earnings_signal"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_earnings_signal_cache(conn, ticker)
        if cached:
            return {"data": cached, "source": cached.get("source"), "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_earnings_signal(ticker)
        _persist_earnings_signal(conn, ticker, result)
        _log_fetch(conn, table, ticker, True, 1)
        return {"data": result, "source": result.get("source"), "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_earnings_signal_cache(conn, ticker)
        return {"data": cached or {}, "source": (cached or {}).get("source"), "cache_hit": bool(cached),
                "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# Real-data sources for the AI Deep-Dive Briefing -- ported from
# ~/trading/new_top.py, REAL-DATA functions only:
#   get_marketbeat_analyst_sentiment_structured -> _fetch_marketbeat_earnings_history
#   get_eps_and_move_summary_real_data          -> fetch_earnings_history_real
#   get_insider_sales_finviz                    -> fetch_insider_sales_finviz
#   get_reddit_posts / get_stocktwits_posts     -> fetch_reddit_posts_public / _fetch_stocktwits_posts
#   calculate_expected_move_iv                  -> already ported verbatim above (earnings_signal)
#
# NOT ported -- new_top.py's ChatGPT-based equivalents ask an LLM to invent
# dates/moves with no real input (get_recent_earnings_dates_via_chatgpt,
# get_expected_move_from_chatgpt, get_eps_and_move_summary_chatgpt). The
# real yfinance/MarketBeat-sourced functions above are used instead
# everywhere a track record or expected move is needed.
#
# ~/trading/new_top.py also hardcodes an Anthropic key, an OpenAI key, and
# Reddit client_id/client_secret in plaintext -- none of that was copied.
# fetch_reddit_posts_public needs no credential at all (Reddit's public
# search JSON is unauthenticated); every other credential here
# (ANTHROPIC_API_KEY, QUIVER_API_KEY) is read from the environment, and
# every fetcher in this section fails soft (never raises) so a blocked/
# changed scrape degrades one section of a briefing instead of crashing it.
#
# ~/trading/marketbeat_scraper.py's institutional-ownership half (Selenium/
# undetected-chromedriver, real-Chrome-window-required) is now ported too
# -- see fetch_marketbeat_institutional_sentiment() further down this file,
# next to _fetch_marketbeat_earnings_history. It's the one deliberate
# exception to this section's "every fetcher is a plain request" rule,
# isolated for that reason; see its own module comment for why.
# get_marketbeat_analyst_sentiment_structured, ported above as
# _fetch_marketbeat_earnings_history, is a plain-requests scrape and never
# needed the browser.
# --------------------------------------------------------------------------

def _earnings_dates_from_history_table(earnings_table, count=6):
    """Real dates parsed straight out of a scraped earnings-history table --
    ported from new_top.py's get_recent_earnings_dates_from_table. The
    non-guessing replacement for that file's ChatGPT-based date lookup."""
    dates = []
    for row in earnings_table:
        try:
            parsed = pd.to_datetime(row["date"].split("(")[0].strip())
            dates.append(parsed)
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(dates, reverse=True)[:count]


def _fetch_marketbeat_earnings_history(ticker):
    """Real scraped earnings history (date, quarter, consensus/actual EPS,
    beat/miss, revenue) from MarketBeat -- ported from new_top.py's
    get_marketbeat_analyst_sentiment_structured (earnings_history part
    only). Tries the NASDAQ URL path first, then NYSE, since MarketBeat's
    URL is exchange-specific and this app has no exchange lookup. Fails
    soft -- [] on a blocked/changed page, never raises."""
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    for exchange in ("NASDAQ", "NYSE"):
        url = f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table", id="earnings-history")
            if not table:
                continue
            history = []
            # Was [1:7] (6 quarters) -- too thin to ever find a genuine
            # historical precedent ("the largest one-day drop since X").
            # MarketBeat's own earnings-history table typically carries
            # several years of quarters; [1:21] takes up to 20 (~5 years)
            # instead of truncating to the most recent year and a half.
            for row in table.find_all("tr")[1:21]:
                cols = row.find_all("td")
                if len(cols) >= 8:
                    history.append({
                        "date": cols[0].text.strip(),
                        "quarter": cols[1].text.strip(),
                        "consensus_eps": cols[2].text.strip(),
                        "actual_eps": cols[3].text.strip(),
                        "beat_miss": cols[4].text.strip(),
                        # cols[5] is GAAP EPS -- not captured, not needed here.
                        "revenue_estimate": cols[6].text.strip(),
                        "revenue_actual": cols[7].text.strip(),
                    })
            if history:
                return history
        except requests.RequestException:
            continue
    return []


# --------------------------------------------------------------------------
# MarketBeat institutional ownership -- Selenium/undetected-chromedriver,
# ported from ~/trading/marketbeat_scraper.py's get_marketbeat_institutional_
# sentiment. Unlike every other fetcher in this file, this one needs a REAL,
# VISIBLE (non-headless -- Cloudflare blocks headless Chrome, confirmed by
# the reference script's own comments) browser window to work at all. That
# means: (1) it only works when this app is running on a machine with an
# actual desktop session, never a headless server/cloud deployment, and
# (2) calling it pops open a visible Chrome window on that machine. This is
# a deliberate, isolated exception to every other source's plain-requests/
# yfinance design -- kept opt-in (only called from the deep-dive tab's
# explicit "Fetch live signals" flow, never from full_refresh's watchlist
# sweep) for exactly that reason.
#
# The reference script's ChromeDriver-version-mismatch bug started as a
# single hardcoded line -- `uc.Chrome(options=options, version_main=145)`
# -- pinning ChromeDriver to Chrome 145 forever, breaking the moment the
# real installed Chrome auto-updated past that. The first fix attempt
# (drop version_main entirely) turned out to be based on a wrong
# assumption: verified directly against this installed library version's
# source (patcher.py's fetch_release_number) that a falsy version_main
# does NOT auto-match the installed browser -- it fetches whatever Chrome-
# for-Testing's LATEST STABLE release is overall, no reference to the
# actually-installed browser at all. That's why the mismatch reappeared
# with a *different* pair of numbers (driver 152 vs. installed Chrome
# 150) even after removing the pin. The real fix:
# _detect_installed_chrome_major_version() below actually runs
# `<chrome binary> --version` and passes the real result as version_main
# explicitly, every launch -- so it self-heals across Chrome auto-updates
# instead of drifting stale (the 145 pin) or drifting ahead (the "no pin"
# non-fix).
#
# selenium/undetected-chromedriver are imported lazily inside these
# functions, not at module load, so the rest of the app works fine on a
# machine without them installed (or without a real Chrome browser).
# --------------------------------------------------------------------------

_marketbeat_uc_driver = None


def _wait_for_marketbeat_real_page(driver, timeout=20):
    """Waits for document.readyState=='complete' and for the page title to
    stop looking like a Cloudflare interstitial ('Just a moment...') --
    ported verbatim from the reference script's _wait_for_real_page."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ready = driver.execute_script("return document.readyState")
            title = (driver.title or "").strip().lower()
            is_cf = title in ("just a moment...", "just a moment", "", "attention required")
            if ready == "complete" and not is_cf:
                time.sleep(random.uniform(0.5, 1.0))
                return
        except Exception:
            pass
        time.sleep(0.8)


def _detect_installed_chrome_major_version():
    """Real detected major version of the locally installed Chrome, via
    `<binary> --version` -- NOT auto-detected by undetected-chromedriver
    itself despite the library's own docs implying version_main=None
    means "auto". Verified directly against this version's source
    (patcher.py's fetch_release_number): when version_main is falsy, it
    fetches whatever Chrome-for-Testing's LATEST STABLE release is
    overall, with no reference to the actually-installed browser at all --
    which is exactly how the original ChromeDriver-mismatch bug
    reappeared even after removing the version_main=145 pin (it started
    grabbing 152 while the real installed Chrome was 150). Returns None
    if detection fails, so the caller can fall back to leaving version_main
    unset rather than crashing."""
    import subprocess
    import undetected_chromedriver as uc
    try:
        binary = uc.find_chrome_executable()
        if not binary:
            return None
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10).stdout
        match = re.search(r"(\d+)\.\d+\.\d+\.\d+", out)
        return int(match.group(1)) if match else None
    except Exception:
        return None


def _get_marketbeat_uc_driver(first_url):
    """Lazily launches (or reuses) a real, visible Chrome window via
    undetected-chromedriver, pinned to the REAL detected installed Chrome
    major version (see _detect_installed_chrome_major_version -- omitting
    version_main does NOT auto-match the installed browser in this
    library version, contrary to what its own docstring implies). Reused
    across multiple tickers in one session rather than relaunched (and
    re-passing Cloudflare's challenge) on every call;
    close_marketbeat_driver() closes it explicitly if needed."""
    global _marketbeat_uc_driver
    import undetected_chromedriver as uc
    if _marketbeat_uc_driver is None:
        options = uc.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1280,900")
        options.add_argument("--lang=en-US")
        options.add_argument("--log-level=3")
        version_main = _detect_installed_chrome_major_version()
        _marketbeat_uc_driver = uc.Chrome(options=options, version_main=version_main)
        _marketbeat_uc_driver.set_page_load_timeout(60)
        _marketbeat_uc_driver.get(first_url)
        _wait_for_marketbeat_real_page(_marketbeat_uc_driver, timeout=45)
    return _marketbeat_uc_driver


def close_marketbeat_driver():
    """Closes the shared Chrome window, if one is open. Not called
    automatically anywhere in this app -- the driver is meant to persist
    across calls within a session (see _get_marketbeat_uc_driver)."""
    global _marketbeat_uc_driver
    if _marketbeat_uc_driver is not None:
        try:
            _marketbeat_uc_driver.quit()
        except Exception:
            pass
        _marketbeat_uc_driver = None


_CHROMEDRIVER_VERSION_MISMATCH_RE = re.compile(
    r"only supports Chrome version (\d+).*Current browser version is ([\d.]+)", re.IGNORECASE | re.DOTALL,
)


def _parse_marketbeat_institutional_page(soup, ticker):
    """Pure HTML-parsing half of fetch_marketbeat_institutional_sentiment --
    split out so the scraping logic is testable against a saved page
    without a real browser.

    The summary-stats section is NOT the reference script's card-scraping
    heuristic (div.col-* text blobs) -- confirmed against a real saved
    MarketBeat institutional-ownership page that MarketBeat has since
    redesigned this section into a plain <dt class="stat-summary-title">/
    <dd class="stat-summary-heading"> label/value definition-list pair,
    which the old heuristic never matched (hence ownership_pct/buyers/
    sellers/inflows/outflows always coming back null). This reads that
    real, current structure directly. The transactions table extraction
    below it was verified against the same saved page and is unchanged."""
    summary = {}
    label_key_map = [
        (("OWNERSHIP", "PERCENTAGE"), "ownership_pct"),
        (("BUYERS",), "buyers"),
        (("SELLERS",), "sellers"),
        (("INFLOWS",), "inflows"),
        (("OUTFLOWS",), "outflows"),
    ]
    for dt in soup.find_all("dt", class_="stat-summary-title"):
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        label_upper = dt.get_text(" ", strip=True).upper()
        value = dd.get_text(strip=True)
        for keywords, key in label_key_map:
            if all(k in label_upper for k in keywords) and key not in summary:
                summary[key] = value
                break

    # Fallback for a layout MarketBeat might revert to or vary by page --
    # the reference script's original div.col-* card-text heuristic.
    if not summary:
        for card in soup.find_all("div", class_=lambda c: c and "col-" in " ".join(c) if c else False):
            text = card.get_text(" ", strip=True)
            for label, key in [
                ("INSTITUTIONAL OWNERSHIP", "ownership_pct"), ("INSTITUTIONAL BUYERS", "buyers"),
                ("INSTITUTIONAL SELLERS", "sellers"), ("INSTITUTIONAL INFLOWS", "inflows"),
                ("INSTITUTIONAL OUTFLOWS", "outflows"),
            ]:
                if label in text.upper():
                    for line in [ln.strip() for ln in text.split("\n") if ln.strip()]:
                        if any(c.isdigit() for c in line) and line not in summary.values():
                            summary[key] = line
                            break

    inst_table = None
    for t in soup.find_all("table"):
        hdrs = " ".join(th.get_text(strip=True).lower() for th in t.find_all("th"))
        if "institution" in hdrs or ("date" in hdrs and "share" in hdrs):
            inst_table = t
            break

    transactions = []
    if inst_table is not None:
        hdrs = [th.get_text(strip=True) for th in inst_table.find_all("th")]

        def col_idx(keywords):
            for i, h in enumerate(hdrs):
                if any(k.lower() in h.lower() for k in keywords):
                    return i
            return None

        # Keywords chosen against MarketBeat's real confirmed header set
        # ('Reporting Date', 'Major Shareholder Name', 'Shares Held',
        # 'Market Value', '% of Portfolio', 'Quarterly Change in Shares',
        # 'Ownership in Company', 'Details') -- a bare "Share" or "Owner"
        # substring-matches the WRONG column ("Major SHAREholder Name" and
        # "OWNERship in Company" respectively), which is exactly the bug
        # that put institution names in the shares field and percentages
        # in the institution field. Longer, more specific phrases first.
        i_date = col_idx(["Reporting Date", "Date"])
        i_inst = col_idx(["Shareholder Name", "Shareholder", "Institution", "Investor Name", "Fund Name"])
        i_shares = col_idx(["Shares Held", "Position"])
        i_val = col_idx(["Market Value", "Value"])
        i_change = col_idx(["Quarterly Change", "Change in Shares", "Activity", "Action", "Type"])

        for row in inst_table.find_all("tr")[1:11]:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            date = (cols[i_date].get_text(strip=True) if i_date is not None and i_date < len(cols)
                    else cols[0].get_text(strip=True))
            inst = (cols[i_inst].get_text(strip=True) if i_inst is not None and i_inst < len(cols)
                    else (cols[1].get_text(strip=True) if len(cols) > 1 else None))
            shares = cols[i_shares].get_text(strip=True) if i_shares is not None and i_shares < len(cols) else None
            value = cols[i_val].get_text(strip=True) if i_val is not None and i_val < len(cols) else None
            change_text = (cols[i_change].get_text(strip=True)
                           if i_change is not None and i_change < len(cols) else "")
            # Prefer a real signed number ("+12,340" / "-8,200") from the
            # Quarterly Change column when present -- more precise than
            # keyword-guessing; falls back to keyword matching (covers
            # phrasing like "New Position" / "Sold Out") when it's not a
            # parseable number (e.g. "N/A").
            numeric_change = re.sub(r"[,\s]", "", change_text)
            if re.fullmatch(r"[+-]?\d+", numeric_change):
                action = "buy" if not numeric_change.startswith("-") and numeric_change != "0" else (
                    "sell" if numeric_change.startswith("-") else "unknown"
                )
            else:
                combined = (change_text + (shares or "")).lower()
                if any(w in combined for w in ["buy", "bought", "new", "incr", "added"]):
                    action = "buy"
                elif any(w in combined for w in ["sell", "sold", "exit", "reduc", "decr"]):
                    action = "sell"
                else:
                    action = "unknown"
            transactions.append({"date": date, "institution": inst, "shares": shares, "value": value,
                                  "action": action})

    buyers = sellers = None
    try:
        if summary.get("buyers"):
            buyers = int(summary["buyers"].replace(",", ""))
        if summary.get("sellers"):
            sellers = int(summary["sellers"].replace(",", ""))
    except ValueError:
        pass
    net_flow_bias_pct = (
        round(buyers / (buyers + sellers) * 100)
        if buyers is not None and sellers is not None and (buyers + sellers) > 0 else None
    )

    if not summary and not transactions:
        return None

    return {
        "ticker": ticker, "ownership_pct": summary.get("ownership_pct"), "buyers": buyers, "sellers": sellers,
        "inflows": summary.get("inflows"), "outflows": summary.get("outflows"),
        "net_flow_bias_pct": net_flow_bias_pct, "recent_transactions": transactions,
        "source": "marketbeat_institutional",
    }


def fetch_marketbeat_institutional_sentiment(ticker):
    """Real institutional-ownership data scraped from MarketBeat -- see the
    module comment above for the real-Chrome-window requirement and the
    ChromeDriver-version-mismatch fix. Tries the NASDAQ URL path first,
    then NYSE (same exchange-fallback pattern as
    _fetch_marketbeat_earnings_history -- this app has no exchange
    lookup). Fails soft: returns None on any error, with a specific,
    actionable message logged for the ChromeDriver-version-mismatch case
    in particular rather than a generic traceback."""
    try:
        import undetected_chromedriver as uc  # noqa: F401
    except ImportError as e:
        print(f"[marketbeat_institutional] required package not installed ({e}). "
              f"Run: pip install --upgrade undetected-chromedriver selenium")
        return None

    ticker = ticker.upper()
    for exchange in ("NASDAQ", "NYSE"):
        url = f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/institutional-ownership/"
        try:
            driver = _get_marketbeat_uc_driver(url)
            if driver.current_url.rstrip("/") != url.rstrip("/"):
                time.sleep(2)
                driver.set_page_load_timeout(60)
                driver.get(url)
                _wait_for_marketbeat_real_page(driver, timeout=45)
            soup = BeautifulSoup(driver.page_source, "html.parser")
        except Exception as e:
            msg = str(e)
            match = _CHROMEDRIVER_VERSION_MISMATCH_RE.search(msg)
            if match:
                print(f"[marketbeat_institutional] ChromeDriver version mismatch (driver supports "
                      f"Chrome {match.group(1)}, installed Chrome is {match.group(2)}) -- run: "
                      f"pip install --upgrade undetected-chromedriver")
            else:
                print(f"[marketbeat_institutional] fetch failed for {ticker} ({exchange}): {e}")
            close_marketbeat_driver()  # a broken/crashed driver shouldn't be reused
            return None

        result = _parse_marketbeat_institutional_page(soup, ticker)
        if result:
            return result
    return None


def _persist_marketbeat_institutional(conn, ticker, record):
    if not record:
        return
    conn.execute(
        """INSERT INTO marketbeat_institutional
               (ticker, ownership_pct, buyers, sellers, inflows, outflows, net_flow_bias_pct,
                transactions_json, source, fetched_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (ticker, record.get("ownership_pct"), record.get("buyers"), record.get("sellers"),
         record.get("inflows"), record.get("outflows"), record.get("net_flow_bias_pct"),
         json.dumps(record.get("recent_transactions") or []), record.get("source"),
         datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_marketbeat_institutional_cache(conn, ticker):
    row = conn.execute(
        """SELECT ownership_pct, buyers, sellers, inflows, outflows, net_flow_bias_pct,
                  transactions_json, source
           FROM marketbeat_institutional WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    return {
        "ticker": ticker, "ownership_pct": row[0], "buyers": row[1], "sellers": row[2],
        "inflows": row[3], "outflows": row[4], "net_flow_bias_pct": row[5],
        "recent_transactions": json.loads(row[6]) if row[6] else [], "source": row[7],
    }


def cached_marketbeat_institutional_sentiment(conn, ticker, max_age_hours=24, force_refresh=False):
    """Same cached_*(conn, ticker, max_age_hours, force_refresh) envelope
    shape as every other fetcher, but the underlying fetch pops open a
    real, visible Chrome window (see fetch_marketbeat_institutional_
    sentiment's module comment) -- the 24h default window (vs. e.g.
    options_flow's 15min) is deliberately long so that window doesn't pop
    up on every rerun; force_refresh (wired to the deep-dive tab's
    explicit "Fetch live signals" button, never full_refresh) is the only
    thing that should trigger it on demand."""
    table = "marketbeat_institutional"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_marketbeat_institutional_cache(conn, ticker)
        if cached:
            return {"data": cached, "source": cached["source"], "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        record = fetch_marketbeat_institutional_sentiment(ticker)
        _persist_marketbeat_institutional(conn, ticker, record)
        _log_fetch(conn, table, ticker, True, 1 if record else 0)
        cached = record or _read_marketbeat_institutional_cache(conn, ticker)
        return {"data": cached, "source": cached["source"] if cached else None, "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_marketbeat_institutional_cache(conn, ticker)
        return {"data": cached, "source": cached["source"] if cached else None,
                "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, ticker)}


def _parse_money_str(s):
    """Real, deterministic parser for MarketBeat's money strings ("$3.70B",
    "$3.31", "-", "N/A") -> a plain float, or None if not parseable. No
    guessing: unrecognized text returns None rather than a fabricated
    number."""
    if not s:
        return None
    s = s.strip().replace("$", "").replace(",", "")
    if not s or s in ("-", "N/A", "n/a"):
        return None
    mult = 1.0
    suffix = s[-1].upper() if s else ""
    if suffix in ("B", "M", "K"):
        mult = {"B": 1e9, "M": 1e6, "K": 1e3}[suffix]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _pct_diff(actual, estimate):
    """Real % difference of actual vs. estimate -- None if either value is
    missing/unparseable or the estimate is zero (can't compute a % base)."""
    if actual is None or estimate is None or estimate == 0:
        return None
    return round((actual - estimate) / abs(estimate) * 100, 2)


def _price_reaction_pct(hist, earnings_date):
    """Real close-to-close % price move from `earnings_date` to the next
    trading day present in `hist` (a yfinance-style DataFrame indexed by
    date). None if either date isn't in range -- never a guessed value."""
    try:
        date_obj = pd.to_datetime(earnings_date).date()
    except (ValueError, TypeError):
        return None
    idx = list(pd.to_datetime(hist.index).date)
    if date_obj not in idx:
        return None
    before_pos = idx.index(date_obj)
    if before_pos + 1 >= len(hist):
        return None
    close_before = float(hist["Close"].iloc[before_pos])
    close_after = float(hist["Close"].iloc[before_pos + 1])
    if not close_before:
        return None
    return round((close_after - close_before) / close_before * 100, 2)


def fetch_earnings_history_real(ticker):
    """Real earnings track record: MarketBeat's scraped beat/miss history,
    each row matched to the REAL yfinance-measured next-trading-day price
    move -- ported from new_top.py's get_eps_and_move_summary_real_data,
    adapted to return structured rows instead of a printed string. This is
    the function that answers "did it beat and still go down, by how much,
    and when" using only real data -- never new_top.py's ChatGPT
    equivalent (get_eps_and_move_summary_chatgpt), which was not ported."""
    earnings_table = _fetch_marketbeat_earnings_history(ticker)
    if not earnings_table:
        return []

    try:
        hist = yf.Ticker(ticker).history(period="2y")
    except Exception:
        hist = pd.DataFrame()
    if not hist.empty:
        hist.index = pd.to_datetime(hist.index.date)

    rows = []
    for e in earnings_table:
        try:
            earnings_date = pd.to_datetime(e["date"].split("(")[0].strip()).date().isoformat()
        except (KeyError, ValueError, TypeError):
            continue
        reaction_pct = _price_reaction_pct(hist, earnings_date) if not hist.empty else None
        eps_beat_miss_pct = _pct_diff(
            _parse_money_str(e.get("actual_eps")), _parse_money_str(e.get("consensus_eps"))
        )
        revenue_beat_miss_pct = _pct_diff(
            _parse_money_str(e.get("revenue_actual")), _parse_money_str(e.get("revenue_estimate"))
        )
        rows.append({
            "earnings_date": earnings_date,
            "quarter": e.get("quarter"),
            "consensus_eps": e.get("consensus_eps"),
            "actual_eps": e.get("actual_eps"),
            "beat_miss": e.get("beat_miss"),
            "revenue_estimate": e.get("revenue_estimate"),
            "revenue_actual": e.get("revenue_actual"),
            "eps_beat_miss_pct": eps_beat_miss_pct,
            "revenue_beat_miss_pct": revenue_beat_miss_pct,
            "price_reaction_pct": reaction_pct,
            "source": "marketbeat+yfinance",
        })
    return rows


def _persist_earnings_history_real(conn, ticker, rows):
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """INSERT INTO earnings_history_real
                   (ticker, earnings_date, quarter, consensus_eps, actual_eps, beat_miss,
                    revenue_estimate, revenue_actual, eps_beat_miss_pct, revenue_beat_miss_pct,
                    price_reaction_pct, source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker, earnings_date) DO UPDATE SET
                   quarter=excluded.quarter, consensus_eps=excluded.consensus_eps,
                   actual_eps=excluded.actual_eps, beat_miss=excluded.beat_miss,
                   revenue_estimate=excluded.revenue_estimate, revenue_actual=excluded.revenue_actual,
                   eps_beat_miss_pct=excluded.eps_beat_miss_pct,
                   revenue_beat_miss_pct=excluded.revenue_beat_miss_pct,
                   price_reaction_pct=excluded.price_reaction_pct,
                   source=excluded.source, fetched_at=excluded.fetched_at""",
            (ticker, r["earnings_date"], r.get("quarter"), r.get("consensus_eps"), r.get("actual_eps"),
             r.get("beat_miss"), r.get("revenue_estimate"), r.get("revenue_actual"),
             r.get("eps_beat_miss_pct"), r.get("revenue_beat_miss_pct"), r.get("price_reaction_pct"),
             r.get("source"), datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_earnings_history_real_cache(conn, ticker, limit=6):
    rows = conn.execute(
        """SELECT earnings_date, quarter, consensus_eps, actual_eps, beat_miss, revenue_estimate,
                  revenue_actual, eps_beat_miss_pct, revenue_beat_miss_pct, price_reaction_pct, source
           FROM earnings_history_real WHERE ticker=? ORDER BY earnings_date DESC LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    keys = ["earnings_date", "quarter", "consensus_eps", "actual_eps", "beat_miss", "revenue_estimate",
            "revenue_actual", "eps_beat_miss_pct", "revenue_beat_miss_pct", "price_reaction_pct", "source"]
    return [dict(zip(keys, r)) for r in rows]


def cached_earnings_history_real(conn, ticker, max_age_hours=12, force_refresh=False):
    table = "earnings_history_real"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_earnings_history_real_cache(conn, ticker)
        if cached:
            return {"data": cached, "source": "marketbeat+yfinance", "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        rows = fetch_earnings_history_real(ticker)
        if rows:
            _persist_earnings_history_real(conn, ticker, rows)
            _log_fetch(conn, table, ticker, True, len(rows))
        else:
            _log_fetch(conn, table, ticker, False, 0, "MarketBeat scrape returned no rows")
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
    cached = _read_earnings_history_real_cache(conn, ticker)
    return {"data": cached, "source": "marketbeat+yfinance" if cached else None,
            "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, ticker)}


def backfill_earnings_history(ticker, conn):
    """EARNINGS SIMULATOR Part 9 -- one-time historical backfill using
    genuinely available data: MarketBeat's scraped EPS/revenue estimate-
    vs-actual (fetch_earnings_history_real -> _fetch_marketbeat_earnings_
    history) cross-referenced against real yfinance price history for the
    actual post-earnings price reaction. This is deliberately a thin
    wrapper around the already-verified fetch_earnings_history_real /
    _persist_earnings_history_real pipeline (already ported from
    new_top.py's get_eps_and_move_summary_real_data, already used by
    cached_earnings_history_real) rather than a second, independent
    implementation of the same real-data cross-reference.

    catalyst_status/buyback_signal/pre_earnings_iv are deliberately left
    NULL on every row this writes -- none of that is available for a
    past quarter unless an AI Briefing was actually generated and a
    simulator prediction reconciled against it at the time (see
    reconcile_earnings_predictions, which is the only path that ever
    populates those three columns going forward). Forcing a fresh fetch
    (bypassing earnings_history_real's normal 12h cache) since this is
    meant to be a deliberate one-time backfill action, not a routine
    refresh.

    Returns a summary dict with an honest, sample-size-aware confidence
    note -- e.g. "EPS/Revenue pattern: based on 6 historical quarters.
    IV/catalyst/buyback signal: based on 0 tracked events (thin, low
    confidence in this specific factor)." -- rather than implying the
    backfilled rows are as complete as a live-pipeline row."""
    rows = fetch_earnings_history_real(ticker)
    if rows:
        _persist_earnings_history_real(conn, ticker, rows)
        _log_fetch(conn, "earnings_history_real", ticker, True, len(rows))
    else:
        _log_fetch(conn, "earnings_history_real", ticker, False, 0, "MarketBeat scrape returned no rows")

    stored = _read_earnings_history_real_cache(conn, ticker, limit=20)
    n_quarters = len(stored)
    n_with_live_signals = sum(
        1 for r in stored if r.get("catalyst_status") or r.get("buyback_signal") or r.get("pre_earnings_iv")
    )
    return {
        "ticker": ticker, "rows_backfilled": len(rows), "total_quarters_on_record": n_quarters,
        "quarters_with_live_qualitative_signal": n_with_live_signals,
        "confidence_note": (
            f"EPS/Revenue pattern: based on {n_quarters} historical quarter"
            f"{'s' if n_quarters != 1 else ''}. IV/catalyst/buyback signal: based on "
            f"{n_with_live_signals} tracked event{'s' if n_with_live_signals != 1 else ''} "
            f"(thin, low confidence in this specific factor)." if n_with_live_signals < 3 else
            f"EPS/Revenue pattern: based on {n_quarters} historical quarters. IV/catalyst/buyback "
            f"signal: based on {n_with_live_signals} tracked events."
        ),
    }


def fetch_quarterly_financials_with_price(ticker, limit=6):
    """Real quarterly financials (revenue, net income, net margin) from
    yfinance's quarterly_income_stmt, each row matched to the REAL closing
    price at that fiscal period end and the REAL price change over the
    following 3 trading days -- so financials and price context sit
    together without cross-referencing another tab.

    Caveat, stated plainly rather than glossed over: quarterly_income_stmt's
    column dates are fiscal PERIOD-END dates, not the actual earnings
    ANNOUNCEMENT date (which is typically weeks later) -- so "price at
    period end" is not the same thing as "price reaction to the earnings
    release." For the real announcement-date price reaction, see
    fetch_earnings_history_real / earnings_track_record_real instead; this
    function is period-end financials + period-end price context, labeled
    as such in the UI.

    Not DB-cached, same "cheap, always fetch fresh" treatment as price
    history elsewhere in this file. Fails soft -- [] on any error."""
    try:
        tk = yf.Ticker(ticker)
        qf = tk.quarterly_income_stmt
    except Exception:
        return []
    if qf is None or qf.empty or "Total Revenue" not in qf.index:
        return []

    try:
        hist = tk.history(period="2y")
    except Exception:
        hist = pd.DataFrame()
    if not hist.empty:
        hist.index = pd.to_datetime(hist.index.date)
        hist_dates = list(hist.index.date)

    rows = []
    for col in qf.columns[:limit]:
        revenue = qf.loc["Total Revenue", col] if "Total Revenue" in qf.index else None
        net_income = qf.loc["Net Income", col] if "Net Income" in qf.index else None
        if pd.isna(revenue) and pd.isna(net_income):
            continue
        revenue = float(revenue) if pd.notna(revenue) else None
        net_income = float(net_income) if pd.notna(net_income) else None
        margin_pct = round(net_income / revenue * 100, 2) if revenue and net_income is not None else None

        period_end = pd.Timestamp(col).date()
        price_at_period_end = None
        price_chg_3d_pct = None
        if not hist.empty and period_end in hist_dates:
            pos = hist_dates.index(period_end)
            price_at_period_end = float(hist["Close"].iloc[pos])
            look_ahead = min(pos + 3, len(hist) - 1)
            if look_ahead > pos and price_at_period_end:
                price_chg_3d_pct = round(
                    (float(hist["Close"].iloc[look_ahead]) / price_at_period_end - 1) * 100, 2
                )

        rows.append({
            "period_end": period_end.isoformat(),
            "revenue": revenue,
            "net_income": net_income,
            "margin_pct": margin_pct,
            "price_at_period_end": price_at_period_end,
            "price_chg_3d_pct": price_chg_3d_pct,
            "source": "yfinance",
        })
    return rows


def fetch_insider_sales_finviz(ticker, limit=10):
    """Real scraped insider-sales table from Finviz -- ported from
    new_top.py's get_insider_sales_finviz. Fails soft -- [] on any scrape
    failure (blocked page, layout change), never raises."""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException:
        return []

    insider_table = None
    for table in soup.find_all("table"):
        if "Insider Trading" in table.get_text():
            insider_table = table
            break
    if not insider_table:
        return []

    sales = []
    for row in insider_table.find_all("tr")[1:]:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) >= 7 and cols[3] in {"Sale", "Proposed Sale"}:
            sales.append({
                "name": cols[0], "relationship": cols[1], "date": cols[2],
                "transaction": cols[3],
                "price": cols[4].replace("$", "").replace(",", ""),
                "shares": cols[5].replace(",", ""),
                "value": cols[6].replace("$", "").replace(",", ""),
            })
    return sales[:limit]


def _persist_insider_sales_finviz(conn, ticker, sales):
    if not sales:
        return
    finviz_url = f"https://finviz.com/quote.ashx?t={ticker}"
    cur = conn.cursor()
    for s in sales:
        try:
            txn_date = pd.to_datetime(s["date"]).date().isoformat()
        except (ValueError, TypeError):
            txn_date = s.get("date")

        def _to_float(v):
            try:
                return float(v) if v not in (None, "") else None
            except ValueError:
                return None

        cur.execute(
            """INSERT OR IGNORE INTO insider_trades
                   (ticker, insider_name, title, transaction_type, shares, price, value,
                    transaction_date, filing_date, source_url, created_at, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker, s.get("name"), s.get("relationship"), "SELL", _to_float(s.get("shares")),
             _to_float(s.get("price")), _to_float(s.get("value")), txn_date, None, finviz_url,
             datetime.utcnow().isoformat(), "finviz_scrape"),
        )
    conn.commit()


def cached_insider_sales_finviz(conn, ticker, max_age_hours=24, force_refresh=False):
    """Real-data (Finviz) insider sales, persisted into the shared
    insider_trades table (source='finviz_scrape') alongside the SEC Form 4
    feed -- richer per-ticker coverage than the sparse market-wide Form 4
    sample. This wrapper only triggers the refresh; read insider_trades
    directly (as the rest of the app already does) to see the rows."""
    table = "insider_sales_finviz"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        return {"source": "finviz_scrape", "cache_hit": True, "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        sales = fetch_insider_sales_finviz(ticker)
        _persist_insider_sales_finviz(conn, ticker, sales)
        _log_fetch(conn, table, ticker, True, len(sales))
        return {"source": "finviz_scrape", "cache_hit": False, "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        return {"source": None, "cache_hit": False, "fetched_at": _last_fetch_info(conn, table, ticker)}


def fetch_apewisdom_sentiment(ticker):
    """Real Reddit mention-tracking data from ApeWisdom (apewisdom.io) --
    no API key required, confirmed live. Replaces the former Devvit-based
    Reddit path entirely: that app was blocked indefinitely in Reddit's
    publish-review queue with no ETA (never actually produced data in
    production), while this is a real, already-working free API.

    This is an ATTENTION/MOMENTUM signal, not a sentiment signal --
    ApeWisdom has no per-ticker filter endpoint (confirmed against their
    real API -- only per-subreddit filters exist, e.g. "all-stocks",
    returning a ranked list) and NO POLARITY FIELD at all (confirmed
    against a real response: only rank/mentions/upvotes/24h-deltas exist,
    no bullish/bearish score of any kind). So this pages through the
    "all-stocks" ranking (aggregates r/wallstreetbets, r/stocks,
    r/options, and other finance subreddits) searching for `ticker`, and
    never fabricates a sentiment field that isn't actually there.

    The actually-useful output is the derived `attention_change_pct` --
    the % change in mention volume vs. 24h ago -- since the raw
    mentions/rank counts alone don't say whether attention is spiking or
    collapsing without that comparison. Confirmed live: ~8 pages, ~2.3s
    total for a full scan. Returns None if the ticker isn't currently
    ranked (zero recent mentions -- a real, common, valid state, not an
    error) or on any fetch error."""
    try:
        page = 1
        while True:
            resp = requests.get(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for r in data.get("results", []):
                if (r.get("ticker") or "").upper() == ticker.upper():
                    mentions = r.get("mentions")
                    mentions_24h_ago = r.get("mentions_24h_ago")
                    attention_change_pct = None
                    if mentions is not None and mentions_24h_ago is not None:
                        attention_change_pct = (
                            (mentions - mentions_24h_ago) / max(mentions_24h_ago, 1) * 100
                        )
                    return {
                        "ticker": ticker, "rank": r.get("rank"), "mentions": mentions,
                        "upvotes": r.get("upvotes"), "rank_24h_ago": r.get("rank_24h_ago"),
                        "mentions_24h_ago": mentions_24h_ago,
                        "attention_change_pct": attention_change_pct, "source": "apewisdom",
                    }
            total_pages = data.get("pages", page)
            if page >= total_pages or page >= 10:  # 10-page hard cap regardless of what the API reports
                return None
            page += 1
    except (requests.RequestException, ValueError) as e:
        print(f"[retail_sentiment] ApeWisdom fetch failed for {ticker}: {e}")
        return None


def fetch_quiverquant_wsb(ticker, limit=25):
    """Real WallStreetBets mention data via QuiverQuant's paid tier --
    requires QUIVER_API_KEY AND a subscription tier that includes this
    specific dataset. Confirmed live (2026-08-16): the current key
    returns {"detail": "Upgrade your subscription plan to access this
    dataset."} -- a real paid-tier gate, not a bug, so this returns []
    under the current plan. Hits the raw REST endpoint directly rather
    than quiverquant.quiver().wallstreetbets(), which has the same
    missing-raise_for_status() defect as congress_trading() (confirmed
    live: an upgrade-required response becomes a confusing
    'ValueError: If using all scalar values, you must pass an index'
    instead of a clean failure).

    NOT independently verified against a real success response -- no
    tier that includes this dataset has been available to test against,
    so the field names below (date/ticker/mentions-shaped) are a
    reasonable guess based on QuiverQuant's other endpoints' conventions,
    not confirmed. Revisit and correct field mapping once a real
    successful response is available to inspect."""
    api_key = os.environ.get("QUIVER_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.get(
            "https://api.quiverquant.com/beta/live/wallstreetbets", params={"count_all": "true"},
            headers={"Accept": "application/json", "Authorization": f"Token {api_key}"}, timeout=15,
        )
    except requests.RequestException as e:
        print(f"[retail_sentiment] QuiverQuant WSB fetch failed: {e}")
        return []
    # Check for the {"detail": "..."} error shape BEFORE raise_for_status()
    # -- confirmed live this is how an upgrade-required response actually
    # looks (HTTP 403 + a real, human-readable detail message), and
    # checking first means the log shows that real message instead of
    # just a generic "403 Forbidden".
    try:
        data = resp.json()
    except ValueError:
        print(f"[retail_sentiment] QuiverQuant WSB returned non-JSON (HTTP {resp.status_code})")
        return []
    if isinstance(data, dict) and "detail" in data:
        print(f"[retail_sentiment] QuiverQuant WSB unavailable: {data['detail']}")
        return []
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[retail_sentiment] QuiverQuant WSB fetch failed: {e}")
        return []
    if not isinstance(data, list):
        print(f"[retail_sentiment] QuiverQuant WSB returned unexpected shape: {type(data)}")
        return []
    matches = [row for row in data if (row.get("Ticker") or row.get("ticker") or "").upper() == ticker.upper()]
    return [
        {"source": "quiverquant_wsb", "date": row.get("Date") or row.get("date"),
         "mentions": row.get("Mentions") or row.get("mentions"),
         "rank": row.get("Rank") or row.get("rank")}
        for row in matches[:limit]
    ]


def _fetch_stocktwits_posts(ticker, limit=25):
    """Real StockTwits posts for `ticker` -- ported from new_top.py's
    get_stocktwits_posts. No auth required. Fails soft -- [] on any error.

    Uses _BROWSER_HEADERS (a realistic full browser User-Agent), not a bare
    "Mozilla/5.0" -- StockTwits sits behind Cloudflare, which serves a bot
    challenge page (HTTP 403, "Just a moment...") to that string but a
    normal 200 with real JSON to a full browser UA. Confirmed live.

    Each message carries StockTwits' own real per-message sentiment tag
    (entities.sentiment.basic -- "Bullish"/"Bearish", or absent/null when
    the poster didn't tag it, a real and common state, not an error) as
    `sentiment`. This is the one retail source with actual tagged
    polarity, unlike ApeWisdom (attention-only, no polarity field) --
    see fetch_stocktwits_sentiment() for the aggregated bull/bear read."""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.upper()}.json"
    try:
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=10)
        resp.raise_for_status()
        messages = resp.json().get("messages", [])
    except (requests.RequestException, ValueError):
        return []
    return [
        # StockTwits' API returns `body` with HTML entities un-decoded
        # (e.g. a literal "&amp;" instead of "&", "&#39;" instead of "'"
        # -- confirmed live, visible verbatim in real post text). Decode
        # once here at the source so both the stored DB text and whatever
        # gets fed into the AI Briefing's context bundle are the real
        # characters, not the raw fetcher's rendering-context artifact.
        {"source": "stocktwits", "text": html.unescape(msg.get("body") or ""), "url": None,
         "posted_at": msg.get("created_at"),
         "sentiment": ((msg.get("entities") or {}).get("sentiment") or {}).get("basic")}
        for msg in messages[:limit]
    ]


def _aggregate_stocktwits_sentiment(posts):
    """Pure aggregation: turns a list of StockTwits posts (each carrying a
    real `sentiment` tag, or None when untagged) into a bull/bear ratio
    plus a few representative message excerpts -- the shape the AI
    Briefing's retail_analysis section actually needs, not just a raw
    message count. Prefers tagged messages for the excerpts (more
    informative), backfilling with untagged ones only if there aren't
    enough tagged posts to fill the quota."""
    bullish = [p for p in posts if p.get("sentiment") == "Bullish"]
    bearish = [p for p in posts if p.get("sentiment") == "Bearish"]
    untagged = [p for p in posts if p.get("sentiment") not in ("Bullish", "Bearish")]
    bull_count, bear_count = len(bullish), len(bearish)
    bull_bear_ratio = round(bull_count / bear_count, 2) if bear_count else None
    tagged_first = bullish + bearish + untagged
    representative = [p["text"] for p in tagged_first if p.get("text")][:5]
    return {
        "source": "stocktwits", "message_count": len(posts), "bullish_count": bull_count,
        "bearish_count": bear_count, "untagged_count": len(untagged),
        "bull_bear_ratio": bull_bear_ratio, "representative_messages": representative,
    }


def fetch_stocktwits_sentiment(ticker, limit=30):
    """The PRIMARY tagged-sentiment source for the AI Briefing's retail
    read: fetches real StockTwits messages and aggregates their real
    per-message sentiment tags into a bull/bear ratio plus a few
    representative message excerpts, rather than exposing only a raw
    message count. Live (uncached) convenience wrapper around
    _fetch_stocktwits_posts + _aggregate_stocktwits_sentiment -- the
    cached path (cached_retail_sentiment) persists the same tagged posts
    to retail_sentiment_posts, and _read_ai_context_from_cache aggregates
    from that cached list instead of calling this live, so the two never
    double-fetch."""
    return _aggregate_stocktwits_sentiment(_fetch_stocktwits_posts(ticker, limit=limit))


def fetch_reddit_posts_public(ticker: str, limit: int = 15) -> list:
    """Real Reddit post text via Reddit's unauthenticated public search
    JSON endpoint (r/wallstreetbets/search.json) -- no API key, no OAuth
    app registration, no Devvit publish-review queue to sit in. This is
    the supplementary, richer-narrative leg of the retail pipeline: raw
    titles/bodies for Claude to read and characterize tone from directly
    during synthesis (no TextBlob or other pre-scoring), same treatment
    as the StockTwits messages already handed to it unscored.

    CRITICAL -- on-demand only: this is called exclusively from
    _bundle_ai_context() (the AI Briefing's confirmation-gated,
    on-demand data-gathering step), never from full_refresh() or any
    background/auto-refresh path. That keeps request frequency low and
    reduces the odds Reddit's anti-bot layer rate-limits or blocks this
    IP -- a real risk for an unauthenticated endpoint hit from a
    server/datacenter address (the same class of block that made the
    Senate EFD free fallback unreliable, see fetch_congressional_
    trades' module docs).

    Fails soft: any block/rate-limit/network error returns [] rather
    than raising, so a AI Briefing generation degrades gracefully (the
    RETAIL section falls back to StockTwits + ApeWisdom) instead of
    failing outright. Not persisted to a DB table -- like
    fetch_quiverquant_wsb, it's cheap enough to call live each time and
    persisting would risk serving stale commentary across briefings for
    a source whose whole value is what's being said right now."""
    try:
        resp = requests.get(
            "https://www.reddit.com/r/wallstreetbets/search.json",
            params={"q": ticker, "sort": "new", "limit": limit, "restrict_sr": "on"},
            headers=_BROWSER_HEADERS, timeout=10,
        )
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
    except (requests.RequestException, ValueError) as e:
        print(f"[retail_sentiment] Reddit public JSON blocked/unavailable for {ticker} "
              f"(best-effort, unauthenticated): {e}")
        return []
    posts = []
    for c in children[:limit]:
        d = c.get("data", {})
        permalink = d.get("permalink")
        posts.append({
            "title": d.get("title"), "body": d.get("selftext") or "", "score": d.get("score"),
            "created": d.get("created_utc"),
            "permalink": f"https://www.reddit.com{permalink}" if permalink else None,
            "source": "reddit_public_json",
        })
    return posts


def fetch_retail_sentiment(ticker):
    """Real retail sentiment posts -- StockTwits only, each tagged with
    its source name, original timestamp, and StockTwits' own real
    per-message sentiment tag -- ported from new_top.py's
    get_stocktwits_posts. No sentiment-score guessing beyond that real
    tag (new_top.py's TextBlob-based analyze_sentiment/
    summarize_retail_sentiment were not ported); untagged posts are
    handed to Claude to characterize themselves, per the
    anti-hallucination rules.

    Reddit is sourced separately, in two distinct roles: see
    fetch_apewisdom_sentiment() (real aggregate mention/rank ATTENTION
    data -- no polarity field, so it's never forced into this post-list
    shape) and fetch_reddit_posts_public() (real raw post text, but
    on-demand-only from _bundle_ai_context, never from this
    background-refreshable path). fetch_quiverquant_wsb() is a third,
    bonus WallStreetBets source, paid-tier-gated. All are combined into
    the RETAIL section separately in _bundle_ai_context/generate_deep_analysis."""
    return _fetch_stocktwits_posts(ticker)


def _persist_retail_sentiment(conn, ticker, posts):
    if not posts:
        return
    cur = conn.cursor()
    for p in posts:
        cur.execute(
            """INSERT OR IGNORE INTO retail_sentiment_posts
                   (ticker, source, text, url, posted_at, sentiment_tag, fetched_at) VALUES (?,?,?,?,?,?,?)""",
            (ticker, p.get("source"), p.get("text"), p.get("url"), p.get("posted_at"), p.get("sentiment"),
             datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_retail_sentiment_cache(conn, ticker, limit=30):
    # ORDER BY posted_at (when the post was actually made), not fetched_at
    # (which fetch batch first saw it). INSERT OR IGNORE means a post seen
    # in an earlier 2h poll keeps that poll's fetched_at forever even once
    # newer posts arrive in a later poll -- sorting by fetched_at mixed
    # batches out of chronological order (confirmed live: a batch fetched
    # at 01:41 containing an Aug 15 22:35 post sorted ahead of an earlier
    # batch fetched at 21:21 whose posts were actually from Aug 14).
    # posted_at is a consistent ISO-8601 UTC string ("YYYY-MM-DDTHH:MM:SSZ")
    # for every row, so lexicographic ORDER BY sorts it correctly.
    rows = conn.execute(
        """SELECT source, text, url, posted_at, sentiment_tag FROM retail_sentiment_posts
           WHERE ticker=? ORDER BY posted_at DESC LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    return [{"source": r[0], "text": r[1], "url": r[2], "posted_at": r[3], "sentiment": r[4]} for r in rows]


def cached_retail_sentiment(conn, ticker, max_age_hours=2, force_refresh=False):
    table = "retail_sentiment"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_retail_sentiment_cache(conn, ticker)
        return {"data": cached, "source": "stocktwits" if cached else None,
                "cache_hit": True, "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        posts = fetch_retail_sentiment(ticker)
        _persist_retail_sentiment(conn, ticker, posts)
        _log_fetch(conn, table, ticker, True, len(posts))
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
    cached = _read_retail_sentiment_cache(conn, ticker)
    return {"data": cached, "source": "stocktwits" if cached else None,
            "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, ticker)}


def _persist_apewisdom_sentiment(conn, ticker, record):
    if not record:
        return
    conn.execute(
        """INSERT INTO apewisdom_sentiment
               (ticker, rank, mentions, upvotes, rank_24h_ago, mentions_24h_ago,
                attention_change_pct, source, fetched_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (ticker, record.get("rank"), record.get("mentions"), record.get("upvotes"),
         record.get("rank_24h_ago"), record.get("mentions_24h_ago"),
         record.get("attention_change_pct"), record.get("source"),
         datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_apewisdom_sentiment_cache(conn, ticker):
    row = conn.execute(
        """SELECT rank, mentions, upvotes, rank_24h_ago, mentions_24h_ago,
                  attention_change_pct, source
           FROM apewisdom_sentiment WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    return {"ticker": ticker, "rank": row[0], "mentions": row[1], "upvotes": row[2],
            "rank_24h_ago": row[3], "mentions_24h_ago": row[4],
            "attention_change_pct": row[5], "source": row[6]}


def cached_apewisdom_sentiment(conn, ticker, max_age_hours=2, force_refresh=False):
    """Same cached_*(conn, ticker, max_age_hours, force_refresh) envelope
    as every other fetcher -- unlike marketbeat_institutional, this is
    fast/free/no-auth/no-browser-popup, so it's safe to call automatically
    from _bundle_ai_context on every briefing generation, same cache
    cadence as StockTwits."""
    table = "apewisdom_sentiment"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_apewisdom_sentiment_cache(conn, ticker)
        if cached:
            return {"data": cached, "source": "apewisdom", "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        record = fetch_apewisdom_sentiment(ticker)
        _persist_apewisdom_sentiment(conn, ticker, record)
        _log_fetch(conn, table, ticker, True, 1 if record else 0)
        cached = record or _read_apewisdom_sentiment_cache(conn, ticker)
        return {"data": cached, "source": "apewisdom" if cached else None, "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_apewisdom_sentiment_cache(conn, ticker)
        return {"data": cached, "source": "apewisdom" if cached else None,
                "cache_hit": bool(cached), "fetched_at": _last_fetch_info(conn, table, ticker)}


def _bucket_options_flow(conn, ticker):
    """Groups the latest cached options_flow rows for `ticker` into weekly
    buckets by days-to-expiry, each summarized by call/put volume skew and
    dominant strikes. Pure DB read against options_flow (populated by
    cached_options_flow() elsewhere in the app) -- no new fetch here."""
    latest_date_row = conn.execute(
        "SELECT MAX(fetch_date) FROM options_flow WHERE ticker=?", (ticker,)
    ).fetchone()
    latest_date = latest_date_row[0] if latest_date_row else None
    if not latest_date:
        return []

    rows = conn.execute(
        """SELECT expiration, option_type, strike, volume, open_interest
           FROM options_flow WHERE ticker=? AND fetch_date=?""",
        (ticker, latest_date),
    ).fetchall()
    if not rows:
        return []

    today = date.today()
    bucket_defs = [
        ("this_week", 0, 7), ("next_week", 8, 14),
        ("2_4_weeks", 15, 28), ("leaps_6mo_plus", 180, 100_000),
    ]
    buckets = {name: {"calls_volume": 0, "puts_volume": 0, "strikes": {}} for name, _, _ in bucket_defs}

    for expiration, option_type, strike, volume, _open_interest in rows:
        try:
            dte = (pd.to_datetime(expiration).date() - today).days
        except (ValueError, TypeError):
            continue
        bucket_name = next((n for n, lo, hi in bucket_defs if lo <= dte <= hi), None)
        if bucket_name is None:
            continue
        b = buckets[bucket_name]
        volume = volume or 0
        if option_type == "call":
            b["calls_volume"] += volume
        else:
            b["puts_volume"] += volume
        key = (option_type, strike)
        b["strikes"][key] = b["strikes"].get(key, 0) + volume

    result = []
    for name, _, _ in bucket_defs:
        b = buckets[name]
        total = b["calls_volume"] + b["puts_volume"]
        if total == 0:
            continue
        dominant = sorted(b["strikes"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        result.append({
            "bucket": name,
            "calls_volume": b["calls_volume"],
            "puts_volume": b["puts_volume"],
            "call_put_skew_pct": round(b["calls_volume"] / total * 100, 1),
            "dominant_strikes": [{"type": t, "strike": s, "volume": v} for (t, s), v in dominant],
        })
    return result


def _iv_term_structure(conn, ticker):
    """ATM IV per real expiration date -- reveals IV TERM STRUCTURE
    (elevated near-term IV vs. further-out expiries signals the market
    pricing in a specific near-term event), distinct from
    _bucket_options_flow's named-range buckets (this-week/next-week/etc.)
    since term structure needs actual calendar dates, not ranges. Pure DB
    read against the same latest options_flow snapshot -- no new fetch.

    For each expiration, finds the strike closest to that day's
    underlying_price (same "true ATM, not chain-wide average" methodology
    as calculate_expected_move_iv) and averages that one strike's
    call+put IV -- never a whole-chain average."""
    latest_date_row = conn.execute(
        "SELECT MAX(fetch_date) FROM options_flow WHERE ticker=?", (ticker,)
    ).fetchone()
    latest_date = latest_date_row[0] if latest_date_row else None
    if not latest_date:
        return []

    rows = conn.execute(
        """SELECT expiration, option_type, strike, implied_volatility, underlying_price
           FROM options_flow WHERE ticker=? AND fetch_date=? AND implied_volatility IS NOT NULL""",
        (ticker, latest_date),
    ).fetchall()
    if not rows:
        return []

    today = date.today()
    by_expiry = {}
    for expiration, option_type, strike, iv, underlying_price in rows:
        by_expiry.setdefault(expiration, {"spot": underlying_price, "calls": {}, "puts": {}})
        target = by_expiry[expiration]["calls"] if option_type == "call" else by_expiry[expiration]["puts"]
        target[strike] = iv

    result = []
    for expiration, d in sorted(by_expiry.items()):
        spot = d["spot"]
        if spot is None or not d["calls"] or not d["puts"]:
            continue
        atm_strike = min(d["calls"].keys(), key=lambda s: abs(s - spot))
        call_iv = d["calls"].get(atm_strike)
        put_iv = d["puts"].get(min(d["puts"].keys(), key=lambda s: abs(s - atm_strike)))
        if call_iv is None or put_iv is None:
            continue
        try:
            dte = (pd.to_datetime(expiration).date() - today).days
        except (ValueError, TypeError):
            dte = None
        result.append({
            "expiration": expiration, "dte": dte, "atm_strike": atm_strike,
            "atm_iv_pct": round((call_iv + put_iv) / 2 * 100, 1),
        })
    return result


# --------------------------------------------------------------------------
# Options-flow interpretive labels (Parts 4-6) -- pure functions mapping a
# numeric Greek/IV/liquidity value to a trader-friendly (label, explanation)
# badge. Shared by the OPTIONS FLOW table's badge columns (dashboard.py)
# and find_option_candidates' filter logic (Part 8) so "risk profile"
# means the exact same delta range in both places, not two copies that can
# drift apart.
# --------------------------------------------------------------------------

DELTA_PROFILES = [
    (0.00, 0.10, "Deep OTM lottery ticket",
     "Cheap, high leverage, low probability -- needs a big move to pay off"),
    (0.10, 0.30, "Lottery ticket", "High leverage, low win rate -- speculative directional bet"),
    (0.30, 0.45, "Speculative", "Moderate leverage, needs a real move but has a real chance"),
    (0.45, 0.60, "Balanced",
     "Near coin-flip probability, moderate leverage -- classic swing-trade profile"),
    (0.60, 0.80, "Conservative/ITM", "High probability, lower leverage -- behaves more like the stock"),
    (0.80, 1.00, "Deep ITM / stock replacement",
     "Very high probability, low leverage -- matches the LEAPS scanner's stock-replacement criteria"),
]

IV_PROFILES = [
    (0, 30, "Low IV", "Cheap premium, limited expected move"),
    (30, 60, "Moderate/Balanced IV", "Fair pricing, typical for active names"),
    (60, 100, "Elevated IV",
     "Market pricing in a real catalyst -- buying pays up for volatility that may not materialize"),
    (100, 999, "Extreme IV", "Major move expected -- high IV crush risk even if directionally correct"),
]

VOL_OI_PROFILES = [
    (0, 2, "Normal activity", "Typical day-to-day options activity"),
    (2, 5, "Unusual -- likely new position",
     "Volume well above open interest -- a new position being built, not just existing contracts trading"),
    (5, float("inf"), "Highly unusual -- strong conviction or urgency",
     "Volume many multiples of open interest -- strong conviction or urgency behind this trade"),
]

DTE_PROFILES = [
    (0, 7, "Gamma zone",
     "Extreme, fast-moving gamma exposure near expiry -- price-sensitive, high risk/reward"),
    (8, 30, "Short-term directional", "A near-term directional bet, limited time for the thesis to play out"),
    (31, 90, "Medium-term swing", "Enough time for a real swing-trade thesis to develop"),
    (91, 365, "Longer-term position", "A longer runway -- less exposed to short-term noise"),
    (365, float("inf"), "LEAPS", "Long-dated, behaves more like a leveraged stock position"),
]

THETA_PCT_PROFILES = [
    (0, 1, "Low decay pressure", "Time decay is a minor drag -- fine to hold"),
    (1, 3, "Moderate decay",
     "Meaningful daily decay -- worth watching if the thesis takes time to play out"),
    (3, float("inf"), "High decay -- short-term trade only", "Decays fast -- only makes sense as a short-dated trade"),
]

SPREAD_PROFILES = [
    (0, 5, "Liquid, tight spread", "Low slippage -- efficient to enter and exit"),
    (5, 15, "Moderate liquidity", "Some slippage on entry/exit -- factor it into sizing"),
    (15, float("inf"), "Illiquid -- high slippage risk",
     "Wide spread -- real risk of a bad fill, size down or use limit orders"),
]


def _classify_by_tiers(value, profiles, use_abs=False):
    """value: the raw number to classify (e.g. delta, IV as a %, DTE in
    days). profiles: a *_PROFILES list of (lo, hi, label, note) tuples,
    lo inclusive/hi exclusive. Falls back to the last tier if value is at
    or beyond its lower bound but the tier list's own hi (a literal
    number like 1.00 or 999 for documentation purposes) doesn't quite
    reach it, so an edge-case value (e.g. delta==1.0 exactly) still gets
    labeled instead of falling through unclassified."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None
    v = abs(value) if use_abs else value
    for lo, hi, label, note in profiles:
        if lo <= v < hi:
            return label, note
    if profiles and v >= profiles[-1][0]:
        return profiles[-1][2], profiles[-1][3]
    return None, None


def classify_delta(delta):
    """delta: raw Black-Scholes delta (signed, -1..1 for a put, 0..1 for a
    call) -- classified on |delta| since the DELTA_PROFILES tiers describe
    probability/leverage profile regardless of direction."""
    return _classify_by_tiers(delta, DELTA_PROFILES, use_abs=True)


def classify_iv(iv_pct):
    """iv_pct: implied volatility as a PERCENT (e.g. 42.5, not 0.425)."""
    return _classify_by_tiers(iv_pct, IV_PROFILES)


def classify_vol_oi(ratio):
    return _classify_by_tiers(ratio, VOL_OI_PROFILES)


def classify_dte(days):
    return _classify_by_tiers(days, DTE_PROFILES)


def classify_theta_pct(theta_pct_per_day):
    """theta_pct_per_day: |theta| / contract premium * 100, i.e. what
    fraction of the option's own price decays per day -- classified on
    magnitude since theta itself is always <=0 for a long option."""
    return _classify_by_tiers(theta_pct_per_day, THETA_PCT_PROFILES, use_abs=True)


def classify_spread_pct(spread_pct):
    return _classify_by_tiers(spread_pct, SPREAD_PROFILES)


def compute_spread_pct(bid, ask, last_price=None):
    """(ask-bid)/mid * 100, mid=(bid+ask)/2 -- real bid/ask spread, not a
    proxy. Falls back to last_price as the mid-point only when bid/ask
    are both missing/zero (no active market, e.g. a stale far-OTM
    contract), same mid-price fallback convention as
    build_itm_calls_table. Returns None (never fabricates a number) if
    neither bid/ask nor last_price is usable."""
    bid = _none_if_nan(bid)
    ask = _none_if_nan(ask)
    if bid is not None and ask is not None and (bid > 0 or ask > 0):
        mid = (bid + ask) / 2.0
        if mid > 0:
            return round((ask - bid) / mid * 100, 2)
    last_price = _none_if_nan(last_price)
    if last_price and last_price > 0 and bid is not None and ask is not None:
        return round((ask - bid) / last_price * 100, 2)
    return None


def annotate_options_badges(df):
    """Adds every derived interpretive field (the classify_*() badges from
    Parts 4-6, plus the raw numbers they're computed from: dte, iv_pct,
    spread_pct, theta_pct_per_day) to a COPY of `df`. Single source of
    truth for per-row badge computation -- used by the OPTIONS FLOW
    table's badge column, the synthesis panel (synthesize_options_flow_
    summary), and the row-level insight sentences (generate_row_insight),
    so all three can never disagree about what badge a given contract
    gets. `df` needs option_type/strike/expiration/delta/
    implied_volatility/volume/open_interest/volume_oi_ratio/theta/
    last_price/bid/ask -- a missing/NaN input just produces a None badge
    for that row (never an error), same fail-soft convention as the
    underlying classify_*() calls."""
    if df.empty:
        return df.copy()
    out = df.copy()
    today = date.today()

    def _dte(exp):
        try:
            return (pd.to_datetime(exp).date() - today).days
        except (ValueError, TypeError):
            return None

    out["dte"] = out["expiration"].apply(_dte)
    out["iv_pct"] = out["implied_volatility"].apply(lambda v: v * 100 if pd.notna(v) else None)
    out["spread_pct"] = out.apply(
        lambda r: compute_spread_pct(r.get("bid"), r.get("ask"), r.get("last_price")), axis=1
    )
    out["theta_pct_per_day"] = out.apply(
        lambda r: (abs(r["theta"]) / r["last_price"] * 100)
        if pd.notna(r.get("theta")) and pd.notna(r.get("last_price")) and r["last_price"] > 0 else None,
        axis=1,
    )
    out["delta_badge"] = out["delta"].apply(lambda v: classify_delta(v)[0])
    out["iv_badge"] = out["iv_pct"].apply(lambda v: classify_iv(v)[0])
    out["vol_oi_badge"] = out["volume_oi_ratio"].apply(lambda v: classify_vol_oi(v)[0])
    out["dte_badge"] = out["dte"].apply(lambda v: classify_dte(v)[0])
    out["theta_badge"] = out["theta_pct_per_day"].apply(lambda v: classify_theta_pct(v)[0])
    out["spread_badge"] = out["spread_pct"].apply(lambda v: classify_spread_pct(v)[0])
    return out


# --------------------------------------------------------------------------
# Row-level and aggregate options-flow insight (Parts 2-3) -- deterministic,
# template-based synthesis over the classify_*() badges above. NOT an LLM
# call: every sentence is picked from a fixed phrase table keyed on real
# computed values, so this renders instantly and for free on every page
# load. Feeds both the dashboard's synthesis panel/row insights AND
# generate_deep_analysis's context bundle (see options_flow_synthesis in
# _bundle_ai_context) so the AI Briefing references the same badge-driven
# read shown mechanically in the table, rather than deriving a second,
# potentially inconsistent one from the raw numbers alone.
# --------------------------------------------------------------------------

_DELTA_INSIGHT_PHRASES = {
    "Deep OTM lottery ticket": "an extremely aggressive, high-leverage, low-probability bet",
    "Lottery ticket": "an aggressive, high-leverage, low-probability bet",
    "Speculative": "a speculative directional bet with real, if modest, odds",
    "Balanced": "a classic, roughly coin-flip directional bet",
    "Conservative/ITM": "a high-probability, lower-leverage bet that behaves more like the stock",
    "Deep ITM / stock replacement": "a stock-replacement position more than a speculative bet",
}

_VOL_OI_INSIGHT_PHRASES = {
    "Normal activity": "in line with ordinary day-to-day activity",
    "Unusual -- likely new position": "as part of what looks like a new position being actively built",
    "Highly unusual -- strong conviction or urgency": "with unusually high urgency and conviction behind it",
}

_DTE_INSIGHT_PHRASES = {
    "Gamma zone": "that needs to play out within days",
    "Short-term directional": "that needs to play out within the next few weeks",
    "Medium-term swing": "with a couple of months for the thesis to develop",
    "Longer-term position": "with a long runway for the thesis to develop",
    "LEAPS": "on a multi-year, stock-like time horizon",
}

_IV_INSIGHT_PHRASES = {
    "Low IV": "priced cheaply, with the market not expecting much movement",
    "Moderate/Balanced IV": "priced fairly typically for an active name",
    "Elevated IV": "priced expensively, with the market bracing for a real event",
    "Extreme IV": "priced for a huge move, with real risk of an IV crush even if the direction is right",
}


def generate_row_insight(row):
    """Deterministic, template-based one-line insight (Part 3) combining
    one contract's own badges (delta profile + IV level + vol/OI level +
    DTE category) into a plain-language sentence -- never free-form
    generation. `row` (a dict or pandas Series) needs option_type/strike/
    expiration plus the *_badge columns annotate_options_badges() adds.
    Returns None if there's no delta badge to build a sentence around."""
    delta_badge = row.get("delta_badge")
    if not delta_badge:
        return None
    iv_badge = row.get("iv_badge")
    vol_oi_badge = row.get("vol_oi_badge")
    dte_badge = row.get("dte_badge")

    option_type = (row.get("option_type") or "").capitalize()
    strike = row.get("strike")
    try:
        expiry_human = pd.to_datetime(row.get("expiration")).strftime("%b %d")
    except (ValueError, TypeError):
        expiry_human = str(row.get("expiration"))

    badge_combo = " + ".join(b for b in [delta_badge, iv_badge, vol_oi_badge, dte_badge] if b)

    clauses = [f"someone is making {_DELTA_INSIGHT_PHRASES.get(delta_badge, 'a directional bet')}"]
    if vol_oi_badge in _VOL_OI_INSIGHT_PHRASES:
        clauses.append(_VOL_OI_INSIGHT_PHRASES[vol_oi_badge])
    if iv_badge in _IV_INSIGHT_PHRASES:
        clauses.append(f"in a contract that's {_IV_INSIGHT_PHRASES[iv_badge]}")
    if dte_badge in _DTE_INSIGHT_PHRASES:
        clauses.append(_DTE_INSIGHT_PHRASES[dte_badge])
    body = ", ".join(clauses) + "."

    if delta_badge in ("Deep OTM lottery ticket", "Lottery ticket") and dte_badge == "Gamma zone":
        risk_note = " High risk -- needs to be right very soon."
    elif delta_badge in ("Deep OTM lottery ticket", "Lottery ticket"):
        risk_note = " High risk."
    elif delta_badge in ("Conservative/ITM", "Deep ITM / stock replacement"):
        risk_note = " Lower risk, more like owning the stock."
    else:
        risk_note = ""

    strike_txt = f"${strike:g}" if strike is not None and pd.notna(strike) else "?"
    return f"{strike_txt} {option_type}, {expiry_human} — {badge_combo}: {body}{risk_note}"


_DELTA_DOMINANT_SENTENCES = {
    "Deep OTM lottery ticket": (
        "Most activity is in Deep OTM Lottery Ticket territory — traders are making very cheap, "
        "very long-shot bets."
    ),
    "Lottery ticket": (
        "Most activity today is in Lottery Ticket territory — traders are making cheap, "
        "speculative bets, not high-conviction ones."
    ),
    "Speculative": (
        "Most activity is in Speculative territory — traders think a real move is coming but "
        "aren't betting the farm on it."
    ),
    "Balanced": "Most activity is in Balanced territory — this looks like classic, textbook directional trading.",
    "Conservative/ITM": (
        "Most activity is in Conservative/ITM territory — traders are positioning for "
        "high-probability, stock-like exposure."
    ),
    "Deep ITM / stock replacement": (
        "Most activity is in Deep ITM / Stock Replacement territory — traders are using options "
        "as a cheaper substitute for owning the stock outright."
    ),
}

_IV_DOMINANT_SENTENCES = {
    "Low IV": "IV is Low across most contracts — the market isn't pricing in anything unusual right now.",
    "Moderate/Balanced IV": "IV is Moderate/Balanced across most contracts — fairly typical options pricing.",
    "Elevated IV": "IV is Elevated across most contracts — the market is pricing in a real event{earnings_clause}.",
    "Extreme IV": (
        "IV is Extreme across most contracts — the market expects a huge move, with real risk of "
        "an IV crush right after the event{earnings_clause}."
    ),
}

# (dominant_side, dominant-delta-badge-on-that-side) -> reading. Not
# exhaustive of every theoretically possible combination -- a fixed
# decision table covering the meaningful cases, with a neutral fallback
# for anything else, exactly as "rules-based, not exhaustive" implies.
_SKEW_RULES = {
    ("call", "Deep OTM lottery ticket"): "this reads as retail chasing a bounce, not institutional positioning",
    ("call", "Lottery ticket"): "this reads as retail chasing a bounce, not institutional positioning",
    ("call", "Speculative"): "this reads as bullish conviction building, but not yet fully committed",
    ("call", "Balanced"): "this reads as real directional conviction on the upside",
    ("call", "Conservative/ITM"): (
        "this reads as high-conviction bullish positioning, closer to stock-like exposure"
    ),
    ("call", "Deep ITM / stock replacement"): (
        "this reads as investors using calls as a stock substitute, not speculation"
    ),
    ("put", "Deep OTM lottery ticket"): "this reads as cheap, speculative bets on a drop, not serious hedging",
    ("put", "Lottery ticket"): "this reads as cheap downside lottery tickets, not serious hedging",
    ("put", "Speculative"): "this reads as bearish conviction building, but not yet aggressive",
    ("put", "Balanced"): "this looks like real downside hedging, not speculation",
    ("put", "Conservative/ITM"): "this looks like real downside hedging, not speculation",
    ("put", "Deep ITM / stock replacement"): "this reads as protective positioning, closer to a synthetic short",
}


def synthesize_options_flow_summary(df, spot=None, earnings_date=None):
    """Rules-based, deterministic, instant (no LLM, no network call)
    summary of what a snapshot of options_flow rows is actually saying --
    Part 2. `df` must already carry the *_badge columns from
    annotate_options_badges() -- an un-annotated frame returns the empty
    result rather than silently computing something wrong. Every sentence
    is picked from a fixed template table keyed on real computed values
    (dominant badge by volume, call/put skew, moneyness vs. `spot`) --
    never freeform generation, so this is safe to call on every render.

    Returns {"dominant_delta", "dominant_iv", "skew_reading",
    "top_unusual"} -- each either None (nothing to say, e.g. no volume)
    or a dict with at least a "sentence" key."""
    empty = {"dominant_delta": None, "dominant_iv": None, "skew_reading": None, "top_unusual": None}
    if df.empty or "delta_badge" not in df.columns:
        return empty

    valid = df[df["volume"].notna() & (df["volume"] > 0)]
    if valid.empty:
        return empty
    total_volume = valid["volume"].sum()

    def _dominant_badge(col):
        by_badge = valid.dropna(subset=[col])["volume"].groupby(valid[col]).sum()
        if by_badge.empty:
            return None, None
        label = by_badge.idxmax()
        share_pct = round(by_badge.max() / total_volume * 100, 1) if total_volume else None
        return label, share_pct

    dom_delta_label, dom_delta_share = _dominant_badge("delta_badge")
    dominant_delta = None
    if dom_delta_label:
        dominant_delta = {
            "label": dom_delta_label, "share_pct": dom_delta_share,
            "sentence": _DELTA_DOMINANT_SENTENCES.get(
                dom_delta_label, f"Most activity is in {dom_delta_label} territory."
            ),
        }

    dom_iv_label, dom_iv_share = _dominant_badge("iv_badge")
    dominant_iv = None
    if dom_iv_label:
        earnings_clause = ""
        if dom_iv_label in ("Elevated IV", "Extreme IV") and earnings_date:
            try:
                days_out = (pd.Timestamp(earnings_date) - pd.Timestamp.now()).days
                if 0 <= days_out <= 30:
                    earnings_clause = f", likely the earnings date on {pd.Timestamp(earnings_date).strftime('%b %d')}"
            except (TypeError, ValueError):
                pass
        template = _IV_DOMINANT_SENTENCES.get(dom_iv_label, f"IV is {dom_iv_label} across most contracts.")
        dominant_iv = {
            "label": dom_iv_label, "share_pct": dom_iv_share,
            "sentence": template.format(earnings_clause=earnings_clause),
        }

    call_volume = valid.loc[valid["option_type"] == "call", "volume"].sum()
    put_volume = valid.loc[valid["option_type"] == "put", "volume"].sum()
    skew_reading = None
    if call_volume + put_volume > 0:
        dominant_side = "call" if call_volume >= put_volume else "put"
        side_df = valid[valid["option_type"] == dominant_side]
        side_share_pct = round(
            (call_volume if dominant_side == "call" else put_volume) / (call_volume + put_volume) * 100, 0
        )
        by_badge = side_df.dropna(subset=["delta_badge"])["volume"].groupby(side_df["delta_badge"]).sum()
        if not by_badge.empty:
            side_label = by_badge.idxmax()
            moneyness = ""
            if spot is not None and pd.notna(spot) and spot > 0:
                sub = side_df[side_df["delta_badge"] == side_label]
                avg_strike = sub["strike"].mean()
                pct_from_spot = (avg_strike - spot) / spot * 100
                if abs(pct_from_spot) <= 2:
                    moneyness = "near-the-money "
                elif (dominant_side == "call" and pct_from_spot > 2) or (
                    dominant_side == "put" and pct_from_spot < -2
                ):
                    moneyness = "OTM "
                else:
                    moneyness = "ITM "
            rule = _SKEW_RULES.get((dominant_side, side_label), "the positioning here is mixed")
            side_txt = "calls" if dominant_side == "call" else "puts"
            skew_reading = {
                "dominant_side": dominant_side, "share_pct": side_share_pct, "delta_badge": side_label,
                "sentence": f"{side_share_pct:.0f}% of volume is in {moneyness}{side_txt} rated {side_label} — {rule}.",
            }

    top_unusual = None
    unusual_pool = valid[valid["unusual"] == 1] if "unusual" in valid.columns and (valid["unusual"] == 1).any() \
        else valid
    if not unusual_pool.empty and unusual_pool["volume_oi_ratio"].notna().any():
        top_row = unusual_pool.loc[unusual_pool["volume_oi_ratio"].idxmax()]
        insight = generate_row_insight(top_row)
        if insight:
            top_unusual = {
                "sentence": f"The single most unusual trade: {insight}",
                "strike": top_row.get("strike"), "type": top_row.get("option_type"),
                "expiration": top_row.get("expiration"), "volume_oi_ratio": top_row.get("volume_oi_ratio"),
            }

    return {
        "dominant_delta": dominant_delta, "dominant_iv": dominant_iv,
        "skew_reading": skew_reading, "top_unusual": top_unusual,
    }


# --------------------------------------------------------------------------
# Option Picker (Part 8) -- "Find an Option" tool for TICKER DEEP-DIVE.
# Pure filter/rank over the same options_flow snapshot _bucket_options_flow
# and _iv_term_structure already read; never fetches -- the caller (the
# dashboard) is responsible for having called cached_options_flow first.
# --------------------------------------------------------------------------

RISK_PROFILE_DELTA_RANGES = {
    # Reuses DELTA_PROFILES' exact tier boundaries (Part 4) so a contract
    # labeled "Balanced" on the OPTIONS FLOW table is exactly what
    # risk_profile="Balanced" returns here -- one definition, not two.
    # Deliberately excludes the 0.00-0.10 "Deep OTM lottery ticket" tier
    # even from "Speculative" -- a near-worthless contract isn't something
    # a picker tool should recommend by default.
    "Conservative": (0.60, 1.00),   # Conservative/ITM + Deep ITM tiers
    "Balanced": (0.45, 0.60),       # Balanced tier
    "Speculative": (0.10, 0.45),    # Lottery ticket + Speculative tiers
}

TIME_HORIZON_DTE_RANGES = {
    "This week": (0, 7),
    "Next few weeks": (8, 30),
    "1-3 months": (31, 90),
    # Diverges from DTE_PROFILES' 365+ "LEAPS" boundary on purpose -- this
    # picker option is literally labeled "6+ months", so it starts there
    # (~180 days), not at 365; the per-contract DTE badge still correctly
    # shows "Longer-term position" vs. "LEAPS" within these results.
    "6+ months (LEAPS)": (180, 100_000),
}

# How many expirations cached_options_flow needs to pull to have any real
# chance of covering a given horizon -- yfinance lists expirations nearest-
# first, and weeklies dominate the front of that list, so reaching even
# ~90 days out on a busy name can take a dozen-plus entries. The dashboard's
# "Find Matches" button passes this straight through to cached_options_flow
# (with force_refresh=True) rather than relying on whatever a different
# tab's narrower default-4 fetch happened to already cache today.
TIME_HORIZON_MAX_EXPIRATIONS = {
    "This week": 6,
    "Next few weeks": 8,
    "1-3 months": 14,
    "6+ months (LEAPS)": 30,
}

OPTION_PICKER_MIN_OPEN_INTEREST = 25
OPTION_PICKER_MAX_SPREAD_PCT = 15.0  # matches SPREAD_PROFILES' "Illiquid" cutoff


def find_option_candidates(conn, ticker, direction, risk_profile, time_horizon,
                            max_budget=None, top_n=5):
    """Filters the latest cached options_flow snapshot for `ticker` down to
    contracts matching the picker's four inputs, ranks by liquidity (open
    interest desc, then tightest spread), and returns the top `top_n` with
    every Part 4-6 interpretive badge attached.

    direction: "Bullish" (-> calls) or "Bearish" (-> puts).
    risk_profile: one of RISK_PROFILE_DELTA_RANGES' keys.
    time_horizon: one of TIME_HORIZON_DTE_RANGES' keys.
    max_budget: optional max cost per contract in dollars (last_price*100).

    If nothing matches, progressively relaxes constraints one at a time --
    liquidity (min OI / max spread) first, then budget -- and reports
    exactly what was relaxed, rather than silently returning an empty list
    or silently ignoring the user's stated constraints."""
    if direction not in ("Bullish", "Bearish"):
        raise ValueError(f"direction must be 'Bullish' or 'Bearish', got {direction!r}")
    if risk_profile not in RISK_PROFILE_DELTA_RANGES:
        raise ValueError(f"risk_profile must be one of {list(RISK_PROFILE_DELTA_RANGES)}, got {risk_profile!r}")
    if time_horizon not in TIME_HORIZON_DTE_RANGES:
        raise ValueError(f"time_horizon must be one of {list(TIME_HORIZON_DTE_RANGES)}, got {time_horizon!r}")

    option_type = "call" if direction == "Bullish" else "put"
    delta_lo, delta_hi = RISK_PROFILE_DELTA_RANGES[risk_profile]
    dte_lo, dte_hi = TIME_HORIZON_DTE_RANGES[time_horizon]

    latest_date_row = conn.execute(
        "SELECT MAX(fetch_date) FROM options_flow WHERE ticker=?", (ticker,)
    ).fetchone()
    latest_date = latest_date_row[0] if latest_date_row else None
    if not latest_date:
        return {"candidates": [], "relaxed": [],
                "note": "No options data cached for this ticker yet -- click 'Fetch live signals'."}

    df = pd.read_sql_query(
        """SELECT expiration, strike, option_type, volume, open_interest, volume_oi_ratio,
                  implied_volatility, last_price, underlying_price, delta, gamma, theta, vega, bid, ask
           FROM options_flow WHERE ticker=? AND fetch_date=? AND option_type=?""",
        conn, params=(ticker, latest_date, option_type),
    )
    if df.empty:
        return {"candidates": [], "relaxed": [],
                "note": f"No {option_type} contracts cached for this ticker yet -- click 'Fetch live signals'."}

    today = date.today()
    df["dte"] = df["expiration"].apply(
        lambda e: (pd.to_datetime(e).date() - today).days if pd.notna(e) else None
    )
    df["spread_pct"] = df.apply(lambda r: compute_spread_pct(r["bid"], r["ask"], r["last_price"]), axis=1)
    df["abs_delta"] = df["delta"].abs()
    df["cost_per_contract"] = df["last_price"] * 100

    def _apply(frame, use_liquidity, use_budget):
        f = frame[
            frame["delta"].notna()
            & (frame["abs_delta"] >= delta_lo) & (frame["abs_delta"] < delta_hi)
            & frame["dte"].notna() & (frame["dte"] >= dte_lo) & (frame["dte"] <= dte_hi)
        ]
        if use_liquidity:
            f = f[
                (f["open_interest"].fillna(0) >= OPTION_PICKER_MIN_OPEN_INTEREST)
                & (f["spread_pct"].fillna(999) <= OPTION_PICKER_MAX_SPREAD_PCT)
            ]
        if use_budget and max_budget:
            f = f[f["cost_per_contract"].notna() & (f["cost_per_contract"] <= max_budget)]
        return f

    relaxed = []
    matches = _apply(df, use_liquidity=True, use_budget=True)
    if matches.empty:
        relaxed.append("liquidity (min open interest / max spread)")
        matches = _apply(df, use_liquidity=False, use_budget=True)
    if matches.empty and max_budget:
        relaxed.append("budget")
        matches = _apply(df, use_liquidity=False, use_budget=False)

    if matches.empty:
        return {
            "candidates": [], "relaxed": relaxed,
            "note": (f"No {option_type} contracts found with delta {delta_lo:.2f}-{delta_hi:.2f} and "
                     f"{dte_lo}-{dte_hi} DTE for {ticker}, even after relaxing "
                     f"{', '.join(relaxed) if relaxed else 'every optional constraint'}."),
        }

    matches = matches.sort_values(["open_interest", "spread_pct"], ascending=[False, True], na_position="last")

    candidates = []
    for _, r in matches.head(top_n).iterrows():
        iv_pct = r["implied_volatility"] * 100 if pd.notna(r["implied_volatility"]) else None
        theta_pct = (
            abs(r["theta"]) / r["last_price"] * 100
            if pd.notna(r["theta"]) and pd.notna(r["last_price"]) and r["last_price"] > 0 else None
        )
        breakeven = (
            (r["strike"] + r["last_price"]) if option_type == "call" else (r["strike"] - r["last_price"])
        ) if pd.notna(r["last_price"]) else None
        delta_label, delta_note = classify_delta(r["delta"])
        iv_label, iv_note = classify_iv(iv_pct)
        vol_oi_label, vol_oi_note = classify_vol_oi(r["volume_oi_ratio"])
        dte_label, dte_note = classify_dte(r["dte"])
        theta_label, theta_note = classify_theta_pct(theta_pct)
        spread_label, spread_note = classify_spread_pct(r["spread_pct"])
        candidates.append({
            "expiration": r["expiration"], "strike": float(r["strike"]), "type": option_type,
            "delta": _none_if_nan(r["delta"]), "iv_pct": _none_if_nan(iv_pct),
            "last_price": _none_if_nan(r["last_price"]), "cost_per_contract": _none_if_nan(r["cost_per_contract"]),
            "breakeven": _none_if_nan(breakeven),
            "open_interest": int(r["open_interest"]) if pd.notna(r["open_interest"]) else None,
            "volume": int(r["volume"]) if pd.notna(r["volume"]) else None,
            "spread_pct": _none_if_nan(r["spread_pct"]), "dte": int(r["dte"]) if pd.notna(r["dte"]) else None,
            "badges": {
                "delta": {"label": delta_label, "note": delta_note},
                "iv": {"label": iv_label, "note": iv_note},
                "vol_oi": {"label": vol_oi_label, "note": vol_oi_note},
                "dte": {"label": dte_label, "note": dte_note},
                "theta": {"label": theta_label, "note": theta_note},
                "spread": {"label": spread_label, "note": spread_note},
            },
        })

    return {"candidates": candidates, "relaxed": relaxed, "note": None}


# --------------------------------------------------------------------------
# EARNINGS SIMULATOR -- combines live market-pricing signals (this
# section), the AI Briefing's own qualitative synthesis (Part 2 below),
# and sample-size-honest historical evidence (earnings_history_real) into
# a two-phase probability engine. See CLAUDE.md's "EARNINGS SIMULATOR"
# section for the full design; each function's docstring covers its own
# piece.
# --------------------------------------------------------------------------

def get_market_pricing_signals(ticker, conn):
    """Live, real-time-priced signals (Part 3) -- current market prices,
    not historical inference, so these are valid from day one with no
    sample-size gating (unlike the historical-evidence side of the
    engine). Pure DB read against whatever's already cached -- no new
    fetch. Returns descriptive market state only; no probability is
    computed here, this feeds INTO estimate_earnings_probability() as
    always-available evidence.

    - iv_implied_move_pct: the IV-based expected move (calculate_
      expected_move_iv, via the cached earnings_signal row) expressed as
      a % of current price -- the market's own probability-weighted
      estimate of the move size, not derived from thin historical
      price-reaction averages.
    - iv_rank: current ATM IV's percentile within this ticker's own
      trailing-90-day ticker_snapshots.iv_snapshot history. None (with a
      note) if fewer than 5 historical snapshots exist -- too thin for a
      reliable rank, same "say so, don't guess" convention as everywhere
      else in this app.
    - skew: 25-delta-ish OTM put IV minus OTM call IV (nearest strikes to
      |delta|=0.25 in the nearest cached expiry) -- a real skew read, not
      a fixed-strike proxy.
    - gamma_concentration: top 3 strikes by |gamma * open_interest * 100|
      across the full cached chain -- same math as render_gamma_exposure_
      chart, computed here as data rather than a plot."""
    earnings_signal = _read_earnings_signal_cache(conn, ticker)
    current_price = (earnings_signal or {}).get("current_price")
    atm_iv_pct = (earnings_signal or {}).get("atm_avg_iv_pct")
    iv_expected_move_usd = (earnings_signal or {}).get("iv_expected_move_usd")

    iv_implied_move_pct = None
    if iv_expected_move_usd is not None and current_price:
        iv_implied_move_pct = round(iv_expected_move_usd / current_price * 100, 2)

    iv_rank = None
    if atm_iv_pct is not None:
        hist_rows = conn.execute(
            """SELECT iv_snapshot FROM ticker_snapshots
               WHERE ticker=? AND iv_snapshot IS NOT NULL AND fetched_at >= datetime('now', '-90 day')
               ORDER BY fetched_at""",
            (ticker,),
        ).fetchall()
        ivs = [r[0] for r in hist_rows]
        if len(ivs) >= 5:
            below_or_equal = sum(1 for v in ivs if v <= atm_iv_pct)
            iv_rank = {
                "percentile": round(below_or_equal / len(ivs) * 100, 1), "sample_size": len(ivs),
                "current_iv_pct": atm_iv_pct, "range_low": round(min(ivs), 1), "range_high": round(max(ivs), 1),
            }
        else:
            iv_rank = {
                "percentile": None, "sample_size": len(ivs),
                "note": f"Only {len(ivs)} historical IV snapshot(s) on record -- too thin for a reliable IV rank.",
            }

    skew = None
    gamma_concentration = []
    latest_date_row = conn.execute(
        "SELECT MAX(fetch_date) FROM options_flow WHERE ticker=?", (ticker,)
    ).fetchone()
    latest_date = latest_date_row[0] if latest_date_row else None
    if latest_date:
        opt_df = pd.read_sql_query(
            """SELECT option_type, strike, expiration, delta, implied_volatility, gamma, open_interest
               FROM options_flow WHERE ticker=? AND fetch_date=?""",
            conn, params=(ticker, latest_date),
        )
        if not opt_df.empty:
            nearest_expiry = min(opt_df["expiration"].unique())
            near = opt_df[opt_df["expiration"] == nearest_expiry]
            calls = near[(near["option_type"] == "call") & near["delta"].notna()]
            puts = near[(near["option_type"] == "put") & near["delta"].notna()]
            if not calls.empty and not puts.empty:
                call_row = calls.loc[(calls["delta"] - 0.25).abs().idxmin()]
                put_row = puts.loc[(puts["delta"] - (-0.25)).abs().idxmin()]
                if pd.notna(call_row["implied_volatility"]) and pd.notna(put_row["implied_volatility"]):
                    skew = {
                        "put_iv_pct": round(put_row["implied_volatility"] * 100, 1),
                        "call_iv_pct": round(call_row["implied_volatility"] * 100, 1),
                        "skew_pts": round((put_row["implied_volatility"] - call_row["implied_volatility"]) * 100, 1),
                        "expiration": nearest_expiry,
                    }
            g = opt_df.dropna(subset=["gamma", "open_interest"]).copy()
            if not g.empty:
                sign = g["option_type"].map({"call": 1, "put": -1}).fillna(0)
                g["gamma_exposure"] = g["gamma"] * g["open_interest"] * 100 * sign
                by_strike = g.groupby("strike")["gamma_exposure"].sum()
                top = by_strike.reindex(by_strike.abs().sort_values(ascending=False).index).head(3)
                gamma_concentration = [
                    {"strike": float(s), "net_gamma_exposure": round(float(v), 1)} for s, v in top.items()
                ]

    return {
        "ticker": ticker, "current_price": current_price, "iv_implied_move_pct": iv_implied_move_pct,
        "atm_iv_pct": atm_iv_pct, "iv_rank": iv_rank, "skew": skew, "gamma_concentration": gamma_concentration,
    }


# --------------------------------------------------------------------------
# Part 2 -- structured signal extraction from the AI Briefing's OWN
# synthesis. Deterministic keyword/pattern matching, not a second LLM
# call -- the whole point is to read what Claude already wrote in the
# Briefing (catalysts[].why_it_matters, institutional_analysis,
# earnings_track_record), never to re-derive a separate, potentially
# inconsistent qualitative judgment from the raw numbers underneath it.
# --------------------------------------------------------------------------

_RISK_DAMPENING_KEYWORDS = [
    "fraud", "investigation", "lawsuit", "litigation", " sec ", " doj", " ftc", "probe",
    "regulatory", "regulator", "sanction", "antitrust", "tariff", "export control",
    "supply chain", "overhang", "pressure", "headwind", "concern", "risk of", "scrutiny",
    "decline", "slowdown", "weakness", "downgrade", "warning", "delist",
]
_RISK_SUPPORTING_KEYWORDS = [
    "launch", "expansion", "partnership", "growth", "new channel", "contract win",
    "approval", "upgrade", "acquisition", "record", "beat", "strong demand", "tailwind",
    "opportunity", "catalyst for upside", "accelerat",
]


def _classify_catalyst_risk(why_it_matters):
    """RISK-DAMPENING / RISK-SUPPORTING / NEUTRAL, read from the
    Briefing's own characterization of why a catalyst matters (its
    why_it_matters text) -- never re-classified independently of what
    Claude actually wrote."""
    text = (why_it_matters or "").lower()
    dampening_hits = sum(1 for kw in _RISK_DAMPENING_KEYWORDS if kw in text)
    supporting_hits = sum(1 for kw in _RISK_SUPPORTING_KEYWORDS if kw in text)
    if dampening_hits > supporting_hits:
        return "RISK-DAMPENING"
    if supporting_hits > dampening_hits:
        return "RISK-SUPPORTING"
    return "NEUTRAL"


_MACRO_OVERHANG_KEYWORDS = [
    "tariff", "export control", "supply chain", "litigation", "lawsuit", "sec investigation",
    "doj", "ftc", "regulatory", "regulator", "antitrust", "sanctions", "geopolitical",
    "recession", "rate hike", "interest rate", "fraud",
]


def _extract_macro_overhang(text_blob):
    """Named macro/sector/legal/regulatory risk explicitly mentioned in
    the Briefing's own text -- NOT derivable from price/volume data, so
    this only ever comes from the Briefing's prose. Returns the actual
    sentence(s) containing a hit, not just the keyword, so the caller can
    show real context rather than a bare label."""
    text = text_blob or ""
    lower = text.lower()
    hits = []
    for kw in _MACRO_OVERHANG_KEYWORDS:
        idx = lower.find(kw)
        if idx == -1:
            continue
        start = lower.rfind(".", 0, idx) + 1
        end = lower.find(".", idx)
        end = end if end != -1 else len(text)
        sentence = text[start:end].strip(" \n-")
        if sentence and sentence not in hits:
            hits.append(sentence)
    return hits


def _extract_buyback_signal(institutional_analysis_text):
    """Buyback trend/magnitude, read from the Briefing's own
    institutional_analysis bullets -- respects the Briefing's own
    'meaningful vs. token' judgment (per Part 2's spec) rather than
    treating any positive dollar figure as automatically bullish; a
    dollar amount with no qualifying language classifies as
    'unspecified', not 'meaningful'."""
    text = institutional_analysis_text or ""
    lines = [
        l.strip("-• ").strip() for l in text.split("\n")
        if "buyback" in l.lower() or "repurchase" in l.lower()
    ]
    if not lines:
        return {"present": False, "magnitude": None, "trend": None, "sentence": None}
    joined = " ".join(lines).lower()
    if any(w in joined for w in ("token", "modest", "small", "minor", "limited", "de minimis")):
        magnitude = "token"
    elif any(w in joined for w in ("meaningful", "significant", "large", "substantial", "aggressive")):
        magnitude = "meaningful"
    else:
        magnitude = "unspecified"
    if any(w in joined for w in ("accelerat", "increas", "stepping up", "ramping", "expanded")):
        trend = "accelerating"
    elif any(w in joined for w in ("decelerat", "slow", "pause", "paused", "halted", "suspend", "cut")):
        trend = "decelerating"
    else:
        trend = "steady"
    return {"present": True, "magnitude": magnitude, "trend": trend, "sentence": lines[0]}


_BEATS_DONT_PAY_KEYWORDS = [
    "beats don't pay", "beat that doesn't pay", "beats that don't pay", "sold off", "sell the news",
    "sell-the-news", "declined despite", "fell despite", "dropped despite", "negative reaction to beats",
    "punished", "didn't reward", "failed to reward", "sold despite",
]


def _detect_beats_dont_pay_pattern(earnings_track_record_text, history_rows):
    """Whether this ticker shows a 'beats that don't pay' pattern (real
    beats followed by a negative price reaction) -- checked BOTH ways:
    qualitative_confirmation is whether the Briefing's own
    earnings_track_record prose says so (checked first, since per Part 2
    this must override naive beat-counting when present); numeric
    cross-check is the same pattern computed directly from real
    earnings_history_real beat_miss/price_reaction_pct pairs, included
    so the caller always has the raw counts even when the Briefing's
    prose didn't happen to use one of the matched phrases."""
    text_lower = (earnings_track_record_text or "").lower()
    qualitative_confirmation = any(kw in text_lower for kw in _BEATS_DONT_PAY_KEYWORDS)

    # beat_miss is a scraped signed-dollar-delta STRING (e.g. "+$0.10",
    # "-$0.06"), not a "BEAT"/"MISS" label -- eps_beat_miss_pct (computed
    # from the real consensus-vs-actual EPS figures) is the reliable
    # signal: positive = beat, negative/zero = miss or inline.
    beats = [
        r for r in history_rows
        if r.get("eps_beat_miss_pct") is not None and r["eps_beat_miss_pct"] > 0
        and r.get("price_reaction_pct") is not None
    ]
    beats_that_sold_off = [r for r in beats if r["price_reaction_pct"] < 0]
    numeric_flag = len(beats) >= 2 and (len(beats_that_sold_off) / len(beats)) >= 0.5

    return {
        "pattern_detected": qualitative_confirmation or numeric_flag,
        "qualitative_confirmation": qualitative_confirmation,
        "numeric_flag": numeric_flag,
        "beats_count": len(beats),
        "beats_that_sold_off_count": len(beats_that_sold_off),
        "sentence": (
            f"{len(beats_that_sold_off)} of {len(beats)} real prior beats were followed by a "
            f"negative price reaction" if beats else None
        ),
    }


def extract_simulator_inputs_from_briefing(ticker, conn):
    """Pulls the most recent AI Briefing (within AI_BRIEF_CACHE_HOURS,
    same freshness window the Briefing itself uses) and extracts
    structured signal for estimate_earnings_probability() -- Part 2.
    Returns None if no recent Briefing exists; the caller (the EARNINGS
    SIMULATOR tab) is responsible for gating on that per Part 1, not
    silently falling back to something else.

    - historical_beat_reaction_pattern: see _detect_beats_dont_pay_pattern.
    - catalyst_risk_factors: every entry in the Briefing's own `catalysts`
      list, each classified RISK-DAMPENING/RISK-SUPPORTING/NEUTRAL from
      its own why_it_matters text (see _classify_catalyst_risk).
    - macro_overhang: named macro/legal/regulatory risk sentences pulled
      from the Briefing's setup/news_summary/catalyst text (see
      _extract_macro_overhang) -- [] if none mentioned.
    - buyback_signal: see _extract_buyback_signal, read from
      institutional_analysis.
    - verdicts / next_earnings_verdict: the Briefing's own three VERDICTS
      passed through as-is (already structured JSON, no parsing needed) --
      the simulator's probability output must never contradict these."""
    brief = get_cached_ai_brief(conn, ticker, max_age_hours=AI_BRIEF_CACHE_HOURS)
    if not brief:
        return None

    history_rows = _read_earnings_history_real_cache(conn, ticker, limit=8)
    beat_pattern = _detect_beats_dont_pay_pattern(brief.get("earnings_track_record"), history_rows)

    catalysts = brief.get("catalysts") or []
    catalyst_risk_factors = [
        {
            "catalyst": c.get("catalyst"), "expected_timing": c.get("expected_timing"),
            "why_it_matters": c.get("why_it_matters"),
            "classification": _classify_catalyst_risk(c.get("why_it_matters")),
        }
        for c in catalysts
    ]

    macro_text_blob = "\n".join(filter(None, [
        brief.get("setup"), brief.get("news_summary"),
        " ".join(c.get("why_it_matters") or "" for c in catalysts),
    ]))
    macro_overhang = _extract_macro_overhang(macro_text_blob)

    buyback_signal = _extract_buyback_signal(brief.get("institutional_analysis"))

    verdicts = brief.get("verdicts") or []
    next_earnings_verdict = next(
        (v for v in verdicts if "earnings" in (v.get("horizon") or "").lower()), None
    )

    return {
        "ticker": ticker,
        "briefing_fetched_at": brief.get("_fetched_at"),
        "briefing_id": brief.get("_briefing_id"),
        "historical_beat_reaction_pattern": beat_pattern,
        "catalyst_risk_factors": catalyst_risk_factors,
        "macro_overhang": macro_overhang,
        "buyback_signal": buyback_signal,
        "verdicts": verdicts,
        "next_earnings_verdict": next_earnings_verdict,
        "bottom_line_setup": brief.get("setup"),
    }


# --------------------------------------------------------------------------
# Part 8 -- trained model, hard-gated on real reconciled data volume.
# sklearn is imported lazily inside these functions (same convention as
# selenium/undetected-chromedriver elsewhere in this file) so the rest of
# the app works fine without it installed; it's an optional dependency
# (see requirements.txt) only exercised once real prediction history
# exists.
# --------------------------------------------------------------------------

TRAINED_MODEL_MIN_SAMPLES = 15
TRAINED_MODEL_GBC_THRESHOLD = 50
TRAINED_MODEL_RETRAIN_EVERY = 5


def _catalyst_status_score(catalyst_status):
    return {"RISK-SUPPORTING": 1.0, "NEUTRAL": 0.0, "MIXED": 0.0, "RISK-DAMPENING": -1.0}.get(catalyst_status, 0.0)


def _buyback_signal_score(buyback_signal_json):
    """buyback_signal_json: the JSON string written by reconcile_earnings_
    predictions (same shape as _extract_buyback_signal's return). A
    'token' buyback scores 0, not positive -- per Part 2's spec, only a
    Briefing-confirmed 'meaningful' buyback counts as real support."""
    if not buyback_signal_json:
        return 0.0
    try:
        sig = json.loads(buyback_signal_json)
    except (TypeError, ValueError):
        return 0.0
    if not sig.get("present") or sig.get("magnitude") != "meaningful":
        return 0.0
    return {"accelerating": 1.0, "steady": 0.5, "decelerating": 0.0}.get(sig.get("trend"), 0.5)


def _build_training_matrix(conn):
    """Joins reconciled, final earnings_predictions rows against their
    matching earnings_history_real row (same ticker+earnings_date) for
    the numeric/qualitative features, plus the nearest divergence_scores
    row at-or-before the earnings date. Returns (X, y, feature_names,
    row_meta) -- row_meta carries ticker/earnings_date/the ORIGINAL
    predicted_direction so the caller can score the Mode 2 baseline on
    the exact same rows without re-deriving anything."""
    rows = conn.execute(
        """SELECT ep.ticker, ep.earnings_date, ep.predicted_direction, ep.actual_direction,
                  ehr.eps_beat_miss_pct, ehr.revenue_beat_miss_pct, ehr.catalyst_status,
                  ehr.buyback_signal, ehr.pre_earnings_iv, ehr.pre_earnings_skew_pts
           FROM earnings_predictions ep
           LEFT JOIN earnings_history_real ehr
             ON ehr.ticker = ep.ticker AND ehr.earnings_date = ep.earnings_date
           WHERE ep.prediction_correct IS NOT NULL AND ep.is_final_prediction = 1
           ORDER BY ep.earnings_date""",
    ).fetchall()

    feature_names = [
        "eps_beat_miss_pct", "revenue_beat_miss_pct", "catalyst_score", "buyback_score",
        "pre_earnings_iv", "pre_earnings_skew_pts", "divergence_score",
    ]
    X, y, row_meta = [], [], []
    for (ticker, earnings_date, predicted_direction, actual_direction, eps_pct, rev_pct,
         catalyst_status, buyback_json, pre_iv, pre_skew) in rows:
        if actual_direction not in ("UP", "DOWN"):
            continue
        div_row = conn.execute(
            """SELECT score FROM divergence_scores WHERE ticker=? AND computed_date <= ?
               ORDER BY computed_date DESC LIMIT 1""",
            (ticker, earnings_date),
        ).fetchone()
        X.append([
            eps_pct or 0.0, rev_pct or 0.0, _catalyst_status_score(catalyst_status),
            _buyback_signal_score(buyback_json), pre_iv or 0.0, pre_skew or 0.0,
            div_row[0] if div_row else 0.0,
        ])
        y.append(1 if actual_direction == "UP" else 0)
        row_meta.append({"ticker": ticker, "earnings_date": earnings_date,
                          "predicted_direction": predicted_direction, "actual_direction": actual_direction})
    return X, y, feature_names, row_meta


def train_earnings_direction_model(conn, min_samples=TRAINED_MODEL_MIN_SAMPLES):
    """Checks reconciled, final rows in earnings_predictions (Part 8).
    Returns None (Mode 2 stays active) if fewer than `min_samples` exist,
    or if a model was already trained at a sample count within
    TRAINED_MODEL_RETRAIN_EVERY of the current one (retrains every 5 new
    reconciled predictions, not on every call).

    Trains LogisticRegression below TRAINED_MODEL_GBC_THRESHOLD samples,
    GradientBoostingClassifier at or above it. Holds out the most recent
    ~20% (chronological, not random -- this is a forecasting task) as a
    test set, and compares its accuracy against the Mode 2 Bayesian
    baseline's ACTUAL historical calls on those same held-out rows
    (predicted_direction as it was really predicted at the time, not
    recomputed). Only persists/returns the trained model if it measurably
    outperforms that baseline; otherwise returns None (with the reason
    logged) so Mode 2 keeps running."""
    X, y, feature_names, row_meta = _build_training_matrix(conn)
    n = len(X)
    if n < min_samples:
        return None

    last = conn.execute(
        "SELECT n_samples_at_train FROM earnings_model_cache ORDER BY trained_at DESC LIMIT 1"
    ).fetchone()
    last_n = last[0] if last else 0
    if last_n and (n - last_n) < TRAINED_MODEL_RETRAIN_EVERY:
        return _read_cached_trained_model(conn)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        return None

    split = max(1, int(n * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    meta_test = row_meta[split:]
    if not X_test or len(set(y_train)) < 2:
        return None  # can't hold out anything, or every training label is identical -- nothing to learn/validate

    model_type = "gradient_boosting" if n >= TRAINED_MODEL_GBC_THRESHOLD else "logistic_regression"
    model = (
        GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
        if model_type == "gradient_boosting" else LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    held_out_accuracy = sum(1 for p, actual in zip(preds, y_test) if p == actual) / len(y_test)

    baseline_correct = sum(
        1 for m in meta_test
        if (m["predicted_direction"] or "").upper() == (m["actual_direction"] or "").upper()
    )
    bayesian_baseline_accuracy = baseline_correct / len(meta_test)

    beats_baseline = held_out_accuracy > bayesian_baseline_accuracy
    model_blob = pickle.dumps(model) if beats_baseline else None

    conn.execute(
        """INSERT INTO earnings_model_cache
               (trained_at, n_samples_at_train, model_type, held_out_accuracy,
                bayesian_baseline_accuracy, beats_baseline, feature_names_json, model_blob)
           VALUES (?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(), n, model_type, held_out_accuracy, bayesian_baseline_accuracy,
         int(beats_baseline), json.dumps(feature_names), model_blob),
    )
    conn.commit()

    if not beats_baseline:
        print(f"[earnings_model] Trained {model_type} on {n} samples ({held_out_accuracy:.0%} held-out) did "
              f"NOT beat the Mode 2 baseline ({bayesian_baseline_accuracy:.0%}) -- keeping Mode 2 active.")
        return None
    return {
        "model": model, "model_type": model_type, "n_samples": n, "held_out_accuracy": held_out_accuracy,
        "bayesian_baseline_accuracy": bayesian_baseline_accuracy, "feature_names": feature_names,
    }


def _read_cached_trained_model(conn):
    """Pure DB read of the last trained model, if it actually beat its
    baseline at training time (model_blob is only ever stored when it
    did) -- used when train_earnings_direction_model's retrain cadence
    says "not yet due for a retrain," so Mode 3 can keep using the last
    validated model instead of silently falling back to Mode 2."""
    row = conn.execute(
        """SELECT model_type, n_samples_at_train, held_out_accuracy, bayesian_baseline_accuracy,
                  feature_names_json, model_blob
           FROM earnings_model_cache WHERE beats_baseline=1 ORDER BY trained_at DESC LIMIT 1""",
    ).fetchone()
    if not row or not row[5]:
        return None
    model_type, n_samples, held_out_acc, baseline_acc, feature_names_json, model_blob = row
    return {
        "model": pickle.loads(model_blob), "model_type": model_type, "n_samples": n_samples,
        "held_out_accuracy": held_out_acc, "bayesian_baseline_accuracy": baseline_acc,
        "feature_names": json.loads(feature_names_json),
    }


# --------------------------------------------------------------------------
# Block B -- "what the current setup says." Restates the AI Briefing's OWN
# synthesized next-earnings verdict as written reasoning, using the exact
# qualitative inputs extract_simulator_inputs_from_briefing() already
# pulled from the Briefing's text. Deliberately independent of Block A's
# (estimate_earnings_probability's) historical-sample-size gate -- this
# must always render once a Briefing exists, full stop. NOT a new
# probability number; it's the Briefing's own verdict, explained.
# --------------------------------------------------------------------------

_QUALITATIVE_LEAN_PHRASES = {
    ("BULLISH", "High"): "Confidently bullish", ("BULLISH", "Medium"): "Cautiously bullish",
    ("BULLISH", "Low"): "Weakly bullish",
    ("BEARISH", "High"): "Confidently bearish", ("BEARISH", "Medium"): "Cautiously bearish",
    ("BEARISH", "Low"): "Weakly bearish",
    ("NEUTRAL", "High"): "Neutral", ("NEUTRAL", "Medium"): "Neutral",
    ("NEUTRAL", "Low"): "Neutral, low-conviction",
}


def generate_qualitative_directional_read(ticker, conn, briefing_inputs, market=None):
    """Block B (Parts 1-2): restates the AI Briefing's own next-earnings
    VERDICT (bias + confidence) as a written paragraph citing the
    specific qualitative inputs that drove it -- historical_beat_
    reaction_pattern, catalyst_risk_factors, macro_overhang,
    buyback_signal, all already extracted by extract_simulator_inputs_
    from_briefing(), plus the live market-pricing context (magnitude,
    skew). Returns None only if briefing_inputs itself is None (no
    recent Briefing at all -- Part 1's gate already handles that
    upstream). Otherwise ALWAYS returns a real reasoning paragraph,
    regardless of how much historical data exists -- this is the fix for
    the bug where qualitative synthesis was computed but never surfaced
    behind Block A's Mode 1 gate."""
    if not briefing_inputs:
        return None
    market = market or get_market_pricing_signals(ticker, conn)
    verdict = briefing_inputs.get("next_earnings_verdict") or {}
    bias = (verdict.get("bias") or "NEUTRAL").upper()
    confidence = verdict.get("confidence") or "Low"
    lean_phrase = _QUALITATIVE_LEAN_PHRASES.get((bias, confidence), bias.title())

    clauses = []
    beat_pattern = briefing_inputs.get("historical_beat_reaction_pattern") or {}
    catalysts = briefing_inputs.get("catalyst_risk_factors") or []
    # Part 5.1: explicitly name the specific thing the market is tracking
    # (when the Briefing actually names one -- never fabricated, see
    # _select_named_operational_catalyst's positive-match requirement)
    # BEFORE the historical-pattern clause, so the reasoning leads with
    # what to watch rather than burying it.
    named_operational = _select_named_operational_catalyst(catalysts)
    if named_operational:
        clauses.append(
            f"The market is specifically watching {_catalyst_short_name(named_operational['catalyst'])} -- "
            f"the Briefing's most recent read: "
            f"{named_operational.get('why_it_matters') or 'status not further detailed'}"
        )

    if beat_pattern.get("pattern_detected") and beat_pattern.get("sentence"):
        clauses.append(f"{beat_pattern['sentence']} — a 'beats that don't pay' pattern")
        # Part 5.2: connect the historical pattern to what would need to
        # be DIFFERENT this time for it to break, rather than leaving the
        # pattern as a bare restated count.
        if named_operational:
            clauses.append(
                f"For this print to actually move the stock up despite that history, the market likely "
                f"needs BOTH a beat AND explicit confirmation that "
                f"{_catalyst_short_name(named_operational['catalyst'])} is on track -- a beat alone, per "
                f"the recent pattern, has not been sufficient"
            )
        else:
            key_risk_txt = verdict.get("key_risk")
            if key_risk_txt:
                clauses.append(
                    f"No single named product/technology catalyst explains the gap this time -- the "
                    f"Briefing's own stated risk is the closest thing to a specific tell: {key_risk_txt}"
                )
            else:
                clauses.append(
                    "No single named catalyst explains the gap -- the historical pattern itself is the "
                    "primary signal here, without a specific confirming/disconfirming event to watch for"
                )

    # Exclude "the earnings event restated" from the dampening/supporting
    # display too -- classified RISK-DAMPENING/SUPPORTING by
    # _classify_catalyst_risk's keyword scan same as any other catalyst,
    # but listing "the earnings report" itself as a "supportive catalyst"
    # is circular and exactly the generic filler Part 5 wants gone.
    dampening = [
        c for c in catalysts
        if c["classification"] == "RISK-DAMPENING" and not _is_earnings_event_restated(c.get("catalyst"))
    ]
    supporting = [
        c for c in catalysts
        if c["classification"] == "RISK-SUPPORTING" and not _is_earnings_event_restated(c.get("catalyst"))
    ]
    if dampening:
        names = "; ".join(c["catalyst"] for c in dampening[:2] if c.get("catalyst"))
        clauses.append(
            f"The AI Briefing flags {'an' if len(dampening) == 1 else len(dampening)} unresolved risk "
            f"factor{'s' if len(dampening) != 1 else ''} as risk-dampening: {names}" if names else
            f"The AI Briefing flags {len(dampening)} catalyst(s) as risk-dampening"
        )
    if supporting:
        names = "; ".join(c["catalyst"] for c in supporting[:2] if c.get("catalyst"))
        if names:
            clauses.append(f"On the supportive side: {names}")

    macro = briefing_inputs.get("macro_overhang") or []
    if macro:
        clauses.append(f"Named macro/regulatory overhang: {macro[0]}")

    buyback = briefing_inputs.get("buyback_signal") or {}
    if buyback.get("present"):
        if buyback.get("magnitude") == "token":
            clauses.append(
                "A buyback is on record, but the Briefing characterizes it as token, non-meaningful support"
            )
        elif buyback.get("magnitude") == "meaningful":
            trend_txt = f", trending {buyback['trend']}" if buyback.get("trend") else ""
            clauses.append(f"Buybacks are Briefing-confirmed meaningful{trend_txt} -- a real supportive factor")

    market_bits = []
    magnitude = (market or {}).get("iv_implied_move_pct")
    if magnitude is not None:
        market_bits.append(f"implied volatility is pricing a ±{magnitude:.1f}% move")
    skew = (market or {}).get("skew") or {}
    if skew.get("skew_pts") is not None:
        skew_dir = (
            "puts pricier than calls (downside hedging demand)" if skew["skew_pts"] > 0
            else "calls pricier than puts (upside demand)"
        )
        market_bits.append(f"skew shows {skew_dir} ({skew['skew_pts']:+.1f} IV pts)")
    if market_bits:
        clauses.append(market_bits[0][0].upper() + market_bits[0][1:] + (
            f", with {market_bits[1]}" if len(market_bits) > 1 else ""
        ) + ".")

    clauses = [c.strip().rstrip(".") + "." for c in clauses]
    reasoning_text = " ".join(clauses) if clauses else (
        "No specific qualitative risk factors were named in the Briefing beyond its stated verdict."
    )
    return {
        "lean_phrase": lean_phrase, "bias": bias, "confidence": confidence,
        "reasoning_text": reasoning_text, "key_risk": verdict.get("key_risk"),
        "sentence": f"Directional lean: {lean_phrase} ({confidence} confidence). Reasoning: {reasoning_text}",
    }


# --------------------------------------------------------------------------
# Part 4 -- the two-phase probability engine itself (Block A -- "what the
# numbers say"). Checks evidence volume FIRST and picks the appropriate
# output mode -- never manufactures a false-precision probability from
# thin data. See generate_qualitative_directional_read above for Block B
# -- always available, computed and returned independently of this
# function's Mode 1/2/3 gate.
# --------------------------------------------------------------------------

BAYESIAN_PRIOR_PSEUDO_COUNT = 4  # "virtual" pseudo-observations the cross-ticker prior counts as; shared with _historical_beat_rate below


# --------------------------------------------------------------------------
# Earnings scenario matrix -- the full EPS x Revenue x (optional) Catalyst
# combinatorial breakdown. This is the mechanism for catching patterns
# like "beats everything but the named catalyst hasn't landed, so the
# stock falls anyway" (e.g. a real WDC/HAMR-shaped setup): a plain
# EPS-beat/miss x Revenue-beat/miss grid can't distinguish "clean beat,
# catalyst confirmed" from "clean beat, catalyst still pending" -- two
# scenarios with the SAME EPS/revenue outcome but opposite market
# reactions in practice. estimate_earnings_probability's Mode 2/3
# probabilities are derived by SUMMING this matrix's rows grouped by
# direction (not computed independently), so the top-line verdict and
# the detailed breakdown can never disagree with each other.
# --------------------------------------------------------------------------

def _historical_beat_rate(conn, ticker, field):
    """Bayesian-shrunk P(beat) for `field` ('eps_beat_miss_pct' or
    'revenue_beat_miss_pct'), blending this ticker's own historical beat
    rate with the cross-ticker pooled rate via the same pseudo-count
    blend used throughout this engine, keyed on beat/miss itself rather
    than the post-beat price reaction. Falls back to 0.65 as the pooled
    prior (most companies beat lowered guidance most quarters -- a
    documented, deliberately-not-measured starting assumption) only when
    there's no pooled data across the whole watchlist yet. Returns
    (rate, ticker_n)."""
    ticker_rows = conn.execute(
        f"SELECT {field} FROM earnings_history_real WHERE ticker=? AND {field} IS NOT NULL", (ticker,)
    ).fetchall()
    ticker_vals = [r[0] for r in ticker_rows]
    ticker_n = len(ticker_vals)
    ticker_beats = sum(1 for v in ticker_vals if v > 0)

    pooled_rows = conn.execute(f"SELECT {field} FROM earnings_history_real WHERE {field} IS NOT NULL").fetchall()
    pooled_vals = [r[0] for r in pooled_rows]
    pooled_rate = (sum(1 for v in pooled_vals if v > 0) / len(pooled_vals)) if pooled_vals else 0.65

    posterior = (BAYESIAN_PRIOR_PSEUDO_COUNT * pooled_rate + ticker_beats) / (BAYESIAN_PRIOR_PSEUDO_COUNT + ticker_n)
    return posterior, ticker_n


# The earnings event itself, restated -- not a distinct catalyst at all
# (a real bug this used to miss: "Q1 FY2027 earnings report (EPS estimate
# $4.04)" isn't macro/legal/M&A, so a pure exclusion-list let it through
# as "the catalyst"). Split out as its own constant since Block B's
# dampening/supporting catalyst display (generate_qualitative_directional_
# read) needs to filter these out too -- listing "the earnings report" as
# a "supportive catalyst" is exactly the circular, generic statement Part
# 5 is meant to eliminate, not just the scenario matrix's catalyst column.
_EARNINGS_EVENT_RESTATED_KEYWORDS = [
    "earnings report", "earnings call", "quarterly report", "eps estimate", " fy20", " fy19", " fy2",
    "fiscal q", "next earnings", "scheduled confirmed event", "next scheduled",
]

# Macro/economic-data events -- market-wide, not company-specific. Missing
# this list was a real bug: "Fed minutes release" matched none of the
# categories below (no macro/legal/M&A keyword, no analyst-action keyword)
# AND matched "release" in _OPERATIONAL_CATALYST_KEYWORDS (a word meant for
# "product release"), so it was selected as WMT's "named operational
# catalyst" ahead of a genuine company-specific one later in the same list.
_MACRO_ECON_EVENT_KEYWORDS = [
    "fed minutes", "fomc", "federal reserve", "rate decision", "rate cut", "rate hike",
    "jobs report", "nonfarm payroll", "cpi report", "cpi data", "consumer price index",
    "ppi report", "ppi data", "producer price index", "central bank", "macro event",
    "economic data", "jobless claims", "treasury yield",
]

# Legal/regulatory/M&A, analyst-action, and options-flow/insider-trade
# exclusions -- checked against catalyst+why_it_matters together (below),
# since these are unambiguous disqualifiers wherever they appear.
_LEGAL_MA_REGULATORY_KEYWORDS = [
    "sec ", "investigation", "lawsuit", "litigation", "acquisition", "buyout", "merger", "acquire",
    "deal", "talks", "regulatory", "regulator", "antitrust", "sanctions", "tariff", "export control",
    "doj", "ftc",
]
_ANALYST_ACTION_KEYWORDS = [
    "rating change", "analyst", "price target", "upgrade", "downgrade", "initiated coverage",
    "coverage initiated", "buy rating", "sell rating", "hold rating", "mean target",
]
_MARKET_SIGNAL_KEYWORDS = [
    "options expiry", "option expiry", "volume/oi", "volume-oi", "gamma zone", "unusual trade",
    "insider sale", "insider sold", "ceo sale", "director sale", "sold shares",
]
# These four categories, checked against the full catalyst+why_it_matters
# blob -- unlike _EARNINGS_EVENT_RESTATED_KEYWORDS below, which is checked
# against the catalyst field ONLY (see _select_named_operational_catalyst).
_NON_EARNINGS_RESTATED_EXCLUSION_KEYWORDS = (
    _MACRO_ECON_EVENT_KEYWORDS + _LEGAL_MA_REGULATORY_KEYWORDS + _ANALYST_ACTION_KEYWORDS
    + _MARKET_SIGNAL_KEYWORDS
)
_NON_OPERATIONAL_CATALYST_KEYWORDS = _EARNINGS_EVENT_RESTATED_KEYWORDS + _NON_EARNINGS_RESTATED_EXCLUSION_KEYWORDS


def _is_earnings_event_restated(catalyst_text):
    text = (catalyst_text or "").lower()
    return any(kw in text for kw in _EARNINGS_EVENT_RESTATED_KEYWORDS)

_OPERATIONAL_CATALYST_KEYWORDS = [
    "technology", "product", "launch", "rollout", "roll-out", "production", "manufactur", "capacity",
    "yield", "shipment", "ship ", "certification", "certified", "approval", "approved", "partnership",
    "contract win", "chip", "node", "platform", "facility", "plant", "expansion", "release", "recall",
    "clinical", "trial", "fda", "patent", "supply agreement", "design win",
]

# When multiple real operational catalysts are available, one of these
# appearing in a catalyst's own why_it_matters marks it as explicitly tied
# to an OBSERVED price reaction (not just a forward-looking milestone) --
# see _select_named_operational_catalyst's preference tier below.
_REACTION_EXPLAINING_KEYWORDS = [
    "sold off", "sell-off", "selloff", "fell despite", "dropped despite", "declined despite",
    "the gap", "margin gap", "reason the stock", "reason for the", "why the stock", "post-earnings",
    "the selloff", "the drop", "stock fell", "stock dropped", "underperformed",
]


def _select_named_operational_catalyst(catalysts):
    """Picks the first entry in the Briefing's own CATALYSTS list that
    reads as a specific product/technology/operational milestone (e.g.
    "HAMR technology rollout status") -- requires a POSITIVE match
    against _OPERATIONAL_CATALYST_KEYWORDS, not just the absence of a
    macro/legal/M&A keyword. A pure exclusion-list (the earlier version
    of this function) let things like "Q1 FY2027 earnings report (EPS
    estimate $4.04)" -- the earnings event itself, restated -- or
    "Wall Street Zen rating change to Buy" -- an analyst action, not a
    catalyst FOR the stock -- slip through as "the named catalyst"
    simply because they don't contain a macro/legal word either.

    Two exclusion checks with DIFFERENT scopes, and this split matters:
    _EARNINGS_EVENT_RESTATED_KEYWORDS is checked against the `catalyst`
    field ONLY, not why_it_matters -- a real bug found live against WMT's
    Briefing: "First in-store automated fulfillment pilot rollout" (a
    genuine, explicitly-labeled 'only company-specific operational item')
    was being wrongly excluded because its OWN why_it_matters happened to
    mention "a talking point on the earnings call" as context, which
    matched "earnings call" in the old full-blob check. Checking the
    catalyst field alone still correctly excludes "Q1 FY2027 earnings
    report (EPS estimate $4.04)" (the phrase is right there in the
    catalyst field itself) without punishing a real catalyst for
    incidentally mentioning the earnings call in its explanation.
    _NON_EARNINGS_RESTATED_EXCLUSION_KEYWORDS (macro/econ, legal/M&A,
    analyst-action, market-signal) is checked against the full blob,
    since those really are unambiguous disqualifiers wherever they land.

    When MULTIPLE catalysts pass both checks, prefers whichever one's own
    why_it_matters explicitly ties to an OBSERVED price reaction (e.g. "the
    same gap ... names as the specific reason the stock fell ~11%") over
    one that's a genuine operational milestone with no stated reaction
    link -- confirmed live against WDC: with both a '40TB UltraSMR' ramp
    catalyst and a 'HAMR ramp' catalyst passing the operational check, only
    HAMR's why_it_matters explicitly explained the real ~11% post-earnings
    drop, making it the more decision-relevant pick for a scenario matrix
    that's specifically modeling earnings-day reactions. Falls back to
    first-list-order when no candidate ties to a reaction (the common
    case -- most operational catalysts are forward-looking, not reaction
    explanations)."""
    candidates = []
    for c in catalysts or []:
        catalyst_field = (c.get("catalyst") or "").lower()
        full_text = f"{catalyst_field} {(c.get('why_it_matters') or '').lower()}"
        if any(kw in catalyst_field for kw in _EARNINGS_EVENT_RESTATED_KEYWORDS):
            continue
        if any(kw in full_text for kw in _NON_EARNINGS_RESTATED_EXCLUSION_KEYWORDS):
            continue
        if any(kw in full_text for kw in _OPERATIONAL_CATALYST_KEYWORDS):
            candidates.append((c, any(kw in full_text for kw in _REACTION_EXPLAINING_KEYWORDS)))
    if not candidates:
        return None
    reaction_tied = [c for c, tied in candidates if tied]
    return reaction_tied[0] if reaction_tied else candidates[0][0]


def _catalyst_short_name(catalyst_text):
    """The grammatical-subject-safe short form of a catalyst name. Some
    Briefings pack a full descriptive clause into the catalyst field
    itself -- e.g. "Klarna membership overhaul -- fees removed, cashback
    boosted, annual value up to EUR6,000" -- rather than a clean short
    name. Splicing that whole string directly into a sentence like
    "without X landing this print" breaks grammatically (a real bug,
    confirmed live: produced "without Klarna membership overhaul -- fees
    removed, cashback boosted, annual value up to EUR6,000 landing this
    print", an unreadable run-on). Takes just the text before the first
    ' -- ' as the short name; the rest is exactly the kind of detail
    why_it_matters already carries separately, so nothing is lost -- it's
    just not forced into the sentence's grammatical subject position."""
    if not catalyst_text:
        return catalyst_text
    return catalyst_text.split(" — ", 1)[0].strip()


_NEAR_TERM_TIMING_KEYWORDS = ["day", "week", "this quarter", "imminent", "shortly", "soon"]
_DISTANT_TIMING_KEYWORDS = ["ongoing", "tbd", "unclear", "future", "long-term", "unresolved", "pending"]


def _catalyst_achievement_probability(catalyst):
    """P(this specific named catalyst is actually achieved/confirmed by
    the earnings print) -- NOT derivable from historical data (it's a
    one-off, catalyst-specific event), so this is a documented heuristic
    default, not measured data: near-term expected_timing language leans
    toward achieved (60%), distant/vague language leans toward delayed
    (35%), otherwise an even 50/50 split."""
    timing = (catalyst.get("expected_timing") or "").lower()
    if any(kw in timing for kw in _NEAR_TERM_TIMING_KEYWORDS):
        return 0.60
    if any(kw in timing for kw in _DISTANT_TIMING_KEYWORDS):
        return 0.35
    return 0.50


# (eps, revenue, catalyst-or-None) -> template. direction/magnitude_range
# only -- the per-combination "reasoning" strings that used to live here
# were pure boilerplate (every ticker's Beat/Beat row read the exact same
# "Clean beat with catalyst confirmation -- the strongest bull case"
# sentence, with zero ticker-specific content). Real reasoning is now
# generated per-ticker by _generate_scenario_reasoning below, grounded in
# this ticker's own tracked earnings history.
# FIXED magnitude ranges (percentage points), per this feature's own
# stated convention: a directional earnings surprise implies roughly a
# 5%+ move, an inline/flat print implies 0-3% -- NOT the current options-
# market-implied move. A real bug (confirmed live against KLAR): this
# used to be a `magnitude_mult` tuple multiplying the ticker's raw,
# volatile iv_implied_move_pct (e.g. WDC's 174% IV -> a ~19% single-
# ticker move), so the "Up"/"Down" scenario magnitude swung with
# whatever IV happened to be that day instead of representing a stable,
# comparable "how big is a typical surprise reaction" scale. The current
# options-market-implied move is still real, useful information -- it's
# now returned separately as market_implied_move_pct, explicitly labeled
# as "what the market is pricing" rather than folded into these rows.
# Ranges keep the ORIGINAL relative severity ordering across templates
# (catalyst-confirmed beat > plain beat > mixed > catalyst-delayed beat >
# clean miss), just anchored to the fixed convention instead of a
# ticker's live IV.
_SCENARIO_TEMPLATES = {
    ("Beat", "Beat", "Achieved"): {"direction": "Up", "magnitude_range": (6.0, 9.0)},
    ("Beat", "Beat", "Delayed"): {"direction": "Down", "magnitude_range": (-7.0, -5.0)},
    ("Beat", "Beat", None): {"direction": "Up", "magnitude_range": (5.0, 8.0)},
    ("Beat", "Miss", None): {"direction": "Flat/Down", "magnitude_range": (-3.0, 0.0)},
    ("Miss", "Beat", None): {"direction": "Flat", "magnitude_range": (0.0, 3.0)},
    ("Miss", "Miss", None): {"direction": "Down", "magnitude_range": (-9.0, -6.0)},
}

_SCENARIO_PATTERN_DESC = {
    (True, True): "beat on both EPS and revenue",
    (True, False): "beat EPS but missed revenue",
    (False, True): "missed EPS but beat revenue",
    (False, False): "missed on both EPS and revenue",
}


def _find_precedent_quarter(history_rows, eps_beat, revenue_beat):
    """The most recent real quarter (from earnings_history_real, newest-
    first) whose actual eps_beat_miss_pct/revenue_beat_miss_pct sign
    matches this exact (eps_beat, revenue_beat) pattern -- the real
    precedent _generate_scenario_reasoning cites, or None if this ticker's
    tracked history genuinely has no such quarter."""
    for r in history_rows:
        eps_pct, rev_pct = r.get("eps_beat_miss_pct"), r.get("revenue_beat_miss_pct")
        if eps_pct is None or rev_pct is None:
            continue
        if (eps_pct > 0) == eps_beat and (rev_pct > 0) == revenue_beat:
            return r
    return None


def _generate_scenario_reasoning(ticker, eps_key, revenue_key, catalyst_state, named_catalyst, history_rows):
    """Real, ticker-specific reasoning for one scenario row -- replaces the
    old static per-EPS/Revenue/Catalyst-combination template strings, which
    read identically for every ticker regardless of that ticker's actual
    history (the exact bug flagged live: WMT's SCENARIO BREAKDOWN showed
    the generic "Clean beat with catalyst confirmation -- the strongest
    bull case" sentence with no WMT-specific content at all).

    Grounds the claim in this ticker's OWN historical beat/miss-to-price-
    reaction pairs (earnings_history_real, up to 16 real quarters deep)
    when a real precedent for this exact EPS/revenue pattern exists,
    citing the actual date and measured next-day %% move; when it doesn't,
    says so explicitly rather than falling back to a fully generic
    sentence with no ticker-specific content. When a named operational
    catalyst is in play, states concretely what "confirmed" vs. "not
    landing this print" means for THIS catalyst, quoting the Briefing's
    own why_it_matters text rather than a generic "the catalyst" phrase.

    Deliberately Python string interpolation, not a second LLM call per
    scenario row: the Earnings Simulator tab has never been gated behind
    the AI Briefing's paid-call confirmation flow (Block A/B and this
    matrix render instantly, for free, on every tab load), and every input
    this function needs -- real beat/miss history, the real named
    catalyst's own text -- is already loaded, so genuine ticker-specific
    grounding is achievable without introducing a new paid-call gate where
    none exists today."""
    eps_beat, revenue_beat = eps_key == "Beat", revenue_key == "Beat"
    pattern_desc = _SCENARIO_PATTERN_DESC[(eps_beat, revenue_beat)]
    precedent = _find_precedent_quarter(history_rows, eps_beat, revenue_beat)

    if precedent:
        quarter_txt = f" ({precedent['quarter']})" if precedent.get("quarter") else ""
        reaction = precedent.get("price_reaction_pct")
        if reaction is not None:
            precedent_clause = (
                f"The last time {ticker} {pattern_desc} was {precedent['earnings_date']}{quarter_txt}, and "
                f"shares moved {reaction:+.1f}% the next trading day (marketbeat+yfinance)."
            )
        else:
            precedent_clause = (
                f"The last time {ticker} {pattern_desc} was {precedent['earnings_date']}{quarter_txt} -- no "
                f"measured next-day price reaction is on record for that print."
            )
    else:
        precedent_clause = (
            f"This exact combination ({pattern_desc}) hasn't occurred in {ticker}'s tracked earnings "
            f"history; the direction/magnitude estimate below is based on the general historical pattern "
            f"for {ticker}, not a specific precedent."
        )

    catalyst_clause = ""
    if named_catalyst and catalyst_state:
        # Short form only -- the raw catalyst field can itself be a full
        # descriptive clause (e.g. "Klarna membership overhaul -- fees
        # removed, cashback boosted, annual value up to EUR6,000"), and
        # splicing that whole thing in as "without X landing this print"
        # produced a real, confirmed run-on/ungrammatical sentence for
        # KLAR. See _catalyst_short_name.
        cname = _catalyst_short_name(named_catalyst.get("catalyst")) or "the named catalyst"
        # Parenthetical, not an em-dash splice -- why_it_matters is a
        # complete sentence of its own (already ends in "."), so splicing
        # it in with " -- " before a trailing clause produced a stray
        # ".," collision (".. earnings call., the market has..."). Wrapping
        # it in parens reads as an aside regardless of its own punctuation.
        why = (named_catalyst.get("why_it_matters") or "").strip().rstrip(".")
        why_txt = f" ({why})" if why else ""
        if catalyst_state == "Achieved/On-track":
            catalyst_clause = (
                f" With {cname} also confirmed on track{why_txt}, the market has both the numbers and the "
                f"specific reason it's been pricing in."
            )
        else:
            catalyst_clause = (
                f" But without {cname} landing this print{why_txt}, a beat alone has historically not been "
                f"enough on its own to hold the stock up."
            )

    return (precedent_clause + catalyst_clause).strip()


def build_earnings_scenario_matrix(ticker, conn, briefing_inputs=None, market=None):
    """The full EPS x Revenue x (optional) Catalyst scenario matrix
    (Parts 1-3). Every row carries an individual probability weight (the
    full set sums to 1.0), an expected direction/magnitude tied to this
    ticker's own live IV-implied move, and a plain-language reasoning
    line -- this is what lets a "beats everything but the catalyst
    lagged" trap show up as its own explicit, differently-weighted row
    instead of disappearing into one blended number.

    The catalyst dimension only splits the Beat+Beat row into two
    (Achieved vs. Delayed) when a real, specific, non-macro/legal/M&A
    catalyst is named in the most recent Briefing (see
    _select_named_operational_catalyst) -- every other combination, and
    the catalyst dimension entirely when none exists, stays "N/A" per
    Part 3's explicit instruction not to fabricate one."""
    market = market or get_market_pricing_signals(ticker, conn)
    # The current options-market-implied move -- real, live, and reported
    # separately (see the return dict below) from the scenario magnitude
    # ranges, which use the fixed convention instead (see _SCENARIO_
    # TEMPLATES' comment for why these must NOT be the same number).
    market_implied_move_pct = market.get("iv_implied_move_pct")
    atm_iv_pct = market.get("atm_iv_pct")

    eps_beat_rate, eps_n = _historical_beat_rate(conn, ticker, "eps_beat_miss_pct")
    rev_beat_rate, rev_n = _historical_beat_rate(conn, ticker, "revenue_beat_miss_pct")

    catalysts = (briefing_inputs or {}).get("catalyst_risk_factors") or []
    named_catalyst = _select_named_operational_catalyst(catalysts)
    catalyst_prob = _catalyst_achievement_probability(named_catalyst) if named_catalyst else None

    def _mag_range(lo, hi):
        return f"{lo:+.1f}% to {hi:+.1f}%"

    # Up to 16 real quarters (see _read_ai_context_from_cache's own
    # earnings_track_record_real bump) -- the real-history grounding
    # _generate_scenario_reasoning searches for a precedent in.
    history_rows = _read_earnings_history_real_cache(conn, ticker, limit=16)

    rows = []
    p_beat_beat = eps_beat_rate * rev_beat_rate
    if named_catalyst:
        for cat_state, cat_p, tmpl_key in [
            ("Achieved/On-track", catalyst_prob, ("Beat", "Beat", "Achieved")),
            ("Delayed/Not achieved", 1 - catalyst_prob, ("Beat", "Beat", "Delayed")),
        ]:
            t = _SCENARIO_TEMPLATES[tmpl_key]
            reasoning = _generate_scenario_reasoning(
                ticker, "Beat", "Beat", cat_state, named_catalyst, history_rows
            )
            rows.append({
                "eps": "Beat", "revenue": "Beat", "catalyst": f"{named_catalyst['catalyst']} — {cat_state}",
                "probability": p_beat_beat * cat_p, "expected_direction": t["direction"],
                "expected_magnitude": _mag_range(*t["magnitude_range"]), "reasoning": reasoning,
            })
    else:
        t = _SCENARIO_TEMPLATES[("Beat", "Beat", None)]
        reasoning = _generate_scenario_reasoning(ticker, "Beat", "Beat", None, None, history_rows)
        rows.append({
            "eps": "Beat", "revenue": "Beat", "catalyst": "N/A", "probability": p_beat_beat,
            "expected_direction": t["direction"], "expected_magnitude": _mag_range(*t["magnitude_range"]),
            "reasoning": reasoning,
        })

    for eps_key, rev_key, tmpl_key in [
        ("Beat", "Miss", ("Beat", "Miss", None)), ("Miss", "Beat", ("Miss", "Beat", None)),
        ("Miss", "Miss", ("Miss", "Miss", None)),
    ]:
        t = _SCENARIO_TEMPLATES[tmpl_key]
        p_eps = eps_beat_rate if eps_key == "Beat" else (1 - eps_beat_rate)
        p_rev = rev_beat_rate if rev_key == "Beat" else (1 - rev_beat_rate)
        reasoning = _generate_scenario_reasoning(ticker, eps_key, rev_key, None, None, history_rows)
        rows.append({
            "eps": eps_key, "revenue": rev_key, "catalyst": "N/A", "probability": p_eps * p_rev,
            "expected_direction": t["direction"], "expected_magnitude": _mag_range(*t["magnitude_range"]),
            "reasoning": reasoning,
        })

    total = sum(r["probability"] for r in rows)
    if total > 0:
        for r in rows:
            r["probability"] = round(r["probability"] / total, 4)

    return {
        "rows": rows, "named_catalyst": named_catalyst["catalyst"] if named_catalyst else None,
        "catalyst_achievement_prob": catalyst_prob,
        "eps_beat_rate": round(eps_beat_rate, 3), "eps_sample_n": eps_n,
        "revenue_beat_rate": round(rev_beat_rate, 3), "revenue_sample_n": rev_n,
        # Scenario rows above use the FIXED convention (~5-9% for a
        # directional surprise, 0-3% for flat/inline) -- market_implied_
        # move_pct is the SEPARATE, real, live options-market-implied move
        # (from atm_iv_pct), shown alongside but never blended into the
        # scenario magnitudes. See _SCENARIO_TEMPLATES' comment.
        "scenario_convention_note": "Scenario magnitudes use a fixed historical-convention range (roughly "
                                     "5-9% for a directional beat/miss, 0-3% for an inline/flat print) -- not "
                                     "the current options-market-implied move, which is shown separately.",
        "market_implied_move_pct": market_implied_move_pct, "atm_iv_pct": atm_iv_pct,
        "sample_size_note": (
            f"EPS beat rate based on {eps_n} of this ticker's own tracked quarters (Bayesian-blended with "
            f"the cross-ticker pool); revenue beat rate based on {rev_n}."
        ),
    }


def _scenario_direction_weights(direction_label):
    """Maps a scenario row's expected_direction label onto Up/Down/Flat
    weights summing to 1.0 -- a compound label like 'Flat/Down' splits
    its probability mass evenly across both named buckets, since the
    template deliberately hedges between them rather than picking one."""
    parts = [p.strip() for p in direction_label.split("/")]
    parts = [p for p in parts if p in ("Up", "Down", "Flat")]
    if not parts:
        return {"Up": 0.0, "Down": 0.0, "Flat": 1.0}
    share = 1.0 / len(parts)
    weights = {"Up": 0.0, "Down": 0.0, "Flat": 0.0}
    for p in parts:
        weights[p] += share
    return weights


def _current_qualitative_feature_row(conn, ticker, market, briefing_inputs):
    """Builds a live feature vector matching _build_training_matrix's
    7-feature schema, for Mode 3's predict_proba call. HONEST CAVEAT:
    eps_beat_miss_pct/revenue_beat_miss_pct are only knowable AFTER a
    quarter reports -- they can't be live inputs to a PRE-earnings
    prediction. This deliberately feeds 0.0 (an "in-line, no surprise
    either way" assumption) for both, so Mode 3's live prediction reads
    as "given the qualitative/market context alone, and no informational
    edge on the EPS/revenue surprise itself, which way does the model
    lean" -- not a claim of knowing the beat/miss in advance. The
    trained model's *training* data legitimately uses the real
    (post-hoc) surprise values, since that's learning "how did the
    market react to this context GIVEN a beat/miss of this size,"
    genuinely different from live prediction's "what's the best guess
    with zero information about the surprise itself.\""""
    catalyst_factors = (briefing_inputs or {}).get("catalyst_risk_factors") or []
    dominant_catalyst = "NEUTRAL"
    if catalyst_factors:
        n_dampening = sum(1 for c in catalyst_factors if c["classification"] == "RISK-DAMPENING")
        n_supporting = sum(1 for c in catalyst_factors if c["classification"] == "RISK-SUPPORTING")
        dominant_catalyst = (
            "RISK-DAMPENING" if n_dampening > n_supporting
            else "RISK-SUPPORTING" if n_supporting > n_dampening else "NEUTRAL"
        )
    buyback = (briefing_inputs or {}).get("buyback_signal") or {}
    div_row = conn.execute(
        "SELECT score FROM divergence_scores WHERE ticker=? ORDER BY computed_date DESC LIMIT 1", (ticker,)
    ).fetchone()
    skew = (market or {}).get("skew") or {}
    return [
        0.0, 0.0, _catalyst_status_score(dominant_catalyst), _buyback_signal_score(json.dumps(buyback)),
        (market or {}).get("atm_iv_pct") or 0.0, skew.get("skew_pts") or 0.0, div_row[0] if div_row else 0.0,
    ]


def estimate_earnings_probability(ticker, conn):
    """Checks evidence volume FIRST, picks the appropriate output mode --
    never manufactures a false-precision probability from thin data
    (Part 4). Magnitude always comes from live market pricing (Part 3 --
    high confidence, the market's own probability-weighted estimate);
    only the directional lean is gated by evidence volume.

    MODE 1 ("raw_pattern", <10 pooled reconciled predictions
    system-wide): no manufactured probability -- raw historical counts
    for THIS ticker instead ("4 of 6 prior quarters beat EPS; 3 of those
    4 saw a next-reaction price increase"), with an explicit
    too-thin-for-a-probability note. prob_up/prob_down/prob_flat are all
    None in this mode -- there's no percentage to show.

    MODE 2 ("bayesian", 10-30 pooled): prob_up/prob_down/prob_flat are
    DERIVED by summing build_earnings_scenario_matrix()'s individual
    scenario probabilities grouped by expected direction (direction_
    totals) -- never computed independently, so this number and the
    SCENARIO BREAKDOWN table it's built from can never disagree with
    each other. The matrix itself already encodes the qualitative
    synthesis (a named catalyst's Achieved/Delayed split, this ticker's
    own Bayesian-blended EPS/revenue beat rates); Extreme IV is the one
    additional dampening applied here (widens uncertainty toward
    neutral, since IV-crush risk isn't part of the matrix's own
    templates). The result is then hard-aligned to never contradict the
    Briefing's own next-earnings verdict (direction and confidence), per
    the "must never contradict, only translate" requirement.

    MODE 3 ("trained_model", train_earnings_direction_model() returns a
    validated model): uses its predict_proba directly (see
    _current_qualitative_feature_row's honest caveat about what "live"
    means for a model trained partly on post-hoc EPS/revenue surprise
    features), states its held-out accuracy alongside the prediction,
    and is still hard-aligned to the Briefing's verdict same as Mode 2."""
    pooled_reconciled_count = conn.execute(
        "SELECT COUNT(*) FROM earnings_predictions WHERE prediction_correct IS NOT NULL"
    ).fetchone()[0]

    market = get_market_pricing_signals(ticker, conn)
    briefing_inputs = extract_simulator_inputs_from_briefing(ticker, conn)
    # Block B -- computed unconditionally, BEFORE the Mode 1/2/3 branch
    # below, and included in every return path regardless of mode. This
    # is the fix: Block B must never be gated behind Block A's historical-
    # sample-size check.
    qualitative_read = generate_qualitative_directional_read(ticker, conn, briefing_inputs, market=market)
    # Scenario matrix -- also computed unconditionally and included in
    # every return path (Part 2: its own always-visible section,
    # independent of Mode 1/2/3). Mode 2/3's prob_up/down/flat below are
    # DERIVED by summing this matrix's rows grouped by direction (Part
    # 4), not computed independently -- see direction_totals.
    scenario_matrix = build_earnings_scenario_matrix(ticker, conn, briefing_inputs=briefing_inputs, market=market)
    direction_totals = {"Up": 0.0, "Down": 0.0, "Flat": 0.0}
    for row in scenario_matrix["rows"]:
        for k, w in _scenario_direction_weights(row["expected_direction"]).items():
            direction_totals[k] += row["probability"] * w
    history_rows = _read_earnings_history_real_cache(conn, ticker, limit=8)
    magnitude_estimate_pct = market.get("iv_implied_move_pct")

    beats = [
        r for r in history_rows
        if r.get("eps_beat_miss_pct") is not None and r["eps_beat_miss_pct"] > 0
        and r.get("price_reaction_pct") is not None
    ]
    beats_up = [r for r in beats if r["price_reaction_pct"] > 0]

    if pooled_reconciled_count < 10:
        sentence = (
            f"{len(beats_up)} of {len(beats)} prior quarters that beat EPS saw a next-reaction price "
            f"increase (of {len(history_rows)} tracked quarters total)." if beats else
            (f"{len(history_rows)} tracked quarters on record, but none with both a real beat and a "
             f"measured price reaction yet." if history_rows else
             "No historical earnings-reaction data on record yet for this ticker.")
        )
        return {
            "ticker": ticker, "mode": "raw_pattern", "magnitude_estimate_pct": magnitude_estimate_pct,
            "prob_up": None, "prob_down": None, "prob_flat": None, "confidence_level": None,
            "raw_counts": {
                "beats": len(beats), "beats_followed_by_up_move": len(beats_up),
                "total_tracked_quarters": len(history_rows),
                "pooled_reconciled_predictions": pooled_reconciled_count,
            },
            "sentence": sentence,
            "note": (f"Sample size too small for a reliable probability ({pooled_reconciled_count} pooled "
                     f"reconciled predictions system-wide, need 10+) -- showing raw counts instead."),
            "reasoning": [], "market_pricing": market, "briefing_inputs": briefing_inputs,
            "qualitative_read": qualitative_read, "scenario_matrix": scenario_matrix,
        }

    trained = train_earnings_direction_model(conn)
    reasoning = []
    if trained is not None:
        mode = "trained_model"
        feature_row = _current_qualitative_feature_row(conn, ticker, market, briefing_inputs)
        proba = trained["model"].predict_proba([feature_row])[0]
        classes = list(trained["model"].classes_)
        prob_up = float(proba[classes.index(1)]) if 1 in classes else 0.5
        reasoning.append(
            f"trained {trained['model_type']} model: {trained['held_out_accuracy']:.0%} held-out accuracy "
            f"vs {trained['bayesian_baseline_accuracy']:.0%} for the Mode 2 Bayesian baseline, on "
            f"{trained['n_samples']} reconciled predictions"
        )
        reasoning.append(
            "live prediction assumes an in-line EPS/revenue surprise (unknowable pre-earnings) -- reflects "
            "the qualitative/market context only, not a beat/miss guess"
        )
        model_meta = {"model_type": trained["model_type"], "held_out_accuracy": trained["held_out_accuracy"],
                      "bayesian_baseline_accuracy": trained["bayesian_baseline_accuracy"],
                      "n_samples": trained["n_samples"]}
    else:
        mode = "bayesian"
        model_meta = None
        # Part 4: derived by SUMMING the scenario matrix's rows grouped
        # by direction (direction_totals, computed once above) -- not a
        # separately-computed heuristic. This is what guarantees Block A's
        # top-line number and the SCENARIO BREAKDOWN table can never
        # disagree: they're the same arithmetic. The old "beats don't
        # pay"/catalyst-dampening/macro-overhang heuristics that used to
        # live here are now redundant with (and would double-count) the
        # matrix's own catalyst-split templates and Bayesian-blended EPS/
        # revenue beat rates.
        non_flat = direction_totals["Up"] + direction_totals["Down"]
        lean = (direction_totals["Up"] / non_flat) if non_flat > 0 else 0.5
        distance = lean - 0.5
        reasoning.append(
            f"scenario matrix (Part 4): {direction_totals['Up']:.0%} Up / {direction_totals['Down']:.0%} Down / "
            f"{direction_totals['Flat']:.0%} Flat, summed across {len(scenario_matrix['rows'])} EPS × Revenue"
            + (f" × {scenario_matrix['named_catalyst']}" if scenario_matrix.get("named_catalyst") else "")
            + " scenarios -- this directional lean is derived directly from that breakdown, not computed "
              "independently"
        )

        atm_iv = market.get("atm_iv_pct")
        if atm_iv is not None and atm_iv >= 100:
            distance *= 0.6
            reasoning.append(f"Extreme IV ({atm_iv:.0f}%) -- high crush risk dampens directional conviction")

        prob_up = 0.5 + distance

    next_verdict = (briefing_inputs or {}).get("next_earnings_verdict") or {}
    verdict_bias = (next_verdict.get("bias") or "").upper()
    verdict_confidence = next_verdict.get("confidence")
    if verdict_bias == "BULLISH" and prob_up < 0.5:
        prob_up = 0.5 + max(abs(prob_up - 0.5), 0.05)
        reasoning.append("re-aligned upward to match the Briefing's own BULLISH next-earnings verdict")
    elif verdict_bias == "BEARISH" and prob_up > 0.5:
        prob_up = 0.5 - max(abs(prob_up - 0.5), 0.05)
        reasoning.append("re-aligned downward to match the Briefing's own BEARISH next-earnings verdict")
    elif verdict_bias == "NEUTRAL" and abs(prob_up - 0.5) > 0.08:
        prob_up = 0.58 if prob_up > 0.5 else 0.42
        reasoning.append("capped near 50/50 to match the Briefing's own NEUTRAL next-earnings verdict")
    prob_up = min(0.95, max(0.05, prob_up))

    # flat_share comes from the SAME scenario matrix direction_totals used
    # for the bayesian lean above (Part 4) -- Mode 3 (trained_model) uses
    # it too now, for consistency, even though its up/down split comes
    # from the trained model's own predict_proba.
    flat_share = round(direction_totals["Flat"], 4)
    remaining = 1 - flat_share
    prob_up_final = round(prob_up * remaining, 4)
    prob_down_final = round((1 - prob_up) * remaining, 4)
    prob_flat_final = round(flat_share, 4)

    confidence_rank = {"Low": 0, "Medium": 1, "High": 2}
    mechanical_confidence = (
        "High" if abs(prob_up - 0.5) >= 0.20 else "Medium" if abs(prob_up - 0.5) >= 0.08 else "Low"
    )
    confidence_level = (
        min([verdict_confidence, mechanical_confidence], key=lambda c: confidence_rank.get(c, 1))
        if verdict_confidence in confidence_rank else mechanical_confidence
    )
    if verdict_confidence and confidence_level != verdict_confidence:
        reasoning.append(
            f"confidence capped at {confidence_level} -- never exceeds the Briefing's own "
            f"{verdict_confidence} confidence in its next-earnings verdict"
        )

    predicted_direction = (
        "UP" if prob_up_final > prob_down_final + 0.05
        else "DOWN" if prob_down_final > prob_up_final + 0.05 else "FLAT"
    )

    return {
        "ticker": ticker, "mode": mode, "magnitude_estimate_pct": magnitude_estimate_pct,
        "prob_up": prob_up_final, "prob_down": prob_down_final, "prob_flat": prob_flat_final,
        "confidence_level": confidence_level, "predicted_direction": predicted_direction,
        "model_meta": model_meta, "reasoning": reasoning,
        "market_pricing": market, "briefing_inputs": briefing_inputs, "qualitative_read": qualitative_read,
        "scenario_matrix": scenario_matrix,
    }


EARNINGS_STRATEGY_RISK_TIERS = ["Balanced", "Speculative"]  # never Conservative/ITM (defeats earnings leverage) or the 0-0.10 deep-OTM tier (rarely survives IV crush)


def _nearest_expirations_at_or_after(conn, ticker, days_to_earnings, n=2):
    """The N real cached expiration dates closest to (but not before) the
    earnings date -- "expiring the week of/after earnings," literally.
    Deliberately NOT built on find_option_candidates' coarse DTE
    *buckets*: a bucket like "1-3 months" (31-90 days) can contain real
    expirations both well before AND (in the worst case) miss the
    nearest one just after a bucket boundary -- e.g. earnings 72 days
    out with real expirations at 33/61/96 days: 96 falls in the 91-179
    gap between "1-3 months" and "6+ months (LEAPS)", so a bucket-based
    filter would silently skip straight to a LEAPS-dated contract
    instead of the genuinely nearest one. This queries real expirations
    directly instead."""
    latest_date_row = conn.execute(
        "SELECT MAX(fetch_date) FROM options_flow WHERE ticker=?", (ticker,)
    ).fetchone()
    latest_date = latest_date_row[0] if latest_date_row else None
    if not latest_date:
        return []
    exp_rows = conn.execute(
        "SELECT DISTINCT expiration FROM options_flow WHERE ticker=? AND fetch_date=? ORDER BY expiration",
        (ticker, latest_date),
    ).fetchall()
    today = date.today()
    floor_days = days_to_earnings if days_to_earnings is not None else 0
    dated = []
    for (exp,) in exp_rows:
        try:
            dte = (pd.to_datetime(exp).date() - today).days
        except (ValueError, TypeError):
            continue
        if dte >= floor_days:
            dated.append((dte, exp))
    dated.sort()
    return [exp for _dte, exp in dated[:n]]


def _earnings_candidate_contracts(conn, ticker, direction, expirations, top_n=3):
    """Delta-tier (Balanced/Speculative) + liquidity-filtered, badge-
    annotated candidates restricted to a specific set of real
    expirations (see _nearest_expirations_at_or_after) -- the earnings-
    picker's own scoped version of find_option_candidates' filter/rank/
    badge logic, reusing the same building blocks (RISK_PROFILE_DELTA_
    RANGES, OPTION_PICKER_MIN_OPEN_INTEREST/MAX_SPREAD_PCT,
    annotate_options_badges) rather than find_option_candidates itself,
    since that function's unit of selection is a DTE bucket, not a fixed
    set of dates."""
    if not expirations:
        return []
    option_type = "call" if direction == "Bullish" else "put"
    latest_date_row = conn.execute(
        "SELECT MAX(fetch_date) FROM options_flow WHERE ticker=?", (ticker,)
    ).fetchone()
    latest_date = latest_date_row[0] if latest_date_row else None
    if not latest_date:
        return []
    placeholders = ",".join("?" * len(expirations))
    df = pd.read_sql_query(
        f"""SELECT expiration, strike, option_type, volume, open_interest, volume_oi_ratio,
                   implied_volatility, last_price, underlying_price, delta, gamma, theta, vega, bid, ask
            FROM options_flow WHERE ticker=? AND fetch_date=? AND option_type=?
                  AND expiration IN ({placeholders})""",
        conn, params=(ticker, latest_date, option_type, *expirations),
    )
    if df.empty:
        return []

    annotated = annotate_options_badges(df)
    delta_los = [RISK_PROFILE_DELTA_RANGES[t][0] for t in EARNINGS_STRATEGY_RISK_TIERS]
    delta_his = [RISK_PROFILE_DELTA_RANGES[t][1] for t in EARNINGS_STRATEGY_RISK_TIERS]
    lo, hi = min(delta_los), max(delta_his)
    annotated["abs_delta"] = annotated["delta"].abs()
    annotated["spread_pct2"] = annotated.apply(
        lambda r: compute_spread_pct(r.get("bid"), r.get("ask"), r.get("last_price")), axis=1
    )
    filtered = annotated[
        annotated["delta"].notna() & (annotated["abs_delta"] >= lo) & (annotated["abs_delta"] < hi)
        & (annotated["open_interest"].fillna(0) >= OPTION_PICKER_MIN_OPEN_INTEREST)
        & (annotated["spread_pct2"].fillna(999) <= OPTION_PICKER_MAX_SPREAD_PCT)
    ]
    # BUG FIX (verified live against real WDC data): sorting by open
    # interest ALONE, across BOTH target expirations combined, let a
    # single outlier-OI strike in the FARTHER expiration (e.g. WDC's
    # $600 Dec-18 call, OI=670 vs. its own expiry's next-highest of 313)
    # rank #1 ahead of every near-the-money, nearer-expiry candidate --
    # producing an 18%-OTM, 124-DTE "Balanced-delta" pick with a
    # breakeven 29% above spot. Sorting by expiration FIRST (nearest
    # preferred), then OI/spread as the tiebreak WITHIN that expiration,
    # keeps candidates concentrated in the nearer, more earnings-
    # relevant expiry unless it genuinely has nothing to offer.
    filtered = filtered.sort_values(
        ["expiration", "open_interest", "spread_pct2"], ascending=[True, False, True], na_position="last"
    )

    today = date.today()
    candidates = []
    for _, r in filtered.head(top_n).iterrows():
        try:
            dte = (pd.to_datetime(r["expiration"]).date() - today).days
        except (ValueError, TypeError):
            dte = None
        breakeven = (
            (r["strike"] + r["last_price"]) if option_type == "call" else (r["strike"] - r["last_price"])
        ) if pd.notna(r["last_price"]) else None
        candidates.append({
            "expiration": r["expiration"], "strike": float(r["strike"]), "type": option_type,
            "delta": _none_if_nan(r["delta"]), "iv_pct": _none_if_nan(r["iv_pct"]),
            # Same real per-contract Greeks already selected in the SQL
            # query above and already used for the delta/theta badges --
            # gamma/theta/vega were being dropped here instead of carried
            # through to the candidate dict, so the OPTIONS STRATEGY cards
            # had no numbers to show alongside their interpretive badges.
            "gamma": _none_if_nan(r["gamma"]), "theta": _none_if_nan(r["theta"]),
            "vega": _none_if_nan(r["vega"]),
            "last_price": _none_if_nan(r["last_price"]),
            "cost_per_contract": _none_if_nan(r["last_price"] * 100) if pd.notna(r["last_price"]) else None,
            "breakeven": _none_if_nan(breakeven),
            "open_interest": int(r["open_interest"]) if pd.notna(r["open_interest"]) else None,
            "volume": int(r["volume"]) if pd.notna(r["volume"]) else None,
            "spread_pct": _none_if_nan(r["spread_pct2"]), "dte": dte,
            "badges": {
                "delta": {"label": r["delta_badge"], "note": None},
                "iv": {"label": r["iv_badge"], "note": None},
                "vol_oi": {"label": r["vol_oi_badge"], "note": None},
                "dte": {"label": r["dte_badge"], "note": None},
                "theta": {"label": r["theta_badge"], "note": None},
                "spread": {"label": r["spread_badge"], "note": None},
            },
        })
    return candidates


def recommend_earnings_strategy(ticker, conn, probability_result=None):
    """Single-leg-only options picker (Part 5) -- never straddles,
    strangles, or any multi-leg strategy. Inherits estimate_earnings_
    probability()'s mode and confidence rather than manufacturing more
    conviction than either the evidence or the AI Briefing itself has:

    - Mode 1 (raw_pattern): no directional call. Returns a balanced-delta
      call AND put both, explicitly framed as magnitude-only (Part 3's
      market-pricing signals don't need historical data).
    - Mode 2/3 with a real lean: calls if prob_up meaningfully exceeds
      prob_down (vice versa for puts), 2-3 candidates from the Balanced/
      Speculative delta tiers (EARNINGS_STRATEGY_RISK_TIERS) expiring
      the week of/after earnings.
    - Always attaches an explicit `caution` string when confidence is
      Low/Medium due to named risk factors (a detected 'beats don't pay'
      pattern, Extreme IV, a named macro overhang) -- the recommendation
      must never look more confident than the Briefing's own stated
      confidence."""
    probability_result = probability_result or estimate_earnings_probability(ticker, conn)
    market = probability_result.get("market_pricing") or get_market_pricing_signals(ticker, conn)
    briefing_inputs = probability_result.get("briefing_inputs") or {}
    mode = probability_result["mode"]

    next_verdict = briefing_inputs.get("next_earnings_verdict") or {}
    earnings_date = next_verdict.get("target_date")
    days_to_earnings = None
    if earnings_date:
        try:
            days_to_earnings = (pd.Timestamp(earnings_date).date() - date.today()).days
        except (TypeError, ValueError):
            days_to_earnings = None
    # Fetch wide enough to reach ANY real earnings date, however far out --
    # a fixed high cap rather than trying to guess the right bucket size.
    cached_options_flow(conn, ticker, max_age_hours=0.25, force_refresh=True, max_expirations=50)
    target_expirations = _nearest_expirations_at_or_after(conn, ticker, days_to_earnings, n=2)

    def _gather(direction):
        return _earnings_candidate_contracts(conn, ticker, direction, target_expirations, top_n=3)

    magnitude_pct = market.get("iv_implied_move_pct")

    if mode == "raw_pattern":
        # Part 8 -- Mode 1 has no statistically validated probability,
        # but Block B's qualitative read (the Briefing's own verdict) is
        # still real signal and shouldn't be thrown away just because
        # Block A's historical sample is thin. A clear non-NEUTRAL bias
        # there gets a directional lean, explicitly labeled as
        # qualitative-only -- never presented as validated.
        qualitative_read = probability_result.get("qualitative_read")
        qual_bias = (qualitative_read or {}).get("bias")
        qual_direction = "Bullish" if qual_bias == "BULLISH" else "Bearish" if qual_bias == "BEARISH" else None

        base_note = (
            f"The market is pricing a ±{magnitude_pct:g}% move either direction." if magnitude_pct
            else "No live IV-based magnitude estimate available."
        )
        if qual_direction:
            factors = []
            beat_pattern = briefing_inputs.get("historical_beat_reaction_pattern") or {}
            if beat_pattern.get("pattern_detected"):
                factors.append("beats-that-don't-pay pattern")
            if briefing_inputs.get("macro_overhang"):
                factors.append("a named macro/regulatory overhang")
            n_dampening = sum(
                1 for c in (briefing_inputs.get("catalyst_risk_factors") or []) if c["classification"] == "RISK-DAMPENING"
            )
            if n_dampening:
                factors.append(f"{n_dampening} risk-dampening catalyst(s)")
            factor_txt = f" ({' + '.join(factors)})" if factors else ""
            note = (
                f"{base_note} Leaning toward {'calls' if qual_direction == 'Bullish' else 'puts'} based on the "
                f"qualitative setup above{factor_txt}, NOT a statistically validated probability -- "
                f"historical sample size is still thin for this ticker."
            )
            candidates = _gather(qual_direction)
        else:
            note = (
                f"{base_note} Historical direction data is too thin to lean toward either side yet, and the "
                f"AI Briefing's own verdict is Neutral. Here's a balanced-delta call and put both worth "
                f"considering."
            )
            candidates = {"calls": _gather("Bullish"), "puts": _gather("Bearish")}

        return {
            "mode": mode, "directional_call": qual_direction is not None, "direction": qual_direction,
            "direction_source": "qualitative_only" if qual_direction else None,
            "earnings_date": earnings_date, "note": note, "candidates": candidates, "caution": None,
        }

    prob_up, prob_down = probability_result.get("prob_up") or 0.5, probability_result.get("prob_down") or 0.5
    direction = "Bullish" if prob_up > prob_down + 0.05 else "Bearish" if prob_down > prob_up + 0.05 else None
    candidates = _gather(direction) if direction else {"calls": _gather("Bullish"), "puts": _gather("Bearish")}

    confidence_level = probability_result.get("confidence_level")
    caution = None
    if confidence_level in ("Low", "Medium"):
        risk_bits = []
        beat_pattern = briefing_inputs.get("historical_beat_reaction_pattern") or {}
        if beat_pattern.get("pattern_detected"):
            risk_bits.append("a historically 'beat that doesn't pay' pattern")
        atm_iv = market.get("atm_iv_pct")
        if atm_iv is not None and atm_iv >= 100:
            risk_bits.append(f"Extreme IV ({atm_iv:.1f}%)")
        if briefing_inputs.get("macro_overhang"):
            risk_bits.append("a named macro/regulatory overhang")
        n_dampening = sum(
            1 for c in (briefing_inputs.get("catalyst_risk_factors") or []) if c["classification"] == "RISK-DAMPENING"
        )
        if n_dampening:
            risk_bits.append(f"{n_dampening} risk-dampening catalyst(s)")
        if risk_bits:
            caution = (
                f"The AI Briefing flags this as a setup with {', '.join(risk_bits)} -- even a correct "
                f"directional call risks being eroded by IV crush or an adverse reaction. Consider sizing "
                f"down or skipping this play."
            )
        else:
            caution = f"The AI Briefing's own confidence in this call is {confidence_level} -- size accordingly."

    return {
        "mode": mode, "directional_call": direction is not None, "direction": direction,
        "earnings_date": earnings_date, "confidence_level": confidence_level,
        "note": (
            f"Directional lean: {prob_up:.0%} up / {prob_down:.0%} down "
            f"({'no strong lean, showing both sides' if direction is None else f'leaning {direction.lower()}'})."
        ),
        "candidates": candidates,
        "caution": caution,
    }


# --------------------------------------------------------------------------
# Part 6 -- P/L simulation with IV crush modeling.
# --------------------------------------------------------------------------

DEFAULT_IV_CRUSH_RATIO = 0.55  # conservative default: post-earnings IV reverts to ~55% of its pre-earnings level


def _estimate_iv_crush_ratio(conn, ticker):
    """Tries to derive a REAL historical pre/post-earnings IV crush ratio
    from this ticker's own ticker_snapshots (iv_snapshot + signed
    hours_to_earnings, grouped by earnings_date) -- averaging every
    snapshot taken before the event vs. every one taken after, per
    event, then averaging across events. Falls back to
    DEFAULT_IV_CRUSH_RATIO when there isn't at least one real event with
    snapshots on both sides of it -- "source" in the return tells the
    caller which happened, so this is never silently presented as more
    precise than it is."""
    rows = conn.execute(
        """SELECT earnings_date, hours_to_earnings, iv_snapshot FROM ticker_snapshots
           WHERE ticker=? AND earnings_date IS NOT NULL AND iv_snapshot IS NOT NULL""",
        (ticker,),
    ).fetchall()
    by_event = {}
    for earnings_date, hours, iv in rows:
        bucket = by_event.setdefault(earnings_date, {"pre": [], "post": []})
        (bucket["pre"] if hours < 0 else bucket["post"]).append(iv)

    ratios = []
    for ev in by_event.values():
        if ev["pre"] and ev["post"]:
            pre_avg = sum(ev["pre"]) / len(ev["pre"])
            post_avg = sum(ev["post"]) / len(ev["post"])
            if pre_avg > 0:
                ratios.append(post_avg / pre_avg)
    if ratios:
        return {"ratio": round(sum(ratios) / len(ratios), 3), "source": "historical", "n_events": len(ratios)}
    return {"ratio": DEFAULT_IV_CRUSH_RATIO, "source": "default_estimate", "n_events": 0}


def simulate_earnings_pl(ticker, conn, candidates, market=None):
    """P/L simulation with IV crush (Part 6, extended for Part 5's
    decomposition). For each candidate option (as returned by recommend_
    earnings_strategy/find_option_candidates -- needs strike/type/
    expiration/last_price/iv_pct/dte) and each outcome scenario (Down/
    Flat/Up, magnitude from the market's own IV-implied move -- Part 3,
    not a hand-picked number), simulates the post-earnings option value
    via Black-Scholes with the scenario's price move applied to spot and
    a modeled IV crush (_estimate_iv_crush_ratio), and returns $ and %
    P/L per scenario -- decomposed into a price-movement component
    (spot moved, IV unchanged) and an IV-crush component (IV crushed on
    top of that), so "how much of this loss is just IV crush" is a real
    visible number, not blended into one figure (Part 5).

    Assumes the position is evaluated ~2 trading days after the report
    (T shrinks by that much from today's DTE) -- a reasonable "how did
    this trade do right after the reaction" checkpoint, not a
    hold-to-expiry assumption. Falls back to intrinsic value if a
    degenerate Black-Scholes input (e.g. T<=0 for a same-day-of-earnings
    expiry) would otherwise return NaN."""
    market = market or get_market_pricing_signals(ticker, conn)
    spot = market.get("current_price")
    magnitude_pct = market.get("iv_implied_move_pct")
    crush = _estimate_iv_crush_ratio(conn, ticker)

    if spot is None or magnitude_pct is None:
        return {"scenarios": [], "iv_crush": crush,
                "note": "Missing current price or IV-implied move -- can't simulate P/L."}

    scenario_moves = [("Down", -magnitude_pct / 100), ("Flat", 0.0), ("Up", magnitude_pct / 100)]
    results = []
    for c in candidates:
        strike, option_type = c.get("strike"), c.get("type")
        entry_price, entry_iv, dte = c.get("last_price"), c.get("iv_pct"), c.get("dte")
        if not entry_price or entry_iv is None or dte is None or entry_price <= 0:
            continue
        remaining_days = max(dte - 2, 1)
        T = remaining_days / 365.0
        entry_iv_frac = entry_iv / 100
        post_iv_frac = max(0.01, entry_iv_frac * crush["ratio"])
        breakeven = (strike + entry_price) if option_type == "call" else (strike - entry_price)

        scenario_pl = []
        for label, move_pct in scenario_moves:
            scenario_spot = spot * (1 + move_pct)

            def _theo(sigma):
                v = bs_price(scenario_spot, strike, T, RISK_FREE_RATE_DEFAULT, sigma, option_type=option_type)
                if math.isnan(v):
                    v = max((scenario_spot - strike) if option_type == "call" else (strike - scenario_spot), 0.0)
                return v

            price_only_value = _theo(entry_iv_frac)   # spot moved, IV unchanged
            full_value = _theo(post_iv_frac)           # spot moved AND IV crushed

            price_component_dollar = (price_only_value - entry_price) * 100
            iv_crush_component_dollar = (full_value - price_only_value) * 100
            pl_dollar = price_component_dollar + iv_crush_component_dollar
            pl_pct = (full_value - entry_price) / entry_price * 100
            scenario_pl.append({
                "scenario": label, "price_move_pct": round(move_pct * 100, 1),
                "scenario_spot": round(scenario_spot, 2), "post_earnings_option_price": round(full_value, 2),
                "pl_dollar": round(pl_dollar, 2), "pl_pct": round(pl_pct, 1),
                "price_component_dollar": round(price_component_dollar, 2),
                "iv_crush_component_dollar": round(iv_crush_component_dollar, 2),
                "entry_iv_pct": entry_iv,
            })
        results.append({
            "expiration": c.get("expiration"), "strike": strike, "type": option_type,
            "entry_price": entry_price, "entry_iv_pct": entry_iv, "spot": spot, "breakeven": round(breakeven, 2),
            "post_earnings_iv_pct": round(post_iv_frac * 100, 1), "scenarios": scenario_pl,
        })

    return {"scenarios": results, "iv_crush": crush, "note": None}


def compute_pl_curve_at_expiration(candidate, spot, price_range_pct=0.15, n_points=35):
    """OptionStrat-style P/L-AT-EXPIRATION curve (Part 3): intrinsic-
    value payoff at n_points prices spanning spot * (1 +/- price_range_
    pct), minus entry cost -- the standard kinked-linear options P/L
    diagram (zero time value left at expiration, so this is intrinsic
    value, not a Black-Scholes-with-remaining-time curve)."""
    strike, entry_price, option_type = candidate.get("strike"), candidate.get("last_price"), candidate.get("type")
    if spot is None or strike is None or entry_price is None:
        return None
    lo, hi = spot * (1 - price_range_pct), spot * (1 + price_range_pct)
    step = (hi - lo) / (n_points - 1)
    prices = [lo + i * step for i in range(n_points)]
    pl_dollar = []
    for p in prices:
        intrinsic = max(p - strike, 0.0) if option_type == "call" else max(strike - p, 0.0)
        pl_dollar.append(round((intrinsic - entry_price) * 100, 2))
    breakeven = (strike + entry_price) if option_type == "call" else (strike - entry_price)
    return {
        "prices": [round(p, 2) for p in prices], "pl_dollar": pl_dollar, "breakeven": round(breakeven, 2),
        "max_loss": round(-entry_price * 100, 2), "cost": round(entry_price * 100, 2), "strike": strike,
        "spot": spot,
    }


# The canonical "directional surprise" / "inline print" move used to mark
# scenario reference points ON the earnings-date P/L curve -- collapses
# _SCENARIO_TEMPLATES' several fixed ranges (5-9% Up, 5-9% Down, 0-3% Flat)
# to one representative midpoint per bucket, so the CHART's three markers
# are the same Down/Flat/Up buckets Block A's prob_down/prob_flat/prob_up
# and the SCENARIO BREAKDOWN table both already use -- not a fourth,
# disconnected set of numbers (Part 2.4 of the P/L-chart rebuild).
_SCENARIO_CONVENTION_ANCHORS_PCT = {"Down": -6.0, "Flat": 1.5, "Up": 6.5}


def compute_pl_curve_at_earnings(candidate, spot, crush, earnings_date, price_range_pct=0.15, n_points=35):
    """P/L curve valued ON THE EARNINGS DATE itself -- Black-Scholes with
    the remaining time from earnings to expiration and the MODELED post-
    earnings-crush IV (_estimate_iv_crush_ratio) -- not held to
    expiration. This is the curve that actually matters for an earnings
    play (Part 2 of the P/L-chart rebuild): most earnings trades are
    evaluated/exited right after the reaction, not held to expiry, so
    compute_pl_curve_at_expiration's zero-time intrinsic-value curve alone
    understates the real time value still left in the contract at that
    point. Falls back to the expiration-style intrinsic payoff if T<=0
    (an earnings date on/after expiration) or if Black-Scholes degenerates
    to NaN for a given price."""
    strike, entry_price, option_type = candidate.get("strike"), candidate.get("last_price"), candidate.get("type")
    entry_iv, expiration = candidate.get("iv_pct"), candidate.get("expiration")
    if None in (spot, strike, entry_price, entry_iv, expiration) or entry_price <= 0:
        return None
    try:
        exp_date = pd.to_datetime(expiration).date()
        earnings_dt = pd.to_datetime(earnings_date).date() if earnings_date else exp_date
    except (ValueError, TypeError):
        return None

    days_remaining = max((exp_date - earnings_dt).days, 0)
    T = days_remaining / 365.0
    post_iv_frac = max(0.01, (entry_iv / 100) * crush["ratio"])

    lo, hi = spot * (1 - price_range_pct), spot * (1 + price_range_pct)
    step = (hi - lo) / (n_points - 1)
    prices = [lo + i * step for i in range(n_points)]
    pl_dollar = []
    for p in prices:
        if T <= 0:
            theo = max(p - strike, 0.0) if option_type == "call" else max(strike - p, 0.0)
        else:
            theo = bs_price(p, strike, T, RISK_FREE_RATE_DEFAULT, post_iv_frac, option_type=option_type)
            if math.isnan(theo):
                theo = max(p - strike, 0.0) if option_type == "call" else max(strike - p, 0.0)
        pl_dollar.append(round((theo - entry_price) * 100, 2))

    # The three scenario reference points, computed on THIS SAME curve
    # (same T, same post-earnings IV) so the chart markers are exact,
    # real values -- not interpolated/approximated from the price grid.
    markers = []
    for label, move_pct in _SCENARIO_CONVENTION_ANCHORS_PCT.items():
        marker_price = spot * (1 + move_pct / 100)
        if T <= 0:
            theo = max(marker_price - strike, 0.0) if option_type == "call" else max(strike - marker_price, 0.0)
        else:
            theo = bs_price(marker_price, strike, T, RISK_FREE_RATE_DEFAULT, post_iv_frac, option_type=option_type)
            if math.isnan(theo):
                theo = max(marker_price - strike, 0.0) if option_type == "call" else max(strike - marker_price, 0.0)
        marker_pl = (theo - entry_price) * 100
        markers.append({
            "label": label, "move_pct": move_pct, "price": round(marker_price, 2),
            "pl_dollar": round(marker_pl, 2), "pl_pct": round(marker_pl / (entry_price * 100) * 100, 1),
        })

    return {
        "prices": [round(p, 2) for p in prices], "pl_dollar": pl_dollar, "spot": spot,
        "post_earnings_iv_pct": round(post_iv_frac * 100, 1), "days_to_earnings": days_remaining,
        "scenario_markers": markers,
    }


def compute_probability_of_profit(spot, breakeven, T, r, sigma, option_type):
    """Genuine 'Chance of Profit' (Part 6) -- the risk-neutral lognormal
    probability the underlying finishes beyond BREAKEVEN (not just ITM
    at strike) by expiration -- the standard industry method. Black-
    Scholes' own d2 gives the risk-neutral P(S_T > K) as N(d2); this
    substitutes breakeven for K, since finishing ITM-but-below-breakeven
    on a debit purchase is still a loss, not a profit."""
    if None in (spot, breakeven, T, sigma) or T <= 0 or sigma <= 0 or spot <= 0 or breakeven <= 0:
        return None
    sqrtT = math.sqrt(T)
    d2 = (math.log(spot / breakeven) + (r - 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    return round(_norm_cdf(d2) if option_type == "call" else _norm_cdf(-d2), 4)


def compute_price_date_heatmap(candidate, spot, crush, earnings_date, n_price_levels=9, n_dates=6):
    """Price(Y) x date(X) grid of estimated P/L% per cell (Part 4) --
    Black-Scholes with time decay applied per date, and the modeled IV
    crush (_estimate_iv_crush_ratio) applied for any date on/after
    earnings_date. Date columns always include today, the earnings
    date (if it falls before expiration), and the expiration itself,
    per Part 4's stated minimum, filled out to n_dates with evenly
    spaced dates in between."""
    strike, entry_price = candidate.get("strike"), candidate.get("last_price")
    entry_iv, option_type = candidate.get("iv_pct"), candidate.get("type")
    expiration = candidate.get("expiration")
    if None in (strike, entry_price, entry_iv, option_type, expiration, spot):
        return None

    today = date.today()
    try:
        exp_date = pd.to_datetime(expiration).date()
    except (ValueError, TypeError):
        return None
    if (exp_date - today).days <= 0:
        return None

    date_points = {today, exp_date}
    earnings_dt = None
    if earnings_date:
        try:
            candidate_dt = pd.to_datetime(earnings_date).date()
            if today < candidate_dt <= exp_date:
                earnings_dt = candidate_dt
                date_points.add(earnings_dt)
        except (ValueError, TypeError):
            pass
    for d in pd.date_range(today, exp_date, periods=n_dates):
        if len(date_points) >= n_dates:
            break
        date_points.add(d.date())
    date_points = sorted(date_points)

    price_pcts = [-0.15 + i * 0.30 / (n_price_levels - 1) for i in range(n_price_levels)]
    price_levels = sorted((spot * (1 + pct) for pct in price_pcts), reverse=True)

    entry_iv_frac = entry_iv / 100
    grid = []
    for p in price_levels:
        row = []
        for d in date_points:
            days_remaining = max((exp_date - d).days, 0)
            T = days_remaining / 365.0
            sigma = max(entry_iv_frac * (crush["ratio"] if (earnings_dt and d >= earnings_dt) else 1.0), 0.01)
            if T <= 0:
                theo = max(p - strike, 0.0) if option_type == "call" else max(strike - p, 0.0)
            else:
                theo = bs_price(p, strike, T, RISK_FREE_RATE_DEFAULT, sigma, option_type=option_type)
                if math.isnan(theo):
                    theo = max(p - strike, 0.0) if option_type == "call" else max(strike - p, 0.0)
            row.append({
                "pl_pct": round((theo - entry_price) / entry_price * 100, 1),
                "pl_dollar": round((theo - entry_price) * 100, 2), "theo_price": round(theo, 2),
            })
        grid.append(row)

    return {
        "price_levels": [round(p, 2) for p in price_levels],
        "dates": [d.isoformat() for d in date_points], "grid": grid, "strike": strike, "spot": spot,
        "earnings_date_in_range": earnings_dt.isoformat() if earnings_dt else None,
    }


def _describe_scenario_condition(row):
    """Turns one scenario-matrix row into a natural-language condition
    clause, e.g. 'beats both EPS and revenue but HAMR technology rollout
    status remains unresolved' -- used by both the risk-scenario callout
    below and could be reused anywhere else a row needs a plain-English
    description rather than its raw eps/revenue/catalyst fields."""
    eps, rev = row["eps"], row["revenue"]
    if eps == "Beat" and rev == "Beat":
        base = "beats both EPS and revenue"
    elif eps == "Beat" and rev == "Miss":
        base = "beats EPS but misses revenue"
    elif eps == "Miss" and rev == "Beat":
        base = "misses EPS but beats revenue"
    else:
        base = "misses both EPS and revenue"
    cat = row.get("catalyst")
    if cat and cat != "N/A":
        # rpartition (last " — "), not partition (first) -- the catalyst
        # name itself can contain its own " — " (e.g. "Klarna membership
        # overhaul -- fees removed, ..."), and build_earnings_scenario_
        # matrix always appends " — {cat_state}" as the FINAL segment, so
        # splitting on the last occurrence is the one that's actually
        # guaranteed to isolate cat_state correctly.
        name, _, state = cat.rpartition(" — ")
        name = _catalyst_short_name(name)
        if "Delayed" in state or "Not achieved" in state:
            base += f" but {name} remains unresolved"
        elif "Achieved" in state:
            base += f" with {name} also confirmed"
    return base


def _highest_probability_risk_scenario(scenario_matrix, option_type, min_probability=0.15):
    """The single scenario-matrix row posing the biggest risk to a given
    option type (Part 4's second requirement) -- Down-leaning rows for a
    call, Up-leaning rows for a put -- ranked by (row probability x
    adverse-direction weight), so a compound label like 'Flat/Down' only
    counts its Down share. Returns None if the biggest such risk is
    below min_probability -- not worth calling out a scenario nobody
    should worry about."""
    if not scenario_matrix or not scenario_matrix.get("rows"):
        return None
    adverse = "Down" if option_type == "call" else "Up"
    best, best_score = None, 0.0
    for row in scenario_matrix["rows"]:
        score = row["probability"] * _scenario_direction_weights(row["expected_direction"]).get(adverse, 0.0)
        if score > best_score:
            best, best_score = row, score
    return (best, best_score) if best and best_score >= min_probability else None


def generate_contract_reasoning(candidate, contract_pl, probability_of_profit, scenario_matrix=None):
    """Deterministic, template-based per-contract reasoning (Part 7) --
    references the ACTUAL computed numbers (cost, the Flat scenario's
    IV-crush-decomposed P/L, breakeven, % move needed, chance of
    profit), never free-form generation. `contract_pl` is one entry from
    simulate_earnings_pl()'s "scenarios" list (already carries spot/
    breakeven/entry_iv_pct/post_earnings_iv_pct + the per-scenario
    price/IV-crush decomposition). When `scenario_matrix` is given
    (build_earnings_scenario_matrix's output), explicitly names the
    single highest-probability RISK scenario for this contract's
    direction -- e.g. the "beats everything but the catalyst hasn't
    landed" trap -- per Part 4 of the scenario-matrix fix."""
    strike, option_type = candidate.get("strike"), candidate.get("type")
    entry_price = candidate.get("last_price")
    cost = entry_price * 100 if entry_price else None
    flat = next((s for s in contract_pl["scenarios"] if s["scenario"] == "Flat"), None)
    breakeven, spot = contract_pl.get("breakeven"), contract_pl.get("spot")

    bits = []
    if cost is not None:
        bits.append(f"This ${strike:g} {option_type} costs ${cost:,.0f} today.")
    if flat:
        pl_d, pl_p = flat["pl_dollar"], flat["pl_pct"]
        iv_component = flat.get("iv_crush_component_dollar") or 0.0
        verb = "gain" if pl_d >= 0 else "lose"
        if pl_d < 0 and abs(iv_component) >= abs(pl_d) * 0.5:
            bits.append(
                f"Even if {candidate.get('ticker', 'the stock')} finishes exactly flat after earnings, this "
                f"option is projected to {verb} about ${abs(pl_d):,.0f} ({abs(pl_p):.0f}%) -- almost entirely "
                f"from IV crush (${abs(iv_component):,.0f} of that ${abs(pl_d):,.0f}), since the market is "
                f"pricing {contract_pl.get('entry_iv_pct', 0):.1f}% IV today but will likely settle closer to "
                f"{contract_pl.get('post_earnings_iv_pct', 0):.0f}% once earnings uncertainty resolves."
            )
        else:
            bits.append(
                f"If the stock finishes exactly flat after earnings, this option is projected to {verb} about "
                f"${abs(pl_d):,.0f} ({abs(pl_p):.0f}%)."
            )
    if breakeven is not None and spot:
        pct_needed = (breakeven - spot) / spot * 100
        direction = "above" if pct_needed > 0 else "below"
        bits.append(
            f"For this trade to be profitable, the stock needs to move enough to overcome both the IV crush "
            f"and time decay -- breakeven is ${breakeven:.2f}, roughly {abs(pct_needed):.1f}% {direction} the "
            f"current price."
        )
    if probability_of_profit is not None:
        bits.append(f"Modeled chance of profit by expiration: {probability_of_profit:.0%}.")

    risk = _highest_probability_risk_scenario(scenario_matrix, option_type)
    if risk:
        row, score = risk
        ticker_txt = candidate.get("ticker") or "the company"
        bits.append(
            f"The single largest risk to this {option_type}: a {score:.0%} probability scenario where "
            f"{ticker_txt} {_describe_scenario_condition(row)} -- {row['reasoning']}"
        )
    return " ".join(bits)


def analyze_earnings_contract(ticker, conn, candidate, market=None, earnings_date=None, scenario_matrix=None):
    """Assembles everything one contract's card needs (Part 9's layout
    order): header stats (net debit, max loss/profit, breakeven, chance
    of profit), the P/L-at-expiration curve (Part 3), the price x date
    heatmap (Part 4), IV-crush-decomposed scenario P/L (Part 5), and
    written reasoning (Part 7, now including the single highest-
    probability risk scenario from `scenario_matrix` when given -- Part
    4 of the scenario-matrix fix) -- one call per contract, so the
    dashboard doesn't have to re-derive any of this or risk the pieces
    disagreeing with each other. `scenario_matrix`: pass estimate_
    earnings_probability()'s scenario_matrix through here so the
    reasoning cites the SAME matrix shown in the SCENARIO BREAKDOWN
    section, not a freshly (and possibly differently) computed one."""
    market = market or get_market_pricing_signals(ticker, conn)
    spot = market.get("current_price")
    pl = simulate_earnings_pl(ticker, conn, [candidate], market=market)
    if not pl["scenarios"]:
        return None
    contract_pl = pl["scenarios"][0]

    dte = candidate.get("dte")
    entry_iv = candidate.get("iv_pct")
    T = max(dte, 1) / 365.0 if dte else None
    pop = (
        compute_probability_of_profit(
            spot, contract_pl.get("breakeven"), T, RISK_FREE_RATE_DEFAULT, entry_iv / 100 if entry_iv else None,
            candidate.get("type"),
        ) if spot and T else None
    )

    pl_curve = compute_pl_curve_at_expiration(candidate, spot)
    pl_curve_earnings = compute_pl_curve_at_earnings(candidate, spot, pl["iv_crush"], earnings_date)
    heatmap = compute_price_date_heatmap(candidate, spot, pl["iv_crush"], earnings_date)
    reasoning = generate_contract_reasoning(
        {**candidate, "ticker": ticker}, contract_pl, pop, scenario_matrix=scenario_matrix
    )

    entry_price = candidate.get("last_price") or 0.0
    net_debit = round(entry_price * 100, 2)
    return {
        "candidate": candidate, "net_debit": net_debit, "max_loss": net_debit,
        "max_profit": None,  # unbounded for a long call, capped-but-large for a long put -- shown as "Uncapped"/"Strike value" in the UI, not a fabricated number
        "breakeven": contract_pl.get("breakeven"), "probability_of_profit": pop,
        "pl_curve": pl_curve, "pl_curve_earnings": pl_curve_earnings, "heatmap": heatmap,
        "contract_pl": contract_pl, "iv_crush": pl["iv_crush"], "reasoning": reasoning,
    }


# --------------------------------------------------------------------------
# Parts 10-11 -- prediction logging (one row per simulator run) and the
# daily reconciliation pass that grades the final pre-earnings prediction
# against the real outcome once it's available.
# --------------------------------------------------------------------------

def log_earnings_prediction(conn, ticker, probability_result, strategy_result=None, is_final_prediction=False):
    """Logs one row to earnings_predictions (Part 10) -- called every
    EARNINGS SIMULATOR run for this ticker, not just once per earnings
    event, so the timeline of how a prediction evolved as new data came
    in is itself preserved (useful data on its own). Only the caller
    should ever pass is_final_prediction=True, for the last check-in
    before the earnings date -- that's the one row reconcile_earnings_
    predictions() later grades against the actual outcome.

    source_briefing_id links back to the exact ai_briefs row that
    informed this prediction (via probability_result['briefing_inputs']
    ['briefing_id'], see _read_ai_brief_cache) -- reconciliation reads it
    back to explain hits/misses in terms of the Briefing's own named
    qualitative factors, not just a bare correct/incorrect label."""
    briefing_inputs = probability_result.get("briefing_inputs") or {}
    earnings_date = (briefing_inputs.get("next_earnings_verdict") or {}).get("target_date")

    scenario_matrix = None
    if strategy_result and strategy_result.get("candidates"):
        raw = strategy_result["candidates"]
        candidates = raw if isinstance(raw, list) else (raw.get("calls", []) + raw.get("puts", []))
        if candidates:
            pl = simulate_earnings_pl(ticker, conn, candidates, market=probability_result.get("market_pricing"))
            scenario_matrix = pl["scenarios"]

    recommended_strategy = None
    if strategy_result:
        if strategy_result.get("direction"):
            side = "call" if strategy_result["direction"] == "Bullish" else "put"
            recommended_strategy = f"{strategy_result['direction']} ({side})"
        else:
            recommended_strategy = "No directional call -- balanced call/put both shown"

    briefing_id = briefing_inputs.get("briefing_id")
    conn.execute(
        """INSERT INTO earnings_predictions
               (ticker, earnings_date, predicted_at, mode, prob_up, prob_down, prob_flat,
                magnitude_estimate_pct, predicted_direction, recommended_strategy, source_briefing_id,
                scenario_matrix_json, confidence_level, is_final_prediction)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ticker, earnings_date, datetime.utcnow().isoformat(), probability_result.get("mode"),
         probability_result.get("prob_up"), probability_result.get("prob_down"),
         probability_result.get("prob_flat"), probability_result.get("magnitude_estimate_pct"),
         probability_result.get("predicted_direction"), recommended_strategy,
         str(briefing_id) if briefing_id is not None else None,
         json.dumps(scenario_matrix, default=str) if scenario_matrix else None,
         probability_result.get("confidence_level"), int(is_final_prediction)),
    )
    conn.commit()


def reconcile_earnings_predictions(conn):
    """Runs daily (Part 11, plus a manual SETTINGS button) -- finds
    earnings_predictions rows where earnings_date has passed (>=1 day
    ago) and reconciled_at IS NULL, restricted to is_final_prediction=1
    (only the last check-in before the event is graded). For each: pulls
    the actual price reaction/EPS/revenue result from earnings_history_real
    (matched by ticker+earnings_date -- the SAME real table used
    everywhere else in this engine, populated by the live pipeline or
    backfill), compares to predicted_direction, sets prediction_correct,
    and writes a plain-language reconciliation_notes string that
    specifically calls out whether the source Briefing's own named
    qualitative factors (a detected 'beats don't pay' pattern,
    risk-dampening catalysts) were consistent with the actual outcome --
    the richer signal a future trained model can learn from, beyond a
    bare correct/incorrect label. Rows with no matching earnings_history_
    real outcome yet (not reported/backfilled) are left alone -- tried
    again on the next daily run, not treated as a failure."""
    rows = conn.execute(
        """SELECT id, ticker, earnings_date, predicted_direction, mode, confidence_level, source_briefing_id
           FROM earnings_predictions
           WHERE is_final_prediction=1 AND reconciled_at IS NULL
                 AND earnings_date IS NOT NULL AND earnings_date <= date('now', '-1 day')""",
    ).fetchall()

    reconciled_count = 0
    for pred_id, ticker, earnings_date, predicted_direction, mode, confidence_level, source_briefing_id in rows:
        history_row = conn.execute(
            """SELECT actual_eps, beat_miss, revenue_actual, price_reaction_pct
               FROM earnings_history_real WHERE ticker=? AND earnings_date=?""",
            (ticker, earnings_date),
        ).fetchone()
        if not history_row or history_row[3] is None:
            continue  # real outcome not backfilled/available yet -- retry on the next daily run

        actual_eps, beat_miss_str, revenue_actual, price_reaction_pct = history_row
        actual_direction = "UP" if price_reaction_pct > 0 else "DOWN" if price_reaction_pct < 0 else "FLAT"
        prediction_correct = (
            int(predicted_direction == actual_direction) if predicted_direction in ("UP", "DOWN", "FLAT") else None
        )

        note_bits = [
            f"Predicted {predicted_direction or 'n/a'} ({mode}, {confidence_level or 'n/a'} confidence); "
            f"actual reaction was {actual_direction} ({price_reaction_pct:+.1f}%). "
            + ("Hit." if prediction_correct else "Miss." if prediction_correct is not None else "Unscored.")
        ]
        if source_briefing_id:
            brief_row = conn.execute(
                "SELECT brief_json FROM ai_briefs WHERE id=?", (source_briefing_id,)
            ).fetchone()
            if brief_row:
                try:
                    brief = json.loads(brief_row[0])
                except (TypeError, ValueError):
                    brief = {}
                beat_pattern = _detect_beats_dont_pay_pattern(
                    brief.get("earnings_track_record"), _read_earnings_history_real_cache(conn, ticker, limit=8)
                )
                catalysts = brief.get("catalysts") or []
                n_dampening = sum(
                    1 for c in catalysts if _classify_catalyst_risk(c.get("why_it_matters")) == "RISK-DAMPENING"
                )
                if beat_pattern.get("pattern_detected"):
                    if prediction_correct:
                        note_bits.append("The Briefing's 'beats don't pay' pattern flag helped call this correctly.")
                    elif actual_direction == "UP":
                        note_bits.append(
                            "The Briefing's 'beats don't pay' caution didn't prevent an incorrect call -- the "
                            "mechanical dampening may need to weight this pattern more heavily."
                        )
                if n_dampening:
                    if prediction_correct and actual_direction != "UP":
                        note_bits.append(
                            f"{n_dampening} risk-dampening catalyst(s) from the Briefing were consistent with "
                            f"the actual outcome."
                        )
                    elif not prediction_correct:
                        note_bits.append(
                            f"{n_dampening} risk-dampening catalyst(s) from the Briefing did not prevent an "
                            f"incorrect call."
                        )

        conn.execute(
            """UPDATE earnings_predictions
               SET actual_eps_result=?, actual_revenue_result=?, actual_price_move_pct=?, actual_direction=?,
                   prediction_correct=?, reconciliation_notes=?, reconciled_at=?
               WHERE id=?""",
            (f"{beat_miss_str or ''} (actual {actual_eps or 'n/a'})".strip(),
             str(revenue_actual) if revenue_actual else None, price_reaction_pct, actual_direction,
             prediction_correct, " ".join(note_bits), datetime.utcnow().isoformat(), pred_id),
        )
        reconciled_count += 1
    conn.commit()
    return {"reconciled_count": reconciled_count, "checked_count": len(rows)}


def get_simulator_track_record(conn):
    """Part 12 -- overall + per-mode + per-ticker accuracy across every
    reconciled earnings_predictions row, for the SETTINGS/EARNINGS
    SIMULATOR track-record panel. Every accuracy figure comes with its
    real sample size attached -- never a bare percentage with no
    denominator, same honesty convention as the rest of this engine.
    by_ticker only includes tickers with >=2 reconciled predictions
    (a single data point isn't a "track record")."""
    rows = conn.execute(
        "SELECT ticker, mode, prediction_correct FROM earnings_predictions WHERE prediction_correct IS NOT NULL"
    ).fetchall()
    if not rows:
        return {"overall": None, "by_mode": {}, "by_ticker": {}, "n_total": 0}

    n_total = len(rows)
    n_correct = sum(1 for r in rows if r[2])

    by_mode = {}
    for _ticker, mode, correct in rows:
        m = by_mode.setdefault(mode, {"n": 0, "correct": 0})
        m["n"] += 1
        m["correct"] += int(bool(correct))
    for m in by_mode.values():
        m["accuracy"] = round(m["correct"] / m["n"], 3)

    by_ticker_raw = {}
    for ticker, _mode, correct in rows:
        t = by_ticker_raw.setdefault(ticker, {"n": 0, "correct": 0})
        t["n"] += 1
        t["correct"] += int(bool(correct))
    by_ticker = {
        t: {**v, "accuracy": round(v["correct"] / v["n"], 3)} for t, v in by_ticker_raw.items() if v["n"] >= 2
    }

    return {
        "overall": {"n": n_total, "correct": n_correct, "accuracy": round(n_correct / n_total, 3)},
        "by_mode": by_mode, "by_ticker": by_ticker, "n_total": n_total,
    }


def _fetch_technical_snapshot(ticker):
    """Real technical snapshot -- current price, RSI14, MACD histogram,
    volume vs. 20-day average, and price change over 1/5/20 days. One
    fresh yfinance .history() call, given the same "cheap, always fetch
    fresh" treatment this app already gives price history elsewhere (see
    cached_fundamentals) -- not DB-cached, since it's a free, lightweight
    call. Fails soft -- {} on any error."""
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
    except Exception:
        return {}
    if hist.empty or len(hist) < 21:
        return {}
    hist = _add_technicals(hist)
    latest = hist.iloc[-1]
    close = hist["Close"]

    def _pct_change(n):
        if len(close) <= n:
            return None
        return round(float((close.iloc[-1] / close.iloc[-1 - n] - 1) * 100), 2)

    avg_vol_20d = float(hist["Volume"].tail(20).mean())
    return {
        "current_price": round(float(latest["Close"]), 2),
        "rsi14": round(float(latest["RSI14"]), 2) if pd.notna(latest["RSI14"]) else None,
        "macd_histogram": round(float(latest["MACD_hist"]), 4) if pd.notna(latest["MACD_hist"]) else None,
        "volume_vs_20d_avg_pct": (
            round(float(latest["Volume"]) / avg_vol_20d * 100 - 100, 1) if avg_vol_20d else None
        ),
        "price_chg_1d_pct": _pct_change(1),
        "price_chg_5d_pct": _pct_change(5),
        "price_chg_20d_pct": _pct_change(20),
    }


def _expected_move_both_horizons(conn, ticker):
    """Real IV-based expected move for both a 1-week (5 calendar day) and a
    4-week (28 calendar day) horizon, computed via calculate_expected_move_iv
    (ported verbatim from new_top.py) from the current price and ATM
    average IV already cached in earnings_signal -- pure DB read, no new
    fetch here (the caller refreshes earnings_signal separately)."""
    cached = _read_earnings_signal_cache(conn, ticker)
    if not cached or cached.get("current_price") is None or cached.get("atm_avg_iv_pct") is None:
        return None
    price, iv = cached["current_price"], cached["atm_avg_iv_pct"]
    return {
        "current_price": price,
        "atm_avg_iv_pct": iv,
        "expected_move_1wk_usd": calculate_expected_move_iv(price, iv, days=5),
        "expected_move_4wk_usd": calculate_expected_move_iv(price, iv, days=28),
    }


# --------------------------------------------------------------------------
# Five data dimensions added specifically to close the gap between the AI
# Briefing's prose and a professional (e.g. Seeking Alpha style) analyst
# note: analyst upgrade/downgrade BREADTH (not just the aggregate rating),
# peer/sector comparison, and benchmark-relative performance + distance
# from high. (The other two of the five -- CapEx/FCF and a deeper earnings
# history for precedent-hunting -- extend fetch_buybacks and
# _fetch_marketbeat_earnings_history above rather than adding new
# fetchers.) All three follow the same cached_*/fetch_log discipline as
# every other fetcher in this file.
# --------------------------------------------------------------------------

_GRADE_RANK = {
    "strong sell": 0, "sell": 1, "underperform": 1, "underweight": 1, "reduce": 1, "negative": 1,
    "hold": 2, "neutral": 2, "market perform": 2, "equal-weight": 2, "equal weight": 2,
    "sector perform": 2, "peer perform": 2, "in-line": 2, "in line": 2,
    "buy": 3, "outperform": 3, "overweight": 3, "positive": 3, "add": 3, "accumulate": 3,
    "strong buy": 4, "top pick": 4, "conviction buy": 4,
}


def _grade_rank(grade):
    return _GRADE_RANK.get((grade or "").strip().lower())


def _classify_analyst_action(action_raw, from_grade, to_grade):
    """upgrade/downgrade/initiated/reiterated, read from yfinance's own
    Action code first ('up'/'down'/'init'/'main'/'reit') and falling back
    to a from-grade/to-grade rank comparison (_GRADE_RANK) only when
    Action is missing or unrecognized -- so a row with no usable Action
    field still gets classified correctly rather than dropped."""
    raw = (action_raw or "").strip().lower()
    if raw == "up":
        return "upgrade"
    if raw == "down":
        return "downgrade"
    if raw in ("init", "initiated"):
        return "initiated"
    if raw in ("main", "maintain", "maintained", "reit", "reiterated"):
        return "reiterated"
    fr, to = _grade_rank(from_grade), _grade_rank(to_grade)
    if fr is not None and to is not None:
        if to > fr:
            return "upgrade"
        if to < fr:
            return "downgrade"
        return "reiterated"
    return "reiterated"


def fetch_analyst_actions(ticker, days_back=180):
    """Real per-firm rating CHANGES (yfinance ticker.upgrades_downgrades) --
    breadth data an aggregate recommendationKey/mean-target figure
    (fetch_analyst_targets) can't show: HOW MANY firms actually moved their
    rating up vs. down recently, not just where the average currently
    sits. Pulls a wider window than any one summary needs (days_back=180
    by default) since cached_analyst_actions re-derives its trailing-90-day
    summary from whatever's persisted, on every read, independent of when
    the underlying fetch last ran. Tagged source='yfinance_upgrades_
    downgrades'."""
    t = yf.Ticker(ticker)
    try:
        df = t.upgrades_downgrades
    except Exception:
        df = None
    if df is None or df.empty:
        return {"actions": [], "source": "yfinance_upgrades_downgrades"}

    df = df.reset_index()
    date_col = "GradeDate" if "GradeDate" in df.columns else df.columns[0]
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days_back)
    actions = []
    for _, r in df.iterrows():
        try:
            dt = pd.to_datetime(r[date_col])
            if dt.tzinfo is not None:
                dt = dt.tz_localize(None)
        except (TypeError, ValueError):
            continue
        if dt < cutoff:
            continue
        from_grade, to_grade = r.get("FromGrade") or None, r.get("ToGrade") or None
        action_raw = r.get("Action")
        actions.append({
            "date": dt.date().isoformat(), "firm": r.get("Firm"), "from_grade": from_grade,
            "to_grade": to_grade, "action": _classify_analyst_action(action_raw, from_grade, to_grade),
        })
    actions.sort(key=lambda a: a["date"], reverse=True)
    return {"actions": actions, "source": "yfinance_upgrades_downgrades"}


def _persist_analyst_actions(conn, ticker, actions, source):
    # Full replace, not upsert: yfinance itself only ever returns a bounded
    # trailing window, so there's no cross-fetch history to preserve --
    # unlike buybacks/earnings_history_real, which report distinct fiscal
    # periods that genuinely accumulate over time.
    conn.execute("DELETE FROM analyst_actions WHERE ticker=?", (ticker,))
    fetched_at = datetime.utcnow().isoformat()
    conn.executemany(
        """INSERT INTO analyst_actions (ticker, action_date, firm, from_grade, to_grade, action,
                                         source, fetched_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        [(ticker, a["date"], a["firm"], a["from_grade"], a["to_grade"], a["action"], source, fetched_at)
         for a in actions],
    )
    conn.commit()


def _read_analyst_actions_cache(conn, ticker, days_back=90):
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).date().isoformat()
    rows = conn.execute(
        """SELECT action_date, firm, from_grade, to_grade, action FROM analyst_actions
           WHERE ticker=? AND action_date >= ? ORDER BY action_date DESC""",
        (ticker, cutoff),
    ).fetchall()
    return [{"date": r[0], "firm": r[1], "from_grade": r[2], "to_grade": r[3], "action": r[4]} for r in rows]


def _summarize_analyst_actions(actions, days_back=90):
    return {
        "window_days": days_back,
        "upgrade_count": sum(1 for a in actions if a["action"] == "upgrade"),
        "downgrade_count": sum(1 for a in actions if a["action"] == "downgrade"),
        "reiteration_count": sum(1 for a in actions if a["action"] == "reiterated"),
        "initiated_count": sum(1 for a in actions if a["action"] == "initiated"),
        "most_recent_actions": actions[:8],
    }


def cached_analyst_actions(conn, ticker, max_age_hours=12, force_refresh=False, days_back=90):
    table = "analyst_actions"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        actions = _read_analyst_actions_cache(conn, ticker, days_back=days_back)
        if actions:
            return {"data": _summarize_analyst_actions(actions, days_back), "source": "yfinance_upgrades_downgrades",
                    "cache_hit": True, "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_analyst_actions(ticker)
        _persist_analyst_actions(conn, ticker, result["actions"], result["source"])
        _log_fetch(conn, table, ticker, True, len(result["actions"]))
        actions = _read_analyst_actions_cache(conn, ticker, days_back=days_back)
        return {"data": _summarize_analyst_actions(actions, days_back), "source": result["source"],
                "cache_hit": False, "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        actions = _read_analyst_actions_cache(conn, ticker, days_back=days_back)
        return {"data": _summarize_analyst_actions(actions, days_back),
                "source": "yfinance_upgrades_downgrades" if actions else None,
                "cache_hit": bool(actions), "fetched_at": _last_fetch_info(conn, table, ticker)}


# A small curated peer map for tickers this app knows well -- yfinance has
# no reliable "get peers" endpoint, and a general industry-wide scanner is
# out of scope. Falls back to same-sector matches among OTHER watchlist
# tickers (a pure DB read against already-cached fundamentals_info, no
# extra fetch) when the ticker isn't in this map.
_PEER_MAP = {
    "WMT": ["TGT", "COST"], "TGT": ["WMT", "COST"], "COST": ["WMT", "TGT"],
    "STX": ["WDC"], "WDC": ["STX"],
    "PYPL": ["SQ", "V", "MA"], "SQ": ["PYPL", "V"],
    "AAPL": ["MSFT", "GOOGL"], "MSFT": ["AAPL", "GOOGL"],
    "GOOGL": ["MSFT", "META"], "META": ["GOOGL", "SNAP"],
    "AMD": ["NVDA", "INTC"], "NVDA": ["AMD", "AVGO"], "INTC": ["AMD", "NVDA"],
    "TSLA": ["RIVN", "GM", "F"], "NFLX": ["DIS"], "DIS": ["NFLX"],
    "JPM": ["BAC", "WFC"], "BAC": ["JPM", "WFC"],
}


def _resolve_peers(ticker, conn, max_peers=4):
    peers = _PEER_MAP.get(ticker.upper())
    if peers:
        return peers[:max_peers]
    own = _read_fundamentals_info_cache(conn, ticker) or {}
    own_sector = own.get("sector")
    if not own_sector:
        return []
    matches = []
    for t in load_watchlist():
        if t.upper() == ticker.upper():
            continue
        info = _read_fundamentals_info_cache(conn, t) or {}
        if info.get("sector") == own_sector:
            matches.append(t)
        if len(matches) >= max_peers:
            break
    return matches


def fetch_peer_comparison(ticker, conn, max_peers=4):
    """2-4 direct peers (see _resolve_peers) with YTD return and trailing
    P/E each, for the AI Briefing's relative-positioning framing (Part 3)
    -- e.g. 'TGT is the sector standout YTD (+62%) against WMT's +4%.'
    Reuses the existing per-ticker caches for both legs (fetch_price_
    history_delta for return, cached_fundamentals for P/E) rather than
    issuing raw yfinance calls, so a peer already tracked elsewhere in the
    watchlist costs nothing extra here."""
    peers = _resolve_peers(ticker, conn, max_peers=max_peers)
    year_start = pd.Timestamp(year=datetime.utcnow().year, month=1, day=1)
    results = []
    for peer in peers:
        try:
            fetch_price_history_delta(conn, peer, full_period="1y", days_back=400, max_age_hours=24)
            hist = _read_price_history(conn, peer, days_back=400)
        except Exception:
            hist = pd.DataFrame()
        ytd_return = None
        if not hist.empty:
            window = hist.sort_index()
            window = window[window.index >= year_start]
            if len(window) >= 2 and window["Close"].iloc[0]:
                ytd_return = round((window["Close"].iloc[-1] / window["Close"].iloc[0] - 1) * 100, 1)
        try:
            fund = (cached_fundamentals(conn, peer, max_age_hours=24).get("data") or {}).get("info") or {}
        except Exception:
            fund = {}
        results.append({
            "ticker": peer, "ytd_return_pct": ytd_return, "pe": fund.get("trailingPE"),
        })
    return {"ticker": ticker, "peers": results, "source": "yfinance"}


def _persist_peer_comparison(conn, ticker, peers, source):
    conn.execute("DELETE FROM peer_comparison WHERE ticker=?", (ticker,))
    fetched_at = datetime.utcnow().isoformat()
    conn.executemany(
        """INSERT INTO peer_comparison (ticker, peer_ticker, ytd_return_pct, pe, source, fetched_at)
           VALUES (?,?,?,?,?,?)""",
        [(ticker, p["ticker"], p["ytd_return_pct"], p["pe"], source, fetched_at) for p in peers],
    )
    conn.commit()


def _read_peer_comparison_cache(conn, ticker):
    rows = conn.execute(
        "SELECT peer_ticker, ytd_return_pct, pe FROM peer_comparison WHERE ticker=? ORDER BY peer_ticker",
        (ticker,),
    ).fetchall()
    return [{"ticker": r[0], "ytd_return_pct": r[1], "pe": r[2]} for r in rows]


def cached_peer_comparison(conn, ticker, max_age_hours=24, force_refresh=False):
    table = "peer_comparison"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        peers = _read_peer_comparison_cache(conn, ticker)
        if peers:
            return {"data": {"ticker": ticker, "peers": peers}, "source": "yfinance", "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_peer_comparison(ticker, conn)
        _persist_peer_comparison(conn, ticker, result["peers"], result["source"])
        _log_fetch(conn, table, ticker, True, len(result["peers"]))
        return {"data": result, "source": result["source"], "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        peers = _read_peer_comparison_cache(conn, ticker)
        return {"data": {"ticker": ticker, "peers": peers}, "source": "yfinance" if peers else None,
                "cache_hit": bool(peers), "fetched_at": _last_fetch_info(conn, table, ticker)}


def fetch_benchmark_relative_performance(ticker, conn):
    """This ticker's total return against a benchmark index (SPY, or QQQ
    for tech-sector names) over YTD/6mo/1y -- both legs computed from the
    SAME stored price_history rows so the comparison is apples-to-apples --
    plus the 52-week high and the DATE it was reached, computed directly
    from stored daily highs (not the coarser fiftyTwoWeekHigh snapshot
    field, which carries no date), falling back to that field only if
    price_history doesn't have enough history yet. Reuses fetch_price_
    history_delta's existing per-ticker cache for both the ticker and the
    benchmark. Tagged source='yfinance_price_history'."""
    fundamentals = _read_fundamentals_info_cache(conn, ticker) or {}
    sector = (fundamentals.get("sector") or "").lower()
    benchmark = "QQQ" if "technology" in sector else "SPY"

    fetch_price_history_delta(conn, ticker, full_period="1y", days_back=400, max_age_hours=24)
    fetch_price_history_delta(conn, benchmark, full_period="1y", days_back=400, max_age_hours=24)
    own_hist = _read_price_history(conn, ticker, days_back=400)
    bench_hist = _read_price_history(conn, benchmark, days_back=400)

    def _return_since(hist, cutoff):
        if hist is None or hist.empty:
            return None
        window = hist.sort_index()
        window = window[window.index >= cutoff]
        if len(window) < 2 or not window["Close"].iloc[0]:
            return None
        return round((window["Close"].iloc[-1] / window["Close"].iloc[0] - 1) * 100, 1)

    now = pd.Timestamp.now().normalize()
    year_start = pd.Timestamp(year=now.year, month=1, day=1)
    windows = {
        "ytd": {"ticker_return_pct": _return_since(own_hist, year_start),
                "benchmark_return_pct": _return_since(bench_hist, year_start)},
        "six_month": {"ticker_return_pct": _return_since(own_hist, now - pd.Timedelta(days=182)),
                      "benchmark_return_pct": _return_since(bench_hist, now - pd.Timedelta(days=182))},
        "one_year": {"ticker_return_pct": _return_since(own_hist, now - pd.Timedelta(days=365)),
                     "benchmark_return_pct": _return_since(bench_hist, now - pd.Timedelta(days=365))},
    }

    fifty_two_wk_high, fifty_two_wk_high_date = None, None
    if own_hist is not None and not own_hist.empty:
        window = own_hist.sort_index()
        window = window[window.index >= now - pd.Timedelta(days=365)]
        if not window.empty and window["High"].notna().any():
            idx = window["High"].idxmax()
            fifty_two_wk_high = float(window.loc[idx, "High"])
            fifty_two_wk_high_date = idx.date().isoformat()
    if fifty_two_wk_high is None:
        fifty_two_wk_high = fundamentals.get("fiftyTwoWeekHigh")

    current_price = fundamentals.get("currentPrice") or fundamentals.get("regularMarketPrice")
    if current_price is None and own_hist is not None and not own_hist.empty:
        current_price = float(own_hist.sort_index()["Close"].iloc[-1])

    pct_below_high = (
        round((1 - current_price / fifty_two_wk_high) * 100, 1)
        if current_price and fifty_two_wk_high else None
    )

    return {
        "ticker": ticker, "benchmark": benchmark, "windows": windows,
        "fifty_two_week_high": fifty_two_wk_high, "fifty_two_week_high_date": fifty_two_wk_high_date,
        "pct_below_52wk_high": pct_below_high, "source": "yfinance_price_history",
    }


def _persist_benchmark_relative_performance(conn, ticker, result):
    conn.execute(
        """INSERT INTO benchmark_relative_performance
               (ticker, benchmark, windows_json, fifty_two_week_high, fifty_two_week_high_date,
                pct_below_52wk_high, source, fetched_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT(ticker) DO UPDATE SET
               benchmark=excluded.benchmark, windows_json=excluded.windows_json,
               fifty_two_week_high=excluded.fifty_two_week_high,
               fifty_two_week_high_date=excluded.fifty_two_week_high_date,
               pct_below_52wk_high=excluded.pct_below_52wk_high,
               source=excluded.source, fetched_at=excluded.fetched_at""",
        (ticker, result.get("benchmark"), json.dumps(result.get("windows") or {}),
         result.get("fifty_two_week_high"), result.get("fifty_two_week_high_date"),
         result.get("pct_below_52wk_high"), result.get("source"), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_benchmark_relative_performance_cache(conn, ticker):
    row = conn.execute(
        """SELECT benchmark, windows_json, fifty_two_week_high, fifty_two_week_high_date,
                  pct_below_52wk_high, source
           FROM benchmark_relative_performance WHERE ticker=?""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    return {
        "ticker": ticker, "benchmark": row[0], "windows": json.loads(row[1]) if row[1] else {},
        "fifty_two_week_high": row[2], "fifty_two_week_high_date": row[3],
        "pct_below_52wk_high": row[4], "source": row[5],
    }


def cached_benchmark_relative_performance(conn, ticker, max_age_hours=24, force_refresh=False):
    table = "benchmark_relative_performance"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_benchmark_relative_performance_cache(conn, ticker)
        if cached:
            return {"data": cached, "source": cached["source"], "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        result = fetch_benchmark_relative_performance(ticker, conn)
        _persist_benchmark_relative_performance(conn, ticker, result)
        _log_fetch(conn, table, ticker, True, 1)
        return {"data": result, "source": result["source"], "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_benchmark_relative_performance_cache(conn, ticker)
        return {"data": cached, "source": cached["source"] if cached else None, "cache_hit": bool(cached),
                "fetched_at": _last_fetch_info(conn, table, ticker)}


def _log_claude_call(purpose, ticker):
    """Printed immediately before EVERY outbound Anthropic API call in this
    file -- a single, greppable tag ([CLAUDE API]) so it's always visible in
    the terminal exactly when real money is being spent, and which feature
    triggered it. Never printed on a cache hit -- only the call sites that
    actually reach client.messages.create() invoke this."""
    print(f"[CLAUDE API] Calling Claude — purpose: {purpose}, ticker: {ticker}, "
          f"timestamp: {datetime.now().isoformat()}")


_CATALYST_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "named_catalysts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": "string"},
                    "description": {"type": "string"},
                    "source_url": {"type": "string"},
                },
                "required": ["name", "status", "description", "source_url"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
        "search_found_nothing": {"type": "boolean"},
    },
    "required": ["named_catalysts", "summary", "search_found_nothing"],
    "additionalProperties": False,
}

_CATALYST_CONTEXT_SYSTEM_PROMPT = (
    "You are researching a stock to find SPECIFIC named operational catalysts -- product/technology "
    "rollouts, manufacturing/capacity ramps, regulatory approvals, named partnerships -- of the kind a "
    "shallow RSS news feed misses but analysts and financial press write about in depth (e.g. a specific "
    "named technology like a storage-media transition, a margin-guidance explanation tied to a named "
    "product line, a capacity ramp at a named facility). Use the web_search tool with SEVERAL distinct, "
    "SPECIFIC queries -- vague single queries like 'TICKER catalyst' routinely miss genuinely specific, "
    "somewhat technical named items (e.g. a specific technology name), so prefer queries that combine the "
    "ticker/company name with a concrete technical/business term (a named technology, a peer comparison, "
    "a product roadmap) over a single generic query.\n\n"
    "RUN AT LEAST 4 DISTINCT SEARCHES before concluding nothing was found -- one or two searches, "
    "especially if the first one or two come back thin or generic, is NOT enough evidence that a "
    "specific catalyst doesn't exist. If an early search surfaces something promising but vague (e.g. "
    "an earnings call mentioning margin pressure without naming why), run a FOLLOW-UP search on that "
    "specific lead (the named product/technology it points to) rather than stopping. Only set "
    "search_found_nothing=true after multiple varied searches have genuinely turned up nothing beyond "
    "generic commentary -- err on the side of searching more, not less.\n\n"
    "Only report catalysts that are REAL, SPECIFIC, and NAMED in a source you actually found -- never "
    "invent one, and never restate the next scheduled earnings date as if it were a catalyst (that's "
    "the event itself, not a catalyst for it). Cite the source URL for each."
)


def _run_catalyst_context_search(ticker, conn, attempt):
    """One single Claude+web_search attempt at catalyst research -- factored
    out of fetch_catalyst_context so it can be retried (see there for why)."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set -- cannot research catalyst context.")

    fundamentals = _read_fundamentals_info_cache(conn, ticker) or {}
    company_name = fundamentals.get("shortName") or ticker
    current_year = datetime.utcnow().year
    peers = _resolve_peers(ticker, conn, max_peers=1)
    peer_txt = peers[0] if peers else None

    query_lines = [
        f'- "{ticker} stock fell despite earnings beat reason" (or "rose despite miss", whichever fits '
        f'this ticker\'s actual recent pattern)',
        f'- "{company_name} {ticker} analyst catalyst {current_year} technology roadmap"',
        f'- "{company_name} {ticker} product technology roadmap {current_year} {current_year + 1}"',
    ]
    if peer_txt:
        query_lines.append(
            f'- "{ticker} {peer_txt} margin guidance difference reason" (a named head-to-head comparison '
            f"often surfaces the specific technology/operational factor analysts cite)"
        )
    user_prompt = (
        f"Research {ticker} ({company_name}) stock for specific named catalysts. Run web searches "
        f"including (adjust wording as needed based on what you find, and run more if the first round "
        f"doesn't surface anything specific):\n" + "\n".join(query_lines) + "\n\n"
        "Report the 6 MOST DECISION-RELEVANT real, specific, named catalysts (not every one you find) "
        "with a source URL for each, and keep each description to 2-3 sentences -- concise over "
        "exhaustive, since a long response risks truncation."
    )

    _log_claude_call(f"Catalyst Context Research (attempt {attempt})", ticker)
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-5",
        # 8000, not 4000 -- a real bug: a rich run (7-8 catalysts, each
        # with a full description) hit 4000 and got cut off mid-JSON-
        # string, raising a JSONDecodeError that silently looked like
        # "search found nothing" once cached_catalyst_context's exception
        # handler fell back to a stale empty cache row. The prompt above
        # also now caps the catalyst count and description length directly
        # so a normal run has real headroom, not just a bigger ceiling.
        max_tokens=8000,
        system=_CATALYST_CONTEXT_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 6}],
        output_config={"format": {"type": "json_schema", "schema": _CATALYST_CONTEXT_SCHEMA}},
        messages=[{"role": "user", "content": user_prompt}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The Claude API declined to research catalyst context.")

    # Last text block, not the first -- search/tool-use blocks and any
    # preamble text precede the final structured-output block in `content`.
    text = next((b.text for b in reversed(response.content) if b.type == "text"), "")
    if not text:
        raise RuntimeError(f"Catalyst context search returned no text output (stop_reason={response.stop_reason})")
    result = json.loads(text)
    result["source"] = "targeted_catalyst_search"
    return result


def fetch_catalyst_context(ticker, conn):
    """Dedicated, cited web-research pass (Claude + the web_search server
    tool) built specifically to surface deep, named product/technology/
    regulatory catalysts that yfinance's shallow RSS headlines (fetch_news)
    structurally can't -- analyst commentary explaining WHY a beat didn't
    pay, or a named rollout/approval/partnership currently being watched.

    This is a SEPARATE, real, paid Claude API call from generate_deep_
    analysis() -- its cited output becomes one more real, cited input in
    that call's context bundle (see _bundle_ai_context), so the main
    synthesis call still makes zero live fetches of its own. Both calls
    are covered by the SAME single user-confirmed 'Generate AI Briefing'
    click (this fetcher is only ever invoked from inside _bundle_ai_
    context, on a 7-day cache -- see cached_catalyst_context), and both
    print a [CLAUDE API] log line right before firing (_log_claude_call).

    Query strategy (fixed a real bug: a single vague query like 'TICKER
    catalyst' can miss a specific, somewhat technical named item -- e.g.
    HAMR for WDC -- even though a direct, specific query surfaces it
    easily): builds several concrete, specific query SUGGESTIONS in the
    user prompt -- a beat-reaction query, a company+technology-roadmap
    query, and, when a real peer is resolvable (_resolve_peers, reusing
    the peer-comparison infrastructure), a head-to-head margin/guidance
    query against that named peer -- and instructs the model to run
    several searches, adjusting as needed, rather than one generic one.

    One automatic retry when the first attempt comes back empty (confirmed
    live: identical queries against the same real ticker (WDC) returned 8
    real, cited catalysts including HAMR on two separate runs and
    search_found_nothing=true with zero catalysts on a third -- genuine
    run-to-run variance in how many searches the model chooses to do
    before concluding nothing exists, not a deterministic "this ticker has
    no catalysts" result). Also retries once on a malformed-JSON response
    (a separate real failure mode found live: a rich run with several
    catalysts hit the old max_tokens=4000 cap and got cut off mid-string --
    raising json.JSONDecodeError, which use to propagate up and get
    mistaken for "nothing found" once the caller's exception handler fell
    back to a stale cache row; max_tokens is now 8000 with the prompt also
    capping catalyst count/length, so this should be rare, but the retry
    stays as a safety net). A second failure of either kind is treated as
    the real answer/error, not retried again. Returns {"named_catalysts":
    [...], "summary": ..., "search_found_nothing": bool, "source":
    "targeted_catalyst_search"}."""
    try:
        result = _run_catalyst_context_search(ticker, conn, attempt=1)
    except (json.JSONDecodeError, ValueError):
        result = _run_catalyst_context_search(ticker, conn, attempt=2)
        return result
    if result.get("search_found_nothing") and not result.get("named_catalysts"):
        result = _run_catalyst_context_search(ticker, conn, attempt=2)
    return result


def _persist_catalyst_context(conn, ticker, result):
    conn.execute(
        "INSERT INTO catalyst_context (ticker, context_json, source, fetched_at) VALUES (?, ?, ?, ?)",
        (ticker, json.dumps(result), result.get("source", "targeted_catalyst_search"),
         datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_catalyst_context_cache(conn, ticker):
    row = conn.execute(
        "SELECT context_json, source, fetched_at FROM catalyst_context WHERE ticker=? "
        "ORDER BY fetched_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    context_json, source, fetched_at = row
    return {"data": json.loads(context_json), "source": source, "fetched_at": fetched_at}


def cached_catalyst_context(conn, ticker, max_age_hours=168, force_refresh=False):
    """7-day cache window -- a deep research pass (like 13F/buybacks), not a
    routine data pull, so re-searching on every AI Briefing click within a
    week would burn real API cost for no benefit; a genuinely new named
    catalyst rarely emerges inside a week."""
    table = "catalyst_context"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_catalyst_context_cache(conn, ticker)
        if cached:
            return {"data": cached["data"], "source": cached["source"], "cache_hit": True,
                    "fetched_at": cached["fetched_at"]}
    try:
        result = fetch_catalyst_context(ticker, conn)
        _persist_catalyst_context(conn, ticker, result)
        _log_fetch(conn, table, ticker, True, len(result.get("named_catalysts") or []))
        return {"data": result, "source": result.get("source"), "cache_hit": False,
                "fetched_at": datetime.utcnow().isoformat()}
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
        cached = _read_catalyst_context_cache(conn, ticker)
        return {"data": cached["data"] if cached else None, "source": cached["source"] if cached else None,
                "cache_hit": bool(cached), "fetched_at": cached["fetched_at"] if cached else None}


# --------------------------------------------------------------------------
# AI Deep-Dive Briefing -- assembles the real-data context above (plus
# everything else already cached in this DB) and sends it to the Anthropic
# API only when explicitly triggered by the user (see dashboard.py's
# confirmation-gated "Generate AI Briefing" button). Never called from
# full_refresh() or any other automatic/background path.
# --------------------------------------------------------------------------

AI_BRIEF_CACHE_HOURS = 4

_AI_BRIEF_SYSTEM_PROMPT = (
    "You are a markets analyst writing a key-takeaway summary a busy discretionary trader can "
    "scan in 20 seconds -- not a report reader who wants a transcript of every data point. "
    "You are given a JSON bundle of REAL data already gathered for one ticker: fundamentals, "
    "technicals, IV-based expected move (1-week and 4-week), options flow grouped into weekly "
    "expiry buckets, options_flow_synthesis (a DETERMINISTIC, rules-based read of today's options "
    "chain -- dominant_delta names the delta-profile badge (e.g. 'Lottery ticket', 'Balanced') "
    "carrying the most volume with a plain-language sentence already written; dominant_iv does the "
    "same for the IV-level badge; skew_reading combines call/put skew with the dominant side's "
    "delta badge and moneyness into one read (e.g. 'retail chasing a bounce' vs. 'real downside "
    "hedging'); top_unusual names the single most unusual contract by volume/OI with its own "
    "combined-badge sentence. This is the SAME badge-driven read the dashboard shows mechanically "
    "in its own synthesis panel -- retail_analysis and institutional_analysis below MUST build on "
    "these exact readings for options-flow-derived claims, never re-derive a separate, possibly "
    "inconsistent interpretation from options_flow_weekly_buckets' raw numbers alone; null if no "
    "options data is cached for this ticker yet), analyst targets, a real MarketBeat-sourced "
    "earnings track record (each row "
    "matched to the REAL yfinance-measured price reaction), buybacks, congressional trades, "
    "insider trades (SEC Form 4 and Finviz, each tagged by source), thirteenf_filings (real SEC "
    "EDGAR full-text-search results for recent 13F-HR filings mentioning this ticker -- each has "
    "an institution name, filing_date, and form_type; this is 'which institutions recently filed "
    "a 13F mentioning this name,' not a share-count/position-size diff), marketbeat_institutional "
    "(real scraped MarketBeat institutional-ownership stats -- ownership_pct, buyers/sellers "
    "counts, inflows/outflows, net_flow_bias_pct, and recent_transactions -- present only if "
    "fetched for this ticker before, since it requires a manual browser-based fetch; null/absent "
    "if not), and THREE DISTINCT retail-sentiment sources, each with a different role -- never "
    "treat them as interchangeable: stocktwits_sentiment (the PRIMARY TAGGED SENTIMENT source -- "
    "real bull/bear counts and ratio aggregated from StockTwits' own real per-message sentiment "
    "tags, plus a few representative message excerpts in representative_messages) and "
    "retail_sentiment_posts (the underlying raw StockTwits posts, each tagged by source, timestamp, "
    "and that same real sentiment tag when present); reddit_posts (SUPPLEMENTARY narrative color -- "
    "real, raw, UNSCORED Reddit post titles/bodies from r/wallstreetbets' public search, fetched "
    "on-demand for this briefing only; read and characterize their tone yourself, exactly as you do "
    "for StockTwits messages -- commonly an empty list, since this is an unauthenticated endpoint "
    "that Reddit may rate-limit or block without warning, a real and expected state, not a bug); "
    "apewisdom_sentiment (an ATTENTION/MOMENTUM signal only -- real Reddit mention-tracking from "
    "ApeWisdom: rank, mentions, upvotes, and each vs. 24h ago, aggregated across r/wallstreetbets "
    "and other finance subreddits, plus the derived attention_change_pct, the % change in mentions "
    "vs. 24h ago. This source has NO SENTIMENT/POLARITY FIELD OF ANY KIND -- never invent or imply "
    "bullish/bearish framing for it beyond describing whether attention is rising or falling; null "
    "if this ticker isn't currently ranked). quiverquant_wsb (real WallStreetBets mention data from "
    "QuiverQuant's paid tier -- usually an empty list, since this dataset requires a subscription "
    "tier beyond what's currently active; treat a non-empty list as real data same as any other "
    "source), dark pool signal, a "
    "divergence-score history, recent news headlines with publisher and date, and "
    "technical_levels -- an ALGORITHMICALLY detected (not AI-guessed) set of support/resistance "
    "levels, trend structure, and measured-move breakdown/breakout targets, computed directly "
    "from price/volume data before this prompt was built. buybacks now also carries capex and "
    "operating_cash_flow per period alongside buyback_value, with fcf (operating_cash_flow minus "
    "capex) computed for you -- buyback_trend/fcf_trend at the top level are the pre-computed "
    "INCREASING/DECREASING/FLAT (or IMPROVING/DECLINING for fcf_trend) direction read off the two "
    "most recent periods, so you don't have to eyeball the raw numbers yourself. "
    "earnings_track_record_real now carries up to 16 real historical quarters (not just the last "
    "4-6) specifically so a genuine historical precedent can be found, not just recent quarters "
    "summarized. Five more real, structured sources: analyst_actions (real per-firm rating CHANGES "
    "from yfinance's upgrades_downgrades feed over a trailing ~90-day window -- upgrade_count/"
    "downgrade_count/reiteration_count/initiated_count plus most_recent_actions, each a real firm "
    "name + date + from-grade-to-grade move; this is BREADTH data distinct from analyst_targets' "
    "single aggregate recommendation/mean-target figure, and null if not yet fetched for this "
    "ticker); peer_comparison (2-4 real direct peer tickers -- from a curated map or same-sector "
    "watchlist matches -- each with its own real ytd_return_pct and trailing pe, for relative "
    "positioning; {'peers': []} if no peer could be resolved for this ticker); benchmark_relative_"
    "performance (this ticker's own total return against a benchmark index -- SPY, or QQQ for "
    "tech-sector names -- over ytd/six_month/one_year windows, both legs computed from the same "
    "real price history so they're directly comparable, plus fifty_two_week_high and the REAL DATE "
    "it was reached and pct_below_52wk_high; null if not yet fetched for this ticker); "
    "catalyst_context (a DEDICATED, CITED web-research pass -- Claude + live web search, run "
    "specifically to find named product/technology/regulatory catalysts that news_headlines' "
    "shallow RSS feed structurally misses, e.g. a specific named technology behind a margin move -- "
    "named_catalysts is a list of {name, status, description, source_url}, each real and sourced; "
    "search_found_nothing=true if the research genuinely turned up nothing beyond generic "
    "commentary; null if not yet researched for this ticker).\n\n"
    "HARD RULES -- these override any general instinct to be comprehensive or engaging:\n"
    "1. Only state facts present in the data provided below. Do not name specific institutions, "
    "analysts, or funds unless they appear explicitly in the provided data.\n"
    "2. If a data category is empty or unavailable, say so explicitly (e.g. 'No insider activity "
    "on record') -- never fill the gap with a plausible-sounding guess.\n"
    "3. Every claim must be traceable to a specific input in the bundle. Do not add outside "
    "knowledge about the company beyond what's provided.\n"
    "4. For earnings beat/miss analysis, explain price reaction using only the real historical "
    "move data provided -- do not speculate about causes unless that reasoning is explicitly "
    "supported by the news headlines in the bundle.\n"
    "5. Prioritize the 3-4 most decision-relevant facts per section over completeness. Write for "
    "a trader scanning quickly, not a report reader -- cut anything that wouldn't change what "
    "they do next.\n"
    "6. For technical_catalyst_setup specifically: every price level you mention MUST come "
    "verbatim from technical_levels (support_levels, resistance_levels, key_level_below, "
    "key_level_above, breakdown_target, breakout_target) -- never invent, round differently, or "
    "estimate a price level that isn't literally present in that data.\n"
    "7. Write with the specificity of a professional sell-side or Seeking Alpha analyst note, not "
    "generic template output -- use exact figures with their source and date, name individual "
    "analyst firms and peer tickers when the data provides them, connect cause and effect "
    "explicitly (e.g. 'FCF fell because CapEx rose,' never two unrelated numbers left for the "
    "reader to connect themselves), state benchmark-relative returns and distance from the 52-week "
    "high (with its date) when opening the setup, and actively search earnings_track_record_real "
    "for a specific historical precedent rather than only describing recent quarters in aggregate. "
    "Avoid restating a generic pattern with no specifics behind it -- every claim should carry a "
    "real number, date, or named entity, not a vague characterization.\n\n"
    "Write exactly these sections:\n"
    "- setup: 3-4 short sentences MAX, grounded in the real technicals and expected-move data "
    "provided (not the earnings track record -- that has its own section below). OPEN with "
    "relative-positioning context when the data supports it: benchmark_relative_performance's "
    "own-return-vs-benchmark-return for whichever window is most relevant (prefer ytd unless "
    "another window tells a clearer story) and, when available, how far below the fifty_two_week_"
    "high (and its date) the stock currently sits -- e.g. 'WMT is up 37% YTD vs. SPY's 11% over "
    "the same period, and sits 4% below its 52-week high reached on [date].' If peer_comparison "
    "has real peers, weave in relative sector positioning too, e.g. 'TGT is the sector standout "
    "YTD (+62%) against WMT's +4%.' Skip any of these openers cleanly (don't force a sentence) if "
    "the underlying data is null/empty rather than inventing a comparison. Bold (markdown "
    "**text**) only the 2-3 single most important numbers in the whole paragraph -- e.g. current "
    "price, the one technical signal that matters most, the expected move -- never bold every "
    "number that appears.\n"
    "- technical_catalyst_setup: a short PLAIN-LANGUAGE narrative in the style of a trader's own "
    "'Final Take' note (not a jargon-heavy technical readout) -- explain what each term means in "
    "context as you use it rather than just naming it. Use markdown **bold** as mini-headers for "
    "each beat, in this exact structure and order:\n"
    "  1. Opening assessment (1-2 sentences, no header): plain-language read of "
    "technical_levels.trend_structure/trend_explanation -- e.g. whether it's a clean trend or a "
    "deteriorating/basing structure -- do not relabel the trend yourself, translate it into plain "
    "words.\n"
    "  2. '**[level] is the level to watch:**' one sentence naming the single most decision-"
    "relevant level (key_level_below or key_level_above from technical_levels, whichever is "
    "nearer/more relevant right now) and briefly why -- this is the 'line in the sand.'\n"
    "  3. '**If it holds:**' one sentence on what a hold implies (e.g. a base forming, room to "
    "retest resistance) -- reasoned from trend_structure/current_zone, not a new invented number.\n"
    "  4. '**If it breaks:**' one sentence on what a break implies, citing the matching "
    "breakdown_target or breakout_target from technical_levels for the projected move -- never a "
    "price level that isn't literally present in technical_levels (see HARD RULE 6).\n"
    "  5. '**Into earnings:**' (only if earnings_calendar has a real upcoming date -- omit this "
    "beat entirely if not) one sentence on the earnings-contingent scenario, using only real "
    "numbers already present elsewhere in the bundle (analyst_targets mean/median, "
    "expected_move's 1wk/4wk figures, or technical_levels) -- never a round number invented for "
    "the occasion.\n"
    "  6. '**Bottom line:**' one sentence stating an overall stance (e.g. constructive, measured, "
    "cautious, bearish) that synthesizes 1-5 -- a real takeaway, not a hedge that avoids picking a "
    "side.\n"
    "If technical_levels shows 'unknown'/insufficient data, skip straight to a single sentence "
    "saying so -- do not invent a trend or levels from other technicals instead.\n"
    "- retail_analysis: SHORT BULLET POINTS (markdown '- ' lines), one line per distinct fact, "
    "never a paragraph. Combine StockTwits sentiment tags, Reddit post text, and ApeWisdom "
    "attention data into ONE retail read, each source playing its own distinct role: "
    "stocktwits_sentiment (bull/bear tag counts + ratio, plus its representative_messages excerpts) "
    "is the SENTIMENT read; reddit_posts (raw post titles/text, when non-empty) adds narrative "
    "color and quotable specifics -- read and characterize their tone yourself, they arrive "
    "unscored; apewisdom_sentiment's attention_change_pct is context on whether that sentiment is "
    "backed by RISING or FALLING attention volume, not a sentiment signal itself. Fold in "
    "quiverquant_wsb (WallStreetBets mentions) too when it's non-empty. NEVER state 'retail "
    "sentiment is bullish/bearish' based on apewisdom_sentiment alone -- it has no sentiment field, "
    "only attention/mention volume; a bullish read must trace to stocktwits_sentiment and/or "
    "reddit_posts. If reddit_posts is empty (a common, expected state for this best-effort, "
    "rate-limitable source -- not a bug), say explicitly that StockTwits + ApeWisdom are the basis "
    "for this read instead of silently omitting the gap. Each bullet cites source and approximate "
    "recency, e.g. '- StockTwits: 12 bullish vs. 3 bearish tagged messages (4:1 ratio)' or "
    "'- ApeWisdom: mentions up 42% vs. 24h ago (attention spike, not sentiment) -- ranked #3, up "
    "from #7'. Max 4 bullets, picking the most decision-relevant. If stocktwits_sentiment has zero "
    "tagged/untagged messages, or apewisdom_sentiment is null because this ticker isn't currently "
    "ranked, state that explicitly rather than omitting it silently or inferring sentiment from "
    "price action instead. If options_flow_synthesis is non-null and its skew_reading or "
    "top_unusual points to retail-style behavior (e.g. dominant_side='call' paired with a "
    "'Lottery ticket'/'Deep OTM lottery ticket' delta_badge, or a top_unusual contract in that same "
    "territory), add ONE bullet using that exact reading verbatim (e.g. '- Options flow: 68% of "
    "volume in OTM calls rated Lottery ticket -- retail chasing a bounce, not institutional "
    "positioning') -- do not re-describe the options chain in your own words instead.\n"
    "- institutional_analysis: SHORT BULLET POINTS (markdown '- ' lines), one line per distinct "
    "fact -- one insider sale, one buyback/FCF figure, one analyst-target datapoint, one analyst-"
    "action breadth line, one 13F filer, one marketbeat_institutional stat if present -- never fold "
    "multiple facts into one run-on sentence. Synthesize across ALL institutional-angle categories "
    "in the bundle (insider_trades, buybacks, analyst_targets, analyst_actions, thirteenf_filings, "
    "and marketbeat_institutional when present), not just insider trades. ALWAYS include one bullet "
    "stating analyst_actions' actual BREADTH when it has any actions on record -- state the real "
    "counts, e.g. '- Analyst activity: 11 upgrades vs. 14 downgrades over the trailing 90 days "
    "(yfinance_upgrades_downgrades)', optionally naming 1-2 of the most recent individual actions "
    "from most_recent_actions (firm + date + from-grade-to-grade) if especially notable -- never "
    "just restate analyst_targets' aggregate recommendation/mean-target as if that were the same "
    "thing; if analyst_actions is null or has zero actions in the window, say so explicitly rather "
    "than omitting the bullet. If buyback_trend or fcf_trend is not 'N/A', include a bullet that "
    "CONNECTS them when the data supports it -- if fcf_trend is 'DECLINING', check whether the same "
    "period's capex rose and say so directly (e.g. '- FCF fell to $X this period, partly due to a "
    "step-up in CapEx to $Y (yfinance)' ), not two separate, unconnected numbers; if capex didn't "
    "rise, still report the FCF trend but don't force a CapEx connection that isn't there. Example "
    "lines: '- CEO William Mosley sold 12,920 shares ($10.38M) -- Aug 3 (finviz_scrape)' and "
    "'- Tybourne Capital Management filed a 13F mentioning this ticker -- Feb 14, 2026 "
    "(sec_edgar_free)'. Cite source and date on every bullet. State explicitly when a category has "
    "no data -- e.g. if thirteenf_filings is empty, say so rather than omitting 13F coverage "
    "silently. If options_flow_synthesis is non-null and its skew_reading or dominant_delta points "
    "to institutional-style positioning (e.g. a 'Balanced'/'Conservative/ITM'/'Deep ITM / stock "
    "replacement' delta_badge dominating volume, or a hedging-flavored put skew reading), add ONE "
    "bullet using that exact reading verbatim rather than re-describing the options chain yourself "
    "-- the same options_flow_synthesis object, read for the opposite (institutional vs. retail) "
    "signal it can also carry. Max 5-6 bullets, picking the most decision-relevant across all these "
    "categories combined.\n"
    "- news_summary: 3-4 sentences MAX on the single dominant narrative driving the stock right "
    "now -- not an exhaustive list of every headline's framing.\n"
    "- catalysts: specific, NAMED forward-looking events -- a product/technology rollout, a "
    "regulatory decision, a named partnership, a capacity/manufacturing milestone -- never generic "
    "statements like 'market conditions' or 'sector trends,' and NEVER the next scheduled earnings "
    "date restated as if it were a catalyst (that's the event itself, not a catalyst for it). "
    "catalyst_context is a DEDICATED, CITED web-research pass built specifically to find these -- "
    "its named_catalysts list is real, sourced, and cited (each entry has a source_url); when it has "
    "entries, PREFER it over news_headlines, since news_headlines' RSS feed is often too shallow to "
    "carry this kind of specific, sometimes technical detail (e.g. a named storage/chip/drug "
    "technology) that catalyst_context was specifically built to surface. When you use a "
    "catalyst_context entry, name it exactly as given (e.g. a specific named technology) and mention "
    "its source domain in why_it_matters so the reader knows it's not from the routine news feed. If "
    "catalyst_context is null or its search_found_nothing is true, fall back to naming any genuinely "
    "specific event visible in news_headlines; if neither source has one, return an empty list rather "
    "than fabricating one. Max 4, picking the most decision-relevant. expected_timing is a short "
    "phrase ('Q3 earnings, ~Nov 2026', 'unspecified'), not a full sentence.\n"
    "- earnings_track_record: 2-3 sentences MAX synthesizing the pattern across the real "
    "beat/miss + real price-reaction pairs given in earnings_track_record_real -- not a "
    "recitation of every quarter. Use only the exact dates and percentages given -- do not alter "
    "or round them differently. Explain a reaction's likely cause only where the news headlines "
    "in the bundle explicitly support that reasoning. ACTIVELY LOOK for a specific historical "
    "precedent before summarizing in aggregate: scan every row in earnings_track_record_real (up "
    "to 16 real quarters may be present) for the single most extreme or most similar prior reaction "
    "-- e.g. the largest one-day move on record, or the quarter whose beat/miss + reaction pattern "
    "most resembles the current setup -- and name that SPECIFIC instance directly with its real "
    "date and percentage (e.g. 'the largest one-day drop since the Q2 2024 print on [date], which "
    "fell X% despite a beat'). Only fall back to a pure aggregate summary if you've genuinely "
    "checked and no single quarter stands out as a real precedent -- say so explicitly rather than "
    "fabricating one, but don't silently skip the precedent search.\n"
    "- verdicts: exactly three entries, in this order -- next earnings, end of this week, end of "
    "this month. Each needs an explicit target_date (derive next earnings' date from the "
    "earnings_calendar data provided; derive the week/month dates from the bundle's 'as_of' "
    "timestamp), a directional bias (BULLISH/BEARISH/NEUTRAL), a confidence level (Low/Medium/"
    "High), and a one-sentence key_risk grounded in the data provided."
)

_AI_BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "setup": {"type": "string"},
        "technical_catalyst_setup": {"type": "string"},
        "retail_analysis": {"type": "string"},
        "institutional_analysis": {"type": "string"},
        "news_summary": {"type": "string"},
        "catalysts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "catalyst": {"type": "string"},
                    "expected_timing": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
                "required": ["catalyst", "expected_timing", "why_it_matters"],
                "additionalProperties": False,
            },
        },
        "earnings_track_record": {"type": "string"},
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "horizon": {"type": "string"},
                    "target_date": {"type": "string"},
                    "bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                    "confidence": {"type": "string", "enum": ["Low", "Medium", "High"]},
                    "key_risk": {"type": "string"},
                },
                "required": ["horizon", "target_date", "bias", "confidence", "key_risk"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "setup", "technical_catalyst_setup", "retail_analysis", "institutional_analysis",
        "news_summary", "catalysts", "earnings_track_record", "verdicts",
    ],
    "additionalProperties": False,
}


def _read_ai_brief_cache(conn, ticker, max_age_hours=AI_BRIEF_CACHE_HOURS):
    row = conn.execute(
        "SELECT id, brief_json, fetched_at FROM ai_briefs WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    briefing_id, brief_json, fetched_at = row
    try:
        fetched_dt = datetime.fromisoformat(fetched_at)
    except (TypeError, ValueError):
        return None
    age_hours = (datetime.utcnow() - fetched_dt).total_seconds() / 3600
    if age_hours > max_age_hours:
        return None
    brief = json.loads(brief_json)
    # A brief cached under an older schema version (e.g. before
    # technical_catalyst_setup was added) is missing one of today's
    # required sections -- treat that as a cache miss rather than
    # silently rendering an incomplete briefing with that section
    # unexplainably absent. This is the real fix for a class of bug where
    # a schema/prompt change ships but a long-running dashboard process's
    # already-imported data_engine module (and any already-cached briefs)
    # don't reflect it until a fresh generate_deep_analysis() call happens
    # -- python doesn't re-import a module just because the .py file on
    # disk changed, and neither does a 4-hour DB cache self-invalidate on
    # a schema change.
    if not set(_AI_BRIEF_SCHEMA["required"]).issubset(brief.keys()):
        return None
    brief["_fetched_at"] = fetched_at
    brief["_cache_hit"] = True
    brief["_briefing_id"] = briefing_id
    return brief


def _persist_ai_brief(conn, ticker, brief):
    payload = {k: v for k, v in brief.items() if not k.startswith("_")}
    conn.execute(
        "INSERT INTO ai_briefs (ticker, brief_json, fetched_at) VALUES (?, ?, ?)",
        (ticker, json.dumps(payload), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_ai_context_from_cache(ticker, conn, technicals=None, technical_levels=None, quiverquant_wsb=None,
                                 reddit_posts=None, analyst_actions=None, peer_comparison=None,
                                 benchmark_relative_performance=None, catalyst_context=None):
    """Pure DB read of everything used to build the AI briefing context --
    no network calls. Used both for the cost estimate (which must never
    trigger a fetch) and as the final read step after _bundle_ai_context
    has refreshed the real-data sources.

    `technical_levels` follows the same pattern as `technicals`: it's a
    real computation (detect_technical_levels), not a DB read, so the
    caller passes it in already-computed rather than this function fetching
    it -- estimate_ai_briefing_cost() leaves it unset (network-free cost
    estimate, may undercount slightly, same as `technicals`), while
    _bundle_ai_context() computes it fresh via detect_technical_levels().
    `analyst_actions`/`peer_comparison`/`benchmark_relative_performance`
    follow the same pattern -- each is a real fetch (via its own cached_*
    wrapper), not a plain DB read, so the caller passes the already-
    resolved envelope's `data` through rather than this function calling
    the network-touching cached_* wrapper itself."""
    fundamentals = _read_fundamentals_info_cache(conn, ticker) or {}
    targets = _read_analyst_targets_cache(conn, ticker) or {}
    earnings_df = _read_earnings_calendar_cache(conn, ticker, limit=6)
    buybacks_df = _read_buybacks_cache(conn, ticker, limit=4)
    news_df = _read_news_cache(conn, ticker, limit=10)
    congress = _read_congressional_cache_by_ticker(conn, ticker, limit=15)
    # 16, not 6 -- a deliberately deep window so the prompt can actually
    # search for a specific historical precedent ("the largest one-day
    # drop since X") instead of only ever seeing the last year and a half.
    earnings_track_record_real = _read_earnings_history_real_cache(conn, ticker, limit=16)
    retail_posts = _read_retail_sentiment_cache(conn, ticker, limit=30)
    thirteenf_filings = _read_13f_cache(conn, ticker, limit=10)
    marketbeat_institutional = _read_marketbeat_institutional_cache(conn, ticker)
    apewisdom = _read_apewisdom_sentiment_cache(conn, ticker)

    insider_rows = conn.execute(
        """SELECT transaction_date, insider_name, title, transaction_type, shares, value, source
           FROM insider_trades WHERE ticker=? AND transaction_date >= date('now','-180 day')
           ORDER BY transaction_date DESC LIMIT 20""",
        (ticker,),
    ).fetchall()

    dark_pool_row = conn.execute(
        """SELECT date, dark_pool_pct, volume_zscore, signal, is_proxy
           FROM dark_pool_signals WHERE ticker=? ORDER BY date DESC LIMIT 1""",
        (ticker,),
    ).fetchone()

    divergence_row = conn.execute(
        """SELECT computed_date, score, label, smart_signal, retail_signal, institutional_magnitude
           FROM divergence_scores WHERE ticker=? ORDER BY computed_date DESC LIMIT 1""",
        (ticker,),
    ).fetchone()

    snapshot_rows = conn.execute(
        """SELECT fetched_at, snapshot_type, hours_to_earnings, conviction_score, divergence_label,
                  smart_call_signals, smart_put_signals
           FROM ticker_snapshots WHERE ticker=? ORDER BY fetched_at DESC LIMIT 10""",
        (ticker,),
    ).fetchall()

    # Same deterministic, badge-driven read the OPTIONS FLOW tab's
    # synthesis panel shows mechanically (Part 6) -- computed from
    # whatever's cached for today, not re-fetched here. Wrapped fail-soft
    # since it's a nice-to-have enrichment, not a required field.
    options_flow_synthesis = None
    try:
        latest_opt_date_row = conn.execute(
            "SELECT MAX(fetch_date) FROM options_flow WHERE ticker=?", (ticker,)
        ).fetchone()
        latest_opt_date = latest_opt_date_row[0] if latest_opt_date_row else None
        if latest_opt_date:
            opt_df = pd.read_sql_query(
                """SELECT option_type, strike, expiration, volume, open_interest, volume_oi_ratio,
                          implied_volatility, last_price, delta, gamma, theta, vega, bid, ask, unusual
                   FROM options_flow WHERE ticker=? AND fetch_date=?""",
                conn, params=(ticker, latest_opt_date),
            )
            if not opt_df.empty:
                earnings_signal = _read_earnings_signal_cache(conn, ticker)
                options_flow_synthesis = synthesize_options_flow_summary(
                    annotate_options_badges(opt_df),
                    spot=(technicals or {}).get("current_price"),
                    earnings_date=(earnings_signal or {}).get("next_earnings_date"),
                )
    except Exception:
        options_flow_synthesis = None

    return {
        "ticker": ticker,
        "as_of": datetime.utcnow().isoformat(),
        "fundamentals": {
            "name": fundamentals.get("shortName"),
            "sector": fundamentals.get("sector"),
            "market_cap": fundamentals.get("marketCap"),
            "trailing_pe": fundamentals.get("trailingPE"),
            "forward_pe": fundamentals.get("forwardPE"),
            "beta": fundamentals.get("beta"),
            "revenue_growth": fundamentals.get("revenueGrowth"),
        },
        "technicals": technicals or {},
        "technical_levels": technical_levels or {},
        "expected_move": _expected_move_both_horizons(conn, ticker),
        "options_flow_weekly_buckets": _bucket_options_flow(conn, ticker),
        # Rules-based, deterministic badge synthesis (Part 2/3) over
        # today's options_flow snapshot -- dominant delta/IV profile,
        # call/put skew reading, and the single most unusual trade. The
        # system prompt below requires retail_analysis/institutional_
        # analysis to build on this SAME read, not re-derive a separate
        # one from options_flow_weekly_buckets alone.
        "options_flow_synthesis": options_flow_synthesis,
        "analyst_targets": targets or {},
        "earnings_calendar": (
            [{k: str(v) for k, v in r.items()} for r in earnings_df.to_dict("records")]
            if not earnings_df.empty else []
        ),
        "earnings_track_record_real": earnings_track_record_real,
        "buybacks": buybacks_df.to_dict("records") if not buybacks_df.empty else [],
        # Trend labels computed from the same buybacks_df rows above (which
        # now carry capex/operating_cash_flow/fcf per period, not just
        # buyback_value) -- surfaced as their own top-level fields so the
        # prompt doesn't have to recompute a trend from the raw numbers
        # itself, same discipline as options_flow_synthesis below.
        "buyback_trend": _buyback_trend(buybacks_df),
        "fcf_trend": _fcf_trend(buybacks_df),
        # Real per-firm rating CHANGE breadth (yfinance upgrades_downgrades),
        # distinct from analyst_targets' single aggregate recommendation --
        # e.g. "11 upgrades vs. 14 downgrades over the trailing 90 days".
        # None if not fetched for this ticker yet.
        "analyst_actions": analyst_actions,
        # 2-4 direct peers' YTD return + trailing P/E, for relative
        # positioning (e.g. "TGT is the sector standout YTD (+62%) against
        # WMT's +4%"). {"peers": []} if no peer could be resolved.
        "peer_comparison": peer_comparison,
        # This ticker's own return vs. a benchmark (SPY, or QQQ for tech
        # names) over YTD/6mo/1y, plus the 52-week high and its date.
        "benchmark_relative_performance": benchmark_relative_performance,
        # Dedicated, cited web-research pass (fetch_catalyst_context) built
        # specifically to surface named product/technology/regulatory
        # catalysts that news_headlines' shallow RSS feed misses --
        # {"named_catalysts": [...], "summary": ..., "search_found_nothing":
        # bool} when fetched, or null if not yet researched for this ticker.
        "catalyst_context": catalyst_context,
        "news_headlines": (
            [
                {"title": r["title"], "publisher": r["publisher"], "published_at": str(r["published_at"])}
                for _, r in news_df.iterrows()
            ] if not news_df.empty else []
        ),
        "congressional_trades": congress,
        "insider_trades": [
            {"date": r[0], "insider": r[1], "title": r[2], "type": r[3], "shares": r[4], "value": r[5],
             "source": r[6] or "sec_form4"}
            for r in insider_rows
        ],
        # SEC EDGAR full-text search for recent 13F-HR filings mentioning
        # this ticker -- "who recently filed", not a holdings-size/share-
        # count diff (see fetch_13f_changes' module docstring). Was fully
        # built (cached_13f_changes, thirteenf_filings table) but never
        # actually wired into this bundle until now.
        "thirteenf_filings": thirteenf_filings,
        # Pure DB read, never fetched from here -- fetching pops a real,
        # visible Chrome window (see fetch_marketbeat_institutional_
        # sentiment's module comment), which a routine "Generate AI
        # Briefing" click shouldn't silently trigger. Only present if the
        # deep-dive tab's explicit MarketBeat-institutional fetch button
        # has been used for this ticker before; None otherwise, and the
        # prompt is written to handle that gracefully.
        "marketbeat_institutional": marketbeat_institutional,
        "retail_sentiment_posts": retail_posts,
        # Aggregated from the same cached StockTwits posts above (never a
        # second live fetch) -- the primary TAGGED SENTIMENT read: real
        # bull/bear counts + ratio from StockTwits' own per-message
        # sentiment tags, plus a few representative excerpts.
        "stocktwits_sentiment": _aggregate_stocktwits_sentiment(retail_posts),
        # Real Reddit mention-tracking -- an ATTENTION/MOMENTUM signal
        # only. No per-post text or sentiment/polarity field exists in
        # this data (ApeWisdom's API doesn't provide one, so none is
        # fabricated here) -- attention_change_pct is the derived,
        # actually-useful figure (% mention-volume change vs. 24h ago).
        # None if this ticker isn't currently ranked (zero recent
        # mentions).
        "apewisdom_sentiment": apewisdom,
        # Real, raw Reddit post titles/bodies via the unauthenticated
        # public search JSON endpoint -- supplementary narrative color,
        # fetched ONLY from _bundle_ai_context (on-demand AI Briefing
        # path), never from a background/auto-refresh path. Fed to
        # Claude as raw text, unscored -- see fetch_reddit_posts_public's
        # docstring. Commonly [] (rate-limited/blocked -- a real, expected
        # state for this best-effort source, not a bug); the prompt is
        # written to say so explicitly rather than silently omitting it.
        "reddit_posts": reddit_posts or [],
        # Real WallStreetBets mention data via QuiverQuant's paid tier --
        # confirmed empty under the current subscription (see
        # fetch_quiverquant_wsb's docstring); present and populated
        # automatically if the plan is ever upgraded, no code change
        # needed.
        "quiverquant_wsb": quiverquant_wsb or [],
        "dark_pool": (
            {"date": dark_pool_row[0], "dark_pool_pct": dark_pool_row[1], "volume_zscore": dark_pool_row[2],
             "signal": dark_pool_row[3], "is_proxy": bool(dark_pool_row[4])}
            if dark_pool_row else None
        ),
        "divergence": (
            {"computed_date": divergence_row[0], "score": divergence_row[1], "label": divergence_row[2],
             "smart_signal": divergence_row[3], "retail_signal": divergence_row[4],
             "institutional_magnitude": divergence_row[5]}
            if divergence_row else None
        ),
        "snapshot_history": [
            {"fetched_at": r[0], "snapshot_type": r[1], "hours_to_earnings": r[2], "conviction_score": r[3],
             "divergence_label": r[4], "smart_call_signals": r[5], "smart_put_signals": r[6]}
            for r in snapshot_rows
        ],
    }


def _bundle_ai_context(ticker, conn):
    """Refreshes the real-data sources ported from new_top.py (each
    respecting its own cache window, so repeated briefings within that
    window don't re-scrape) -- MarketBeat earnings history, Finviz insider
    sales, StockTwits + ApeWisdom retail sentiment, earnings_signal (for
    the IV expected move), SEC EDGAR 13F filings -- plus a fresh technical
    snapshot and a live (uncached) QuiverQuant WSB check, then reads
    everything via _read_ai_context_from_cache. This is the version
    generate_deep_analysis() uses; it can make real network calls, each
    fail-soft."""
    cached_earnings_history_real(conn, ticker, max_age_hours=12)
    cached_insider_sales_finviz(conn, ticker, max_age_hours=24)
    cached_retail_sentiment(conn, ticker, max_age_hours=2)
    cached_apewisdom_sentiment(conn, ticker, max_age_hours=2)
    cached_earnings_signal(conn, ticker, max_age_hours=12)
    cached_13f_changes(conn, ticker, max_age_hours=168)
    cached_buybacks(conn, ticker, max_age_hours=168)
    analyst_actions = cached_analyst_actions(conn, ticker, max_age_hours=12).get("data")
    peer_comparison = cached_peer_comparison(conn, ticker, max_age_hours=24).get("data")
    benchmark_relative_performance = cached_benchmark_relative_performance(conn, ticker, max_age_hours=24).get("data")
    # Real, paid Claude API call on a cold 7-day cache (logs its own
    # [CLAUDE API] line via _log_claude_call inside fetch_catalyst_context)
    # -- covered by the same "Generate AI Briefing" confirmation as the main
    # synthesis call below, not a separate gate.
    catalyst_context = cached_catalyst_context(conn, ticker, max_age_hours=168).get("data")
    technicals = _fetch_technical_snapshot(ticker)
    # DEFAULT_INTERVAL/DEFAULT_LOOKBACK (1d/6mo) -- the same daily-candle,
    # 6-month swing-horizon window the dashboard shows by default, so the
    # AI's narrative lines up with what a trader sees on first load.
    technical_levels = detect_technical_levels(ticker, DEFAULT_INTERVAL, DEFAULT_LOOKBACK, conn=conn)
    # Not cached (see fetch_quiverquant_wsb's docstring -- no verified
    # success schema yet to design a table around); cheap to call live
    # since it returns [] instantly under the current plan tier.
    quiverquant_wsb = fetch_quiverquant_wsb(ticker)
    # Reddit public JSON: called ONLY here (the on-demand AI Briefing
    # path), never from full_refresh -- see fetch_reddit_posts_public's
    # docstring. Logged to fetch_log as a best-effort unauthenticated
    # source regardless of outcome; the fetcher itself already fails
    # soft, so this call never raises.
    reddit_posts = fetch_reddit_posts_public(ticker)
    _log_fetch(conn, "reddit_public_json", ticker, True, len(reddit_posts))
    return _read_ai_context_from_cache(
        ticker, conn, technicals=technicals, technical_levels=technical_levels, quiverquant_wsb=quiverquant_wsb,
        reddit_posts=reddit_posts, analyst_actions=analyst_actions, peer_comparison=peer_comparison,
        benchmark_relative_performance=benchmark_relative_performance, catalyst_context=catalyst_context,
    )


def _estimate_ai_briefing_cost(context):
    """Rough token/cost estimate for the confirmation popup -- no API call is
    made to produce this estimate, just a size-based approximation."""
    bundle_chars = len(json.dumps(context, default=str)) + len(_AI_BRIEF_SYSTEM_PROMPT)
    # ~4 chars/token is a standard rough approximation for English/JSON text.
    est_input_tokens = max(1, bundle_chars // 4)
    est_output_tokens = 1400  # six-section brief incl. catalysts + 3 verdicts, generous upper bound
    # claude-opus-5 list pricing: $5.00 / 1M input tokens, $25.00 / 1M output tokens.
    est_cost = (est_input_tokens / 1_000_000) * 5.00 + (est_output_tokens / 1_000_000) * 25.00
    return {"input_tokens": est_input_tokens, "output_tokens": est_output_tokens, "cost_usd": est_cost}


def estimate_ai_briefing_cost(ticker, conn):
    """Rough cost estimate for the confirmation popup. Pure DB read -- never
    makes a network call (Claude or otherwise), so the estimate may
    undercount slightly for a ticker whose real-data sources haven't been
    fetched yet (they'll be freshened when generate_deep_analysis runs)."""
    return _estimate_ai_briefing_cost(_read_ai_context_from_cache(ticker, conn))


def get_cached_ai_brief(conn, ticker, max_age_hours=AI_BRIEF_CACHE_HOURS):
    """Returns the cached brief dict if one exists within max_age_hours, else
    None. Pure DB read -- never makes an API call, safe to call on every page
    load/rerun."""
    return _read_ai_brief_cache(conn, ticker, max_age_hours=max_age_hours)


def generate_deep_analysis(ticker: str, conn) -> dict:
    """Builds the full real-data AI Deep-Dive Briefing context for `ticker`
    (see _bundle_ai_context) and sends it to the Anthropic API -- reads
    ANTHROPIC_API_KEY from the environment, never hardcoded -- to produce a
    structured 6-section brief: setup, retail_analysis, institutional_analysis
    (each citing source + date), news_summary, catalysts (dynamically
    derived, never hardcoded per ticker), earnings_track_record (real
    beat/miss + real price-reaction pairs), and exactly three dated
    verdicts (next earnings / end of week / end of month).

    The system prompt enforces hard anti-hallucination rules: only state
    facts present in the bundle, name no institution/analyst/fund not
    explicitly in the data, explicitly flag empty categories instead of
    guessing, and explain earnings reactions using only the real move data
    (or news headlines) provided.

    Returns structured JSON for clean rendering, and caches the result in
    ai_briefs (4h window). Always makes a fresh (paid) Claude API call when
    invoked -- callers must check get_cached_ai_brief() first and only call
    this after explicit user confirmation.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set -- cannot generate an AI briefing.")

    context = _bundle_ai_context(ticker, conn)

    _log_claude_call("AI Briefing", ticker)
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=6000,
        system=_AI_BRIEF_SYSTEM_PROMPT,
        output_config={"format": {"type": "json_schema", "schema": _AI_BRIEF_SCHEMA}},
        messages=[{"role": "user", "content": json.dumps(context, default=str)}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("The Claude API declined to generate this briefing.")

    text = next((b.text for b in response.content if b.type == "text"), "")
    brief = json.loads(text)

    # Real numeric fields, computed by us (never LLM-generated text) --
    # kept separate from the prose sections so the dashboard can render them
    # with plain Python f-string formatting via st.metric (immune to
    # Streamlit's markdown $-as-LaTeX parsing) instead of ever needing to
    # extract a number out of AI-written prose.
    brief["technicals_snapshot"] = context.get("technicals") or {}
    brief["expected_move_snapshot"] = context.get("expected_move") or {}
    brief["technical_levels_snapshot"] = context.get("technical_levels") or {}

    brief["_fetched_at"] = datetime.utcnow().isoformat()
    brief["_cache_hit"] = False
    _persist_ai_brief(conn, ticker, brief)
    return brief


# --------------------------------------------------------------------------
# RUNNERS tab -- Day Prediction: a same-day forecast-cone target, built for
# the (at most 4) tickers RUNNERS flags as today's unusual movers. Reuses
# the Earnings Simulator's evidence-tiered mode system (raw_pattern /
# bayesian / trained_model, same thresholds) and its qualitative-briefing
# extraction (extract_simulator_inputs_from_briefing), recalibrated for a
# single trading day: magnitude from this ticker's own trailing 1-year
# DAILY realized volatility (never claimed as an intraday-pattern model --
# only daily bars exist in the 1-year backfill), direction from a blend of
# early-session momentum + technical structure + the Briefing's nearest-
# term qualitative read.
# --------------------------------------------------------------------------

DAY_PREDICTION_BUFFER_MINUTES_DEFAULT = 30


def get_session_prediction_window(market_open_time, buffer_minutes=DAY_PREDICTION_BUFFER_MINUTES_DEFAULT):
    """The timestamp the Day Prediction panel is allowed to commit to a
    target at -- market_open_time + buffer_minutes, so the model waits
    for real opening volume/direction to establish itself instead of
    reacting to the first few noisy seconds/minutes of the session.
    Before this passes, the caller shows a 'waiting for session data'
    message rather than an empty or placeholder chart."""
    return market_open_time + timedelta(minutes=buffer_minutes)


def _infer_bar_minutes(hist):
    """Median spacing between consecutive bars in `hist`, in minutes --
    used to correctly time-scale today's own realized intraday volatility
    to whatever step_minutes simulate_intraday_path is asked for, even if
    that ever differs from the bars' native interval."""
    if hist is None or len(hist) < 2:
        return None
    diffs = hist.index.to_series().diff().dropna()
    if diffs.empty:
        return None
    minutes = diffs.median().total_seconds() / 60
    return minutes if minutes > 0 else None


def simulate_intraday_path(current_price, target_price, minutes_remaining, daily_volatility,
                            step_minutes=5, start_time=None, intraday_hist=None, stored_intraday_step_vol=None):
    """A real drift-adjusted random walk from current_price toward
    target_price -- NOT a straight ruler-line, which implies false
    precision a genuine simulation doesn't have. Fixes exactly that bug:
    the chart used to draw a literal 2-point line from model-start to the
    end-of-day target.

    - steps = minutes_remaining / step_minutes (step_minutes floored at
      5 -- never coarser, regardless of the UI's own refresh-interval
      selection, which controls redraw cadence, not simulation
      granularity)
    - drift_per_step is the constant per-step LOG return that, compounded
      over every step, averages to log(target_price / current_price) --
      so the path lands APPROXIMATELY on target in expectation, not
      forced there with an artificial final-step jump
    - step_vol prefers, in order: (1) TODAY'S OWN realized intraday
      volatility so far (from intraday_hist's real closes, once >=4
      bars/3 log-returns exist) -- most relevant for a RUNNERS ticker
      specifically, since these are flagged for an unusually active day
      RIGHT NOW, which a multi-week average would understate; (2) the
      real PERSISTED intraday history's own realized vol
      (stored_intraday_step_vol, pre-computed by the caller via
      _compute_intraday_realized_vol from real stored bars -- Part 4's
      backfill is what makes this tier possible at all, and it keeps
      improving as real trading days accumulate); (3) the 1-year DAILY
      volatility scaled down via sqrt(step_minutes/(6.5*60)) -- 6.5*60 =
      390 minutes in a trading day, the standard time-scaling -- only
      when neither real intraday source is available yet
    - each step applies np.random.normal(drift_per_step, step_vol) as a
      log-return shock, producing genuine up/down wiggle texture

    Generate this ONCE per (ticker, session_date) and cache the result
    (see log_day_prediction) -- regenerating on every refresh would mean
    comparing evolving reality against a constantly-redrawn prediction,
    defeating the entire point of a committed target.

    Returns a list of (timestamp, price) tuples at step_minutes intervals
    from start_time (default: now) through minutes_remaining."""
    step_minutes = max(5, step_minutes)
    steps = max(1, int(round(minutes_remaining / step_minutes)))

    total_log_return = (
        math.log(target_price / current_price) if current_price and target_price and current_price > 0
        else 0.0
    )
    drift_per_step = total_log_return / steps

    step_vol = None
    if intraday_hist is not None and len(intraday_hist) >= 4:
        closes = intraday_hist["Close"]
        log_rets = np.log(closes / closes.shift(1)).dropna()
        if len(log_rets) >= 3:
            bar_minutes = _infer_bar_minutes(intraday_hist) or step_minutes
            today_bar_std = float(log_rets.std())
            if math.isfinite(today_bar_std) and today_bar_std > 0:
                step_vol = today_bar_std * math.sqrt(step_minutes / bar_minutes)
    if step_vol is None and stored_intraday_step_vol is not None and math.isfinite(stored_intraday_step_vol) \
            and stored_intraday_step_vol > 0:
        step_vol = stored_intraday_step_vol
    if step_vol is None or not math.isfinite(step_vol) or step_vol <= 0:
        step_vol = (daily_volatility or 0.01) * math.sqrt(step_minutes / (6.5 * 60))
    step_vol = max(step_vol, 1e-5)

    start_time = start_time or pd.Timestamp.now(tz="America/New_York")
    rng = np.random.default_rng()
    price = float(current_price)
    t = start_time
    path = [(t, round(price, 2))]
    for _ in range(steps):
        shock = rng.normal(drift_per_step, step_vol)
        price = price * math.exp(shock)
        t = t + timedelta(minutes=step_minutes)
        path.append((t, round(float(price), 2)))
    return path


def _predicted_price_at_time(path, when):
    """Reads the predicted price at `when` off the CACHED simulated path
    (Part 3) -- linear interpolation between the two path points bracketing
    `when`, not a fresh straight-line calc from the endpoints. This is what
    guarantees the chart and the live comparison table can never disagree:
    both read this exact same path."""
    if not path:
        return None
    if when <= path[0][0]:
        return path[0][1]
    if when >= path[-1][0]:
        return path[-1][1]
    for (t0, p0), (t1, p1) in zip(path, path[1:]):
        if t0 <= when <= t1:
            span = (t1 - t0).total_seconds()
            frac = (when - t0).total_seconds() / span if span > 0 else 0.0
            return p0 + frac * (p1 - p0)
    return path[-1][1]


def fetch_intraday_bars(ticker, interval="5m"):
    """TODAY's session bars only, via yfinance (period='1d') -- for the
    RUNNERS tab's live candlestick-so-far and early-momentum signal. Not
    DB-cached: this is a live "what does the chart look like right now"
    read, distinct from fetch_intraday_history_delta below, which
    persists the real trailing ~60-day intraday window (including today,
    once the session ends) for volatility estimation. Returns an empty
    DataFrame on any error, a closed market, or before today's first bar
    exists."""
    try:
        hist = yf.Ticker(ticker).history(period="1d", interval=interval)
    except Exception:
        return pd.DataFrame()
    return hist if hist is not None else pd.DataFrame()


def fetch_intraday_history_delta(conn, ticker, interval="5m", period="60d"):
    """Delta-aware persisted intraday backfill (Part 4): a full
    period='60d' pull only the first time this ticker has no stored
    intraday_price_history rows, then just `start=<the minute after our
    last stored bar>` after that -- same delta-loading pattern as
    fetch_price_history_delta, just at 5-minute granularity. yfinance's
    real free-tier limit for 5-minute bars is ~60 days; this never claims
    or attempts more than that. Returns (n_new_rows, request_desc) for
    logging/verification, same shape as fetch_price_history_delta."""
    table = "intraday_price_history"
    last_ts = conn.execute(
        "SELECT MAX(datetime) FROM intraday_price_history WHERE ticker=?", (ticker,)
    ).fetchone()[0]
    try:
        if last_ts is None:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            request_desc = f'FULL PULL: history(period="{period}", interval="{interval}")'
        else:
            start = pd.Timestamp(last_ts) + pd.Timedelta(minutes=1)
            hist = yf.Ticker(ticker).history(start=start, interval=interval)
            request_desc = f'DELTA PULL: history(start="{start.isoformat()}", interval="{interval}")'
    except Exception as e:
        hist, request_desc = pd.DataFrame(), f"FAILED: {e}"

    if hist is not None and not hist.empty:
        _persist_intraday_history(conn, ticker, hist)
    _log_fetch(conn, table, ticker, hist is not None, len(hist) if hist is not None else 0)
    return (len(hist) if hist is not None else 0), request_desc


def _persist_intraday_history(conn, ticker, hist):
    cur = conn.cursor()
    fetched_at = datetime.utcnow().isoformat()
    for idx, row in hist.iterrows():
        cur.execute(
            """INSERT INTO intraday_price_history (ticker, datetime, open, high, low, close, volume, fetched_at)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(ticker, datetime) DO UPDATE SET
                   open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                   volume=excluded.volume, fetched_at=excluded.fetched_at""",
            (ticker, pd.Timestamp(idx).isoformat(), _safe_num(row.get("Open"), None),
             _safe_num(row.get("High"), None), _safe_num(row.get("Low"), None), _safe_num(row.get("Close"), None),
             int(row["Volume"]) if pd.notna(row.get("Volume")) else None, fetched_at),
        )
    conn.commit()


def _read_intraday_history(conn, ticker, days_back=60):
    df = pd.read_sql_query(
        """SELECT datetime, open, high, low, close, volume FROM intraday_price_history
           WHERE ticker=? AND datetime >= datetime('now', ?) ORDER BY datetime ASC""",
        conn, params=(ticker, f"-{int(days_back)} day"),
    )
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")
    return df.set_index("datetime").rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume",
    })


def _compute_intraday_realized_vol(conn, ticker, step_minutes=5, lookback_days=60):
    """Real intraday-bar realized volatility from the PERSISTED history
    (not just today's live bars) -- the genuine improvement Part 4's
    backfill is meant to unlock: as real trading days of stored 5-minute
    bars accumulate, this becomes a richer, less-noisy volatility
    estimate than either today's handful of bars or the daily-vol
    fallback. Log-returns are computed only WITHIN each trading day
    (grouped by date) -- never across the overnight/weekend gap between
    one day's last bar and the next day's first, which would otherwise
    inflate the estimate with a return that isn't really intraday at
    all. Returns None if there isn't enough real stored history yet
    (<3 trading days worth of bars) -- simulate_intraday_path falls back
    to today's-bars-only, then daily-vol-scaled, in that case."""
    hist = _read_intraday_history(conn, ticker, days_back=lookback_days)
    if hist.empty or len(hist) < 50:
        return None
    hist = hist.sort_index()
    session_date = hist.index.tz_convert("America/New_York").date
    all_rets = []
    for _day, group in hist.groupby(session_date):
        closes = group["Close"]
        if len(closes) < 3:
            continue
        rets = np.log(closes / closes.shift(1)).dropna()
        all_rets.extend(rets.tolist())
    if len(all_rets) < 30:
        return None
    bar_minutes = _infer_bar_minutes(hist) or step_minutes
    bar_std = float(np.std(all_rets))
    if not math.isfinite(bar_std) or bar_std <= 0:
        return None
    step_vol = bar_std * math.sqrt(step_minutes / bar_minutes)
    n_days = len(set(session_date))
    return {"step_vol": step_vol, "n_bars": len(all_rets), "n_days": n_days}


def _prior_session_close(ticker):
    """Yesterday's close, for computing today's opening GAP -- deliberately
    a fresh small daily pull rather than reusing _fetch_technical_
    snapshot's 6-month history, since that series can itself include
    today's still-forming daily bar as its last row during market hours,
    which would make 'the prior close' silently mean 'today's live price'
    instead. Takes the last row strictly before today by date, not just
    index -2, so this stays correct regardless of whether today's bar has
    landed in the series yet."""
    try:
        hist = yf.Ticker(ticker).history(period="5d", interval="1d")
    except Exception:
        return None
    if hist is None or hist.empty:
        return None
    today = date.today()
    prior = hist[hist.index.date < today]
    if prior.empty:
        return None
    return float(prior["Close"].iloc[-1])


def _realized_vol_from_closes(closes, lookback_days=252):
    """Pure math half of _compute_realized_daily_vol -- trailing
    log-return std, annualized via sqrt(252) then rescaled back down to a
    1-day figure via sqrt(1/252). Split out from the live fetch so the
    exact same formula can be reused against a historical, strictly-
    prior-to-day-D slice of closes (backtest_day_predictions) without
    duplicating the math."""
    closes = closes.tail(lookback_days + 1)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if log_returns.empty:
        return None
    daily_std = float(log_returns.std())
    if not math.isfinite(daily_std) or daily_std <= 0:
        return None
    annualized_vol_pct = daily_std * math.sqrt(252) * 100
    one_day_expected_move_pct = annualized_vol_pct * math.sqrt(1 / 252)
    return {
        "annualized_vol_pct": round(annualized_vol_pct, 2),
        "one_day_expected_move_pct": round(one_day_expected_move_pct, 2),
        "n_days": len(log_returns),
    }


def _compute_realized_daily_vol(conn, ticker, lookback_days=252):
    """Trailing ~1-year realized daily volatility -> a 1-day expected
    move (see _realized_vol_from_closes for the actual math). Honest by
    construction: this is projected from DAILY bars only, never a claim
    of any trained intraday pattern (see the panel's own caption)."""
    fetch_price_history_delta(conn, ticker, full_period="1y", days_back=lookback_days + 30, max_age_hours=24)
    hist = _read_price_history(conn, ticker, days_back=lookback_days + 30)
    if hist.empty or len(hist) < 30:
        return None
    return _realized_vol_from_closes(hist.sort_index()["Close"], lookback_days)


_DAY_TREND_LEAN_SCORE = {
    "uptrend": 1.0, "breaking out": 1.0, "downtrend": -1.0, "breaking down": -1.0,
    "consolidating": 0.0, "unknown": 0.0,
}


def _day_prediction_directional_lean(prior_close, intraday_hist, technical_levels, technicals, briefing_inputs):
    """Blends three signals into one -1..+1 lean, each contributing only
    when its own input is actually available -- weights renormalize over
    whichever signals are present, never padded with a fabricated neutral
    standing in for a missing one:

    (a) early-session momentum, weight 0.40 -- the opening gap (today's
        open vs. yesterday's real close) blended with intraday momentum
        so far (last vs. open), compressed to -1..+1 over a +/-2% span;
    (b) short-term technical structure, weight 0.35 -- detect_technical_
        levels' trend_structure label, nudged by an RSI mean-reversion
        signal (overbought/oversold);
    (c) the AI Briefing's NEAREST-TERM qualitative verdict, weight 0.25 --
        prefers the 'end of week' verdict over 'next earnings' (closer in
        horizon to a single trading day), reusing extract_simulator_
        inputs_from_briefing()'s own verdicts list rather than a new
        extractor."""
    components = []

    if intraday_hist is not None and not intraday_hist.empty:
        open_price = float(intraday_hist["Open"].iloc[0])
        last_price = float(intraday_hist["Close"].iloc[-1])
        gap_pct = (open_price / prior_close - 1) * 100 if prior_close else None
        momentum_pct = (last_price / open_price - 1) * 100 if open_price else None
        combined = None
        if gap_pct is not None and momentum_pct is not None:
            combined = gap_pct * 0.4 + momentum_pct * 0.6
        elif momentum_pct is not None:
            combined = momentum_pct
        elif gap_pct is not None:
            combined = gap_pct
        if combined is not None:
            components.append((max(-1.0, min(1.0, combined / 2.0)), 0.40))

    if technical_levels and technical_levels.get("trend_structure"):
        trend_score = _DAY_TREND_LEAN_SCORE.get(technical_levels["trend_structure"], 0.0)
        rsi = (technicals or {}).get("rsi14")
        rsi_nudge = 0.3 if (rsi is not None and rsi <= 30) else -0.3 if (rsi is not None and rsi >= 70) else 0.0
        components.append((max(-1.0, min(1.0, trend_score + rsi_nudge)), 0.35))

    if briefing_inputs:
        verdicts = briefing_inputs.get("verdicts") or []
        nearest = (
            next((v for v in verdicts if "week" in (v.get("horizon") or "").lower()), None)
            or briefing_inputs.get("next_earnings_verdict")
        )
        if nearest:
            bias = (nearest.get("bias") or "NEUTRAL").upper()
            conf_mag = {"High": 1.0, "Medium": 0.6, "Low": 0.3}.get(nearest.get("confidence"), 0.3)
            qual_lean = conf_mag if bias == "BULLISH" else -conf_mag if bias == "BEARISH" else 0.0
            components.append((qual_lean, 0.25))

    if not components:
        return None, {}
    total_weight = sum(w for _, w in components)
    lean = round(max(-1.0, min(1.0, sum(s * w for s, w in components) / total_weight)), 4)
    trend_score_out = _DAY_TREND_LEAN_SCORE.get((technical_levels or {}).get("trend_structure"), 0.0)
    return lean, {"trend_score": trend_score_out}


# --------------------------------------------------------------------------
# Historical backtest -- bootstraps a real reconciled sample for Mode 1/2/3
# and calibration immediately, instead of waiting weeks for a ticker to
# qualify as a live RUNNERS "unusual" day often enough to accumulate one.
# --------------------------------------------------------------------------

BACKTEST_DEFAULT_LOOKBACK_DAYS = 330
# Extra trading days of daily history needed BEFORE the earliest backtest
# day so RSI/MACD/SMA200-based trend structure have real warmup on day 1
# of the window too, not "unknown" for lack of prior history.
_BACKTEST_WARMUP_TRADING_DAYS = 260


def _historical_technicals_and_lean(hist_before, day_row, prior_close):
    """Reconstructs _day_prediction_directional_lean's inputs for ONE
    historical trading day, using ONLY hist_before (real daily OHLCV
    strictly before that day -- no lookahead) plus that day's own OPEN
    (known at the market open, i.e. before any lookahead too).

    This is a real but narrower approximation of the live pipeline, and
    is deliberately labeled as such everywhere it surfaces:
      - momentum component: the opening GAP only (today's open vs.
        yesterday's real close) -- live's own intraday-momentum-so-far
        half of this same component isn't reconstructable this far back
        (5-min bars aren't retained beyond ~60 days), so this uses
        exactly the same gap-only fallback _day_prediction_directional_
        lean itself already has for when momentum_pct is unavailable.
      - technical-structure component: identical to live -- RSI14/MACD
        histogram/trend_structure computed via the exact same functions
        (_rsi, _macd, _compute_technical_levels), just fed a truncated
        history ending the day before.
      - qualitative (AI Briefing) component: dropped entirely -- no
        historical AI Briefings exist. Weights renormalize over the two
        remaining components, same mechanism _day_prediction_directional_
        lean already uses for any missing signal.

    Returns (lean, lean_features, technicals) or (None, {}, {}) if there
    isn't enough prior history to compute anything meaningful."""
    if len(hist_before) < 15:
        return None, {}, {}

    h = hist_before.copy()
    h["RSI14"] = _rsi(h["Close"])
    _, _, macd_hist = _macd(h["Close"])
    h["MACD_hist"] = macd_hist
    rsi14 = float(h["RSI14"].iloc[-1]) if pd.notna(h["RSI14"].iloc[-1]) else None
    macd_histogram = float(h["MACD_hist"].iloc[-1]) if pd.notna(h["MACD_hist"].iloc[-1]) else None

    trend_structure = "unknown"
    if len(h) >= _BACKTEST_WARMUP_TRADING_DAYS:
        cfg = dict(_INTERVAL_CONFIG["1d"])
        cfg["interval"], cfg["lookback"] = "1d", "backtest"
        short_w, long_w = cfg["sma_windows"]
        h["SMA_short"] = h["Close"].rolling(short_w).mean()
        h["SMA_long"] = h["Close"].rolling(long_w).mean()
        levels = _compute_technical_levels(h, cfg, "")
        trend_structure = levels.get("trend_structure") or "unknown"

    open_price = float(day_row["Open"])
    gap_pct = (open_price / prior_close - 1) * 100 if prior_close else None

    components = []
    if gap_pct is not None:
        components.append((max(-1.0, min(1.0, gap_pct / 2.0)), 0.40))
    trend_score = _DAY_TREND_LEAN_SCORE.get(trend_structure, 0.0)
    rsi_nudge = 0.3 if (rsi14 is not None and rsi14 <= 30) else -0.3 if (rsi14 is not None and rsi14 >= 70) else 0.0
    components.append((max(-1.0, min(1.0, trend_score + rsi_nudge)), 0.35))

    total_weight = sum(w for _, w in components)
    lean = round(max(-1.0, min(1.0, sum(s * w for s, w in components) / total_weight)), 4) if total_weight else None
    return lean, {"trend_score": trend_score, "trend_structure": trend_structure}, {
        "rsi14": rsi14, "macd_histogram": macd_histogram,
    }


def backtest_day_predictions(ticker, conn, lookback_days=BACKTEST_DEFAULT_LOOKBACK_DAYS):
    """Simulates the Day Prediction pipeline against real historical daily
    sessions, day by day, using only data strictly BEFORE each day (no
    lookahead bias) -- see _historical_technicals_and_lean for the exact
    reconstruction and its documented, honest gaps vs. the live pipeline
    (no 5-min intraday bars or AI Briefing this far back; model_start_price
    is approximated by the session's real Open).

    Only days that would have genuinely qualified as a live RUNNERS
    "unusual" day (volume >=2x the trailing 20-day average, computed from
    data strictly before that day, OR a full-session move >=3%) are
    backtested -- not every day -- so the sample matches real-world usage
    instead of a flat historical average. Rows are tagged source='backtest'
    (never overwriting an existing 'live' row for the same session_date --
    live stays the higher-trust category) and are reconciled immediately,
    since the actual outcome is already known history.

    Idempotent per (ticker, session_date): re-running skips any day
    already on record (from a prior backtest OR a live prediction) and
    only backfills genuinely new qualifying days -- see Part 6's "Re-run
    Backtest," meant to keep the trailing window current, not duplicate
    work. Always logs one row to backtest_runs (Part 6 registry), even if
    zero new sessions were logged this run."""
    total_days_back = lookback_days + _BACKTEST_WARMUP_TRADING_DAYS + 140  # + calendar padding for weekends/holidays
    fetch_price_history_delta(conn, ticker, full_period="2y", days_back=total_days_back, max_age_hours=24)
    full_hist = _read_price_history(conn, ticker, days_back=total_days_back)
    if full_hist.empty or len(full_hist) < 45:
        return {
            "ticker": ticker, "ok": False,
            "reason": f"Not enough daily price history for {ticker} to run a backtest "
                      f"({len(full_hist)} trading day(s) available; need at least 45).",
        }
    full_hist = full_hist.sort_index()
    # Never backtest TODAY: a backtest row and a live prediction share the
    # same UNIQUE(ticker, session_date) slot, so if today's daily bar is
    # already closed and stored (e.g. running this late in the evening,
    # after the session ended) AND today happens to qualify as "unusual,"
    # a backtest row would claim today's session_date before the live
    # pipeline ever runs -- then log_day_prediction's path-backfill would
    # attach a live (tz-aware) simulated path onto a row whose
    # model_start_time is still the backtest's naive midnight timestamp,
    # producing "Cannot compare tz-naive and tz-aware timestamps" the
    # moment the comparison table tries to grade a snapshot against it.
    # Confirmed live: exactly this happened to LULU's 2026-08-18 row.
    # Backtesting is explicitly historical reconstruction; today is
    # always live-only territory.
    today_et = pd.Timestamp.now(tz="America/New_York").date()
    candidate_start = max(0, len(full_hist) - lookback_days)
    qualifying = []
    for i in range(candidate_start, len(full_hist)):
        if i < 15:
            continue
        day_row = full_hist.iloc[i]
        if full_hist.index[i].date() >= today_et:
            continue
        hist_before = full_hist.iloc[:i]
        prior_close = float(hist_before["Close"].iloc[-1])
        avg_vol_20d = hist_before["Volume"].tail(20).mean()
        vol_ratio = (day_row["Volume"] / avg_vol_20d) if avg_vol_20d else None
        chg_pct = (day_row["Close"] / prior_close - 1) * 100 if prior_close else None
        unusual_vol = vol_ratio is not None and vol_ratio >= 2.0
        big_move = chg_pct is not None and abs(chg_pct) >= 3.0
        if unusual_vol or big_move:
            qualifying.append((full_hist.index[i], day_row, hist_before, prior_close))

    logged, skipped_existing = 0, 0
    for ts, day_row, hist_before, prior_close in qualifying:
        session_date = ts.date().isoformat()
        exists = conn.execute(
            "SELECT 1 FROM day_predictions WHERE ticker=? AND session_date=?", (ticker, session_date)
        ).fetchone()
        if exists:
            skipped_existing += 1
            continue

        lean, lean_features, tech = _historical_technicals_and_lean(hist_before, day_row, prior_close)
        if lean is None:
            continue
        vol = _realized_vol_from_closes(hist_before["Close"], lookback_days=252)
        if vol is None:
            continue

        model_start_price = float(day_row["Open"])
        expected_move_pct = vol["one_day_expected_move_pct"]
        target_price = round(model_start_price * (1 + lean * expected_move_pct / 100), 2)
        predicted_direction = (
            "UP" if target_price > model_start_price else "DOWN" if target_price < model_start_price else "FLAT"
        )
        actual_close = float(day_row["Close"])
        actual_direction = (
            "UP" if actual_close > model_start_price else "DOWN" if actual_close < model_start_price else "FLAT"
        )
        prediction_correct = int(predicted_direction == actual_direction)
        error_pct = round((actual_close - target_price) / target_price * 100, 2) if target_price else None

        conn.execute(
            """INSERT OR IGNORE INTO day_predictions
                   (ticker, session_date, model_start_time, model_start_price, target_price,
                    predicted_direction, magnitude_estimate_pct, lean_pre_model, rsi14, macd_histogram,
                    trend_score, mode, source, actual_close_price, actual_direction,
                    prediction_correct_direction, error_pct, reconciliation_notes, predicted_at, reconciled_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ticker, session_date, ts.isoformat(), model_start_price, target_price, predicted_direction,
             expected_move_pct, lean, tech.get("rsi14"), tech.get("macd_histogram"),
             lean_features.get("trend_score"), "backtest", "backtest", actual_close, actual_direction,
             prediction_correct, error_pct,
             "Backtested from historical daily OHLCV -- model_start_price approximated by the session's real "
             "Open (no persisted 5-min intraday bars this far back); no intraday path to grade.",
             datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
        )
        logged += 1
    conn.commit()

    agg_rows = conn.execute(
        "SELECT error_pct, prediction_correct_direction FROM day_predictions WHERE ticker=? AND source='backtest'",
        (ticker,),
    ).fetchall()
    errors = [r[0] for r in agg_rows if r[0] is not None]
    correct = sum(1 for r in agg_rows if r[1])
    result = {
        "ticker": ticker, "ok": True, "lookback_days": lookback_days,
        "qualifying_sessions_found": len(qualifying), "sessions_logged": logged,
        "sessions_skipped_existing": skipped_existing, "total_backtested_on_record": len(agg_rows),
        "date_range_start": qualifying[0][0].date().isoformat() if qualifying else None,
        "date_range_end": qualifying[-1][0].date().isoformat() if qualifying else None,
        "mean_error_pct": round(sum(errors) / len(errors), 2) if errors else None,
        "mean_abs_error_pct": round(sum(abs(e) for e in errors) / len(errors), 2) if errors else None,
        "direction_accuracy": round(correct / len(agg_rows), 3) if agg_rows else None,
    }
    conn.execute(
        """INSERT INTO backtest_runs
               (ticker, run_at, lookback_days, date_range_start, date_range_end, qualifying_sessions_found,
                sessions_backtested, mean_error_pct, mean_abs_error_pct, direction_accuracy)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (ticker, datetime.utcnow().isoformat(), lookback_days, result["date_range_start"],
         result["date_range_end"], result["qualifying_sessions_found"], len(agg_rows),
         result["mean_error_pct"], result["mean_abs_error_pct"], result["direction_accuracy"]),
    )
    conn.commit()
    return result


def get_day_prediction_sample_composition(conn, ticker=None):
    """Splits the pooled reconciled day_predictions sample by source
    (Part 3) -- backtested rows bootstrap Mode 1/2/3 thresholds and
    calibration far faster than live sessions alone, but live rows remain
    the higher-trust category as they accumulate: they capture real
    same-day intraday momentum and the AI Briefing's qualitative read,
    neither of which a historical daily-OHLCV reconstruction can supply.
    Scoped to one ticker when given, else system-wide -- matching how
    Mode selection itself is pooled system-wide, not per-ticker."""
    query = (
        "SELECT COALESCE(source, 'live') AS src, COUNT(*) FROM day_predictions "
        "WHERE prediction_correct_direction IS NOT NULL"
    )
    params = ()
    if ticker:
        query += " AND ticker=?"
        params = (ticker,)
    query += " GROUP BY src"
    counts = dict(conn.execute(query, params).fetchall())
    backtest_n = counts.get("backtest", 0)
    live_n = sum(n for src, n in counts.items() if src != "backtest")
    return {"backtest_n": backtest_n, "live_n": live_n, "total": backtest_n + live_n}


def get_current_day_prediction_mode(conn):
    """The system-wide pooled Mode compute_day_target would currently
    select (Part 6) -- exposed standalone so the SETTINGS registry can
    show "current mode" without needing any specific ticker's live
    session data."""
    pooled_reconciled_count = conn.execute(
        "SELECT COUNT(*) FROM day_predictions WHERE prediction_correct_direction IS NOT NULL"
    ).fetchone()[0]
    if pooled_reconciled_count < 10:
        return "raw_pattern", pooled_reconciled_count
    if train_day_direction_model(conn) is not None:
        return "trained_model", pooled_reconciled_count
    return "bayesian", pooled_reconciled_count


def get_backtested_tickers_summary(conn):
    """Latest backtest_runs row per ticker (Part 6 registry table)."""
    rows = conn.execute(
        """SELECT b.ticker, b.run_at, b.lookback_days, b.qualifying_sessions_found, b.sessions_backtested,
                  b.mean_abs_error_pct, b.direction_accuracy
           FROM backtest_runs b
           INNER JOIN (SELECT ticker, MAX(id) AS max_id FROM backtest_runs GROUP BY ticker) latest
             ON b.ticker = latest.ticker AND b.id = latest.max_id
           ORDER BY b.ticker"""
    ).fetchall()
    keys = ["ticker", "run_at", "lookback_days", "qualifying_sessions_found", "sessions_backtested",
            "mean_abs_error_pct", "direction_accuracy"]
    return [dict(zip(keys, r)) for r in rows]


def get_backtest_report_data(conn, ticker):
    """Everything the Backtest Report (Part 4) needs for one ticker: every
    backtested day_predictions row (for the distribution/best-worst
    table) plus the latest backtest_runs summary row. Returns (rows,
    latest_run) -- rows is [] and latest_run is None if never backtested."""
    rows = conn.execute(
        """SELECT session_date, model_start_price, target_price, actual_close_price, error_pct,
                  prediction_correct_direction
           FROM day_predictions WHERE ticker=? AND source='backtest' AND error_pct IS NOT NULL
           ORDER BY session_date""",
        (ticker,),
    ).fetchall()
    keys = ["session_date", "model_start_price", "target_price", "actual_close_price", "error_pct",
            "prediction_correct_direction"]
    rows = [dict(zip(keys, r)) for r in rows]
    latest_run = conn.execute(
        """SELECT run_at, lookback_days, date_range_start, date_range_end, qualifying_sessions_found,
                  sessions_backtested, mean_error_pct, mean_abs_error_pct, direction_accuracy
           FROM backtest_runs WHERE ticker=? ORDER BY id DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if latest_run:
        run_keys = ["run_at", "lookback_days", "date_range_start", "date_range_end",
                    "qualifying_sessions_found", "sessions_backtested", "mean_error_pct",
                    "mean_abs_error_pct", "direction_accuracy"]
        latest_run = dict(zip(run_keys, latest_run))
    return rows, latest_run


def _build_day_training_matrix(conn):
    """Joins reconciled day_predictions rows -- unlike earnings_
    predictions, there's no separate 'real outcome features' table to
    join against for a daily session, so the feature values captured AT
    PREDICTION TIME on the row itself (lean_pre_model/rsi14/macd_
    histogram/volume_vs_20d_avg_pct/trend_score/magnitude_estimate_pct)
    are what train_day_direction_model learns from. Returns (X, y,
    feature_names, row_meta), same shape as _build_training_matrix."""
    rows = conn.execute(
        """SELECT ticker, session_date, predicted_direction, actual_direction, lean_pre_model,
                  magnitude_estimate_pct, rsi14, macd_histogram, volume_vs_20d_avg_pct, trend_score
           FROM day_predictions WHERE prediction_correct_direction IS NOT NULL
           ORDER BY session_date""",
    ).fetchall()
    feature_names = [
        "lean_pre_model", "magnitude_estimate_pct", "rsi14", "macd_histogram",
        "volume_vs_20d_avg_pct", "trend_score",
    ]
    X, y, row_meta = [], [], []
    for (ticker, session_date, predicted_direction, actual_direction, lean, mag, rsi, macd_hist,
         vol_vs_avg, trend_score) in rows:
        if actual_direction not in ("UP", "DOWN"):
            continue
        X.append([lean or 0.0, mag or 0.0, rsi or 50.0, macd_hist or 0.0, vol_vs_avg or 0.0, trend_score or 0.0])
        y.append(1 if actual_direction == "UP" else 0)
        row_meta.append({"ticker": ticker, "session_date": session_date,
                          "predicted_direction": predicted_direction, "actual_direction": actual_direction})
    return X, y, feature_names, row_meta


def train_day_direction_model(conn, min_samples=TRAINED_MODEL_MIN_SAMPLES):
    """Day Prediction's Mode 3 -- structurally identical to train_earnings_
    direction_model (same min_samples/GBC threshold/retrain cadence
    constants, same chronological 80/20 holdout, same 'must beat the Mode
    2 baseline's actual historical calls or stay None' gate), pointed at
    day_predictions/day_model_cache instead. Will realistically stay None
    for a long time -- day_predictions starts empty -- which is the
    correct, honest behavior, not a bug: Mode 1/2 carry this feature until
    real reconciled history accumulates, exactly like the Earnings
    Simulator's own bootstrapping."""
    X, y, feature_names, row_meta = _build_day_training_matrix(conn)
    n = len(X)
    if n < min_samples:
        return None

    last = conn.execute(
        "SELECT n_samples_at_train FROM day_model_cache ORDER BY trained_at DESC LIMIT 1"
    ).fetchone()
    last_n = last[0] if last else 0
    if last_n and (n - last_n) < TRAINED_MODEL_RETRAIN_EVERY:
        return _read_cached_day_trained_model(conn)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        return None

    split = max(1, int(n * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    meta_test = row_meta[split:]
    if not X_test or len(set(y_train)) < 2:
        return None

    model_type = "gradient_boosting" if n >= TRAINED_MODEL_GBC_THRESHOLD else "logistic_regression"
    model = (
        GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
        if model_type == "gradient_boosting" else LogisticRegression(max_iter=1000)
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    held_out_accuracy = sum(1 for p, actual in zip(preds, y_test) if p == actual) / len(y_test)

    baseline_correct = sum(
        1 for m in meta_test
        if (m["predicted_direction"] or "").upper() == (m["actual_direction"] or "").upper()
    )
    baseline_accuracy = baseline_correct / len(meta_test)
    beats_baseline = held_out_accuracy > baseline_accuracy
    model_blob = pickle.dumps(model) if beats_baseline else None

    conn.execute(
        """INSERT INTO day_model_cache
               (trained_at, n_samples_at_train, model_type, held_out_accuracy, baseline_accuracy,
                beats_baseline, feature_names_json, model_blob)
           VALUES (?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(), n, model_type, held_out_accuracy, baseline_accuracy,
         int(beats_baseline), json.dumps(feature_names), model_blob),
    )
    conn.commit()

    if not beats_baseline:
        print(f"[day_model] Trained {model_type} on {n} samples ({held_out_accuracy:.0%} held-out) did NOT beat "
              f"the Mode 2 baseline ({baseline_accuracy:.0%}) -- keeping Mode 2 active.")
        return None
    return {
        "model": model, "model_type": model_type, "n_samples": n, "held_out_accuracy": held_out_accuracy,
        "bayesian_baseline_accuracy": baseline_accuracy, "feature_names": feature_names,
    }


def _read_cached_day_trained_model(conn):
    row = conn.execute(
        """SELECT model_type, n_samples_at_train, held_out_accuracy, baseline_accuracy,
                  feature_names_json, model_blob
           FROM day_model_cache WHERE beats_baseline=1 ORDER BY trained_at DESC LIMIT 1""",
    ).fetchone()
    if not row or not row[5]:
        return None
    model_type, n_samples, held_out_acc, baseline_acc, feature_names_json, model_blob = row
    return {
        "model": pickle.loads(model_blob), "model_type": model_type, "n_samples": n_samples,
        "held_out_accuracy": held_out_acc, "bayesian_baseline_accuracy": baseline_acc,
        "feature_names": json.loads(feature_names_json),
    }


def ensure_ticker_data_ready(ticker, conn, daily_max_age_hours=24, intraday_max_age_hours=24):
    """Synchronous, blocking prerequisite (Part 2 of the data-backfill
    fix) -- called BEFORE compute_day_target for any ticker, every time,
    from the UI layer (render_day_prediction_panel, and any future Day
    Prediction entry point). A brand-new ticker must never show
    "insufficient data" as a dead end when the fix is one real fetch
    away: this fetches whatever's missing/stale RIGHT NOW and returns
    once done -- a couple of real API calls, seconds, not a background
    job or a multi-day wait.

    Checks two things:
    1. 1-year daily OHLCV (price_history) -- the hard prerequisite;
       compute_day_target cannot run at all without it.
    2. Persisted intraday history (intraday_price_history, ~60 real
       days) -- an ENRICHMENT, not a hard blocker: compute_day_target/
       simulate_intraday_path already have two working fallbacks
       (today's live bars, then daily-vol scaling), so a ticker with no
       intraday history yet still gets a real prediction. This fetches
       it now purely so it starts accumulating from day one, per Part
       4, rather than never.

    Returns {"ready": bool, "fetched_now": [...], "still_missing": [...]}
    -- "ready" is True iff daily history is usable. "still_missing" is
    populated ONLY when a real fetch was just attempted and genuinely
    came back empty (illiquid/newly-listed/no free-source coverage) --
    never for 'we simply haven't tried yet,' which is exactly the gap
    this function closes."""
    fetched_now, still_missing = [], []

    daily_hist = _read_price_history(conn, ticker, days_back=380)
    need_daily = (
        daily_hist.empty or len(daily_hist) < 30
        or should_refetch(conn, "price_history", ticker, daily_max_age_hours)
    )
    if need_daily:
        try:
            fetch_price_history_delta(conn, ticker, full_period="1y", days_back=400, max_age_hours=0)
            fetched_now.append("1-year daily price history")
        except Exception:
            pass
        daily_hist = _read_price_history(conn, ticker, days_back=380)
    daily_ready = not daily_hist.empty and len(daily_hist) >= 30
    if not daily_ready:
        still_missing.append(
            f"1-year daily price history for {ticker} -- no usable data returned by the free source "
            f"(yfinance) just now; this ticker may be illiquid, too newly listed, or delisted."
        )

    intraday_count_before = conn.execute(
        "SELECT COUNT(*) FROM intraday_price_history WHERE ticker=?", (ticker,)
    ).fetchone()[0]
    need_intraday = should_refetch(conn, "intraday_price_history", ticker, intraday_max_age_hours)
    if need_intraday:
        try:
            n_new, _desc = fetch_intraday_history_delta(conn, ticker)
            if n_new:
                fetched_now.append(f"intraday history ({n_new} new 5-min bars)")
        except Exception:
            pass
    intraday_count_after = conn.execute(
        "SELECT COUNT(*) FROM intraday_price_history WHERE ticker=?", (ticker,)
    ).fetchone()[0]
    if need_intraday and intraday_count_before == 0 and intraday_count_after == 0:
        # Not a hard blocker (see docstring) -- surfaced so the UI can
        # explain why the volatility estimate is using daily-vol scaling
        # instead of real intraday patterns.
        still_missing.append(
            f"Intraday (5-min) history for {ticker} -- no bars returned by the free source just now; "
            f"volatility estimates will use daily-vol scaling until this resolves."
        )

    # Historical backtest bootstrap: once daily history is usable, run it
    # ONCE, ever, per ticker -- not on every call, since it's real work
    # (a 2y fetch + a lookback_days scan). This is what replaces "wait a
    # week for live RUNNERS days to accumulate" with an immediate,
    # sizeable reconciled sample for a brand-new ticker. Re-running later
    # to roll the window forward is Part 6's explicit manual control, not
    # something this synchronous prerequisite repeats automatically.
    if daily_ready:
        already_backtested = conn.execute(
            "SELECT 1 FROM backtest_runs WHERE ticker=? LIMIT 1", (ticker,)
        ).fetchone()
        if not already_backtested:
            try:
                bt_result = backtest_day_predictions(ticker, conn)
                if bt_result.get("ok"):
                    fetched_now.append(
                        f"historical backtest ({bt_result['sessions_logged']} qualifying session(s) logged "
                        f"from the last {bt_result['lookback_days']} trading days)"
                    )
            except Exception:
                pass

    return {"ready": daily_ready, "fetched_now": fetched_now, "still_missing": still_missing}


def compute_day_target(ticker, conn, buffer_minutes=DAY_PREDICTION_BUFFER_MINUTES_DEFAULT):
    """The Day Prediction target (Part 2): {"ready": False, "wait_message":
    ...} before the buffer window passes or if real session data isn't
    available yet; otherwise a full target dict with model_start_price,
    target_price, predicted_direction, magnitude_estimate_pct, mode,
    confidence_level -- honesty-tiered exactly like estimate_earnings_
    probability (same <10/15-sample thresholds, same raw_pattern/
    bayesian/trained_model names), just evaluated against day_predictions
    instead of earnings_predictions."""
    now_et = pd.Timestamp.now(tz="America/New_York")
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    model_start_time = get_session_prediction_window(market_open, buffer_minutes)

    if now_et < model_start_time:
        return {
            "ticker": ticker, "ready": False, "model_start_time": model_start_time.isoformat(),
            "market_open": market_open.isoformat(), "market_close": market_close.isoformat(),
            "reason": "waiting_for_buffer",
            "wait_message": f"Waiting for session data — prediction starts at "
                             f"{model_start_time.strftime('%-I:%M %p')} ET",
        }

    intraday_hist = fetch_intraday_bars(ticker, interval="5m")
    vol = _compute_realized_daily_vol(conn, ticker)
    # Two DISTINCT honest failure reasons (Part 3) -- by the time this
    # runs, ensure_ticker_data_ready has already made a real, synchronous
    # fetch attempt, so a missing `vol` here means genuine data
    # unavailability (case 1: illiquid/newly-listed/no free-source
    # coverage), not "we haven't tried yet." Empty `intraday_hist` is a
    # separate, narrower condition -- no live quote for TODAY specifically
    # (e.g. checked in the first instant after the buffer window, or a
    # transient yfinance hiccup), distinct from a genuine multi-year data
    # gap.
    if vol is None:
        return {
            "ticker": ticker, "ready": False, "model_start_time": model_start_time.isoformat(),
            "market_open": market_open.isoformat(), "market_close": market_close.isoformat(),
            "reason": "no_price_history",
            "wait_message": f"No reliable 1-year daily price history available for {ticker} via free sources -- "
                             f"this ticker may be too illiquid, newly listed, or delisted for a volatility "
                             f"estimate (not a 'haven't fetched it yet' issue -- a fetch was just attempted).",
        }
    if intraday_hist.empty:
        return {
            "ticker": ticker, "ready": False, "model_start_time": model_start_time.isoformat(),
            "market_open": market_open.isoformat(), "market_close": market_close.isoformat(),
            "reason": "no_live_session_quote",
            "wait_message": f"No live session quote for {ticker} right now -- retry shortly; the market may be "
                             f"between bars or the live feed may be briefly unavailable.",
        }

    technicals = _fetch_technical_snapshot(ticker)
    technical_levels = detect_technical_levels(ticker, DEFAULT_INTERVAL, DEFAULT_LOOKBACK, conn=conn)
    briefing_inputs = extract_simulator_inputs_from_briefing(ticker, conn)
    prior_close = _prior_session_close(ticker)

    current_price = float(intraday_hist["Close"].iloc[-1])
    lean, lean_features = _day_prediction_directional_lean(
        prior_close, intraday_hist, technical_levels, technicals, briefing_inputs
    )
    lean_pre_model = lean if lean is not None else 0.0

    pooled_reconciled_count = conn.execute(
        "SELECT COUNT(*) FROM day_predictions WHERE prediction_correct_direction IS NOT NULL"
    ).fetchone()[0]

    lean_final, model_meta = lean_pre_model, None
    if pooled_reconciled_count < 10:
        mode = "raw_pattern"
        confidence_level = None
    else:
        trained = train_day_direction_model(conn)
        if trained is not None:
            mode = "trained_model"
            feature_row = [
                lean_pre_model, vol["one_day_expected_move_pct"], (technicals or {}).get("rsi14") or 50.0,
                (technicals or {}).get("macd_histogram") or 0.0,
                (technicals or {}).get("volume_vs_20d_avg_pct") or 0.0, lean_features.get("trend_score") or 0.0,
            ]
            proba = trained["model"].predict_proba([feature_row])[0]
            classes = list(trained["model"].classes_)
            prob_up = float(proba[classes.index(1)]) if 1 in classes else 0.5
            lean_final = round(max(-1.0, min(1.0, (prob_up - 0.5) * 2)), 4)
            model_meta = {"model_type": trained["model_type"], "held_out_accuracy": trained["held_out_accuracy"],
                          "baseline_accuracy": trained["bayesian_baseline_accuracy"], "n_samples": trained["n_samples"]}
            confidence_level = "High" if abs(prob_up - 0.5) >= 0.20 else "Medium" if abs(prob_up - 0.5) >= 0.08 else "Low"
        else:
            mode = "bayesian"
            confidence_level = "High" if abs(lean_pre_model) >= 0.6 else "Medium" if abs(lean_pre_model) >= 0.25 else "Low"

    # Part 4: apply the calibration loop's own persisted, evidence-based
    # scaling factors (default 1.0/1.0, i.e. no-op, until enough reconciled
    # sessions exist to justify a real adjustment -- see
    # calibrate_day_prediction_model).
    calibration = get_active_day_prediction_calibration(conn)
    expected_move_pct = vol["one_day_expected_move_pct"] * calibration["vol_scale"]
    lean_calibrated = round(max(-1.0, min(1.0, lean_final * calibration["drift_scale"])), 4)
    target_price = round(current_price * (1 + lean_calibrated * expected_move_pct / 100), 2)
    predicted_direction = "UP" if target_price > current_price else "DOWN" if target_price < current_price else "FLAT"

    backtest = get_day_prediction_track_record(conn, ticker=ticker)

    return {
        "ticker": ticker, "ready": True, "session_date": now_et.date().isoformat(),
        "model_start_time": model_start_time.isoformat(), "model_start_price": current_price,
        "current_price": current_price, "prior_close": prior_close, "target_price": target_price,
        "predicted_direction": predicted_direction, "lean": lean_calibrated, "lean_pre_model": lean_pre_model,
        "magnitude_estimate_pct": expected_move_pct, "realized_vol": vol,
        "rsi14": (technicals or {}).get("rsi14"), "macd_histogram": (technicals or {}).get("macd_histogram"),
        "volume_vs_20d_avg_pct": (technicals or {}).get("volume_vs_20d_avg_pct"),
        "trend_score": lean_features.get("trend_score"), "trend_structure": (technical_levels or {}).get("trend_structure"),
        "mode": mode, "confidence_level": confidence_level, "model_meta": model_meta,
        "backtest_error_pct": backtest.get("mean_abs_error_pct"), "backtest_n": backtest.get("n"),
        "market_open": market_open.isoformat(), "market_close": market_close.isoformat(),
        "intraday_bars": intraday_hist, "calibration": calibration,
    }


def log_day_prediction(conn, ticker, prediction_result):
    """Commits ONE prediction row per (ticker, session_date) -- 'wait,
    then commit to a target, then track it' (Part 1 of the original
    build). Checks for an existing row FIRST (rather than relying only on
    INSERT OR IGNORE) specifically so simulate_intraday_path's random walk
    is only ever generated once per session -- generating it unconditionally
    on every call and letting INSERT OR IGNORE discard all but the first
    would still be correct, just wasteful, since this function runs on
    every refresh. Later same-day refreshes are a no-op: the target (and
    its cached simulated path) never move out from under an in-progress
    comparison table."""
    if not prediction_result.get("ready"):
        return
    # Checks simulated_path_json SPECIFICALLY, not just row existence --
    # a row can exist without a path (e.g. one committed before the
    # simulate_intraday_path caching feature existed, or a prior call
    # whose path generation failed). The old "SELECT 1 ... if existing:
    # return" treated any row as fully done and never revisited it,
    # which was the confirmed root cause of "no simulated path cached"
    # persisting forever for a ticker whose row predated this feature.
    existing = conn.execute(
        "SELECT simulated_path_json, source FROM day_predictions WHERE ticker=? AND session_date=?",
        (ticker, prediction_result["session_date"]),
    ).fetchone()
    # Self-heal: backtest_day_predictions now refuses to ever backtest
    # today's date (see its own docstring), but a DB from before that fix
    # can still hold a stale source='backtest' row squatting on today's
    # (ticker, session_date) slot -- its model_start_time is a naive
    # historical-bar timestamp, structurally incompatible with the
    # tz-aware live simulated path that would otherwise just get patched
    # onto it (confirmed live: this is exactly what produced "Cannot
    # compare tz-naive and tz-aware timestamps" for LULU). Rather than
    # patching only the path and leaving the naive model_start_time
    # behind, this promotes the WHOLE row to a proper live row (full
    # UPDATE, not INSERT OR IGNORE -- the row already exists, so an
    # insert would just be silently ignored by the UNIQUE constraint).
    promote_from_backtest = bool(existing) and existing[1] == "backtest"
    if promote_from_backtest:
        pass
    elif existing and existing[0]:
        return

    model_start_time = pd.Timestamp(prediction_result["model_start_time"])
    market_close = pd.Timestamp(prediction_result["market_close"])
    minutes_remaining = max(5, (market_close - model_start_time).total_seconds() / 60)
    daily_vol_fraction = (prediction_result.get("realized_vol") or {}).get("one_day_expected_move_pct")
    daily_vol_fraction = (daily_vol_fraction / 100) if daily_vol_fraction is not None else 0.01
    # Real persisted intraday history's own realized vol (Part 4) -- the
    # middle tier between today's live bars and the daily-vol fallback;
    # None until enough real stored trading days accumulate.
    stored_intraday = _compute_intraday_realized_vol(conn, ticker, step_minutes=5)
    simulated_path = simulate_intraday_path(
        prediction_result["model_start_price"], prediction_result["target_price"], minutes_remaining,
        daily_vol_fraction, step_minutes=5, start_time=model_start_time,
        intraday_hist=prediction_result.get("intraday_bars"),
        stored_intraday_step_vol=(stored_intraday or {}).get("step_vol"),
    )
    if not simulated_path:
        # No silent straight-line degradation (Part 3): if the simulator
        # genuinely produced nothing, that's a real bug to surface, not
        # paper over with a fallback that implies false precision.
        raise RuntimeError(
            f"simulate_intraday_path produced no points for {ticker} "
            f"({prediction_result['session_date']}) -- refusing to commit a pathless prediction."
        )
    path_json = json.dumps([[t.isoformat(), p] for t, p in simulated_path])

    if promote_from_backtest:
        # Full overwrite, not just the path -- model_start_time/price,
        # target, mode, etc. all need to become the real live values;
        # leaving the backtest row's naive model_start_time in place
        # while only patching the path is exactly what caused the bug.
        conn.execute(
            """UPDATE day_predictions SET
                   model_start_time=?, model_start_price=?, target_price=?, predicted_direction=?,
                   magnitude_estimate_pct=?, lean_pre_model=?, rsi14=?, macd_histogram=?,
                   volume_vs_20d_avg_pct=?, trend_score=?, mode=?, confidence_level=?,
                   backtest_error_pct=?, simulated_path_json=?, source='live', predicted_at=?,
                   actual_close_price=NULL, actual_direction=NULL, prediction_correct_direction=NULL,
                   error_pct=NULL, reconciliation_notes=NULL, reconciled_at=NULL, path_mean_abs_error_pct=NULL
               WHERE ticker=? AND session_date=?""",
            (prediction_result["model_start_time"], prediction_result["model_start_price"],
             prediction_result["target_price"], prediction_result["predicted_direction"],
             prediction_result["magnitude_estimate_pct"], prediction_result["lean_pre_model"],
             prediction_result.get("rsi14"), prediction_result.get("macd_histogram"),
             prediction_result.get("volume_vs_20d_avg_pct"), prediction_result.get("trend_score"),
             prediction_result["mode"], prediction_result["confidence_level"],
             prediction_result.get("backtest_error_pct"), path_json, datetime.utcnow().isoformat(),
             ticker, prediction_result["session_date"]),
        )
        conn.commit()
        return

    if existing:
        # Pre-existing pathless LIVE row -- backfill just the path in
        # place instead of leaving it stuck forever.
        conn.execute(
            "UPDATE day_predictions SET simulated_path_json=? WHERE ticker=? AND session_date=?",
            (path_json, ticker, prediction_result["session_date"]),
        )
        conn.commit()
        return

    conn.execute(
        """INSERT OR IGNORE INTO day_predictions
               (ticker, session_date, model_start_time, model_start_price, target_price, predicted_direction,
                magnitude_estimate_pct, lean_pre_model, rsi14, macd_histogram, volume_vs_20d_avg_pct,
                trend_score, mode, confidence_level, backtest_error_pct, simulated_path_json, source, predicted_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ticker, prediction_result["session_date"], prediction_result["model_start_time"],
         prediction_result["model_start_price"], prediction_result["target_price"],
         prediction_result["predicted_direction"], prediction_result["magnitude_estimate_pct"],
         prediction_result["lean_pre_model"], prediction_result.get("rsi14"),
         prediction_result.get("macd_histogram"), prediction_result.get("volume_vs_20d_avg_pct"),
         prediction_result.get("trend_score"), prediction_result["mode"], prediction_result["confidence_level"],
         prediction_result.get("backtest_error_pct"), path_json, "live", datetime.utcnow().isoformat()),
    )
    conn.commit()


def get_committed_day_prediction(conn, ticker, session_date):
    """Pure DB read of the ONE committed target for (ticker, session_date)
    -- the dashboard renders against this (not a fresh compute_day_target
    call) once a prediction has been committed for today, so the target
    line/badge/simulated path never change mid-session even as compute_
    day_target's own live inputs (intraday bars, technicals) keep changing
    on every refresh. `simulated_path` is decoded back into (Timestamp,
    price) tuples -- the exact same list simulate_intraday_path returned
    at commit time, read by both the chart and the comparison table so
    they can never disagree (Part 3)."""
    row = conn.execute(
        """SELECT model_start_time, model_start_price, target_price, predicted_direction,
                  magnitude_estimate_pct, mode, confidence_level, backtest_error_pct, simulated_path_json
           FROM day_predictions WHERE ticker=? AND session_date=?""",
        (ticker, session_date),
    ).fetchone()
    if not row:
        return None
    keys = ["model_start_time", "model_start_price", "target_price", "predicted_direction",
            "magnitude_estimate_pct", "mode", "confidence_level", "backtest_error_pct", "simulated_path_json"]
    result = dict(zip(keys, row))
    path_json = result.pop("simulated_path_json")
    if path_json:
        result["simulated_path"] = [(pd.Timestamp(t), p) for t, p in json.loads(path_json)]
    else:
        result["simulated_path"] = []
    return result


def get_most_recent_day_prediction(conn, ticker):
    """The most recent LIVE (source='live'/NULL, i.e. a real same-day
    prediction, never a backtest reconstruction) day_predictions row for
    `ticker`, regardless of session_date -- same shape as get_committed_
    day_prediction (model_start_time + decoded simulated_path included)
    plus the outcome fields, so the UI can render the EXACT SAME rich
    chart + comparison table for "last session's results" that it does
    for today's live session, not a stripped-down summary. Lets that view
    stay visible even outside today's active buffer window (e.g. before
    10am ET, or before today's own prediction has committed yet) -- this
    and the Backtest Report are both independent of whether TODAY'S
    prediction is ready, and shouldn't be hidden just because today's own
    chart can't render yet. Returns None if this ticker has never had a
    live prediction."""
    row = conn.execute(
        """SELECT session_date, model_start_time, model_start_price, target_price, predicted_direction,
                  magnitude_estimate_pct, mode, confidence_level, backtest_error_pct, simulated_path_json,
                  actual_close_price, actual_direction, prediction_correct_direction, error_pct,
                  reconciliation_notes, reconciled_at
           FROM day_predictions WHERE ticker=? AND COALESCE(source,'live')='live'
           ORDER BY session_date DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    keys = ["session_date", "model_start_time", "model_start_price", "target_price", "predicted_direction",
            "magnitude_estimate_pct", "mode", "confidence_level", "backtest_error_pct", "simulated_path_json",
            "actual_close_price", "actual_direction", "prediction_correct_direction", "error_pct",
            "reconciliation_notes", "reconciled_at"]
    result = dict(zip(keys, row))
    path_json = result.pop("simulated_path_json")
    result["simulated_path"] = [(pd.Timestamp(t), p) for t, p in json.loads(path_json)] if path_json else []
    return result


def get_most_recent_backtest_prediction(conn, ticker):
    """Most recent BACKTESTED (source='backtest') day_predictions row for
    `ticker`, regardless of session_date -- a fallback reference when a
    ticker has never had a real live session yet (a brand-new pin, or one
    that just hasn't hit today's buffer window), so there's still
    something concrete on screen instead of nothing. Always shown
    labeled as backtest-based, never mistaken for live: backtest rows
    carry no simulated_path_json (no intraday path was ever simulated for
    them), so this is metrics-only, not chart-eligible."""
    row = conn.execute(
        """SELECT session_date, model_start_price, target_price, predicted_direction, mode,
                  actual_close_price, actual_direction, prediction_correct_direction, error_pct,
                  reconciliation_notes, reconciled_at
           FROM day_predictions WHERE ticker=? AND source='backtest'
           ORDER BY session_date DESC LIMIT 1""",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    keys = ["session_date", "model_start_price", "target_price", "predicted_direction", "mode",
            "actual_close_price", "actual_direction", "prediction_correct_direction", "error_pct",
            "reconciliation_notes", "reconciled_at"]
    return dict(zip(keys, row))


def get_intraday_bars_for_session(conn, ticker, session_date):
    """Persisted 5-min bars (intraday_price_history, ~60-day real window)
    filtered to ONE calendar session -- lets the "last session" chart
    show real candles for a closed day, not just today's live feed
    (fetch_intraday_bars only ever returns TODAY's bars). Empty DataFrame
    if that session falls outside the ~60-day retention window or was
    never persisted (e.g. the app wasn't running that day)."""
    hist = _read_intraday_history(conn, ticker, days_back=60)
    if hist.empty:
        return hist
    target_date = pd.Timestamp(session_date).date()
    return hist[hist.index.date == target_date]


def should_log_day_prediction_snapshot(conn, ticker, session_date, now_et=None):
    """The hard gate a bug report confirmed was completely missing:
    real market hours (9:30-4:00 ET) on a real trading weekday, PLUS
    exactly one additional snapshot allowed right at/after close to
    serve as that session's final closing print. Once that closing
    snapshot exists, every later call (an after-hours page reload, a
    stale auto-refresh fragment still firing) is a no-op -- this is
    what stops the repeated identical-price rows logged hours after the
    session ended (confirmed live: MU/WDC both had 6 identical rows
    between 6:17 PM and 7:39 PM ET)."""
    now_et = now_et or pd.Timestamp.now(tz="America/New_York")
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if market_open <= now_et <= market_close:
        return True
    if now_et > market_close:
        close_cutoff_utc = market_close.tz_convert("UTC").tz_localize(None).isoformat()
        already_closed = conn.execute(
            """SELECT 1 FROM day_prediction_snapshots
               WHERE ticker=? AND session_date=? AND snapshot_time>=?""",
            (ticker, session_date, close_cutoff_utc),
        ).fetchone()
        return already_closed is None
    return False


def log_day_prediction_snapshot(conn, ticker, session_date, actual_price):
    """One row per refresh past model-start (Part 4) -- DB-persisted (not
    st.session_state) so the live comparison table survives a page
    reload within the same session_date. Gated by should_log_day_
    prediction_snapshot (Part 2): silently returns outside market hours
    once the session's closing print is already logged. The moment that
    closing snapshot is actually written, this also triggers same-day
    reconciliation immediately -- not the separate next-calendar-day
    auto-hook, which is what let a session sit unreconciled for up to a
    full extra day."""
    now_et = pd.Timestamp.now(tz="America/New_York")
    if not should_log_day_prediction_snapshot(conn, ticker, session_date, now_et):
        return
    conn.execute(
        """INSERT OR IGNORE INTO day_prediction_snapshots (ticker, session_date, snapshot_time, actual_price)
           VALUES (?,?,?,?)""",
        (ticker, session_date, datetime.utcnow().isoformat(), actual_price),
    )
    conn.commit()
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if now_et > market_close:
        reconcile_day_predictions(conn)
        calibrate_day_prediction_model(conn)


def get_day_prediction_snapshots(conn, ticker, session_date):
    rows = conn.execute(
        """SELECT snapshot_time, actual_price FROM day_prediction_snapshots
           WHERE ticker=? AND session_date=? ORDER BY snapshot_time""",
        (ticker, session_date),
    ).fetchall()
    return [{"snapshot_time": r[0], "actual_price": r[1]} for r in rows]


def reconcile_day_predictions(conn):
    """Runs once per day, shortly after close or on next app load if the
    app wasn't running then (mirrors reconcile_earnings_predictions'
    pattern) -- grades every day_predictions row whose session is over
    (session_date <= yesterday, so a still-live today is never graded
    early) and not yet reconciled, against the real close for that date
    in price_history (populated by the normal fetch_price_history_delta
    path once the next session's fetch runs). Rows with no backfilled
    close yet are left alone -- tried again on the next run, not treated
    as a failure.

    Part 5's addition: reconciles the INTRADAY PATH itself, not just the
    final target -- mean |error%| between the cached simulated path
    (simulated_path_json) and the real logged snapshots
    (day_prediction_snapshots) for that session, via the same
    _predicted_price_at_time both the chart and live table use. This is
    what lets the system eventually learn whether the path SHAPE was
    realistic, not just whether the final direction call was right.
    Also writes a plain-language note that names the approximate time of
    the single largest real reversal in the predicted-vs-actual gap, when
    one is genuinely present in the logged snapshots -- never a fabricated
    narrative when snapshots are too sparse to support one.

    Part 2.3 fix: the cutoff used to be SQLite's `date('now', '-1 day')`,
    which is UTC-dated (not ET-aware) and always excluded today's own
    session regardless of wall-clock time -- so a session could never
    reconcile until the calendar date rolled over, even hours after the
    4pm ET close already happened. Now computed from real ET wall-clock
    time: today's session becomes eligible the moment close has passed,
    which is also the exact trigger point log_day_prediction_snapshot
    calls this from right after logging the final closing print."""
    now_et = pd.Timestamp.now(tz="America/New_York")
    market_close_today = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    cutoff_date = now_et.date() if now_et >= market_close_today else (now_et - timedelta(days=1)).date()
    rows = conn.execute(
        """SELECT id, ticker, session_date, model_start_price, target_price, predicted_direction,
                  simulated_path_json
           FROM day_predictions WHERE reconciled_at IS NULL AND session_date <= ?""",
        (cutoff_date.isoformat(),),
    ).fetchall()
    reconciled_count = 0
    for (pred_id, ticker, session_date, model_start_price, target_price, predicted_direction,
         path_json) in rows:
        close_row = conn.execute(
            "SELECT close FROM price_history WHERE ticker=? AND date=?", (ticker, session_date)
        ).fetchone()
        if not close_row or close_row[0] is None:
            continue
        actual_close = close_row[0]
        actual_direction = (
            "UP" if actual_close > model_start_price else "DOWN" if actual_close < model_start_price else "FLAT"
        )
        prediction_correct = int(predicted_direction == actual_direction) if predicted_direction else None
        error_pct = round((actual_close - target_price) / target_price * 100, 2) if target_price else None
        actual_move_pct = (actual_close / model_start_price - 1) * 100 if model_start_price else None
        target_move_pct = (target_price / model_start_price - 1) * 100 if model_start_price else None

        path_mean_abs_error_pct = None
        reversal_note = ""
        snaps = conn.execute(
            """SELECT snapshot_time, actual_price FROM day_prediction_snapshots
               WHERE ticker=? AND session_date=? ORDER BY snapshot_time""",
            (ticker, session_date),
        ).fetchall()
        if snaps and path_json:
            path = [(pd.Timestamp(t), p) for t, p in json.loads(path_json)]
            path_tz = path[0][0].tz if path else None
            signed_errs = []
            for snap_time, actual_price in snaps:
                if actual_price is None:
                    continue
                t = pd.Timestamp(snap_time)
                if t.tzinfo is None and path_tz is not None:
                    t = t.tz_localize("UTC").tz_convert(path_tz)
                predicted_price = _predicted_price_at_time(path, t)
                if predicted_price:
                    signed_errs.append((t, (actual_price - predicted_price) / predicted_price * 100))
            if signed_errs:
                path_mean_abs_error_pct = round(sum(abs(e) for _, e in signed_errs) / len(signed_errs), 2)
                # A real, derivable "something changed here" signal --
                # the single largest sign flip in the predicted-vs-actual
                # gap, named by its own logged time -- not an invented
                # narrative. Only reported if one genuinely occurred.
                best_flip, best_swing = None, 1.0
                for (t0, e0), (t1, e1) in zip(signed_errs, signed_errs[1:]):
                    if e0 * e1 < 0 and abs(e1 - e0) > best_swing:
                        best_flip, best_swing = (t1, e1), abs(e1 - e0)
                if best_flip:
                    reversal_note = (
                        f" The predicted-vs-actual gap flipped direction around "
                        f"{best_flip[0].strftime('%-I:%M %p')} ET (logged snapshot), consistent with a real "
                        f"intraday move the model's morning-only inputs couldn't have captured."
                    )

        notes = (
            f"Predicted {target_move_pct:+.2f}% to ${target_price:.2f}; actual close ${actual_close:.2f} "
            f"({actual_move_pct:+.2f}% vs. predicted {predicted_direction or 'n/a'}) -- "
            + ("the directional call was correct." if prediction_correct else "the directional call was wrong.")
            + (f" Path tracking: mean |error| of {path_mean_abs_error_pct:.2f}% across {len(snaps)} logged "
               f"snapshot(s)." if path_mean_abs_error_pct is not None else " No logged snapshots to grade the "
               f"path shape against for this session.")
            + reversal_note
        )
        conn.execute(
            """UPDATE day_predictions SET actual_close_price=?, actual_direction=?,
                   prediction_correct_direction=?, error_pct=?, path_mean_abs_error_pct=?,
                   reconciliation_notes=?, reconciled_at=?
               WHERE id=?""",
            (actual_close, actual_direction, prediction_correct, error_pct, path_mean_abs_error_pct, notes,
             datetime.utcnow().isoformat(), pred_id),
        )
        reconciled_count += 1
    conn.commit()
    return {"reconciled_count": reconciled_count, "checked_count": len(rows)}


def get_day_prediction_track_record(conn, ticker=None):
    """Mirrors get_simulator_track_record -- accuracy + mean |error%|
    (final target) + mean path |error%| (intraday shape, Part 5) across
    reconciled day_predictions rows, scoped to one ticker (used by
    compute_day_target to surface THIS ticker's own backtested error on
    the chart) or system-wide when ticker is omitted."""
    query = ("SELECT ticker, error_pct, prediction_correct_direction, path_mean_abs_error_pct "
             "FROM day_predictions WHERE reconciled_at IS NOT NULL")
    params = ()
    if ticker:
        query += " AND ticker=?"
        params = (ticker,)
    rows = conn.execute(query, params).fetchall()
    if not rows:
        return {"n": 0, "accuracy": None, "mean_abs_error_pct": None, "mean_path_abs_error_pct": None}
    n = len(rows)
    correct = sum(1 for r in rows if r[2])
    errors = [abs(r[1]) for r in rows if r[1] is not None]
    path_errors = [r[3] for r in rows if r[3] is not None]
    return {
        "n": n, "accuracy": round(correct / n, 3),
        "mean_abs_error_pct": round(sum(errors) / len(errors), 2) if errors else None,
        "mean_path_abs_error_pct": round(sum(path_errors) / len(path_errors), 2) if path_errors else None,
    }


# Below this sample floor, calibrate_day_prediction_model logs an honest
# "not enough data yet" entry and makes NO parameter adjustment -- a
# handful of reconciled sessions is not a real signal, and adjusting on
# noise would just add a second source of error on top of the model's own.
DAY_CALIBRATION_MIN_SAMPLES = 8
DAY_CALIBRATION_DRIFT_BOUNDS = (0.5, 1.5)
DAY_CALIBRATION_VOL_BOUNDS = (0.6, 1.8)


def get_active_day_prediction_calibration(conn):
    """The currently-active drift/vol scaling factors -- just the most
    recent model_calibration_log row, defaulting to a neutral 1.0/1.0
    (no adjustment) until at least one real calibration pass has run."""
    row = conn.execute(
        "SELECT drift_scale, vol_scale FROM model_calibration_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return {"drift_scale": row[0], "vol_scale": row[1]} if row else {"drift_scale": 1.0, "vol_scale": 1.0}


def get_day_prediction_calibration_history(conn, limit=20):
    """Most-recent-first log of every calibration pass, for the SETTINGS
    UI -- shows whether/how the model's assumptions have actually shifted
    as more sessions reconcile, not just the current snapshot."""
    rows = conn.execute(
        """SELECT calibrated_at, n_samples, mean_abs_error_pct, target_error_pct, drift_scale, vol_scale,
                  magnitude_ratio, direction_accuracy, note
           FROM model_calibration_log ORDER BY id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    keys = ["calibrated_at", "n_samples", "mean_abs_error_pct", "target_error_pct", "drift_scale", "vol_scale",
            "magnitude_ratio", "direction_accuracy", "note"]
    return [dict(zip(keys, r)) for r in rows]


def _log_day_calibration_run(conn, result):
    conn.execute(
        """INSERT INTO model_calibration_log
               (n_samples, mean_abs_error_pct, target_error_pct, drift_scale, vol_scale, magnitude_ratio,
                direction_accuracy, note, calibrated_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (result["n"], result.get("mean_abs_error_pct"), result["target_error_pct"], result["drift_scale"],
         result["vol_scale"], result.get("magnitude_ratio"), result.get("direction_accuracy"), result["note"],
         datetime.utcnow().isoformat()),
    )
    conn.commit()


def calibrate_day_prediction_model(conn, target_error_pct=3.0):
    """Uses the reconciled day_predictions history (error_pct per past
    session, from reconcile_day_predictions()) to tune the model's
    drift/volatility assumptions toward a 0-3% end-of-day error target:

    1. Computes mean |error%| system-wide (per-ticker breakdown is
       available via get_day_prediction_track_record(conn, ticker=...)
       for the same reconciled rows).
    2. Detects two independent, real biases and nudges a persisted
       scaling factor for each, in small bounded steps (never a single-
       session overreaction, never amplified beyond the fixed bounds
       above):
         - magnitude bias: mean |actual move %| vs mean |predicted move
           %| across reconciled sessions. Actual systematically larger
           => the model under-predicts magnitude => vol_scale nudged up.
           Systematically smaller => nudged down.
         - direction bias: directional accuracy vs. the 50% chance
           baseline. Only ever nudges drift_scale DOWN (trusts the lean
           less) when accuracy is reliably worse than chance with enough
           samples to mean something -- deliberately never nudges it up
           automatically, since amplifying a confident wrong call is
           worse than a timid right one.
    3. Every run -- adjusted or not -- is appended to model_calibration_
       log so the trend is visible over time, not just the latest state.
    4. Returns a result dict the UI renders directly, always naming the
       real N and real mean error against the stated target -- never
       implying more confidence than the sample size supports.

    Called automatically right after reconcile_day_predictions() inside
    log_day_prediction_snapshot's same-day close trigger (Part 2.3), and
    from SETTINGS' manual reconciliation button."""
    rows = conn.execute(
        """SELECT model_start_price, target_price, actual_close_price, prediction_correct_direction, error_pct
           FROM day_predictions WHERE reconciled_at IS NOT NULL AND actual_close_price IS NOT NULL"""
    ).fetchall()
    n = len(rows)
    result = {
        "n": n, "target_error_pct": target_error_pct, "mean_abs_error_pct": None, "adjusted": False,
        "drift_scale": 1.0, "vol_scale": 1.0, "magnitude_ratio": None, "direction_accuracy": None,
    }
    composition = get_day_prediction_sample_composition(conn)
    comp_txt = f"{composition['backtest_n']} backtested + {composition['live_n']} live"
    if n < DAY_CALIBRATION_MIN_SAMPLES:
        prev = get_active_day_prediction_calibration(conn)
        result["drift_scale"], result["vol_scale"] = prev["drift_scale"], prev["vol_scale"]
        result["note"] = (
            f"Only {n} reconciled session(s) system-wide ({comp_txt}) -- below the "
            f"{DAY_CALIBRATION_MIN_SAMPLES}-sample floor for a real calibration adjustment. No parameters "
            f"changed; carrying forward the existing drift_scale={prev['drift_scale']:.2f}/"
            f"vol_scale={prev['vol_scale']:.2f}."
        )
        _log_day_calibration_run(conn, result)
        return result

    errors = [abs(r[4]) for r in rows if r[4] is not None]
    mean_abs_error = sum(errors) / len(errors) if errors else None
    result["mean_abs_error_pct"] = round(mean_abs_error, 2) if mean_abs_error is not None else None

    actual_moves, target_moves = [], []
    for msp, tp, acp, _correct, _err in rows:
        if msp:
            actual_moves.append(abs((acp - msp) / msp * 100))
            target_moves.append(abs((tp - msp) / msp * 100))
    magnitude_ratio = (sum(actual_moves) / sum(target_moves)) if target_moves and sum(target_moves) else None
    result["magnitude_ratio"] = round(magnitude_ratio, 3) if magnitude_ratio is not None else None

    correct = sum(1 for r in rows if r[3])
    direction_accuracy = correct / n
    result["direction_accuracy"] = round(direction_accuracy, 3)

    prev = get_active_day_prediction_calibration(conn)
    drift_scale, vol_scale = prev["drift_scale"], prev["vol_scale"]
    adjusted = False
    if magnitude_ratio is not None:
        if magnitude_ratio > 1.15:
            vol_scale = min(DAY_CALIBRATION_VOL_BOUNDS[1], round(vol_scale * 1.05, 4))
            adjusted = True
        elif magnitude_ratio < 0.85:
            vol_scale = max(DAY_CALIBRATION_VOL_BOUNDS[0], round(vol_scale * 0.95, 4))
            adjusted = True
    if direction_accuracy < 0.45:
        drift_scale = max(DAY_CALIBRATION_DRIFT_BOUNDS[0], round(drift_scale * 0.90, 4))
        adjusted = True

    result["adjusted"], result["drift_scale"], result["vol_scale"] = adjusted, drift_scale, vol_scale
    result["composition"] = composition
    within_target = mean_abs_error is not None and mean_abs_error <= target_error_pct
    result["note"] = (
        f"{n} reconciled session(s) ({comp_txt}); mean |error| "
        f"{mean_abs_error:.2f}% ({'within' if within_target else 'above'} the {target_error_pct:.1f}% target)"
        + (f"; magnitude ratio (actual/predicted move size) {magnitude_ratio:.2f}x" if magnitude_ratio else "")
        + f"; directional accuracy {direction_accuracy:.0%}. "
        + ("Adjusted drift/vol scaling this run "
           f"(drift_scale={drift_scale:.2f}, vol_scale={vol_scale:.2f})." if adjusted else
           "No adjustment triggered this run.")
    )
    _log_day_calibration_run(conn, result)
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def full_refresh(tickers=None, db_path=DEFAULT_DB_PATH, progress_callback=None, force_refresh=False):
    """Refreshes every table for the given tickers, respecting each
    fetcher's cache window unless force_refresh=True bypasses it entirely.
    progress_callback(msg) is called with a short status string after each
    step, if provided."""
    tickers = tickers or load_watchlist()
    init_db(db_path)
    conn = get_connection(db_path)

    def _report(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    try:
        try:
            result = cached_congressional_trades(conn, max_age_hours=1, force_refresh=force_refresh)
            tag = "cached" if result["cache_hit"] else "fetched"
            _report(f"Congressional trades: {len(result['data'])} rows ({tag}, source={result['source']})")
        except Exception as e:
            _report(f"Congressional trades failed: {e}")

        try:
            n = fetch_insider_trades(conn)
            _report(f"Insider trades: {n} new")
        except Exception as e:
            _report(f"Insider trades failed: {e}")

        try:
            n = fetch_polymarket(conn)
            _report(f"Polymarket events: {n} synced")
        except Exception as e:
            _report(f"Polymarket failed: {e}")

        try:
            n = fetch_polymarket(conn, limit=60, keywords=MACRO_KEYWORDS)
            _report(f"Polymarket macro-relevant events: {n} synced")
        except Exception as e:
            _report(f"Polymarket macro fetch failed: {e}")

        for i, ticker in enumerate(tickers):
            _report(f"[{i + 1}/{len(tickers)}] {ticker}: options flow...")
            try:
                cached_options_flow(conn, ticker, max_age_hours=0.25, force_refresh=force_refresh)
            except Exception as e:
                _report(f"  {ticker} options_flow failed: {e}")

            try:
                cached_dark_pool(conn, ticker, max_age_hours=1, force_refresh=force_refresh)
            except Exception as e:
                _report(f"  {ticker} dark_pool failed: {e}")

            try:
                # compute_divergence's earnings-proximity term reads this
                # from the DB rather than fetching live, so it needs to be
                # populated here for every watchlist ticker, not just
                # on-demand from the deep-dive tab.
                cached_earnings_signal(conn, ticker, max_age_hours=12, force_refresh=force_refresh)
            except Exception as e:
                _report(f"  {ticker} earnings_signal failed: {e}")

            try:
                # Part 4 of the Day Prediction data-backfill fix: keeps
                # every watchlist ticker's daily + persisted intraday
                # history fresh via the normal refresh cycle, so RUNNERS'
                # Day Prediction feature never hits a cold-start ticker
                # that needs its OWN synchronous fetch later.
                dr = ensure_ticker_data_ready(ticker, conn)
                if dr["fetched_now"]:
                    _report(f"  {ticker} day-prediction data backfilled: {', '.join(dr['fetched_now'])}")
            except Exception as e:
                _report(f"  {ticker} day-prediction data backfill failed: {e}")

            time.sleep(0.3)  # rate limit yfinance

        for ticker in tickers:
            try:
                result = compute_divergence(ticker, conn)
                _report(f"{ticker}: {result['label']} ({result['score']})")
            except Exception as e:
                _report(f"  {ticker} divergence failed: {e}")

    finally:
        conn.close()


# --------------------------------------------------------------------------
# check_source_health — lightweight connectivity probe per configured
# source (not a full data pull). Each probe only checks the specific
# source named in DATA_SOURCE_CONFIG; if a config value is swapped to a
# source with no probe implemented yet, that entry reports
# status="not_configured" rather than silently probing the wrong thing.
# --------------------------------------------------------------------------

def _probe_yfinance():
    t0 = time.time()
    try:
        hist = yf.Ticker("SPY").history(period="1d")
        latency_ms = int((time.time() - t0) * 1000)
        if hist is not None and not hist.empty:
            return {"status": "up", "source": "yfinance", "latency_ms": latency_ms}
        return {"status": "degraded", "source": "yfinance", "latency_ms": latency_ms,
                "error": "empty response for SPY 1d history"}
    except Exception as e:
        return {"status": "down", "source": "yfinance", "error": str(e)}


def _probe_finra_proxy():
    """Checks the real FINRA ATS endpoint. Reports the dark_pool feature's
    effective status: 'up' with source=finra_ats if the real endpoint
    answers, else 'degraded' with source=finra_proxy since the volume
    z-score fallback still works as long as yfinance itself is reachable
    (this probe doesn't re-check that -- see the options_flow probe)."""
    t0 = time.time()
    try:
        resp = requests.post(
            FINRA_ATS_URL,
            json={
                "compareFilters": [
                    {"compareType": "EQUAL", "fieldName": "issueSymbolIdentifier", "fieldValue": "SPY"}
                ],
                "limit": 1,
            },
            headers={"Content-Type": "application/json"},
            timeout=8,
        )
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "finra_ats", "latency_ms": latency_ms}
        return {"status": "degraded", "source": "finra_proxy", "latency_ms": latency_ms,
                "error": f"FINRA ATS returned HTTP {resp.status_code}; falling back to volume z-score proxy"}
    except Exception as e:
        return {"status": "degraded", "source": "finra_proxy",
                "error": f"FINRA ATS unreachable ({e}); falling back to volume z-score proxy"}


def _probe_senate_efd():
    """The free congressional fallback. Flagged as known-unreliable even
    when 'up' -- efdsearch.senate.gov sits behind Akamai bot protection
    that returns a blanket 403 to non-browser traffic from many server/
    datacenter IPs (confirmed during development)."""
    note = ("known-unreliable: efdsearch.senate.gov sits behind bot protection "
            "that blocks many server/datacenter IPs")
    t0 = time.time()
    try:
        resp = requests.get(f"{SENATE_EFD_BASE}/search/", headers=_BROWSER_HEADERS, timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "senate_efd_free", "latency_ms": latency_ms, "note": note}
        return {"status": "down", "source": "senate_efd_free", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}", "note": note}
    except Exception as e:
        return {"status": "down", "source": "senate_efd_free", "error": str(e), "note": note}


def _extract_api_error_detail(resp):
    try:
        body = resp.json()
        if isinstance(body, dict):
            return body.get("detail") or body.get("message") or str(body)[:200]
        return str(body)[:200]
    except Exception:
        return (resp.text or "")[:200]


def _probe_congressional():
    """Hits the QuiverQuant API directly rather than going through
    quiverquant.quiver().congress_trading() -- that helper never calls
    raise_for_status(), so it silently returns a malformed 1-column
    DataFrame (e.g. {"detail": "Invalid token."}) on 401/403 instead of
    raising, which would make this probe falsely report "up" on a bad key.
    Going direct lets us surface the *actual* API error (auth failure,
    rate limit, ...) instead of a generic status."""
    fallback = _probe_senate_efd()
    api_key = os.environ.get("QUIVER_API_KEY")
    if not api_key:
        return {"status": "not_configured", "source": "quiverquant",
                "error": "QUIVER_API_KEY not set in environment", "fallback": fallback}

    t0 = time.time()
    try:
        resp = requests.get(
            "https://api.quiverquant.com/beta/historical/congresstrading/AAPL",
            headers={"accept": "application/json", "Authorization": f"Token {api_key}"},
            timeout=10,
        )
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "quiverquant", "latency_ms": latency_ms, "fallback": fallback}
        if resp.status_code in (401, 403):
            return {"status": "down", "source": "quiverquant", "latency_ms": latency_ms,
                    "error": f"authentication failed: {_extract_api_error_detail(resp)}", "fallback": fallback}
        if resp.status_code == 429:
            return {"status": "degraded", "source": "quiverquant", "latency_ms": latency_ms,
                    "error": f"rate limited (HTTP 429): {_extract_api_error_detail(resp)}", "fallback": fallback}
        return {"status": "down", "source": "quiverquant", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}: {_extract_api_error_detail(resp)}", "fallback": fallback}
    except Exception as e:
        return {"status": "down", "source": "quiverquant", "error": str(e), "fallback": fallback}


def _probe_sec_edgar():
    t0 = time.time()
    try:
        resp = requests.get(SEC_FULLTEXT_SEARCH_URL.format(ticker="AAPL"), headers=SEC_HEADERS, timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "sec_edgar_free", "latency_ms": latency_ms}
        return {"status": "down", "source": "sec_edgar_free", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "down", "source": "sec_edgar_free", "error": str(e)}


def _probe_news_feeds():
    """News is no longer a single DATA_SOURCE_CONFIG dispatch -- it's a
    user-editable registry (news_sources). Opens its own short-lived
    connection (every other probe is self-contained/arg-less, and
    check_source_health() isn't otherwise passed a conn) to report
    not_configured with zero enabled feeds, else tests the first enabled
    feed live (same test_news_feed() used by the SETTINGS 'Test Feed'
    button) as a representative sample -- a full per-feed probe would
    defeat the 'lightweight, not a full data pull' point of a health
    check as feed count grows."""
    conn = get_connection()
    try:
        feeds = _read_news_sources(conn, enabled_only=True)
    finally:
        conn.close()
    if not feeds:
        return {"status": "not_configured", "source": None,
                "error": "no enabled feeds in news_sources -- add one in SETTINGS → News Feeds"}
    primary = feeds[0]
    result = test_news_feed(primary["feed_url"])
    if result["ok"]:
        return {"status": "up", "source": primary["name"],
                "note": f"{len(feeds)} enabled feed(s); tested '{primary['name']}'"}
    return {"status": "down", "source": primary["name"], "error": result["error"],
            "note": f"{len(feeds)} enabled feed(s); tested '{primary['name']}'"}


def check_source_health():
    """Tests connectivity to every configured data source with a lightweight
    probe (not a full data pull). Returns a dict keyed by fetcher name, each
    with status ("up"/"down"/"degraded"/"not_configured"), source,
    latency_ms and/or error, plus a top-level "checked_at" timestamp."""
    results = {}

    if DATA_SOURCE_CONFIG.get("options_flow") == "yfinance":
        results["options_flow"] = _probe_yfinance()
    else:
        results["options_flow"] = {"status": "not_configured", "source": DATA_SOURCE_CONFIG.get("options_flow"),
                                    "error": "no health probe implemented for this source"}

    results["news"] = _probe_news_feeds()

    if DATA_SOURCE_CONFIG.get("dark_pool") == "finra_proxy":
        results["dark_pool"] = _probe_finra_proxy()
    else:
        results["dark_pool"] = {"status": "not_configured", "source": DATA_SOURCE_CONFIG.get("dark_pool"),
                                 "error": "no health probe implemented for this source"}

    results["congressional"] = _probe_congressional()

    if DATA_SOURCE_CONFIG.get("13f") == "sec_edgar_free":
        results["13f"] = _probe_sec_edgar()
    else:
        results["13f"] = {"status": "not_configured", "source": DATA_SOURCE_CONFIG.get("13f"),
                           "error": "no health probe implemented for this source"}

    results["checked_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return results


# --------------------------------------------------------------------------
# Full source registry -- every source this app discusses, including ones
# only used by the AI Briefing feature (anthropic, reddit, stocktwits,
# marketbeat, finviz) and ones not wired into any fetcher at all (openai,
# financialjuice). Distinct from check_source_health() above, which stays
# feature-scoped (options_flow/dark_pool/congressional/news/13f) since
# congressional_empty_reason() and the CONGRESS & INSIDER / TICKER
# DEEP-DIVE tabs depend on that exact shape.
# --------------------------------------------------------------------------

def _probe_anthropic():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"status": "not_configured", "source": "anthropic", "error": "ANTHROPIC_API_KEY not set"}
    try:
        import anthropic
        t0 = time.time()
        # Not _log_claude_call -- that tag is reserved for real, billed
        # generation calls; this is the free Models API health probe, so
        # it gets its own clearly-labeled line instead of looking like spend.
        print(f"[CLAUDE API] Health probe (free, not billed) — timestamp: {datetime.now().isoformat()}")
        client = anthropic.Anthropic(api_key=api_key)
        client.models.retrieve("claude-opus-5")  # free -- the Models API is not billed
        return {"status": "up", "source": "anthropic", "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"status": "down", "source": "anthropic", "error": str(e)}


def _probe_reddit_public():
    """Best-effort connectivity probe for Reddit's unauthenticated public
    search JSON endpoint (see fetch_reddit_posts_public). A server/
    datacenter IP can get rate-limited or blocked by Reddit's anti-bot
    layer even with a realistic browser User-Agent -- the same class of
    issue documented for the Senate EFD free fallback (see
    fetch_congressional_trades' module docs) -- so a non-200 here is
    reported as 'degraded', not 'down': it's an expected, real possible
    state for this source, not necessarily evidence something's broken."""
    t0 = time.time()
    try:
        resp = requests.get(
            "https://www.reddit.com/r/wallstreetbets/search.json",
            params={"q": "AAPL", "sort": "new", "limit": 1, "restrict_sr": "on"},
            headers=_BROWSER_HEADERS, timeout=8,
        )
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "reddit_public_json", "latency_ms": latency_ms}
        return {"status": "degraded", "source": "reddit_public_json", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code} -- unauthenticated, best-effort endpoint; may be "
                         f"rate-limited/blocked from this IP without warning"}
    except Exception as e:
        return {"status": "degraded", "source": "reddit_public_json",
                "error": f"{e} (best-effort, unauthenticated -- expected to sometimes fail)"}


def _probe_quiverquant_wsb():
    """Same direct-REST approach as _probe_congressional, for the same
    reason: quiverquant.quiver().wallstreetbets() has the same missing-
    raise_for_status() defect (confirmed live -- an upgrade-required
    response becomes a confusing ValueError instead of a clean error)."""
    api_key = os.environ.get("QUIVER_API_KEY")
    if not api_key:
        return {"status": "not_configured", "source": "quiverquant_wsb", "error": "QUIVER_API_KEY not set"}
    t0 = time.time()
    try:
        resp = requests.get(
            "https://api.quiverquant.com/beta/live/wallstreetbets", params={"count_all": "true"},
            headers={"accept": "application/json", "Authorization": f"Token {api_key}"}, timeout=10,
        )
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "quiverquant_wsb", "latency_ms": latency_ms}
        if resp.status_code in (401, 402, 403):
            return {"status": "not_configured", "source": "quiverquant_wsb", "latency_ms": latency_ms,
                    "error": f"plan tier doesn't include this dataset: {_extract_api_error_detail(resp)}"}
        return {"status": "down", "source": "quiverquant_wsb", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}: {_extract_api_error_detail(resp)}"}
    except Exception as e:
        return {"status": "down", "source": "quiverquant_wsb", "error": str(e)}


def _probe_apewisdom():
    t0 = time.time()
    try:
        resp = requests.get("https://apewisdom.io/api/v1.0/filter/all-stocks/page/1", timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200 and isinstance(resp.json().get("results"), list):
            return {"status": "up", "source": "apewisdom", "latency_ms": latency_ms}
        return {"status": "degraded", "source": "apewisdom", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code} or unexpected response shape"}
    except Exception as e:
        return {"status": "down", "source": "apewisdom", "error": str(e)}


def _probe_stocktwits():
    """Mirrors _fetch_stocktwits_posts exactly -- same URL shape and same
    _BROWSER_HEADERS. A bare "Mozilla/5.0" UA gets a Cloudflare bot-challenge
    403 here; a full browser UA gets a real 200, confirmed live."""
    t0 = time.time()
    try:
        resp = requests.get("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json",
                             headers=_BROWSER_HEADERS, timeout=8)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "stocktwits", "latency_ms": latency_ms}
        return {"status": "down", "source": "stocktwits", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "down", "source": "stocktwits", "error": str(e)}


def _probe_marketbeat():
    t0 = time.time()
    try:
        resp = requests.get(
            "https://www.marketbeat.com/stocks/NASDAQ/AAPL/earnings/",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"},
            timeout=10,
        )
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200 and 'id="earnings-history"' in resp.text:
            return {"status": "up", "source": "marketbeat", "latency_ms": latency_ms}
        if resp.status_code == 200:
            return {"status": "degraded", "source": "marketbeat", "latency_ms": latency_ms,
                    "error": "page loaded but earnings-history table not found (layout change or block page)"}
        return {"status": "down", "source": "marketbeat", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code} (scraping -- may be blocked)"}
    except Exception as e:
        return {"status": "down", "source": "marketbeat", "error": f"{e} (scraping -- may be blocked)"}


def _probe_marketbeat_institutional():
    """Deliberately a capability check only, not a live scrape: actually
    running fetch_marketbeat_institutional_sentiment pops a real, visible
    Chrome window (Cloudflare blocks headless Chrome), which a health
    probe shouldn't trigger unprompted -- especially not on every
    SETTINGS-tab 'Recheck All Sources' click. Checks that the required
    packages import and a Chrome/Chromium binary exists on this machine,
    which is everything that CAN be verified without actually launching a
    session; Cloudflare/page-layout issues can only surface when the
    source is really used (see its own [marketbeat_institutional] log
    lines when that happens)."""
    try:
        import undetected_chromedriver  # noqa: F401
        import selenium  # noqa: F401
    except ImportError as e:
        return {"status": "not_configured", "source": "marketbeat_institutional",
                "error": f"required package not installed ({e}). Run: pip install --upgrade "
                         f"undetected-chromedriver selenium"}

    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium-browser",
    ]
    if not any(os.path.exists(p) for p in chrome_paths):
        return {"status": "not_configured", "source": "marketbeat_institutional",
                "error": "no Chrome/Chromium binary found on this machine"}

    return {
        "status": "degraded", "source": "marketbeat_institutional",
        "note": ("packages + a Chrome binary are present, but this source only works with a "
                 "real, visible (non-headless) browser window on an active desktop session -- "
                 "never in a headless/server deployment -- and this probe doesn't launch a real "
                 "session to confirm a live scrape actually succeeds (that would pop a browser "
                 "window during every health check). See ChromeDriver-version-mismatch handling "
                 "in fetch_marketbeat_institutional_sentiment if a real fetch fails."),
    }


def _probe_finviz():
    t0 = time.time()
    try:
        resp = requests.get("https://finviz.com/quote.ashx?t=AAPL",
                             headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200 and "Insider Trading" in resp.text:
            return {"status": "up", "source": "finviz", "latency_ms": latency_ms}
        if resp.status_code == 200:
            return {"status": "degraded", "source": "finviz", "latency_ms": latency_ms,
                    "error": "page loaded but insider trading table not found"}
        return {"status": "down", "source": "finviz", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "down", "source": "finviz", "error": str(e)}


def _probe_openai():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"status": "not_configured", "source": "openai", "error": "OPENAI_API_KEY not set"}
    return {"status": "up", "source": "openai",
            "note": "key present but not used by any feature in this app (legacy from new_top.py)"}


SOURCE_REGISTRY = [
    {"key": "yfinance", "purpose": "options / price / technicals / fundamentals",
     "env_var": None, "endpoint": "yfinance lib -> query2.finance.yahoo.com (no raw URL in this codebase)",
     "probe": _probe_yfinance},
    {"key": "finra_ats", "purpose": "dark pool proxy", "env_var": None, "endpoint": FINRA_ATS_URL,
     "probe": _probe_finra_proxy},
    {"key": "quiverquant", "purpose": "congressional trades",
     "env_var": "QUIVER_API_KEY",
     "endpoint": "https://api.quiverquant.com/beta/historical/congresstrading/{ticker}",
     "probe": _probe_congressional},
    {"key": "senate_efd", "purpose": "congressional trades fallback",
     "env_var": None, "endpoint": f"{SENATE_EFD_BASE}/search/", "probe": _probe_senate_efd},
    {"key": "sec_edgar_free", "purpose": "13F / insider filings", "env_var": None,
     "endpoint": SEC_FULLTEXT_SEARCH_URL, "probe": _probe_sec_edgar},
    {"key": "13f_edgar", "purpose": "13F institutional filings (AI Briefing institutional_analysis)",
     "env_var": None, "endpoint": SEC_FULLTEXT_SEARCH_URL,
     # Same endpoint/probe as sec_edgar_free above (fetch_13f_changes hits
     # the identical SEC full-text-search URL) -- kept as its own registry
     # row per Part 4 so this specific AI-Briefing feature has its own
     # visible status, not just the shared infrastructure-level one.
     "probe": _probe_sec_edgar},
    {"key": "anthropic", "purpose": "AI briefing", "env_var": "ANTHROPIC_API_KEY",
     "endpoint": "anthropic SDK -> api.anthropic.com/v1/messages (claude-opus-5)", "probe": _probe_anthropic},
    {"key": "apewisdom", "purpose": "Retail attention/momentum (mention volume) -- not sentiment polarity",
     "env_var": None,
     "endpoint": "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}",
     "probe": _probe_apewisdom},
    {"key": "stocktwits", "purpose": "Retail sentiment -- tagged bullish/bearish messages", "env_var": None,
     "endpoint": "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json", "probe": _probe_stocktwits},
    {"key": "reddit_public_json",
     "purpose": "Unauthenticated, best-effort raw commentary -- used only in on-demand AI Briefing, "
                "may be rate-limited or blocked by Reddit without warning",
     "env_var": None,
     "endpoint": "https://www.reddit.com/r/wallstreetbets/search.json?q={ticker}&sort=new&limit={limit}",
     "probe": _probe_reddit_public},
    {"key": "quiverquant_wsb", "purpose": "retail sentiment (WallStreetBets mentions, bonus 3rd source)",
     "env_var": "QUIVER_API_KEY",
     "endpoint": "https://api.quiverquant.com/beta/live/wallstreetbets?count_all=true",
     "probe": _probe_quiverquant_wsb},
    {"key": "marketbeat", "purpose": "earnings history scrape", "env_var": None,
     "endpoint": "https://www.marketbeat.com/stocks/NASDAQ/{ticker}/earnings/", "probe": _probe_marketbeat},
    {"key": "marketbeat_institutional", "purpose": "institutional ownership % + recent buy/sell (AI Briefing)",
     "env_var": None,
     "endpoint": "Selenium/undetected-chromedriver -> https://www.marketbeat.com/stocks/{NASDAQ,NYSE}/"
                 "{ticker}/institutional-ownership/ -- requires a real, visible desktop browser session, "
                 "never headless/server",
     "probe": _probe_marketbeat_institutional},
    {"key": "finviz", "purpose": "insider sales scrape", "env_var": None,
     "endpoint": "https://finviz.com/quote.ashx?t={ticker}", "probe": _probe_finviz},
    {"key": "openai", "purpose": "not currently used (legacy from new_top.py)",
     "env_var": "OPENAI_API_KEY", "endpoint": "n/a -- key checked only, no feature calls this",
     "probe": _probe_openai},
    {"key": "financialjuice", "purpose": "reference links only, no API", "env_var": None,
     "endpoint": "https://www.financialjuice.com/home (manual link, not fetched)", "probe": None},
]


def check_full_source_registry():
    """Every source this app discusses -- a registry, not just a health
    check. Sources with no probe (financialjuice) are reported as
    'not_applicable' rather than a failure state."""
    results = []
    for entry in SOURCE_REGISTRY:
        row = {k: v for k, v in entry.items() if k != "probe"}
        if entry["probe"] is None:
            results.append({**row, "status": "not_applicable", "error": None, "latency_ms": None})
            continue
        results.append({**row, **entry["probe"]()})
    return {"sources": results, "checked_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}


if __name__ == "__main__":
    full_refresh(load_watchlist())
