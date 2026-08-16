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
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone

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
    "news": "yfinance",
    "13f": "sec_edgar_free",
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
"""

# Migrations for columns added after a table's original CREATE TABLE, so
# existing databases from earlier versions of this app pick them up too.
_COLUMN_MIGRATIONS = {
    "options_flow": [("source", "TEXT")],
    "dark_pool_signals": [("source", "TEXT")],
    "congressional_trades": [("source", "TEXT")],
    "insider_trades": [("source", "TEXT")],
    "analyst_targets": [
        ("rec_period", "TEXT"), ("strong_buy", "INTEGER"), ("buy", "INTEGER"),
        ("hold", "INTEGER"), ("sell", "INTEGER"), ("strong_sell", "INTEGER"),
    ],
    "earnings_history_real": [
        ("revenue_estimate", "TEXT"), ("eps_beat_miss_pct", "REAL"), ("revenue_beat_miss_pct", "REAL"),
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
    finally:
        conn.close()


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
]


def fetch_options_flow(ticker, max_expirations=4):
    source = DATA_SOURCE_CONFIG.get("options_flow", "yfinance")
    if source == "yfinance":
        return _fetch_options_flow_yfinance(ticker, max_expirations=max_expirations)
    raise NotImplementedError(
        f"options_flow source '{source}' is not implemented. Intended: e.g. "
        "Polygon.io GET /v3/snapshot/options/{ticker} or Tradier GET /v1/markets/options/chains"
    )


def _fetch_options_flow_yfinance(ticker, max_expirations=4):
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
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            for _, row in df.iterrows():
                volume = int(_safe_num(row.get("volume")))
                oi = int(_safe_num(row.get("openInterest")))
                ratio = (volume / oi) if oi else None
                rows.append({
                    "ticker": ticker, "expiry": exp, "strike": float(row["strike"]),
                    "type": option_type, "volume": volume, "open_interest": oi,
                    "vol_oi_ratio": ratio, "implied_volatility": _safe_num(row.get("impliedVolatility")),
                    "last_price": _safe_num(row.get("lastPrice")), "unusual": False,
                    "underlying_price": underlying_price, "source": "yfinance",
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
                 volume_oi_ratio, implied_volatility, last_price, underlying_price, unusual, source, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker, fetch_date, expiration, strike, option_type) DO UPDATE SET
                volume=excluded.volume, open_interest=excluded.open_interest,
                volume_oi_ratio=excluded.volume_oi_ratio, implied_volatility=excluded.implied_volatility,
                last_price=excluded.last_price, underlying_price=excluded.underlying_price,
                unusual=excluded.unusual, source=excluded.source
            """,
            (ticker, fetch_date, row["expiry"], float(row["strike"]), row["type"], int(row["volume"]),
             int(row["open_interest"]), row["vol_oi_ratio"], row["implied_volatility"], row["last_price"],
             row.get("underlying_price"), int(bool(row["unusual"])), row.get("source"),
             datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_options_flow_cache(conn, ticker):
    fetch_date = date.today().isoformat()
    df = pd.read_sql_query(
        """SELECT ticker, expiration, strike, option_type, volume, open_interest, volume_oi_ratio,
                  implied_volatility, last_price, underlying_price, unusual, source
           FROM options_flow WHERE ticker=? AND fetch_date=?""",
        conn, params=(ticker, fetch_date),
    )
    return df.rename(columns=_OPTIONS_FLOW_DB_TO_SPEC)


def cached_options_flow(conn, ticker, max_age_hours=0.25, force_refresh=False):
    """Options chains are a full-state snapshot -- yfinance has no delta/
    incremental fetch for a chain, you get the whole current chain or
    nothing -- so the meaningful cache boundary is "once per trading day,"
    not a fine-grained TTL. If we already have today's fetch_date stored
    for this ticker, skip the network call entirely regardless of how many
    hours old that snapshot is. `max_age_hours` is accepted for signature
    compatibility with every other cached_* wrapper but no longer gates
    this one -- the day-based check below does."""
    table = "options_flow"
    last_fetch_date = get_last_timestamp(conn, table, ticker, "fetch_date")
    have_today = last_fetch_date == date.today().isoformat()

    if not force_refresh and have_today:
        cached_df = _read_options_flow_cache(conn, ticker)
        if not cached_df.empty:
            return {"data": cached_df, "source": cached_df["source"].iloc[0], "cache_hit": True,
                    "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        df = fetch_options_flow(ticker)
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


def fetch_buybacks(ticker):
    t = yf.Ticker(ticker)
    try:
        cf = t.cashflow
    except Exception:
        cf = None

    empty = {"ticker": ticker, "history": pd.DataFrame(), "trend": "N/A", "source": "yfinance"}
    if cf is None or cf.empty:
        return empty

    row_name = next((c for c in _BUYBACK_ROW_CANDIDATES if c in cf.index), None)
    if row_name is None:
        row_name = next(
            (idx for idx in cf.index if "repurchase" in str(idx).lower() and "stock" in str(idx).lower()),
            None,
        )
    if row_name is None:
        return empty

    series = cf.loc[row_name].dropna()
    if series.empty:
        return empty

    history = series.reset_index()
    history.columns = ["period", "buyback_value"]
    # yfinance reports repurchases as a cash outflow (negative); magnitude is the spend.
    history["buyback_magnitude"] = history["buyback_value"].abs()
    history = history.sort_values("period").reset_index(drop=True)
    history["ticker"] = ticker

    trend = _buyback_trend(history)
    return {"ticker": ticker, "history": history.tail(4), "trend": trend, "source": "yfinance"}


def _buyback_trend(history):
    if history is None or len(history) < 2:
        return "N/A"
    recent, prior = history["buyback_magnitude"].iloc[-1], history["buyback_magnitude"].iloc[-2]
    if prior == 0:
        return "NEW" if recent > 0 else "FLAT"
    if recent > prior * 1.05:
        return "INCREASING"
    if recent < prior * 0.95:
        return "DECREASING"
    return "FLAT"


def _persist_buybacks(conn, ticker, result):
    history = result.get("history")
    if history is None or history.empty:
        return
    cur = conn.cursor()
    for _, row in history.iterrows():
        cur.execute(
            """INSERT INTO buybacks (ticker, period, buyback_value, source, fetched_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(ticker, period) DO UPDATE SET
                   buyback_value=excluded.buyback_value, source=excluded.source, fetched_at=excluded.fetched_at""",
            (ticker, str(row["period"]), float(row["buyback_value"]), result.get("source"),
             datetime.utcnow().isoformat()),
        )
    conn.commit()


def _read_buybacks_cache(conn, ticker, limit=4):
    df = pd.read_sql_query(
        "SELECT period, buyback_value FROM buybacks WHERE ticker=? ORDER BY period DESC LIMIT ?",
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
            return {"data": {"ticker": ticker, "history": history, "trend": _buyback_trend(history)},
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
        return {"data": {"ticker": ticker, "history": history, "trend": _buyback_trend(history)},
                "source": "yfinance" if not history.empty else None, "cache_hit": not history.empty,
                "fetched_at": _last_fetch_info(conn, table, ticker)}


# --------------------------------------------------------------------------
# fetch_news(ticker, limit=8) -> DataFrame[ticker, title, publisher, link,
# published_at, source]. Source-agnostic: dispatches on
# DATA_SOURCE_CONFIG["news"].
# --------------------------------------------------------------------------

def fetch_news(ticker, limit=8):
    source = DATA_SOURCE_CONFIG.get("news", "yfinance")
    if source == "yfinance":
        return _fetch_news_yfinance(ticker, limit=limit)
    raise NotImplementedError(
        f"news source '{source}' is not implemented. "
        "Intended: e.g. NewsAPI.org GET /v2/everything?q={ticker}&apiKey=..."
    )


def _fetch_news_yfinance(ticker, limit=8):
    t = yf.Ticker(ticker)
    try:
        raw = t.news or []
    except Exception:
        raw = []

    items = []
    for entry in raw[:limit]:
        # yfinance has nested news payloads under "content" in newer releases;
        # fall back to the flat top-level schema from older ones.
        content = entry.get("content") if isinstance(entry.get("content"), dict) else entry

        title = content.get("title") or entry.get("title")
        if not title:
            continue

        provider = content.get("provider")
        publisher = provider.get("displayName") if isinstance(provider, dict) else entry.get("publisher")

        canonical = content.get("canonicalUrl")
        link = canonical.get("url") if isinstance(canonical, dict) else entry.get("link")

        pub_raw = content.get("pubDate", entry.get("providerPublishTime"))
        published_at = None
        if isinstance(pub_raw, str):
            published_at = pd.to_datetime(pub_raw, utc=True, errors="coerce")
        elif isinstance(pub_raw, (int, float)):
            published_at = pd.to_datetime(pub_raw, unit="s", utc=True, errors="coerce")

        items.append({
            "ticker": ticker, "title": title, "publisher": publisher,
            "link": link, "published_at": published_at, "source": "yfinance",
        })

    df = pd.DataFrame(items)
    if not df.empty:
        df = df.sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)
    return df


def _persist_news(conn, ticker, df):
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
            """INSERT OR IGNORE INTO news_cache (ticker, title, publisher, link, published_at, source, fetched_at)
               VALUES (?,?,?,?,?,?,?)""",
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
        df = fetch_news(ticker, limit=limit)
        new_count = len(df)
        if last_ts and not df.empty:
            new_count = int((df["published_at"] > pd.Timestamp(last_ts)).sum())
        _persist_news(conn, ticker, df)
        _log_fetch(conn, table, ticker, True, new_count)
        # Re-read so newly-fetched items merge with any previously cached
        # headlines still within the requested limit, sorted by publish date.
        merged = _read_news_cache(conn, ticker, limit=limit)
        result_df = merged if not merged.empty else df
        source = df["source"].iloc[0] if not df.empty else DATA_SOURCE_CONFIG.get("news")
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

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _bs_call_delta_theta(S, K, T, r, sigma, q=0.0):
    """Black-Scholes call delta and theta (per day, per share)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan"), float("nan")

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    delta = math.exp(-q * T) * _norm_cdf(d1)

    theta_per_year = (
        -(S * math.exp(-q * T) * _norm_pdf(d1) * sigma) / (2.0 * sqrtT)
        - r * K * math.exp(-r * T) * _norm_cdf(d2)
        + q * S * math.exp(-q * T) * _norm_cdf(d1)
    )
    return delta, theta_per_year / 365.0


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
    a percent (e.g. 42.5, not 0.425). days: horizon in calendar days."""
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
#   get_reddit_posts / get_stocktwits_posts     -> _fetch_reddit_posts / _fetch_stocktwits_posts
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
# Every credential here (ANTHROPIC_API_KEY, DEVVIT_SENTIMENT_URL,
# DEVVIT_APP_TOKEN) is read from the environment, and every fetcher in
# this section fails soft (never raises) so a blocked/changed scrape
# degrades one section of a briefing instead of crashing it.
#
# ~/trading/marketbeat_scraper.py (imported by new_top.py, not ported) is a
# separate Selenium/undetected-chromedriver institutional scraper --
# currently broken in this environment (ChromeDriver/Chrome version
# mismatch). It is treated as degraded/unavailable; nothing here depends on
# it. get_marketbeat_analyst_sentiment_structured, ported above, is a
# plain-requests scrape defined directly in new_top.py and does not need it.
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
            for row in table.find_all("tr")[1:7]:
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


def _fetch_reddit_posts_devvit(ticker, limit=25):
    """Real Reddit posts (r/wallstreetbets, r/stocks, r/investing) mentioning
    `ticker`, via the smartmoneydashboard Devvit app's published
    /external/sentiment endpoint -- see
    smartmoneydashboard/src/server/sentiment.ts and devvit.json's
    server.externalEndpoints.sentiment (scopes: ["global"], a long-lived
    managed app token per @devvit/external-endpoints' docs). Requires
    DEVVIT_SENTIMENT_URL (the published app's full external URL, e.g.
    "https://smartmoneydashboard-<id>-external.devvit.net/external/sentiment")
    and DEVVIT_APP_TOKEN in the environment. Fails soft -- [] if either is
    unset or the call errors, so this degrades cleanly until the app is
    published and the token is minted."""
    base_url = os.environ.get("DEVVIT_SENTIMENT_URL")
    app_token = os.environ.get("DEVVIT_APP_TOKEN")
    if not base_url or not app_token:
        print(f"[retail_sentiment] Devvit path skipped for {ticker}: "
              f"DEVVIT_SENTIMENT_URL{'' if base_url else ' (unset)'} / "
              f"DEVVIT_APP_TOKEN{'' if app_token else ' (unset)'} -- "
              f"no Reddit sentiment for this ticker until published")
        return []
    print(f"[retail_sentiment] Devvit GET {base_url}?ticker={ticker}")
    try:
        resp = requests.get(
            base_url, params={"ticker": ticker},
            headers={"Authorization": f"bearer {app_token}"}, timeout=15,
        )
        resp.raise_for_status()
        mentions = resp.json().get("mentions", [])
        print(f"[retail_sentiment] Devvit returned {len(mentions)} mentions for {ticker}")
    except (requests.RequestException, ValueError) as e:
        print(f"[retail_sentiment] Devvit sentiment fetch failed for {ticker}: {e}")
        return []
    posts = []
    for m in mentions[:limit]:
        created_ms = m.get("createdAt")
        posted_at = datetime.utcfromtimestamp(created_ms / 1000).isoformat() if created_ms else None
        posts.append({
            "source": "devvit", "text": m.get("title"), "url": m.get("permalink"), "posted_at": posted_at,
        })
    return posts


def _fetch_reddit_posts(ticker, limit=25):
    """Real Reddit posts mentioning `ticker`, via the Devvit app's
    published external endpoint (source="devvit"). Fails soft to [] if
    that's not configured or errors -- Reddit stays a secondary source;
    StockTwits is the always-on primary retail-sentiment feed regardless.

    The former PRAW/script-app fallback was removed: its REDDIT_CLIENT_ID/
    REDDIT_CLIENT_SECRET pair was a dead, permanently-401ing credential
    from a deleted Reddit app (confirmed live), so it never produced data
    and only added a confusing extra hop to the trace."""
    print(f"[retail_sentiment] _fetch_reddit_posts({ticker}) called")
    return _fetch_reddit_posts_devvit(ticker, limit=limit)


def _fetch_stocktwits_posts(ticker, limit=25):
    """Real StockTwits posts for `ticker` -- ported from new_top.py's
    get_stocktwits_posts. No auth required. Fails soft -- [] on any error.

    Uses _BROWSER_HEADERS (a realistic full browser User-Agent), not a bare
    "Mozilla/5.0" -- StockTwits sits behind Cloudflare, which serves a bot
    challenge page (HTTP 403, "Just a moment...") to that string but a
    normal 200 with real JSON to a full browser UA. Confirmed live."""
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
         "posted_at": msg.get("created_at")}
        for msg in messages[:limit]
    ]


def fetch_retail_sentiment(ticker):
    """Real retail sentiment: Reddit + StockTwits posts, each tagged with
    its source name and original timestamp -- ported from new_top.py's
    get_reddit_posts / get_stocktwits_posts. No sentiment-score guessing
    (new_top.py's TextBlob-based analyze_sentiment/summarize_retail_sentiment
    were not ported); the raw, source-tagged posts are handed to Claude to
    characterize themselves, per the anti-hallucination rules."""
    return _fetch_reddit_posts(ticker) + _fetch_stocktwits_posts(ticker)


def _persist_retail_sentiment(conn, ticker, posts):
    if not posts:
        return
    cur = conn.cursor()
    for p in posts:
        cur.execute(
            """INSERT OR IGNORE INTO retail_sentiment_posts
                   (ticker, source, text, url, posted_at, fetched_at) VALUES (?,?,?,?,?,?)""",
            (ticker, p.get("source"), p.get("text"), p.get("url"), p.get("posted_at"),
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
        """SELECT source, text, url, posted_at FROM retail_sentiment_posts
           WHERE ticker=? ORDER BY posted_at DESC LIMIT ?""",
        (ticker, limit),
    ).fetchall()
    return [{"source": r[0], "text": r[1], "url": r[2], "posted_at": r[3]} for r in rows]


def cached_retail_sentiment(conn, ticker, max_age_hours=2, force_refresh=False):
    table = "retail_sentiment"
    if not force_refresh and not should_refetch(conn, table, ticker, max_age_hours):
        cached = _read_retail_sentiment_cache(conn, ticker)
        return {"data": cached, "source": "reddit+stocktwits" if cached else None,
                "cache_hit": True, "fetched_at": _last_fetch_info(conn, table, ticker)}
    try:
        posts = fetch_retail_sentiment(ticker)
        _persist_retail_sentiment(conn, ticker, posts)
        _log_fetch(conn, table, ticker, True, len(posts))
    except Exception as e:
        _log_fetch(conn, table, ticker, False, 0, str(e))
    cached = _read_retail_sentiment_cache(conn, ticker)
    return {"data": cached, "source": "reddit+stocktwits" if cached else None,
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
    "expiry buckets, analyst targets, a real MarketBeat-sourced earnings track record (each row "
    "matched to the REAL yfinance-measured price reaction), buybacks, congressional trades, "
    "insider trades (SEC Form 4 and Finviz, each tagged by source), real Reddit/StockTwits "
    "retail sentiment posts (each tagged by source and timestamp), dark pool signal, a "
    "divergence-score history, recent news headlines with publisher and date, and "
    "technical_levels -- an ALGORITHMICALLY detected (not AI-guessed) set of support/resistance "
    "levels, trend structure, and measured-move breakdown/breakout targets, computed directly "
    "from price/volume data before this prompt was built.\n\n"
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
    "estimate a price level that isn't literally present in that data.\n\n"
    "Write exactly these sections:\n"
    "- setup: 3-4 short sentences MAX, grounded in the real technicals and expected-move data "
    "provided (not the earnings track record -- that has its own section below). Bold (markdown "
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
    "never a paragraph. Each bullet cites source and approximate recency, e.g. '- Reddit "
    "(Aug 14): ...'. Max 4 bullets, picking the most decision-relevant. If "
    "retail_sentiment_posts is empty, a single bullet saying 'No retail sentiment data "
    "available' -- do not infer sentiment from price action instead.\n"
    "- institutional_analysis: SHORT BULLET POINTS (markdown '- ' lines), one line per distinct "
    "fact -- one insider sale, one buyback figure, one analyst-target datapoint -- never fold "
    "multiple facts into one run-on sentence. Example line: '- CEO William Mosley sold 12,920 "
    "shares ($10.38M) -- Aug 3 (finviz_scrape)'. Cite source and date on every bullet. State "
    "explicitly when a category has no data. Max 4-5 bullets, picking the most decision-relevant.\n"
    "- news_summary: 3-4 sentences MAX on the single dominant narrative driving the stock right "
    "now -- not an exhaustive list of every headline's framing.\n"
    "- catalysts: specific, NAMED forward-looking events pulled from the news headlines in the "
    "bundle (e.g. a named product launch, a specific earnings date, a court ruling, a regulatory "
    "decision) -- never generic statements like 'market conditions' or 'sector trends.' Max 4, "
    "picking the most decision-relevant -- empty list if the headlines genuinely contain no such "
    "event. expected_timing is a short phrase ('Q3 earnings, ~Nov 2026', 'unspecified'), not a "
    "full sentence.\n"
    "- earnings_track_record: 2-3 sentences MAX synthesizing the pattern across the real "
    "beat/miss + real price-reaction pairs given in earnings_track_record_real -- not a "
    "recitation of every quarter. Use only the exact dates and percentages given -- do not alter "
    "or round them differently. Explain a reaction's likely cause only where the news headlines "
    "in the bundle explicitly support that reasoning.\n"
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
        "SELECT brief_json, fetched_at FROM ai_briefs WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    if not row:
        return None
    brief_json, fetched_at = row
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
    return brief


def _persist_ai_brief(conn, ticker, brief):
    payload = {k: v for k, v in brief.items() if not k.startswith("_")}
    conn.execute(
        "INSERT INTO ai_briefs (ticker, brief_json, fetched_at) VALUES (?, ?, ?)",
        (ticker, json.dumps(payload), datetime.utcnow().isoformat()),
    )
    conn.commit()


def _read_ai_context_from_cache(ticker, conn, technicals=None, technical_levels=None):
    """Pure DB read of everything used to build the AI briefing context --
    no network calls. Used both for the cost estimate (which must never
    trigger a fetch) and as the final read step after _bundle_ai_context
    has refreshed the real-data sources.

    `technical_levels` follows the same pattern as `technicals`: it's a
    real computation (detect_technical_levels), not a DB read, so the
    caller passes it in already-computed rather than this function fetching
    it -- estimate_ai_briefing_cost() leaves it unset (network-free cost
    estimate, may undercount slightly, same as `technicals`), while
    _bundle_ai_context() computes it fresh via detect_technical_levels()."""
    fundamentals = _read_fundamentals_info_cache(conn, ticker) or {}
    targets = _read_analyst_targets_cache(conn, ticker) or {}
    earnings_df = _read_earnings_calendar_cache(conn, ticker, limit=6)
    buybacks_df = _read_buybacks_cache(conn, ticker, limit=4)
    news_df = _read_news_cache(conn, ticker, limit=10)
    congress = _read_congressional_cache_by_ticker(conn, ticker, limit=15)
    earnings_track_record_real = _read_earnings_history_real_cache(conn, ticker, limit=6)
    retail_posts = _read_retail_sentiment_cache(conn, ticker, limit=30)

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
        "analyst_targets": targets or {},
        "earnings_calendar": (
            [{k: str(v) for k, v in r.items()} for r in earnings_df.to_dict("records")]
            if not earnings_df.empty else []
        ),
        "earnings_track_record_real": earnings_track_record_real,
        "buybacks": buybacks_df.to_dict("records") if not buybacks_df.empty else [],
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
        "retail_sentiment_posts": retail_posts,
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
    sales, Reddit/StockTwits retail sentiment, earnings_signal (for the IV
    expected move) -- plus a fresh technical snapshot, then reads
    everything via _read_ai_context_from_cache. This is the version
    generate_deep_analysis() uses; it can make real network calls, each
    fail-soft."""
    cached_earnings_history_real(conn, ticker, max_age_hours=12)
    cached_insider_sales_finviz(conn, ticker, max_age_hours=24)
    cached_retail_sentiment(conn, ticker, max_age_hours=2)
    cached_earnings_signal(conn, ticker, max_age_hours=12)
    technicals = _fetch_technical_snapshot(ticker)
    # DEFAULT_INTERVAL/DEFAULT_LOOKBACK (1d/6mo) -- the same daily-candle,
    # 6-month swing-horizon window the dashboard shows by default, so the
    # AI's narrative lines up with what a trader sees on first load.
    technical_levels = detect_technical_levels(ticker, DEFAULT_INTERVAL, DEFAULT_LOOKBACK, conn=conn)
    return _read_ai_context_from_cache(ticker, conn, technicals=technicals, technical_levels=technical_levels)


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

    if DATA_SOURCE_CONFIG.get("news") == "yfinance":
        results["news"] = _probe_yfinance()
    else:
        results["news"] = {"status": "not_configured", "source": DATA_SOURCE_CONFIG.get("news"),
                            "error": "no health probe implemented for this source"}

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
        client = anthropic.Anthropic(api_key=api_key)
        client.models.retrieve("claude-opus-5")  # free -- the Models API is not billed
        return {"status": "up", "source": "anthropic", "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"status": "down", "source": "anthropic", "error": str(e)}


def _probe_reddit():
    """Checks the Devvit external-endpoint path -- the only Reddit path
    left (see _fetch_reddit_posts_devvit). The former PRAW/script-app
    fallback was removed (dead credentials, see _fetch_reddit_posts)."""
    devvit_url = os.environ.get("DEVVIT_SENTIMENT_URL")
    devvit_token = os.environ.get("DEVVIT_APP_TOKEN")
    if not devvit_url or not devvit_token:
        return {"status": "not_configured", "source": "reddit",
                "error": "DEVVIT_SENTIMENT_URL/DEVVIT_APP_TOKEN not set"}
    t0 = time.time()
    try:
        resp = requests.get(
            devvit_url, params={"ticker": "SPY"},
            headers={"Authorization": f"bearer {devvit_token}"}, timeout=10,
        )
        latency_ms = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return {"status": "up", "source": "devvit", "latency_ms": latency_ms}
        return {"status": "down", "source": "devvit", "latency_ms": latency_ms,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except Exception as e:
        return {"status": "down", "source": "devvit", "error": str(e)}


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
    {"key": "anthropic", "purpose": "AI briefing", "env_var": "ANTHROPIC_API_KEY",
     "endpoint": "anthropic SDK -> api.anthropic.com/v1/messages (claude-opus-5)", "probe": _probe_anthropic},
    {"key": "reddit", "purpose": "retail sentiment (Reddit)",
     "env_var": "DEVVIT_SENTIMENT_URL, DEVVIT_APP_TOKEN",
     "endpoint": "Devvit: $DEVVIT_SENTIMENT_URL/external/sentiment (unpublished -- not set)",
     "probe": _probe_reddit},
    {"key": "stocktwits", "purpose": "retail sentiment (StockTwits)", "env_var": None,
     "endpoint": "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json", "probe": _probe_stocktwits},
    {"key": "marketbeat", "purpose": "earnings history scrape", "env_var": None,
     "endpoint": "https://www.marketbeat.com/stocks/NASDAQ/{ticker}/earnings/", "probe": _probe_marketbeat},
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
