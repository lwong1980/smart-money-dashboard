"""Streamlit UI for the Smart Money Intelligence Dashboard."""

import html
import json
import os
import re
import threading
import time
from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone, time as dt_time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import auth
import charts
import data_engine as de
from digest import healthchecks
from digest.core import run_daily_digest_send, run_digests_for_all_users
from charts import (
    ACCENT,
    ANALYST_BREAKDOWN_CID,
    ANALYST_FAN_CID,
    _MACRO_IMPACT_COLORS,
    COLOR_BULLISH,
    COLOR_BEARISH,
    COLOR_INSTITUTIONAL,
    COLOR_NEUTRAL,
    COLOR_RETAIL,
    FEATURED_STOCK_CHART_CID,
    GRID,
    PANEL,
    SWING_CHART_CID,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    add_macro_event_markers,
    apply_theme,
    build_analyst_chart_images,
    build_featured_stock_chart_bytes,
    build_swing_chart_image,
    render_chart_as_image,
    render_entry_zone_mini_chart,
    render_price_target_fan,
    render_price_volume_chart,
    render_recommendation_breakdown,
)
from data_engine import (
    AI_BRIEF_CACHE_HOURS,
    DATA_SOURCE_CONFIG,
    DEFAULT_DB_PATH,
    DEFAULT_STARTER_WATCHLIST,
    DIVERGENCE_LABELS,
    cached_analyst_targets,
    cached_buybacks,
    cached_congressional_trades,
    cached_congressional_trades_by_ticker,
    cached_dark_pool,
    cached_earnings_calendar,
    cached_earnings_signal,
    cached_fundamentals,
    cached_leaps_candidates,
    cached_news,
    cached_earnings_history_real,
    cached_options_flow,
    cached_retail_sentiment,
    cached_apewisdom_sentiment,
    _aggregate_stocktwits_sentiment,
    classify_delta,
    classify_iv,
    classify_vol_oi,
    classify_dte,
    classify_theta_pct,
    classify_spread_pct,
    compute_spread_pct,
    find_option_candidates,
    RISK_PROFILE_DELTA_RANGES,
    TIME_HORIZON_DTE_RANGES,
    TIME_HORIZON_MAX_EXPIRATIONS,
    DELTA_PROFILES,
    IV_PROFILES,
    VOL_OI_PROFILES,
    DTE_PROFILES,
    THETA_PCT_PROFILES,
    SPREAD_PROFILES,
    DEFAULT_OPTIONS_MAX_EXPIRATIONS,
    annotate_options_badges,
    synthesize_options_flow_summary,
    generate_row_insight,
    _iv_term_structure,
    extract_simulator_inputs_from_briefing,
    get_market_pricing_signals,
    estimate_earnings_probability,
    recommend_earnings_strategy,
    analyze_earnings_contract,
    backfill_earnings_history,
    log_earnings_prediction,
    reconcile_earnings_predictions,
    get_simulator_track_record,
    train_earnings_direction_model,
    cached_13f_changes,
    cached_marketbeat_institutional_sentiment,
    _compute_technical_levels,
    _read_marketbeat_institutional_cache,
    _read_news_sources,
    add_news_source,
    remove_news_source,
    set_news_source_enabled,
    test_news_feed,
    detect_technical_levels,
    fetch_session_price_summary,
    check_full_source_registry,
    check_source_health,
    compute_divergence,
    estimate_ai_briefing_cost,
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK,
    fetch_analyst_price_target_breakdown,
    fetch_polymarket,
    fetch_price_history,
    fetch_quarterly_financials_with_price,
    full_refresh,
    generate_deep_analysis,
    get_cached_ai_brief,
    get_connection,
    init_db,
    load_watchlist,
    save_watchlist,
    INTERVAL_OPTIONS,
    LOOKBACK_OPTIONS,
    MACRO_KEYWORDS,
    DAY_PREDICTION_BUFFER_MINUTES_DEFAULT,
    compute_day_target,
    log_day_prediction,
    get_committed_day_prediction,
    log_day_prediction_snapshot,
    get_day_prediction_snapshots,
    reconcile_day_predictions,
    get_day_prediction_track_record,
    _predicted_price_at_time,
    ensure_ticker_data_ready,
    calibrate_day_prediction_model,
    get_active_day_prediction_calibration,
    get_day_prediction_calibration_history,
    DAY_CALIBRATION_MIN_SAMPLES,
    backtest_day_predictions,
    BACKTEST_DEFAULT_LOOKBACK_DAYS,
    get_day_prediction_sample_composition,
    get_current_day_prediction_mode,
    get_backtested_tickers_summary,
    get_backtest_report_data,
    get_most_recent_day_prediction,
    get_most_recent_backtest_prediction,
    get_intraday_bars_for_session,
    cached_macro_events,
    get_macro_events_in_range,
    get_macro_events_near,
    get_macro_events_by_id,
    get_watchlist_relevant_tickers,
    get_cpi_bracket_odds,
    get_macro_release_banner_state,
    check_and_send_macro_release_alerts,
    _format_et_time_12h,
    FOMC_MEETING_DATES,
    load_pinned_runners,
    save_pinned_runners,
    load_runners_refresh_interval,
    save_runners_refresh_interval,
    SWING_HORIZON_OPTIONS,
    get_multiday_realized_volatility,
    check_earnings_collision,
    get_divergence_score_trend,
    get_relative_strength_trend,
    get_recency_weighted_activity,
    get_dark_pool_trend,
    get_gamma_evolution,
    backtest_swing_forecasts,
    compute_swing_forecast,
    log_swing_forecast,
    reconcile_swing_forecasts,
    recommend_swing_strategy,
    get_swing_forecast_track_record,
    get_swing_backtest_report_data,
    get_active_swing_calibration,
    get_swing_calibration_history,
    calibrate_swing_forecast_model,
    SWING_CALIBRATION_MIN_SAMPLES,
    analyze_swing_contract,
    get_history,
    get_committed_swing_path,
    get_macro_event_grounded_reasoning,
    get_entry_zones,
    get_latest_cached_price,
    refresh_entry_zone,
    refresh_all_entry_zones,
    check_entry_zones,
    check_rating_changes,
    synthesize_rating_signal,
    get_watchlist_signals_since,
    load_watchlist_signals_last_viewed,
    save_watchlist_signals_last_viewed,
    send_daily_digest_email,
    cached_fed_funds_rate,
    get_fed_target_range_text,
    get_firm_tier,
    compute_and_log_runner_scan,
    log_runners_scan,
    get_email_config,
    save_email_config,
    is_daily_digest_ready,
    should_send_daily_digest_now,
    build_digest_content,
    is_market_holiday,
    get_market_holiday_name,
    DIGEST_SECTION_LABELS,
    DEFAULT_DIGEST_SECTIONS,
    select_featured_stock,
    get_featured_stock_section,
    DEFAULT_DIGEST_FROM_EMAIL,
    unsubscribe_by_token,
    add_email_subscriber,
    get_active_subscribers,
    get_all_subscribers,
    remove_email_subscriber,
    resubscribe_email,
    is_valid_email_format,
    get_subscriber_by_token,
    get_subscriber_sections_included,
    set_subscriber_active,
    set_subscriber_sections_included,
    get_or_build_todays_edition,
    build_digest_edition,
    get_digest_edition,
    get_last_digest_edition,
    _get_edition_admin_email,
    build_substack_markdown,
    render_substack_image_slots,
    render_edition_for_recipient,
    send_edition_to_substack,
    export_substack_zip,
    get_substack_posts,
    get_last_substack_post,
    get_substack_post_for_edition,
    test_substack_connection,
    today_et,
    load_chart_defaults,
    save_chart_defaults,
    CHART_REFRESH_OPTIONS,
    RUNNERS_REFRESH_INTERVAL_OPTIONS,
    get_app_setting,
    set_app_setting,
    list_users,
    SnapshotCache,
    get_featured_override,
    set_featured_override,
    clear_featured_override,
    get_todays_runner_tickers,
    USER_ROLES,
    pmset_wake_command_for,
    UNIVERSE_GATE_MODE,
    VOLATILITY_MOVE_DISLOCATION_FLAG,
    VOLATILITY_UNUSUAL_OPTVOL_FLAG,
    get_universe,
    get_scan,
    start_volatility_scan_on_demand,
    get_latest_volatility_scan_run,
)

# --------------------------------------------------------------------------
# Theme / palette
# --------------------------------------------------------------------------

BG = "#030609"
PANEL_ALT = "#0a121b"
BORDER = "rgba(0,229,255,0.14)"
# PANEL, ACCENT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, GRID, COLOR_INSTITUTIONAL,
# COLOR_RETAIL, COLOR_BULLISH, COLOR_BEARISH, COLOR_NEUTRAL: imported from charts.py
# (the one place that defines them -- see that file's own comment).
COLOR_CONFLICTED = "#a86fe0"      # violet -- distinct from NEUTRAL's gray so a
                                  # high-conviction-but-disagreeing read doesn't
                                  # visually collapse into "nothing going on"

LABEL_COLORS = {
    "SMART_BULLISH": COLOR_BULLISH,
    "SMART_BEARISH_RETAIL_LONG": COLOR_BEARISH,
    "RETAIL_FRENZY": COLOR_RETAIL,
    "INSTITUTIONAL_ACTIVE": COLOR_INSTITUTIONAL,
    "CONFLICTED_SIGNALS": COLOR_CONFLICTED,
    "NEUTRAL": COLOR_NEUTRAL,
}

# Source -> (display label, "free"/"paid")
SOURCE_LABELS = {
    "yfinance": ("yfinance", "free"),
    "finra_ats": ("FINRA ATS", "free"),
    "finra_proxy": ("FINRA (proxy)", "free"),
    "sec_edgar_free": ("SEC EDGAR", "free"),
    "senate_efd_free": ("Senate EFD", "free"),
    "quiverquant": ("QuiverQuant", "paid"),
    "Yahoo Finance RSS": ("Yahoo Finance RSS", "free"),  # default seeded news_sources feed
    None: ("no data", None),
}

# Status -> (dot glyph, color)
STATUS_STYLE = {
    "up": ("●", COLOR_BULLISH, "UP"),
    "degraded": ("●", COLOR_RETAIL, "DEGRADED"),
    "down": ("●", COLOR_BEARISH, "DOWN"),
    "not_configured": ("○", COLOR_NEUTRAL, "NOT CONFIGURED"),
    "not_applicable": ("○", COLOR_NEUTRAL, "N/A — REFERENCE ONLY"),
}

FIX_HINTS = {
    "congressional": "export QUIVER_API_KEY=... to enable live QuiverQuant data (falls back to a free, "
                      "often-blocked Senate EFD scrape otherwise)",
}

# Sector -> Polymarket keyword set, for TICKER DEEP-DIVE's "Related
# Polymarket Markets" section. Ticker-specific overrides win when present
# (e.g. STX/WDC/MU care about chip export controls specifically, not just
# "Technology" broadly); otherwise falls back by yfinance sector string,
# and finally to a generic macro default.
SECTOR_KEYWORDS = {
    "Technology": ["china", "taiwan", "chip", "chips", "export", "tariff", "semiconductor"],
    "Financial Services": ["fed", "rate", "recession", "bank", "treasury"],
    "Energy": ["oil", "opec", "tariff", "sanctions"],
    "Consumer Cyclical": ["tariff", "inflation", "recession", "consumer"],
    "Consumer Defensive": ["inflation", "tariff", "recession"],
    "Healthcare": ["fda", "drug pricing", "healthcare"],
    "Industrials": ["tariff", "china", "export", "supply chain"],
    "Communication Services": ["china", "regulation", "antitrust"],
    "Real Estate": ["fed", "rate", "recession"],
    "Utilities": ["fed", "rate", "inflation"],
    "Basic Materials": ["china", "tariff", "export"],
}

TICKER_KEYWORD_OVERRIDES = {
    "STX": ["china", "taiwan", "chip", "chips", "export", "storage"],
    "WDC": ["china", "taiwan", "chip", "chips", "export", "storage"],
    "MU": ["china", "taiwan", "chip", "chips", "export", "memory"],
}

NOISY_CATEGORIES = {"sports", "entertainment", "pop culture", "celebrities", "crypto memes", "gaming"}


def keywords_for_ticker(ticker, sector=None):
    if ticker in TICKER_KEYWORD_OVERRIDES:
        return TICKER_KEYWORD_OVERRIDES[ticker]
    return SECTOR_KEYWORDS.get(sector, MACRO_KEYWORDS)


def filter_by_keywords(df, keywords, exclude_noisy=True):
    """Case-insensitive substring match of `keywords` against df['question'].
    Also strips known-noisy categories (sports/entertainment/pop-culture)
    that might coincidentally contain a keyword, so the default MACRO
    SIGNALS view never surfaces novelty markets."""
    if df.empty or not keywords:
        return df.iloc[0:0]
    pattern = "|".join(re.escape(k) for k in keywords)
    mask = df["question"].fillna("").str.contains(pattern, case=False, regex=True)
    result = df[mask]
    if exclude_noisy and "category" in result.columns:
        result = result[~result["category"].fillna("").str.lower().isin(NOISY_CATEGORIES)]
    return result

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, .stButton button,
.stTextInput input, .stSelectbox, .stMultiSelect, div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"], .stDataFrame {{
    font-family: 'Space Mono', 'Courier New', monospace !important;
}}

.stApp {{
    background-color: {BG};
    color: {TEXT_PRIMARY};
}}
section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {BORDER};
}}
div.block-container {{
    padding-top: 1.1rem;
    max-width: 100%;
}}

h1, h2, h3, h4, h5 {{
    color: {TEXT_PRIMARY} !important;
    letter-spacing: 0.03em;
}}
p, span, label, .stCaption {{
    color: {TEXT_SECONDARY};
}}

/* header */
.smd-header {{
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px solid {BORDER}; padding-bottom: 12px; margin-bottom: 10px;
}}
.smd-title {{
    font-size: 26px; font-weight: 700; color: {ACCENT};
    text-shadow: 0 0 14px rgba(0,229,255,0.35);
    letter-spacing: 0.04em;
}}
.smd-subtitle {{
    font-size: 12px; color: {TEXT_MUTED}; letter-spacing: 0.08em; margin-top: 2px;
}}
.smd-updated {{
    font-size: 11px; color: {TEXT_MUTED}; text-align: right;
}}

/* system status strip */
.status-strip {{
    padding: 7px 4px; border-bottom: 1px solid {BORDER}; margin-bottom: 4px;
    display: flex; flex-wrap: wrap; align-items: center;
}}
.status-chip {{
    margin-right: 18px; font-size: 11px; color: {TEXT_SECONDARY}; white-space: nowrap;
}}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent; color: {TEXT_MUTED};
    font-size: 12px; letter-spacing: 0.06em; padding: 10px 14px;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* metrics */
div[data-testid="stMetric"] {{
    background-color: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 10px 14px;
}}
div[data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY}; font-size: 20px; }}
div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; font-size: 11px; letter-spacing: 0.06em; }}

/* buttons */
.stButton button {{
    background-color: {PANEL_ALT}; color: {ACCENT}; border: 1px solid {ACCENT};
    border-radius: 3px; letter-spacing: 0.05em; font-size: 12px;
}}
.stButton button:hover {{ background-color: rgba(0,229,255,0.12); color: {ACCENT}; }}

/* chips / badges */
.chip {{
    display: inline-block; border: 1px solid; border-radius: 3px;
    padding: 2px 9px; font-size: 10.5px; letter-spacing: 0.06em; font-weight: 700;
}}

/* signal cards */
.smcard {{
    background-color: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px;
    padding: 12px 14px; margin-bottom: 12px;
}}
.smcard-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
.smcard-ticker {{ font-size: 16px; font-weight: 700; color: {TEXT_PRIMARY}; letter-spacing: 0.03em; }}
.smcard-score {{ font-size: 28px; font-weight: 700; color: {ACCENT}; }}
.smcard-score-max {{ font-size: 13px; color: {TEXT_MUTED}; }}
.smcard-row {{
    display: flex; justify-content: space-between; font-size: 11px;
    color: {TEXT_SECONDARY}; padding: 2px 0; border-top: 1px solid rgba(255,255,255,0.04);
}}

.smd-section {{
    font-size: 13px; color: {ACCENT}; letter-spacing: 0.08em; margin: 18px 0 4px 0;
    border-left: 3px solid {ACCENT}; padding-left: 8px;
    display: flex; justify-content: space-between; align-items: baseline;
}}
.smd-note {{ font-size: 11px; color: {TEXT_MUTED}; }}

div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 4px; }}
hr {{ border-color: {BORDER}; }}
</style>
"""

st.set_page_config(page_title="SMART MONEY INTELLIGENCE", layout="wide", page_icon="📡")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Startup: log QUIVER_API_KEY detection once per server process (not once
# per rerun -- Streamlit re-execs this whole script on every interaction,
# so this is cache_resource-gated to fire exactly once at real app launch).
# --------------------------------------------------------------------------

@st.cache_resource
def _log_startup_status():
    if os.environ.get("QUIVER_API_KEY"):
        print("[startup] QUIVER_API_KEY detected -- congressional trades will use QuiverQuant (paid).")
    else:
        print("[startup] QUIVER_API_KEY not set -- congressional trades will use the free Senate EFD "
              "fallback (often blocked/unreliable). Set QUIVER_API_KEY to enable live data.")
    return True


_log_startup_status()


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------

_db_thread_local = threading.local()


def get_conn():
    """A connection PRIVATE to the current thread -- was previously a
    single @st.cache_resource-cached sqlite3.Connection shared globally
    across every session, tab, AND every RUNNERS Day Prediction fragment
    auto-refresh (st.fragment(run_every=...) executes on Streamlit's own
    background scheduling, genuinely concurrently with the main script
    thread, not just "later"). One Python sqlite3.Connection object is
    NOT safe for two threads to call .execute() on at the same instant
    -- check_same_thread=False only disables Python's own same-thread
    guard, it does not make the underlying connection thread-safe for
    true concurrent use. Confirmed live: the moment the "5 min" auto-
    refresh fragment actually started firing on its own timer (right
    after fixing a separate bug where that setting silently reverted to
    "On demand" on every reload, so this race was previously almost
    never triggered), a normal page load hit
    'sqlite3.InterfaceError: bad parameter or other API misuse' in the
    NEWS tab while the fragment's background thread was also live.

    threading.local() gives each thread (main script thread, and each
    fragment's own execution thread) its own real connection to the
    same WAL-mode file -- multiple separate connections to a WAL
    database ARE safe to use concurrently, which is the whole reason
    WAL mode exists; a single connection OBJECT shared across threads
    is a different, unsafe thing entirely. init_db()'s schema/migration
    checks are cheap and idempotent, so re-running them once per new
    thread (not once per rerun on the same thread) costs nothing
    noticeable."""
    if not hasattr(_db_thread_local, "conn"):
        init_db(DEFAULT_DB_PATH)
        _db_thread_local.conn = get_connection(DEFAULT_DB_PATH)
    return _db_thread_local.conn


# --------------------------------------------------------------------------
# Unsubscribe query-param handler (Part 4.2) -- checks for ?unsubscribe=
# <token> on EVERY load, before any tab content renders, and shows a
# dedicated confirmation/error page instead of the dashboard when present
# (st.stop() below short-circuits the rest of this script). Only actually
# reachable while this app is running somewhere with a real, stable
# public URL (see get_unsubscribe_url's own APP_PUBLIC_URL / mailto:
# fallback) -- a link clicked from an email obviously can't reach a
# purely local `streamlit run` process on someone's own machine.
# --------------------------------------------------------------------------

_unsubscribe_token = st.query_params.get("unsubscribe")
if _unsubscribe_token:
    # NOTE: st.set_page_config() already ran once, earlier in this script
    # (it must be the first Streamlit call and can only run once per
    # script execution) -- not repeated here.
    _unsubscribed_email = unsubscribe_by_token(get_conn(), _unsubscribe_token)
    st.markdown("### 📧 Smart Money Intelligence — Daily Digest")
    if _unsubscribed_email:
        st.success(f"✅ **{_unsubscribed_email}** has been unsubscribed from the daily digest. "
                   f"You will not receive further emails.")
        st.caption("Changed your mind? Re-subscribe from the SETTINGS → Newsletter panel, or contact "
                   "the sender directly.")
    else:
        st.error("⚠️ This unsubscribe link isn't valid (or has already been used). If you're still receiving "
                 "emails you don't want, reply to the digest directly and ask to be removed.")
    st.stop()


# --------------------------------------------------------------------------
# Manage-preferences query-param handler (2026-10-13, Part 1.3) -- checks
# for ?manage=<token> on every load, same tokenization/no-login pattern as
# the unsubscribe handler above. A subscriber (identified ONLY by their
# token, never requiring a dashboard account/login) can toggle which
# sections they receive and whether they receive the newsletter at all.
# --------------------------------------------------------------------------

_manage_token = st.query_params.get("manage")
if _manage_token:
    st.markdown("### 📧 Smart Money Intelligence — Manage Preferences")
    _sub = get_subscriber_by_token(get_conn(), _manage_token)
    if not _sub:
        st.error("⚠️ This preferences link isn't valid. If you're still receiving emails you don't want, "
                 "reply to the digest directly and ask to be removed.")
        st.stop()
    st.caption(f"Preferences for **{_sub['email']}**. No login needed — this link is unique to you.")
    _active = st.toggle("Receive the newsletter", value=not _sub["unsubscribed"], key="manage_active_toggle")
    st.markdown("**Sections**")
    _current_sections = set(get_subscriber_sections_included(_sub))
    _chosen_sections = []
    for _sec_key in DEFAULT_DIGEST_SECTIONS:
        if st.checkbox(DIGEST_SECTION_LABELS.get(_sec_key, _sec_key), value=_sec_key in _current_sections,
                       key=f"manage_section_{_sec_key}"):
            _chosen_sections.append(_sec_key)
    if st.button("💾 Save preferences", key="manage_save_btn"):
        set_subscriber_active(get_conn(), _manage_token, _active)
        set_subscriber_sections_included(get_conn(), _manage_token, _chosen_sections)
        st.success("✅ Preferences saved.")
    st.stop()


# --------------------------------------------------------------------------
# AUTH GATE (multi-user). Everything below this point requires a signed-in,
# registered, active Google account -- require_login() renders the setup /
# login / denied screen and st.stop()s otherwise, so NO tab, sidebar
# widget, background scan, or digest send below can run for an
# unauthenticated visitor. (The unsubscribe handler above is intentionally
# the one exception: an emailed unsubscribe link must work with no account.)
#
# ACTOR_EMAIL   = who is actually logged in (drives permissions).
# EFFECTIVE_EMAIL = whose data the UI shows/edits. Always == ACTOR_EMAIL for
#   a regular user; an admin may switch it via the ADMIN tab's "View as
#   user" selector (their permissions stay admin's -- only the data
#   context changes). Every per-user read/write below uses EFFECTIVE_EMAIL.
# --------------------------------------------------------------------------

CURRENT_USER = auth.require_login(get_conn())
ACTOR_EMAIL = CURRENT_USER["email"]
IS_ADMIN = CURRENT_USER["role"] == "admin"


def _resolve_effective_email():
    if IS_ADMIN:
        target = de.normalize_email(st.session_state.get("view_as_email"))
        # Only honor a target that still exists; a stale selection (user
        # deleted in another session) silently falls back to the admin's own.
        if target and target != ACTOR_EMAIL and de.get_user(get_conn(), target):
            return target
    return ACTOR_EMAIL


EFFECTIVE_EMAIL = _resolve_effective_email()


def _uk(name):
    """Namespace a per-user widget key / session_state key by whose data is
    on screen. Streamlit keeps widget state under the key and the browser
    re-sends it every rerun -- with a fixed key like "wl_textarea", switching
    "View as user" could leave the PREVIOUS user's text in the box (and Save
    would then write it onto the wrong user). A key that includes the user
    makes that impossible: each user's widgets are different widgets."""
    return f"{name}::{EFFECTIVE_EMAIL}"


# Session-state isolation. Streamlit widget values persist in
# st.session_state BY KEY, ignoring a changed `value=` default -- so when
# the data context changes (an admin's "View as user" switch, or a
# different account signing in within the same browser session), the
# previous user's toggles / text boxes / pinned tickers would silently
# carry over onto the new user's screen, and a Save click could write them
# onto that user. On any context change, drop every session key except the
# few that define identity/context itself.
_SESSION_KEEP_KEYS = {
    "_auth_email", "view_as_email", "_ctx_email", "_admin_view_as_select",
    # GLOBAL (not per-user) session caches: the once-a-day reconcile/macro-refresh marker and the
    # data-source health probes. Wiping these would re-run network work on every "View as" switch.
    "last_auto_reconcile_date", "health", "full_registry",
}
if st.session_state.get("_ctx_email") != EFFECTIVE_EMAIL:
    for _k in [k for k in st.session_state.keys() if k not in _SESSION_KEEP_KEYS]:
        del st.session_state[_k]
    st.session_state["_ctx_email"] = EFFECTIVE_EMAIL

watchlist = load_watchlist(get_conn(), EFFECTIVE_EMAIL)


@st.cache_data(ttl=30, show_spinner=False)
def q(sql, params=()):
    return pd.read_sql_query(sql, get_conn(), params=params)


def latest_divergence(tickers):
    if not tickers:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(tickers))
    return q(
        f"""
        SELECT d.* FROM divergence_scores d
        INNER JOIN (
            SELECT ticker, MAX(computed_date) AS max_date FROM divergence_scores
            WHERE ticker IN ({placeholders}) GROUP BY ticker
        ) latest ON d.ticker = latest.ticker AND d.computed_date = latest.max_date
        """,
        tuple(tickers),
    )


def last_refreshed():
    df = q("SELECT MAX(computed_at) AS t FROM divergence_scores")
    val = df["t"].iloc[0] if not df.empty else None
    return val or "never"


@st.cache_data(ttl=600, show_spinner=False)
def watchlist_fundamentals_table(tickers):
    """Bulk multi-ticker preview -- routes through cached_fundamentals
    (DB-cached info + delta-loaded price_df via fetch_price_history_delta)
    instead of the pure, uncached fetch_fundamentals. A full-watchlist
    sweep of uncached live calls -- 2 yfinance requests per ticker,
    .history() + .get_info() -- every time this 10-minute st.cache_data
    wrapper expired is exactly what triggered yfinance's rate limit; the
    underlying data no longer costs a full live pull per ticker even when
    this Streamlit-level cache expires or a new session starts fresh."""
    conn = get_conn()
    rows = []
    for t in tickers:
        f = cached_fundamentals(conn, t, max_age_hours=12)["data"]
        if f["price_df"].empty:
            continue
        snap, info = f["snapshot"], f["info"]
        rows.append(
            {
                "Ticker": t,
                "Price": snap["last_price"],
                "Chg %": snap["change_pct"],
                "RSI14": snap["rsi14"],
                "Trend": snap["trend"],
                "Sector": info.get("sector"),
                "Mkt Cap": info.get("marketCap"),
                "P/E": info.get("trailingPE"),
                "Beta": info.get("beta"),
            }
        )
        time.sleep(0.3)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# System health: probed once per session, cached in session_state so it
# doesn't re-run (and re-hit every source) on every unrelated rerun.
# --------------------------------------------------------------------------

def get_health(force=False):
    if force or "health" not in st.session_state:
        st.session_state["health"] = check_source_health()
    return st.session_state["health"]


def get_full_registry(force=False):
    if force or "full_registry" not in st.session_state:
        st.session_state["full_registry"] = check_full_source_registry()
    return st.session_state["full_registry"]


def congressional_empty_reason(health):
    """Explains *why* congressional data is empty, sourced from the same
    check_source_health() probe the SYSTEM STATUS panel uses, so every tab
    that shows an empty congressional section gives the real reason (e.g.
    an actual QuiverQuant auth error, or the Senate EFD fallback's own
    status) instead of a generic "no data" message -- never guessed
    independently per section."""
    info = (health or {}).get("congressional", {})
    status = info.get("status")
    fallback = info.get("fallback") or {}

    if status == "not_configured":
        base = "No data — QUIVER_API_KEY not set."
        if fallback.get("status") == "up":
            return base + " Using free Senate EFD fallback."
        fb_error = fallback.get("error", "unavailable")
        return f"{base} Senate EFD fallback also down ({fb_error})."
    if status == "down":
        detail = info.get("error", "unknown error")
        if fallback.get("status") == "up":
            return f"No data — quiverquant: {detail}. Senate EFD fallback is up but returned nothing."
        fb_error = fallback.get("error", "unavailable")
        return f"No data — quiverquant: {detail}; senate_efd fallback: {fb_error}."
    if status == "degraded":
        return f"No data — quiverquant degraded ({info.get('error', 'unknown error')})."
    return "No recent congressional trades captured."


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def human_money(v):
    if v is None or pd.isna(v):
        return "—"
    v = float(v)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(v) >= div:
            return f"${v / div:.2f}{unit}"
    return f"${v:,.0f}"


def md_safe(text):
    """Escapes '$' before markdown rendering. Streamlit's markdown renderer
    treats a run of text between two unescaped '$' as LaTeX math and drops
    the enclosed characters -- AI-generated prose that mentions two dollar
    amounts in one paragraph (e.g. "...at $61.66 after a move from $60.50
    ...") renders as garbled, space-eaten text ("61.66after a...60.50")
    without this. Applied to every AI-generated string before st.write /
    st.markdown; numeric values that must render exactly are shown via
    st.metric instead (immune to markdown parsing) rather than extracted
    from prose."""
    if not text:
        return text
    return str(text).replace("$", "\\$")


def html_safe_snippet(text):
    """HTML-escapes raw source text (e.g. a StockTwits/Reddit post body)
    before embedding it in a raw <div> via st.markdown(..., unsafe_allow_html=True).
    md_safe()'s backslash-escape trick doesn't work here: that escape is a
    CommonMark convention Streamlit's markdown parser resolves in prose
    context, but inside a raw HTML block the literal backslash survives
    and shows up in the rendered page. Swapping '$' for its HTML entity
    instead sidesteps the LaTeX-$ scanner (which runs over the markdown
    source before HTML entities decode) without leaving any stray
    characters visible."""
    # quote=False: only '&'/'<'/'>' need escaping in HTML text-node content;
    # escaping quotes too (html.escape's default) double-encodes -- Streamlit's
    # markdown-to-HTML pass already resolves entities like '&quot;' back into
    # a literal '"' by the time it reaches the browser, so escaping the quote
    # itself just leaves the raw entity text ('&quot;') visible on the page.
    escaped = html.escape(text or "", quote=False)
    escaped = escaped.replace("$", "&#36;")
    return escaped.replace("\n", "<br>")


def _is_substantive_post_text(text):
    """Filters out raw retail-sentiment posts that are just a run of
    cashtags with no real commentary (e.g. '$DRAL $KORU $SNDK $SOXL
    $WDC') -- pure noise in the raw post list, no actual signal, and
    visually crowds out the posts that do say something."""
    if not text:
        return False
    stripped = re.sub(r"\$[A-Za-z]+", "", text)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return len(stripped) >= 15


def _truncate_post_text(text, max_len=240):
    """Caps a raw post body at max_len chars so one long sprawling post
    (e.g. a multi-paragraph pump writeup) doesn't dominate the raw post
    list -- the AI Briefing's own synthesized bullets above already
    summarize the substance; this list is meant to be a scannable
    verification trail, not a second full-text feed."""
    text = text or ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "…"


def ai_brief_header(text, subtitle=None):
    """AI Briefing sub-section header, styled to match the app's existing
    .smd-section class (accent color, left border, letter-spacing) so
    headers read clearly larger/bolder than the body text beneath them,
    with consistent spacing between sections.

    `subtitle`, if given, renders as a small muted tag next to the header
    -- used to mark a section as e.g. 'AI SYNTHESIS' vs 'AUTOMATED' so the
    hard-computed numbers and the AI's interpretive narrative stay
    visually distinct (same discipline as render_technical_structure_panel's
    'Automated' label)."""
    tag = (
        f'<span style="font-size:10px;color:{TEXT_MUTED};font-weight:400;letter-spacing:0.04em;">'
        f' &nbsp;{subtitle}</span>'
    ) if subtitle else ""
    st.markdown(f'<div class="smd-section" style="margin-top:22px;">{text}{tag}</div>', unsafe_allow_html=True)


def ai_section_or_fallback(value):
    """AI-briefing prose sections should never render as a bare '—' --
    generate_deep_analysis's prompt already asks Claude to say so
    explicitly when a category is empty, but this is the Python-side
    backstop for a blank/None response."""
    text = (value or "").strip()
    return md_safe(text) if text else "_No data available for this section._"


def fmt_num(v, digits=2):
    return "—" if v is None or pd.isna(v) else f"{v:.{digits}f}"


def fmt_pct(v, digits=1):
    return "—" if v is None or pd.isna(v) else f"{v * 100:.{digits}f}%"


def fmt_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return pd.Timestamp(v).strftime("%b %d, %Y")
    except (TypeError, ValueError):
        return str(v)


def time_ago(dt):
    if dt is None or (isinstance(dt, float) and pd.isna(dt)):
        return "—"
    try:
        dt = pd.Timestamp(dt)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(dt):
        return "—"
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    seconds = (pd.Timestamp.now(tz="UTC") - dt).total_seconds()
    if seconds < 0:
        return dt.strftime("%b %d, %Y")
    if seconds < 3600:
        mins = max(1, int(seconds // 60))
        return "just now" if seconds < 60 else f"{mins} min{'s' if mins != 1 else ''} ago"
    if seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(seconds // 86400)
    if days < 30:
        return f"{days} day{'s' if days != 1 else ''} ago"
    return dt.strftime("%b %d, %Y")


def badge_html(label, score=None):
    color = LABEL_COLORS.get(label, COLOR_NEUTRAL)
    score_txt = f" &middot; {score:.0f}" if score is not None else ""
    return (
        f'<span class="chip" style="border-color:{color};color:{color};">'
        f'{label.replace("_", " ")}{score_txt}</span>'
    )


def source_badge(meta):
    """meta: an envelope dict with 'source'/'cache_hit'/'fetched_at' (or
    None/empty). Renders: SOURCE: X (FREE) · cached 3h ago."""
    source = (meta or {}).get("source")
    label, kind = SOURCE_LABELS.get(source, (source or "no data", None))
    kind_color = COLOR_BULLISH if kind == "free" else COLOR_RETAIL if kind == "paid" else TEXT_MUTED
    kind_txt = f" ({kind.upper()})" if kind else ""

    age_txt = ""
    if meta and meta.get("fetched_at"):
        cache_word = "cached" if meta.get("cache_hit") else "fetched"
        age_txt = f" &middot; {cache_word} {time_ago(meta['fetched_at'])}"

    return (
        f'<span class="smd-note">SOURCE: <span style="color:{kind_color};">{label}{kind_txt}</span>{age_txt}</span>'
    )


# --------------------------------------------------------------------------
# Options-flow interpretive badges (Parts 4-6) -- color-codes the (label,
# note) pairs data_engine's classify_*() functions return. Delta/IV colors
# are the explicit red/yellow/green/blue scheme from the spec; the Part 6
# badges (vol/OI, DTE, theta%, spread%) have no mandated scheme, so they
# use the same red/yellow/green escalation language by tier position
# (first/least-notable tier reads neutral, escalating to red for the most
# extreme tier) for a consistent look.
# --------------------------------------------------------------------------

DELTA_BADGE_COLORS = {
    "Deep OTM lottery ticket": COLOR_BEARISH,
    "Lottery ticket": COLOR_BEARISH,
    "Speculative": COLOR_RETAIL,
    "Balanced": COLOR_BULLISH,
    "Conservative/ITM": COLOR_INSTITUTIONAL,
    "Deep ITM / stock replacement": COLOR_INSTITUTIONAL,
}

IV_BADGE_COLORS = {
    "Low IV": COLOR_BULLISH,
    "Moderate/Balanced IV": COLOR_NEUTRAL,
    "Elevated IV": COLOR_RETAIL,
    "Extreme IV": COLOR_BEARISH,
}

_TIER_ESCALATION_COLORS = [COLOR_NEUTRAL, COLOR_RETAIL, COLOR_BEARISH]


def _tier_escalation_color(label, profiles):
    """label: the string a classify_*() call returned. profiles: the
    matching *_PROFILES list it was classified against -- used only to
    find that label's tier index (position = severity), not its numeric
    bounds."""
    for i, (_, _, lbl, _note) in enumerate(profiles):
        if lbl == label:
            return _TIER_ESCALATION_COLORS[min(i, len(_TIER_ESCALATION_COLORS) - 1)]
    return COLOR_NEUTRAL


def interp_badge_html(label, color, note=None):
    """Generic interpretive badge -- reuses the .chip CSS class (same visual
    language as badge_html's divergence-label chips) but takes an explicit
    color instead of a LABEL_COLORS lookup, plus an optional hover tooltip
    (the plain-English 'why' from a classify_*() call) via the title attr.

    `not label` alone doesn't catch a NaN float slipping in from an
    .iterrows()-sourced badge column (confirmed live: NaN is truthy in
    Python, so it passed this check and crashed html.escape() -- "'float'
    object has no attribute 'replace'" -- since html.escape assumes a str
    and just calls .replace() with no type check). isinstance guards both
    label and note explicitly, regardless of what a caller passes."""
    if not isinstance(label, str) or not label:
        return ""
    note_txt = note if isinstance(note, str) else None
    title_attr = f' title="{html.escape(note_txt)}"' if note_txt else ""
    return f'<span class="chip" style="border-color:{color};color:{color};"{title_attr}>{html.escape(label)}</span>'


# --------------------------------------------------------------------------
# Chart helpers
# --------------------------------------------------------------------------

def render_macro_event_expander(events):
    """Part 3 -- collapsed, on-demand full detail for whatever events a
    chart's overlay actually plotted (already impact-filtered for wide
    ranges by get_macro_events_in_range's overlay_impact_filter="auto").
    No-op if there are none in this chart's range."""
    if not events:
        return
    with st.expander(f"📅 {len(events)} macro event(s) in this range", expanded=False):
        for ev in events:
            color = _MACRO_IMPACT_COLORS.get(ev["impact_level"], TEXT_SECONDARY)
            time_label = f" {_format_et_time_12h(ev['scheduled_time_et'])}" if ev.get("scheduled_time_et") else ""
            st.markdown(
                f"<span style='color:{color};'>●</span> **{ev['event_name']}** — "
                f"{ev['scheduled_date']}{time_label} ({ev['impact_level']} impact)",
                unsafe_allow_html=True,
            )


def render_price_chart(df, ticker, mode="line", height=380):
    fig = go.Figure()
    if mode == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color=COLOR_BULLISH, decreasing_line_color=COLOR_BEARISH,
                increasing_fillcolor=COLOR_BULLISH, decreasing_fillcolor=COLOR_BEARISH,
                name=ticker,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close",
                       line=dict(color=ACCENT, width=2))
        )
    if "SMA50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA50"], mode="lines", name="SMA 50",
                       line=dict(color=COLOR_INSTITUTIONAL, width=1.4, dash="dot"))
        )
    if "SMA200" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA200"], mode="lines", name="SMA 200",
                       line=dict(color=COLOR_RETAIL, width=1.4, dash="dot"))
        )
    fig.update_layout(xaxis_rangeslider_visible=False)
    macro_events = []
    if not df.empty:
        macro_events = get_macro_events_in_range(get_conn(), df.index.min(), df.index.max(),
                                                  overlay_impact_filter="auto")
        lo = float(df["Low"].min()) if mode == "candlestick" else float(df["Close"].min())
        hi = float(df["High"].max()) if mode == "candlestick" else float(df["Close"].max())
        y_bottom = lo - (hi - lo) * 0.03
        add_macro_event_markers(fig, macro_events, y_bottom=y_bottom)
    apply_theme(
        fig, height=height,
        title=dict(text=f"{ticker} &mdash; Price", font=dict(color=TEXT_PRIMARY, size=14)),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig, macro_events


def render_technical_structure_panel(levels):
    """Renders the automated TECHNICAL STRUCTURE panel from a
    detect_technical_levels()-shaped dict -- pure display of already-
    computed numbers, no synthesis of any kind. Explicitly labeled
    'Automated' so it reads as visually distinct from the AI Briefing's
    narrative paragraph (ai_brief.get('technical_catalyst_setup')), which
    is AI-synthesized interpretation of the same numbers plus news
    context -- same hard-numbers-vs-narrative separation as the rest of
    the briefing's anti-hallucination discipline."""
    st.markdown(
        '<div class="smd-section">TECHNICAL STRUCTURE'
        f'<span style="font-size:10px;color:{TEXT_MUTED};font-weight:400;letter-spacing:0.04em;">'
        ' &nbsp;AUTOMATED — DERIVED FROM PRICE/VOLUME DATA</span></div>',
        unsafe_allow_html=True,
    )

    trend = levels.get("trend_structure")
    if trend in (None, "unknown"):
        st.caption(levels.get("trend_explanation") or "Not enough price history to determine structure.")
        return

    trend_color = {
        "uptrend": COLOR_BULLISH, "breaking out": COLOR_BULLISH,
        "downtrend": COLOR_BEARISH, "breaking down": COLOR_BEARISH,
        "consolidating": TEXT_SECONDARY,
    }.get(trend, TEXT_SECONDARY)
    st.markdown(
        f'<div style="font-size:15px;font-weight:700;color:{trend_color};text-transform:uppercase;'
        f'letter-spacing:0.05em;">{html.escape(trend)}</div>'
        f'<div style="font-size:12px;color:{TEXT_SECONDARY};margin-top:2px;margin-bottom:10px;">'
        f'{html.escape(levels.get("trend_explanation") or "")}</div>',
        unsafe_allow_html=True,
    )

    kcol1, kcol2 = st.columns(2)
    with kcol1:
        below = levels.get("key_level_below")
        st.markdown(md_safe(metric_chip_html(
            "KEY SUPPORT BELOW", f"${below:,.2f}" if below is not None else "—", color=COLOR_BULLISH,
        )), unsafe_allow_html=True)
        if below is not None and levels.get("breakdown_target") is not None:
            st.caption(f"Break below {below:,.2f} → next measured-move target ~{levels['breakdown_target']:,.2f}")
    with kcol2:
        above = levels.get("key_level_above")
        st.markdown(md_safe(metric_chip_html(
            "KEY RESISTANCE ABOVE", f"${above:,.2f}" if above is not None else "—", color=COLOR_BEARISH,
        )), unsafe_allow_html=True)
        if above is not None and levels.get("breakout_target") is not None:
            st.caption(f"Break above {above:,.2f} → next measured-move target ~{levels['breakout_target']:,.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(levels.get("current_zone") or "")

    chips = sorted(
        [(lv["level"], lv["touches"], COLOR_BULLISH, "S") for lv in levels.get("support_levels", [])]
        + [(lv["level"], lv["touches"], COLOR_BEARISH, "R") for lv in levels.get("resistance_levels", [])],
        key=lambda c: c[0],
    )
    if chips:
        chip_html = "".join(
            f'<span class="chip" style="border-color:{color};color:{color};margin:2px 5px 2px 0;">'
            f'{tag} {level:,.2f} &middot; {touches}x</span>'
            for level, touches, color, tag in chips
        )
        st.markdown(chip_html, unsafe_allow_html=True)
    else:
        st.caption("No confirmed (2+ touch) support/resistance levels detected in this window.")


@st.cache_data(ttl=300, show_spinner=False)
def _cached_session_price_summary(ticker):
    """5-minute client-side cache (st.cache_data, not the DB-backed
    cached_* layer) around fetch_session_price_summary -- a single cheap
    5d/5m yfinance pull that doesn't justify a full DB table + fetch_log
    migration, but still shouldn't re-fetch on every Streamlit rerun."""
    return fetch_session_price_summary(ticker)


def render_session_price_summary(ticker):
    """Compact stats row above the price chart, shown regardless of which
    chart interval/lookback is selected: CURRENT PRICE (the most recent
    trade -- live during market hours, last close when shut, colored
    green/red off the change vs. the prior session's close and drawn
    larger than the other chips since it's the single most important
    number in the row), the current session's Open and running VWAP (real
    cumulative price*volume / cumulative volume, not a simple average),
    plus the prior session's Open/Close. Labels switch from 'Today'/
    'Yesterday' to 'Session'/'Prior Session' whenever the most recent
    available session isn't actually today (market closed, weekend,
    holiday) -- see fetch_session_price_summary's is_today flag."""
    summary = _cached_session_price_summary(ticker)
    if not summary:
        return

    is_today = summary["is_today"]
    open_label = "TODAY'S OPEN" if is_today else "SESSION OPEN"
    vwap_label = "TODAY'S VWAP" if is_today else "SESSION VWAP"
    prior_open_label = "YESTERDAY'S OPEN" if is_today else "PRIOR SESSION OPEN"
    prior_close_label = "YESTERDAY'S CLOSE" if is_today else "PRIOR SESSION CLOSE"

    def _fmt(v):
        return f"${v:,.2f}" if v is not None else "—"

    change_pct = summary.get("current_price_change_pct")
    price_color = TEXT_PRIMARY
    change_str = None
    if change_pct is not None:
        price_color = COLOR_BULLISH if change_pct >= 0 else COLOR_BEARISH
        change_str = f"{change_pct:+.2f}%"

    scol0, scol1, scol2, scol3, scol4 = st.columns([1.3, 1, 1, 1, 1])
    with scol0:
        st.markdown(
            metric_chip_html(
                "CURRENT PRICE", _fmt(summary.get("current_price")),
                color=price_color, value_size=28, sub_value=change_str,
            ),
            unsafe_allow_html=True,
        )
    with scol1:
        st.markdown(metric_chip_html(open_label, _fmt(summary["session_open"])), unsafe_allow_html=True)
    with scol2:
        st.markdown(metric_chip_html(vwap_label, _fmt(summary["session_vwap"]), color=ACCENT),
                    unsafe_allow_html=True)
    with scol3:
        st.markdown(metric_chip_html(prior_open_label, _fmt(summary["prior_session_open"])),
                    unsafe_allow_html=True)
    with scol4:
        st.markdown(metric_chip_html(prior_close_label, _fmt(summary["prior_session_close"])),
                    unsafe_allow_html=True)

    if not is_today:
        st.caption(
            f"Market's most recent session is {summary['session_date']}, not today — "
            f"showing that session's stats instead."
        )


# Sensible default Lookback per candle Interval -- applied via the
# Interval selectbox's on_change so picking "5m" defaults to "show me the
# most recent session" (1d) rather than silently carrying over whatever
# Lookback was last selected (which used to mean "5m" always meant 5 days
# of 5-minute bars, not "today"). The user can still widen it manually
# right after -- this only sets the default, it doesn't lock anything.
_INTERVAL_DEFAULT_LOOKBACK = {"5m": "1d", "15m": "1d", "1h": "1mo", "1d": DEFAULT_LOOKBACK}


def _chart_defaults():
    """The effective user's saved timeframe/chart defaults (see SETTINGS ->
    CHART DEFAULTS); app-wide defaults for anyone who hasn't chosen."""
    return load_chart_defaults(get_conn(), EFFECTIVE_EMAIL)


def render_timeframe_chart_section(ticker, key_prefix, mode="line", show_technical_structure=True):
    """Shared chart section, used by both the FUNDAMENTALS tab and TICKER
    DEEP-DIVE's price chart: two independent controls -- candle Interval
    (granularity: 5m/15m/1h/1d) and Lookback (visible range: 1d/5d/1mo/
    3mo/6mo/1y) -- rather than one dropdown conflating both, matching how
    TradingView actually separates the two. Also a refresh-rate selector
    (on-demand / 1 min / 5 min via st.fragment), the price+volume chart
    (with support/resistance overlay lines when show_technical_structure),
    RSI/MACD framed as 'intraday setup' vs 'swing setup', and the
    automated TECHNICAL STRUCTURE panel.

    Technical levels are computed fresh on every call -- never wrapped in
    st.cache_data -- so switching interval/lookback always recomputes
    rather than showing a stale prior selection's levels."""
    interval_key = _uk(f"{key_prefix}_interval")
    lookback_key = _uk(f"{key_prefix}_lookback")

    def _on_interval_change():
        st.session_state[lookback_key] = _INTERVAL_DEFAULT_LOOKBACK.get(
            st.session_state.get(interval_key), DEFAULT_LOOKBACK,
        )

    tcol1, tcol2, tcol3 = st.columns([1.2, 1.2, 1])
    with tcol1:
        interval = st.selectbox(
            "Interval", INTERVAL_OPTIONS, index=INTERVAL_OPTIONS.index(_chart_defaults()["interval"]),
            key=interval_key, on_change=_on_interval_change,
        )
    with tcol2:
        lookback = st.selectbox(
            "Lookback", LOOKBACK_OPTIONS, index=LOOKBACK_OPTIONS.index(_chart_defaults()["lookback"]),
            key=lookback_key,
        )
    with tcol3:
        refresh_choice = st.selectbox(
            "Refresh", CHART_REFRESH_OPTIONS, key=_uk(f"{key_prefix}_refresh"),
            index=CHART_REFRESH_OPTIONS.index(_chart_defaults()["refresh"]),
        )

    def _render():
        render_session_price_summary(ticker)
        hist, cfg = fetch_price_history(ticker, interval, lookback, conn=get_conn())
        if hist.empty:
            st.warning(f"No price data available for {ticker} at this interval/lookback.")
            return
        if cfg.get("fallback_from"):
            st.caption(
                f"'{cfg['fallback_from']}' isn't available for {ticker} right now "
                f"(yfinance lookback limit) — showing {cfg['interval']}/{cfg['lookback']} instead.",
            )
        if cfg.get("lookback_note"):
            st.caption(cfg["lookback_note"])
        short_w, long_w = cfg["sma_windows"]
        levels = _compute_technical_levels(hist, cfg, ticker) if show_technical_structure else None
        cache_suffix = f"{ticker}_{interval}_{lookback}"
        pv_fig, pv_macro_events = render_price_volume_chart(
            hist, ticker, mode=mode,
            short_label=f"SMA {short_w}", long_label=f"SMA {long_w}", levels=levels, conn=get_conn(),
        )
        st.plotly_chart(pv_fig, width='stretch', key=f"{key_prefix}_pv_{cache_suffix}")
        render_macro_event_expander(pv_macro_events)
        horizon_label = "Intraday setup" if cfg["horizon"] == "intraday" else "Swing setup"
        st.caption(horizon_label)
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.plotly_chart(render_rsi_chart(hist), width='stretch',
                            key=f"{key_prefix}_rsi_{cache_suffix}")
        with rcol2:
            st.plotly_chart(render_macd_chart(hist), width='stretch',
                            key=f"{key_prefix}_macd_{cache_suffix}")

        if show_technical_structure and levels:
            render_technical_structure_panel(levels)

    if refresh_choice == "On demand":
        _render()
    else:
        interval_s = 60 if refresh_choice == "1 min" else 300

        @st.fragment(run_every=interval_s)
        def _frag():
            _render()

        _frag()


def render_rsi_chart(df, height=240):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI 14",
                              line=dict(color=ACCENT, width=2)))
    fig.add_hline(y=70, line=dict(color=COLOR_BEARISH, width=1, dash="dash"),
                  annotation_text="70", annotation_font_color=TEXT_MUTED)
    fig.add_hline(y=30, line=dict(color=COLOR_BULLISH, width=1, dash="dash"),
                  annotation_text="30", annotation_font_color=TEXT_MUTED)
    fig.update_yaxes(range=[0, 100])
    apply_theme(fig, height=height, title=dict(text="RSI (14)", font=dict(color=TEXT_PRIMARY, size=13)),
                showlegend=False)
    return fig


def render_macd_chart(df, height=240):
    hist = df["MACD_hist"].fillna(0)
    colors = [COLOR_BULLISH if v >= 0 else COLOR_BEARISH for v in hist]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=hist, marker_color=colors, name="Histogram"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD",
                              line=dict(color=ACCENT, width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="Signal",
                              line=dict(color=COLOR_RETAIL, width=1.5)))
    apply_theme(fig, height=height, title=dict(text="MACD (12,26,9)", font=dict(color=TEXT_PRIMARY, size=13)),
                legend=dict(orientation="h", y=1.18, bgcolor="rgba(0,0,0,0)"))
    return fig


def render_gauge(pct, ticker, is_proxy=False, height=220):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(pct, 1),
            number={"suffix": "%", "font": {"color": ACCENT, "size": 26}},
            gauge={
                "axis": {"range": [0, 70], "tickcolor": TEXT_MUTED, "tickfont": {"color": TEXT_MUTED, "size": 9}},
                "bar": {"color": ACCENT, "thickness": 0.35},
                "bgcolor": PANEL_ALT,
                "borderwidth": 1,
                "bordercolor": BORDER,
                "steps": [
                    {"range": [0, 35], "color": "#0a1420"},
                    {"range": [35, 45], "color": "#132433"},
                    {"range": [45, 70], "color": "#241418"},
                ],
                "threshold": {"line": {"color": COLOR_BEARISH, "width": 3}, "thickness": 0.8, "value": 45},
            },
            title={"text": ticker + (" (proxy)" if is_proxy else ""),
                   "font": {"color": TEXT_PRIMARY, "size": 13}},
        )
    )
    apply_theme(fig, height=height, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def render_divergence_map(df, tickers, height=420):
    fig = go.Figure()
    for label in DIVERGENCE_LABELS:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=sub["ticker"], y=sub["score"], name=label.replace("_", " "),
                marker_color=LABEL_COLORS[label],
                text=[f"{v:.0f}" for v in sub["score"]],
                textposition="outside",
                textfont=dict(color=TEXT_SECONDARY, size=10),
            )
        )
    fig.update_xaxes(categoryorder="array", categoryarray=tickers)
    fig.update_yaxes(range=[0, 108], title=dict(text="Conviction score", font=dict(color=TEXT_MUTED)))
    apply_theme(fig, height=height, legend=dict(orientation="h", y=1.14, bgcolor="rgba(0,0,0,0)"))
    return fig


def render_signal_card(row):
    label = row["label"]
    color = LABEL_COLORS.get(label, COLOR_NEUTRAL)
    try:
        comps = json.loads(row["components_json"]) if row["components_json"] else {}
    except (TypeError, ValueError):
        comps = {}
    return f"""
    <div class="smcard" style="border-left:3px solid {color};">
      <div class="smcard-head">
        <span class="smcard-ticker">{row['ticker']}</span>
        <span class="chip" style="border-color:{color};color:{color};">{label.replace('_', ' ')}</span>
      </div>
      <div class="smcard-score">{row['score']:.0f}<span class="smcard-score-max">/100</span></div>
      <div class="smcard-row"><span>Smart signal</span><span>{comps.get('smart_signal', '—')}</span></div>
      <div class="smcard-row"><span>Retail signal</span><span>{comps.get('retail_signal', '—')}</span></div>
      <div class="smcard-row"><span>Institutional</span><span>{comps.get('institutional_magnitude', '—')}</span></div>
    </div>
    """


def render_earnings_probability_chart(prob_down, prob_flat, prob_up, magnitude_pct, current_price, height=130):
    """EARNINGS SIMULATOR Part 7 -- a horizontal range showing current
    price and shaded Down/Flat/Up zones sized by their real probability
    weight, reusing render_price_target_fan's color language (COLOR_
    BEARISH/NEUTRAL/BULLISH) and label style (colored badge with the
    dollar level) rather than the fan chart's time-forward shape, which
    doesn't fit a 3-way probability split."""
    fig = go.Figure()
    if prob_down is None or prob_flat is None or prob_up is None:
        apply_theme(fig, height=height)
        return fig
    down_price = current_price * (1 - magnitude_pct / 100) if current_price and magnitude_pct is not None else None
    up_price = current_price * (1 + magnitude_pct / 100) if current_price and magnitude_pct is not None else None
    segments = [
        ("Down", prob_down, COLOR_BEARISH, down_price), ("Flat", prob_flat, COLOR_NEUTRAL, current_price),
        ("Up", prob_up, COLOR_BULLISH, up_price),
    ]
    for name, p, color, level in segments:
        label = f"<b>{name} {p:.0%}</b>" + (f"<br>${level:,.2f}" if level is not None else "")
        fig.add_trace(go.Bar(
            x=[p], y=["Outcome"], orientation="h", name=name, marker_color=color,
            text=label, textposition="inside", insidetextanchor="middle", textfont=dict(color=PANEL, size=11),
            hovertemplate=f"{name}: {p:.0%}" + (f" (${level:,.2f})" if level is not None else "") + "<extra></extra>",
        ))
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False)
    return apply_theme(fig, height=height, margin=dict(l=10, r=10, t=10, b=10))


def _price_pct_ticks(spot, lo, hi, n_ticks=7):
    """Evenly spaced (price, '% move from current price') tick pairs
    across [lo, hi] -- shared by the P/L chart's x-axis and used as the
    template for the heatmap's y-axis labels, so 'current price is the
    reference frame' means the same thing in both places (Parts 2.1/3.2
    of the chart-rebuild fix)."""
    if not spot:
        return [], []
    step = (hi - lo) / (n_ticks - 1)
    vals = [lo + i * step for i in range(n_ticks)]
    # round() then + 0.0 normalizes a real "-0.0" float (e.g. from a price
    # a hair below spot rounding to 0.0%) so it prints "+0.0%", not "-0.0%".
    text = [f"${v:,.2f}<br>({round((v / spot - 1) * 100, 1) + 0.0:+.1f}%)" for v in vals]
    return vals, text


def render_pl_curve_chart(pl_curve, pl_curve_earnings, spot, height=340):
    """Rebuilt P/L chart (fixes: axis labels never require mental math,
    current price is prominent, TWO curves shown, and the three fixed-
    convention scenario points are marked on the earnings-date curve --
    see the P/L-chart-rebuild request):

    1. X-axis ticks show BOTH price and % move from current price
       ("$22.00 (+5.8%)"), not a bare dollar figure.
    2. Current price gets a solid, thick, clearly-labeled line -- not a
       thin dotted one indistinguishable from strike/breakeven.
    3. TWO curves: the earnings-date curve (pl_curve_earnings -- Black-
       Scholes with post-earnings-crush IV and time remaining to
       expiration, the one that actually matters for a trade evaluated
       near the event) is the PRIMARY filled line; the expiration curve
       (pl_curve -- the old intrinsic-value-only payoff) is a secondary,
       thinner reference line. Both are legended.
    4. The three canonical Down/Flat/Up scenario points (same fixed
       ~-6%/+1.5%/+6.5% convention as the SCENARIO BREAKDOWN table) are
       marked directly on the earnings-date curve with labeled markers.
    """
    if not pl_curve and not pl_curve_earnings:
        return None
    fig = go.Figure()

    # Expiration curve -- secondary reference line, thinner and dashed so
    # it doesn't visually compete with the earnings-date curve.
    if pl_curve:
        fig.add_trace(go.Scatter(
            x=pl_curve["prices"], y=pl_curve["pl_dollar"], mode="lines",
            line=dict(color=TEXT_SECONDARY, width=1.5, dash="dot"), name="At expiration",
            hovertemplate="Price $%{x:.2f}<br>P/L at expiration: $%{y:,.2f}<extra></extra>",
        ))

    # Earnings-date curve -- PRIMARY: filled, bold, this is the one that
    # matters for a trade evaluated near the event, not held to expiry.
    if pl_curve_earnings:
        prices_e, pl_e = pl_curve_earnings["prices"], pl_curve_earnings["pl_dollar"]
        fig.add_trace(go.Scatter(
            x=prices_e, y=[max(v, 0) for v in pl_e], mode="lines", line=dict(width=0), fill="tozeroy",
            fillcolor="rgba(12,163,12,0.18)", showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=prices_e, y=[min(v, 0) for v in pl_e], mode="lines", line=dict(width=0), fill="tozeroy",
            fillcolor="rgba(230,103,103,0.18)", showlegend=False, hoverinfo="skip",
        ))
        post_iv = pl_curve_earnings.get("post_earnings_iv_pct")
        fig.add_trace(go.Scatter(
            x=prices_e, y=pl_e, mode="lines", line=dict(color=TEXT_PRIMARY, width=2.5),
            name=f"On earnings date (~{post_iv:.0f}% IV)" if post_iv is not None else "On earnings date",
            hovertemplate="Price $%{x:.2f}<br>P/L on earnings date: $%{y:,.2f}<extra></extra>",
        ))

        # Part 2.4 -- the three fixed-convention scenario points, marked
        # directly on the earnings-date curve, matching the SCENARIO
        # BREAKDOWN table's own convention (not the raw IV-implied move).
        markers = pl_curve_earnings.get("scenario_markers") or []
        marker_color = {"Down": COLOR_BEARISH, "Flat": COLOR_NEUTRAL, "Up": COLOR_BULLISH}
        if markers:
            fig.add_trace(go.Scatter(
                x=[m["price"] for m in markers], y=[m["pl_dollar"] for m in markers], mode="markers+text",
                marker=dict(size=11, color=[marker_color.get(m["label"], ACCENT) for m in markers],
                            line=dict(color=PANEL, width=1.5), symbol="diamond"),
                text=[f"{m['label']} {m['move_pct']:+.1f}%" for m in markers],
                textposition="top center", textfont=dict(size=10, color=TEXT_PRIMARY),
                name="Scenario (fixed convention)",
                hovertemplate="%{text}<br>Price $%{x:.2f}<br>P/L $%{y:,.2f}<extra></extra>",
            ))

    fig.add_hline(y=0, line=dict(color=TEXT_MUTED, width=1))

    # Current price -- solid, thick, unambiguous; no longer a thin dotted
    # line easily confused with strike/breakeven.
    if spot is not None:
        fig.add_vline(x=spot, line=dict(color=ACCENT, width=2.5),
                      annotation_text=f"Current: ${spot:.2f}", annotation_position="top",
                      annotation_font=dict(size=12, color=ACCENT))

    strike = (pl_curve or pl_curve_earnings or {}).get("strike")
    if strike is not None:
        fig.add_vline(x=strike, line=dict(color=TEXT_SECONDARY, dash="dot", width=1.5),
                      annotation_text=f"Strike: ${strike:g}", annotation_position="top left")
    be = (pl_curve or {}).get("breakeven")
    if be is not None:
        fig.add_vline(x=be, line=dict(color=COLOR_RETAIL, dash="dash", width=1.5),
                      annotation_text=f"Breakeven: ${be:.2f}", annotation_position="bottom")

    all_prices = (pl_curve or {}).get("prices") or (pl_curve_earnings or {}).get("prices") or []
    if all_prices and spot:
        tickvals, ticktext = _price_pct_ticks(spot, min(all_prices), max(all_prices))
        fig.update_xaxes(title_text="Underlying stock price ($) — % move from current price shown below",
                          tickvals=tickvals, ticktext=ticktext)
    else:
        fig.update_xaxes(title_text="Underlying stock price ($)")
    fig.update_yaxes(title_text="P/L ($)")
    return apply_theme(fig, height=height, legend=dict(orientation="h", y=-0.32, bgcolor="rgba(0,0,0,0)"))


def render_pl_heatmap_chart(heatmap, height=340):
    """Rebuilt price(Y) x date(X) grid (fixes: current price and % move
    are now the primary y-axis reference frame, and the color scale no
    longer washes out to near-invisible in the mid-range -- see the
    heatmap-rebuild request). heatmap comes from data_engine.compute_
    price_date_heatmap, which already applies time decay per date and IV
    crush for dates on/after earnings.

    Y-axis labels show BOTH price and % move from current price
    ("$22.00 (+5.8%)"), and the current-price-nearest row gets its own
    "◀ CURRENT" marker plus a highlighted horizontal guide line, distinct
    from the existing "◀ STRIKE" marker (both can appear on the same row
    if strike and spot happen to be close). Categorical axes (both of
    these are, by price/date label) don't reliably support per-tick font-
    weight changes, so a visible marker glyph -- now backed by an actual
    guide line for the current-price row -- is the robust way to make
    "this is where we are now" obvious without relying on the caption
    below.

    Color scale is now a 5-stop diverging palette (was a 3-stop one whose
    midpoint, #1c2a36, was close enough to the panel background color
    that small/mid-range P/L cells visually disappeared) with zmin/zmax
    capped at the 90th percentile of |P/L%| (was the raw max, so a single
    extreme corner cell could compress every other cell toward that same
    washed-out midpoint)."""
    if not heatmap:
        return None
    z = [[cell["pl_pct"] for cell in row] for row in heatmap["grid"]]
    text = [[f"{cell['pl_pct']:+.0f}%" for cell in row] for row in heatmap["grid"]]
    flat_abs = sorted(abs(v) for row in z for v in row)
    if flat_abs:
        cap_idx = min(int(len(flat_abs) * 0.90), len(flat_abs) - 1)
        max_abs = flat_abs[cap_idx] or flat_abs[-1] or 1
    else:
        max_abs = 1

    spot = heatmap.get("spot")
    strike = heatmap.get("strike")
    price_levels = heatmap["price_levels"]
    strike_idx = (
        min(range(len(price_levels)), key=lambda i: abs(price_levels[i] - strike))
        if strike is not None else None
    )
    current_idx = (
        min(range(len(price_levels)), key=lambda i: abs(price_levels[i] - spot))
        if spot else None
    )
    y_labels = []
    for i, p in enumerate(price_levels):
        pct_txt = f" ({round((p / spot - 1) * 100, 1) + 0.0:+.1f}%)" if spot else ""
        marker = ""
        if i == current_idx:
            marker += " ◀ CURRENT"
        if i == strike_idx:
            marker += " ◀ STRIKE"
        y_labels.append(f"${p:,.2f}{pct_txt}{marker}")

    earnings_date = heatmap.get("earnings_date_in_range")
    x_labels = [f"📅 {d} (earnings)" if d == earnings_date else d for d in heatmap["dates"]]

    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=10, color=TEXT_PRIMARY),
        colorscale=[
            [0, COLOR_BEARISH], [0.25, "#8a4a4a"], [0.5, "#3a4452"], [0.75, "#3f7a3f"], [1, COLOR_BULLISH],
        ],
        zmin=-max_abs, zmax=max_abs, zmid=0, colorbar=dict(title="P/L %"),
        hovertemplate="Date %{x}<br>Price %{y}<br>P/L %{z:+.1f}%<extra></extra>",
    ))
    if current_idx is not None:
        # Bold horizontal guide line across the whole grid at the current-
        # price row, on top of the "◀ CURRENT" text marker -- categorical
        # y-axes accept the category's own label as a shape position.
        fig.add_shape(
            type="line", xref="paper", x0=0, x1=1, yref="y", y0=y_labels[current_idx], y1=y_labels[current_idx],
            line=dict(color=ACCENT, width=2.5),
        )
    fig.update_xaxes(title_text="Date", type="category")
    fig.update_yaxes(title_text="Underlying price ($) and % move from current price")
    return apply_theme(fig, height=height, showlegend=False)


def render_buyback_chart(history_df, height=220):
    fig = go.Figure(
        go.Bar(
            x=history_df["period"].astype(str), y=history_df["buyback_magnitude"],
            marker_color=ACCENT,
            text=[human_money(v) for v in history_df["buyback_magnitude"]],
            textposition="outside", textfont=dict(color=TEXT_SECONDARY, size=10),
        )
    )
    apply_theme(fig, height=height, title=dict(text="Share buybacks (quarterly)", font=dict(color=TEXT_PRIMARY, size=13)),
                showlegend=False)
    return fig


# --------------------------------------------------------------------------
# Calls-vs-puts summary (OPTIONS FLOW tab)
# --------------------------------------------------------------------------

def metric_chip_html(label, value, color=None, value_size=20, sub_value=None):
    """value_size/sub_value let a chip stand out as the row's most
    prominent number (e.g. CURRENT PRICE, drawn larger with a % change
    line beneath it) without changing every other call site's default
    look."""
    color = color or TEXT_PRIMARY
    sub_html = (
        f'<div style="font-size:12px;color:{color};font-weight:600;margin-top:1px;">{sub_value}</div>'
        if sub_value else ""
    )
    return f"""
    <div style="background-color:{PANEL_ALT};border:1px solid {BORDER};border-radius:4px;padding:10px 14px;">
      <div style="font-size:11px;color:{TEXT_MUTED};letter-spacing:0.06em;">{label}</div>
      <div style="font-size:{value_size}px;color:{color};font-weight:700;">{value}</div>
      {sub_html}
    </div>
    """


def render_call_put_split(call_vol, put_vol, height=64):
    fig = go.Figure()
    total = call_vol + put_vol
    if total <= 0:
        apply_theme(fig, height=height)
        return fig

    call_pct, put_pct = call_vol / total * 100, put_vol / total * 100
    fig.add_trace(go.Bar(
        x=[call_pct], y=["Volume"], orientation="h", name="Calls", marker_color=COLOR_BULLISH,
        text=[f"{call_pct:.0f}% CALLS" if call_pct >= 12 else ""], textposition="inside",
        insidetextfont=dict(color=PANEL, size=11),
        hovertemplate=f"Calls: {call_vol:,.0f} ({call_pct:.1f}%)<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[put_pct], y=["Volume"], orientation="h", name="Puts", marker_color=COLOR_BEARISH,
        text=[f"{put_pct:.0f}% PUTS" if put_pct >= 12 else ""], textposition="inside",
        insidetextfont=dict(color=PANEL, size=11),
        hovertemplate=f"Puts: {put_vol:,.0f} ({put_pct:.1f}%)<extra></extra>",
    ))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(visible=False, range=[0, 100])
    fig.update_yaxes(visible=False)
    apply_theme(fig, height=height, margin=dict(l=6, r=6, t=6, b=6), showlegend=False)
    return fig


def render_options_chain_table(chain_df, spot):
    """Per-strike calls-vs-puts view for a single expiration: one row per
    strike, call side on the left and put side on the right, each with a
    volume bar scaled to the chain's max single-side volume so relative
    size is scannable at a glance -- same calls-vs-puts skew visual
    language as render_call_put_split, applied per strike."""
    if chain_df.empty:
        return ""
    calls = chain_df[chain_df["option_type"] == "call"].set_index("strike")
    puts = chain_df[chain_df["option_type"] == "put"].set_index("strike")
    strikes = sorted(set(calls.index) | set(puts.index))
    max_vol = max(
        calls["volume"].max() if not calls.empty else 0,
        puts["volume"].max() if not puts.empty else 0,
        1,
    )

    rows_html = []
    for strike in strikes:
        c = calls.loc[strike] if strike in calls.index else None
        p = puts.loc[strike] if strike in puts.index else None
        c_vol = int(c["volume"]) if c is not None else 0
        c_oi = int(c["open_interest"]) if c is not None else 0
        p_vol = int(p["volume"]) if p is not None else 0
        p_oi = int(p["open_interest"]) if p is not None else 0
        c_bar_pct = (c_vol / max_vol) * 100
        p_bar_pct = (p_vol / max_vol) * 100
        is_atm = spot is not None and not pd.isna(spot) and abs(strike - spot) <= (spot * 0.005)
        strike_style = f"font-weight:700;color:{ACCENT};" if is_atm else f"color:{TEXT_PRIMARY};"
        rows_html.append(f"""
        <tr>
          <td style="text-align:right;padding:2px 6px;color:{TEXT_MUTED};font-size:11px;">{c_oi:,}</td>
          <td style="width:110px;padding:2px 6px;">
            <div style="display:flex;justify-content:flex-end;">
              <div style="background:{COLOR_BULLISH};height:10px;width:{c_bar_pct:.0f}%;border-radius:2px;"></div>
            </div>
          </td>
          <td style="text-align:right;padding:2px 8px;color:{TEXT_SECONDARY};font-size:11px;">{c_vol:,}</td>
          <td style="text-align:center;padding:2px 10px;{strike_style}font-size:12px;">{strike:g}</td>
          <td style="text-align:left;padding:2px 8px;color:{TEXT_SECONDARY};font-size:11px;">{p_vol:,}</td>
          <td style="width:110px;padding:2px 6px;">
            <div style="display:flex;">
              <div style="background:{COLOR_BEARISH};height:10px;width:{p_bar_pct:.0f}%;border-radius:2px;"></div>
            </div>
          </td>
          <td style="text-align:left;padding:2px 6px;color:{TEXT_MUTED};font-size:11px;">{p_oi:,}</td>
        </tr>""")

    header = f"""
    <tr style="color:{TEXT_MUTED};font-size:10px;letter-spacing:0.05em;">
      <th style="text-align:right;padding:4px 6px;">CALL OI</th>
      <th></th>
      <th style="text-align:right;padding:4px 8px;">CALL VOL</th>
      <th style="text-align:center;padding:4px 10px;">STRIKE</th>
      <th style="text-align:left;padding:4px 8px;">PUT VOL</th>
      <th></th>
      <th style="text-align:left;padding:4px 6px;">PUT OI</th>
    </tr>"""

    return (
        f'<div style="max-height:520px;overflow-y:auto;">'
        f'<table style="width:100%;border-collapse:collapse;">{header}{"".join(rows_html)}</table>'
        f'</div>'
    )


def render_options_badge_table(opt_raw, max_rows=None):
    """Per-contract interpretive-badge view (Parts 4-6): Profile (delta),
    IV Level, Vol/OI, DTE, Theta decay, and Spread liquidity, each a
    color-coded chip with a hover tooltip carrying the plain-English
    'why'. opt_raw needs option_type/strike/expiration/delta/
    implied_volatility/volume_oi_ratio/theta/last_price/bid/ask -- rows
    missing the Greek columns (a same-day snapshot fetched before this
    feature shipped) simply render blank badges rather than erroring.

    max_rows=None (default) renders every row it's given -- the caller's
    own SQL query is already the single source of truth for how many rows
    this sees (LIMIT 40 for 'Unusual only', a 500-row cap for 'All
    contracts', no limit at all for a single selected expiry). A second,
    independent cap here previously silently cut a single-expiry chain
    off mid-strike (e.g. PYPL's 2026-08-21 chain has 80 rows across
    strikes up to 100+, but a hardcoded max_rows=40 stopped at strike 54)
    even though nothing upstream said to. Pass max_rows explicitly only
    if a caller genuinely wants a *second*, tighter cap on top of its
    query's own limit."""
    if opt_raw.empty:
        return ""
    today = date.today()
    view = opt_raw.head(max_rows) if max_rows is not None else opt_raw
    rows_html = []
    for _, r in view.iterrows():
        try:
            dte = (pd.to_datetime(r.get("expiration")).date() - today).days
        except (ValueError, TypeError):
            dte = None
        iv_pct = r.get("implied_volatility") * 100 if pd.notna(r.get("implied_volatility")) else None
        spread_pct = compute_spread_pct(r.get("bid"), r.get("ask"), r.get("last_price"))
        theta_pct = (
            abs(r["theta"]) / r["last_price"] * 100
            if pd.notna(r.get("theta")) and pd.notna(r.get("last_price")) and r["last_price"] > 0 else None
        )

        delta_label, delta_note = classify_delta(r.get("delta"))
        iv_label, iv_note = classify_iv(iv_pct)
        vol_oi_label, vol_oi_note = classify_vol_oi(r.get("volume_oi_ratio"))
        dte_label, dte_note = classify_dte(dte)
        theta_label, theta_note = classify_theta_pct(theta_pct)
        spread_label, spread_note = classify_spread_pct(spread_pct)

        type_color = COLOR_BULLISH if r["option_type"] == "call" else COLOR_BEARISH
        badges = "".join([
            interp_badge_html(delta_label, DELTA_BADGE_COLORS.get(delta_label, COLOR_NEUTRAL), delta_note),
            interp_badge_html(iv_label, IV_BADGE_COLORS.get(iv_label, COLOR_NEUTRAL), iv_note),
            interp_badge_html(vol_oi_label, _tier_escalation_color(vol_oi_label, VOL_OI_PROFILES), vol_oi_note),
            interp_badge_html(dte_label, COLOR_NEUTRAL, dte_note),
            interp_badge_html(theta_label, _tier_escalation_color(theta_label, THETA_PCT_PROFILES), theta_note),
            interp_badge_html(spread_label, _tier_escalation_color(spread_label, SPREAD_PROFILES), spread_note),
        ])
        rows_html.append(f"""
        <tr style="border-bottom:1px solid {BORDER};">
          <td style="padding:5px 8px;font-size:11px;color:{type_color};font-weight:700;">
            {r['option_type'].upper()}</td>
          <td style="padding:5px 8px;font-size:12px;color:{TEXT_PRIMARY};">{r['strike']:g}</td>
          <td style="padding:5px 8px;font-size:11px;color:{TEXT_SECONDARY};">{html.escape(str(r['expiration']))}</td>
          <td style="padding:5px 8px;"><div style="display:flex;flex-wrap:wrap;gap:4px;">{badges}</div></td>
        </tr>""")

    header = f"""
    <tr style="color:{TEXT_MUTED};font-size:10px;letter-spacing:0.05em;">
      <th style="text-align:left;padding:4px 8px;">TYPE</th>
      <th style="text-align:left;padding:4px 8px;">STRIKE</th>
      <th style="text-align:left;padding:4px 8px;">EXPIRY</th>
      <th style="text-align:left;padding:4px 8px;">INTERPRETIVE BADGES (hover for detail)</th>
    </tr>"""
    return (
        f'<div style="max-height:520px;overflow-y:auto;">'
        f'<table style="width:100%;border-collapse:collapse;">{header}{"".join(rows_html)}</table>'
        f'</div>'
    )


def render_options_synthesis_panel(opt_raw, spot=None, earnings_date=None):
    """Part 2 -- rules-based, deterministic, INSTANT (no API call, no
    confirmation popup) aggregate read of what the currently-loaded
    options_flow rows are actually saying: dominant delta profile,
    dominant IV level, call/put skew combined with the dominant side's
    delta profile, and the single most unusual trade. Pure presentation
    over data_engine.synthesize_options_flow_summary -- every sentence in
    it is picked from a fixed template table, never generated. Renders
    nothing if there's no volume to synthesize from (e.g. an empty
    selection), rather than showing an empty panel."""
    if opt_raw.empty:
        return
    annotated = annotate_options_badges(opt_raw)
    summary = synthesize_options_flow_summary(annotated, spot=spot, earnings_date=earnings_date)
    lines = [
        entry["sentence"] for key in ("dominant_delta", "dominant_iv", "skew_reading", "top_unusual")
        if (entry := summary.get(key)) and entry.get("sentence")
    ]
    if not lines:
        return
    st.markdown('<div class="smd-section">WHAT IS THIS TABLE TELLING ME?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="smcard">'
        + "".join(
            f'<div style="font-size:12.5px;color:{TEXT_PRIMARY};padding:4px 0;line-height:1.5;">'
            f'• {html.escape(line)}</div>' for line in lines
        )
        + '</div>',
        unsafe_allow_html=True,
    )


def render_top_unusual_insights(opt_raw, top_n=5):
    """Part 3 -- one expandable, deterministic one-line insight per
    top-`top_n` most unusual contract (ranked by vol/OI ratio), combining
    that row's own badges (delta profile + IV level + vol/OI level + DTE
    category) via data_engine.generate_row_insight. Restricts to rows
    flagged unusual=1 when that flag is present and set on at least one
    row in the current selection; otherwise ranks whatever's in view."""
    if opt_raw.empty:
        return
    annotated = annotate_options_badges(opt_raw)
    pool = annotated
    if "unusual" in annotated.columns and (annotated["unusual"] == 1).any():
        pool = annotated[annotated["unusual"] == 1]
    pool = pool.dropna(subset=["volume_oi_ratio", "delta_badge"]).sort_values(
        "volume_oi_ratio", ascending=False
    )
    if pool.empty:
        return
    st.markdown('<div class="smd-section">TOP UNUSUAL CONTRACTS -- WHAT THEY MEAN</div>', unsafe_allow_html=True)
    for _, row in pool.head(top_n).iterrows():
        insight = generate_row_insight(row)
        if not insight:
            continue
        vor = row.get("volume_oi_ratio")
        header = (
            f"{row['option_type'].upper()} ${row['strike']:g} — {row['expiration']}"
            f"{f' (Vol/OI {vor:.1f}x)' if vor is not None else ''}"
        )
        with st.expander(header):
            st.markdown(insight)


# Part 1 -- dedicated plain-language legend copy, deliberately distinct
# from each badge's technical `note` (which stays the hover-tooltip text
# used everywhere else in the table, unchanged). Keyed by exact badge
# label so a mismatch would surface immediately as a missing entry.
_LEGEND_PLAIN_TERMS = {
    "Deep OTM lottery ticket": "Very unlikely to pay off, but cheap and pays big if it does",
    "Lottery ticket": "Unlikely to pay off, cheap, high risk/high reward",
    "Speculative": "Needs the stock to move your way, but it's a real possibility",
    "Balanced": "Roughly a coin flip — the classic \"normal\" trade",
    "Conservative/ITM": "Likely to pay off, moves more like the actual stock",
    "Deep ITM / stock replacement": (
        "Nearly guaranteed to track the stock — used as a cheaper substitute for buying shares outright"
    ),
    "Low IV": "Cheap, calm market, not expecting big news",
    "Moderate/Balanced IV": "Normal pricing for an active stock",
    "Elevated IV": "Market is bracing for a real event — you're paying extra for that",
    "Extreme IV": (
        "Market expects a huge move — high chance the price crashes back down right after the "
        "event, even if you were right about direction"
    ),
    "Normal activity": "Ordinary day, nothing stands out",
    "Unusual -- likely new position": "Someone is actively building a new position here",
    "Highly unusual -- strong conviction or urgency": "A lot of money moving fast — someone feels strongly",
    "Gamma zone": "Very fast-moving, small price changes cause big swings — high risk",
    "Short-term directional": "A near-term bet with limited time to be right",
    "Medium-term swing": "Enough time for a real trade idea to develop",
    "Longer-term position": "More patient, less affected by day-to-day noise",
    "LEAPS": "Long-dated — behaves almost like owning the stock itself",
    "Low decay pressure": "Barely loses value sitting still — fine to hold",
    "Moderate decay": "Losing real value each day — don't sit on this too long",
    "High decay -- short-term trade only": "Bleeding value fast — only makes sense if you're right very soon",
    "Liquid, tight spread": "Easy to get in and out at a fair price",
    "Moderate liquidity": "Some cost to entering/exiting — worth factoring in",
    "Illiquid -- high slippage risk": "Hard to trade cleanly — you may get a worse fill than expected",
}

_LEGEND_SECTIONS = [
    {
        "title": "1️⃣ How likely is this option to pay off? (Delta)",
        "intro": (
            "Delta measures the probability the option finishes profitable — lower delta means a "
            "bigger, less likely payoff; higher delta means a smaller, more likely payoff, closer "
            "to just owning the stock."
        ),
        "profiles": DELTA_PROFILES,
        "range_fmt": lambda i, lo, hi, n: f"Δ {lo:.2f}–{hi:.2f}",
        "color_fn": lambda label: DELTA_BADGE_COLORS.get(label, COLOR_NEUTRAL),
    },
    {
        "title": "2️⃣ How expensive/risky is this option right now? (IV)",
        "intro": (
            "IV reflects how much movement the market is pricing in. Higher IV = more expensive "
            "option = the market expects something big (like earnings) to happen."
        ),
        "profiles": IV_PROFILES,
        "range_fmt": lambda i, lo, hi, n: f"{lo:.0f}%+" if i == n - 1 else f"{lo:.0f}–{hi:.0f}%",
        "color_fn": lambda label: IV_BADGE_COLORS.get(label, COLOR_NEUTRAL),
    },
    {
        "title": "3️⃣ Is unusual money moving into this trade? (Volume/OI)",
        "intro": (
            "This compares today's trading volume to how many contracts already exist — a spike "
            "suggests someone is opening a brand new position, not just regular trading."
        ),
        "profiles": VOL_OI_PROFILES,
        "range_fmt": lambda i, lo, hi, n: f"{lo:.0f}x+" if i == n - 1 else f"{lo:.0f}–{hi:.0f}x",
        "color_fn": lambda label: _tier_escalation_color(label, VOL_OI_PROFILES),
    },
    {
        "title": "4️⃣ How long until expiry, and what does that mean? (DTE)",
        "intro": (
            "Shorter time frames mean faster-moving, riskier bets; longer time frames give more "
            "room for a thesis to play out."
        ),
        "profiles": DTE_PROFILES,
        "range_fmt": lambda i, lo, hi, n: f"{lo:.0f}+ days" if i == n - 1 else f"{lo:.0f}–{hi:.0f} days",
        "color_fn": lambda label: COLOR_NEUTRAL,
    },
    {
        "title": "5️⃣ How fast does this lose value from time alone? (Theta)",
        "intro": (
            "Every option loses a little value each day just from the clock ticking, regardless of "
            "stock price — this shows how fast."
        ),
        "profiles": THETA_PCT_PROFILES,
        "range_fmt": lambda i, lo, hi, n: f"{lo:.0f}%+/day" if i == n - 1 else f"{lo:.0f}–{hi:.0f}%/day",
        "color_fn": lambda label: _tier_escalation_color(label, THETA_PCT_PROFILES),
    },
    {
        "title": "6️⃣ How easy is it to buy and sell this? (Spread)",
        "intro": (
            "This shows the gap between buy and sell prices — a wide gap means you'll likely get a "
            "worse price than you expect."
        ),
        "profiles": SPREAD_PROFILES,
        "range_fmt": lambda i, lo, hi, n: f"{lo:.0f}%+" if i == n - 1 else f"{lo:.0f}–{hi:.0f}%",
        "color_fn": lambda label: _tier_escalation_color(label, SPREAD_PROFILES),
    },
]


def render_badge_legend():
    """Part 1 -- six labeled, collapsible sections (one per underlying
    metric: Delta/IV/Volume-OI/DTE/Theta/Spread), each opening with a
    plain-language intro sentence naming the metric before any jargon, so
    a first-time reader can open just one topic instead of facing one
    flat 24-row table. Ranges/badges/colors are still driven live from
    the *_PROFILES lists (can't drift from what classify_*() actually
    returns) -- only the "in plain terms" column is dedicated legend copy
    (_LEGEND_PLAIN_TERMS), deliberately distinct from each badge's
    technical hover-tooltip note used elsewhere in the table. Renders
    real st.expander()s directly (not nested inside one outer expander --
    Streamlit doesn't support nesting them), so call this under its own
    section header, not inside another expander."""
    for section in _LEGEND_SECTIONS:
        with st.expander(section["title"]):
            st.caption(section["intro"])
            profiles = section["profiles"]
            n = len(profiles)
            rows_html = []
            for i, (lo, hi, label, note) in enumerate(profiles):
                rng = section["range_fmt"](i, lo, hi, n)
                plain = _LEGEND_PLAIN_TERMS.get(label, note)
                color = section["color_fn"](label)
                rows_html.append(
                    f'<tr style="border-bottom:1px solid {BORDER};">'
                    f'<td style="padding:5px 10px;white-space:nowrap;">{interp_badge_html(label, color)}</td>'
                    f'<td style="padding:5px 10px;font-size:11px;color:{TEXT_SECONDARY};white-space:nowrap;">'
                    f'{html.escape(rng)}</td>'
                    f'<td style="padding:5px 10px;font-size:11px;color:{TEXT_SECONDARY};">'
                    f'{html.escape(plain)}</td>'
                    f'</tr>'
                )
            header = f"""
            <tr style="color:{TEXT_MUTED};font-size:10px;letter-spacing:0.05em;">
              <th style="text-align:left;padding:4px 10px;">BADGE</th>
              <th style="text-align:left;padding:4px 10px;">RANGE</th>
              <th style="text-align:left;padding:4px 10px;">IN PLAIN TERMS</th>
            </tr>"""
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;">{header}{"".join(rows_html)}</table>',
                unsafe_allow_html=True,
            )


def compute_strike_bias(opt_raw, spot):
    """opt_raw: DataFrame with raw (unrenamed) option_type/strike columns
    for today's unusual activity. OTM calls above spot = bullish bet; OTM
    puts below spot = bearish bet; everything else (ITM) is less
    directional and left out of the tally."""
    if opt_raw.empty or spot is None or pd.isna(spot):
        return None
    bullish = opt_raw[(opt_raw["option_type"] == "call") & (opt_raw["strike"] > spot)]
    bearish = opt_raw[(opt_raw["option_type"] == "put") & (opt_raw["strike"] < spot)]
    n_bull, n_bear = len(bullish), len(bearish)
    if n_bull > n_bear:
        lean = "BULLISH"
    elif n_bear > n_bull:
        lean = "BEARISH"
    else:
        lean = "NEUTRAL"
    return {"bullish_count": n_bull, "bearish_count": n_bear, "lean": lean}


def render_gamma_exposure_chart(opt_raw, spot, height=280):
    """Aggregate gamma-by-strike bar chart (Part 7): sum of
    (gamma * open_interest * 100) per strike, calls and puts combined --
    a rough 'gamma wall' visualization of where dealer hedging pressure
    concentrates. Calls contribute positive dealer gamma, puts negative
    (standard market-maker-is-short-the-position convention), so a call
    bar and a put bar at the same strike partially offset rather than
    stack -- net exposure is what actually drives hedging flow."""
    df = opt_raw.dropna(subset=["gamma", "open_interest"]).copy()
    if df.empty:
        return None
    sign = df["option_type"].map({"call": 1, "put": -1}).fillna(0)
    df["gamma_exposure"] = df["gamma"] * df["open_interest"] * 100 * sign
    by_strike = df.groupby("strike", as_index=False)["gamma_exposure"].sum().sort_values("strike")
    colors = [COLOR_BULLISH if v >= 0 else COLOR_BEARISH for v in by_strike["gamma_exposure"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_strike["strike"], y=by_strike["gamma_exposure"], marker_color=colors,
                          name="Net gamma exposure"))
    if spot is not None and not pd.isna(spot):
        fig.add_vline(x=spot, line_dash="dot", line_color=ACCENT, annotation_text="Spot",
                      annotation_position="top")
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="Net $ Gamma Exposure (per 1pt move)")
    return apply_theme(fig, height=height, showlegend=False)


# IV_PROFILES tier -> background band color (Part 4): cool/blue for low IV,
# warm/red for elevated/extreme, low alpha so the line stays the primary
# visual. Keyed by label (not tier index) so a reorder of IV_PROFILES in
# data_engine can't silently mismatch a band to the wrong tier.
IV_BAND_FILL_COLORS = {
    "Low IV": "rgba(57,135,229,0.16)",             # cool blue
    "Moderate/Balanced IV": "rgba(90,100,114,0.12)",  # neutral
    "Elevated IV": "rgba(201,133,0,0.16)",          # amber
    "Extreme IV": "rgba(230,103,103,0.20)",         # warm red
}


def render_iv_term_structure_chart(term_structure, height=260):
    """ATM IV by expiry (Part 3's line) with a heatmap-style background
    (Part 4): horizontal bands shaded per IV_PROFILES tier so it's
    immediately obvious which expiries sit in 'elevated'/'extreme'
    territory without reading exact numbers. The open-ended last tier
    (100+) is capped at the data's own max (with headroom) rather than
    literally 999, so the band -- and the chart's y-range -- stay
    proportionate to what's actually being plotted."""
    if not term_structure:
        return None
    ivs = [t["atm_iv_pct"] for t in term_structure if t.get("atm_iv_pct") is not None]
    y_max = max(150.0, (max(ivs) * 1.15) if ivs else 150.0)

    fig = go.Figure()
    for lo, hi, label, _note in IV_PROFILES:
        band_top = min(hi, y_max)
        if lo >= y_max:
            continue
        fig.add_shape(
            type="rect", xref="paper", x0=0, x1=1, yref="y", y0=lo, y1=band_top,
            fillcolor=IV_BAND_FILL_COLORS.get(label, "rgba(90,100,114,0.10)"),
            line_width=0, layer="below",
        )
    fig.add_trace(go.Scatter(
        x=[t["expiration"] for t in term_structure], y=[t["atm_iv_pct"] for t in term_structure],
        mode="lines+markers", line=dict(color=ACCENT, width=2), marker=dict(size=8),
        name="ATM IV %",
    ))
    fig.update_yaxes(title_text="ATM IV %", range=[0, y_max])
    return apply_theme(fig, height=height, showlegend=False)


# --------------------------------------------------------------------------
# SYSTEM STATUS -- rendered at the very top of the page, above everything
# else. Compact one-row badge strip always visible; a collapsible expander
# underneath carries latency/error/fix-hint detail. Health is probed once
# per session (see get_health) so this never re-hits every source on an
# unrelated rerun -- only the "Recheck" button forces a fresh probe.
# --------------------------------------------------------------------------

_STATUS_ORDER = ["options_flow", "dark_pool", "congressional", "news", "13f"]


def render_news_feeds_section():
    """SETTINGS tab's 'News Feeds' section (Part 2): every news_sources
    row with an Enabled toggle and Remove button, plus an Add Feed form
    with a real Test Feed check (feedparser.parse against the candidate
    URL) before saving. This is the ONE place feed management happens --
    the NEWS tab only filters/views by feed name, it never adds/removes
    (see the 'Manage feeds →' pointer there)."""
    st.markdown('<div class="smd-section">NEWS FEEDS</div>', unsafe_allow_html=True)
    st.caption(
        "Real feed URLs, not publisher names -- add/remove a feed here and it takes effect "
        "immediately for every ticker's news."
    )
    conn_nf = get_conn()
    sources = _read_news_sources(conn_nf)

    if not sources:
        st.caption("No news feeds configured.")
    else:
        for src in sources:
            rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns([2, 3.5, 1, 1, 1])
            rcol1.markdown(f"**{md_safe(src['name'])}**")
            rcol2.markdown(
                f'<span style="font-size:11px;color:{TEXT_SECONDARY};word-break:break-all;">'
                f'{html.escape(src["feed_url"])}</span>',
                unsafe_allow_html=True,
            )
            rcol3.markdown(f'<span style="font-size:11px;">{html.escape(src["url_type"])}</span>',
                           unsafe_allow_html=True)
            new_enabled = rcol4.checkbox(
                "On", value=bool(src["enabled"]), key=f"news_src_enabled_{src['id']}", label_visibility="visible",
            )
            if new_enabled != bool(src["enabled"]):
                set_news_source_enabled(conn_nf, src["id"], new_enabled)
                st.cache_data.clear()
                st.rerun()
            if rcol5.button("Remove", key=f"news_src_remove_{src['id']}"):
                remove_news_source(conn_nf, src["id"])
                st.cache_data.clear()
                st.rerun()

    st.markdown("**Add feed**")
    acol1, acol2 = st.columns([1, 2])
    with acol1:
        new_name = st.text_input("Name", key="news_add_name", placeholder="e.g. Yahoo Finance RSS")
    with acol2:
        new_url = st.text_input(
            "Feed URL", key="news_add_url",
            placeholder="https://example.com/rss?ticker={ticker}  (use {ticker} if the feed is per-symbol)",
        )
    tcol1, tcol2 = st.columns([1, 3])
    with tcol1:
        test_clicked = st.button("Test Feed", key="news_add_test", width='stretch')
    if test_clicked:
        if not new_url.strip():
            st.warning("Enter a feed URL first.")
        else:
            with st.spinner("Testing feed..."):
                st.session_state["news_test_result"] = test_news_feed(new_url.strip())
                st.session_state["news_test_url"] = new_url.strip()

    test_result = st.session_state.get("news_test_result")
    if test_result and st.session_state.get("news_test_url") == new_url.strip():
        if test_result["ok"]:
            st.success(f"✅ {test_result['entry_count']} entries found. Preview:")
            for title in test_result["preview"]:
                st.caption(f"• {title}")
        else:
            st.error(f"❌ Feed test failed: {test_result['error']}")

    if st.button("Save Feed", key="news_add_save"):
        if not new_name.strip() or not new_url.strip():
            st.warning("Name and Feed URL are both required.")
        elif not (test_result and test_result.get("ok") and st.session_state.get("news_test_url") == new_url.strip()):
            st.warning("Click 'Test Feed' first and confirm it returns real entries before saving.")
        else:
            add_news_source(conn_nf, new_name.strip(), new_url.strip(), url_type="rss")
            st.session_state.pop("news_test_result", None)
            st.session_state.pop("news_test_url", None)
            st.cache_data.clear()
            st.success(f"Saved '{new_name.strip()}'.")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)


def render_firm_tiers_section():
    """RATING SIGNAL FIRM TIERS (Part 2): editable firm_tiers table --
    classifies each firm's real MARKET IMPACT (attention/volume a call
    typically generates), NOT analyst accuracy. synthesize_rating_signal
    (WATCHLIST SIGNALS tab) reads this via get_firm_tier on every call, so
    an edit here takes effect immediately on the next scan/render, no
    cache to clear. Any firm with no row here defaults to Tier 3 / 0.25 --
    that default lives in data_engine.get_firm_tier, not here."""
    st.markdown('<div class="smd-section">RATING SIGNAL FIRM TIERS</div>', unsafe_allow_html=True)
    st.caption(
        "Classifies each firm's real market-impact tier (attention/volume a rating call typically "
        "generates) -- not a claim about analyst accuracy. Used to weight WATCHLIST SIGNALS' synthesized "
        "per-ticker rating signal. Any firm not listed here defaults to Tier 3 / weight 0.25. Edit directly "
        "and Save; use the blank bottom row to add a new firm."
    )
    conn = get_conn()
    rows = conn.execute(
        "SELECT firm_name, tier, tier_label, weight FROM firm_tiers ORDER BY tier, firm_name"
    ).fetchall()
    df = pd.DataFrame(rows, columns=["Firm", "Tier", "Tier label", "Weight"])
    edited = st.data_editor(
        df, width='stretch', hide_index=True, num_rows="dynamic", key="firm_tiers_editor",
        column_config={
            "Tier": st.column_config.NumberColumn(min_value=1, max_value=3, step=1),
            "Weight": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
        },
    )
    if st.button("Save changes", key="firm_tiers_save_btn"):
        conn.execute("DELETE FROM firm_tiers")
        for _, r in edited.iterrows():
            firm = str(r["Firm"]).strip() if pd.notna(r["Firm"]) else ""
            if not firm:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO firm_tiers (firm_name, tier, tier_label, weight) VALUES (?,?,?,?)",
                (firm, int(r["Tier"]) if pd.notna(r["Tier"]) else 3,
                 str(r["Tier label"]) if pd.notna(r["Tier label"]) else "Smaller/Regional",
                 float(r["Weight"]) if pd.notna(r["Weight"]) else 0.25),
            )
        conn.commit()
        st.success("Firm tier classifications saved.")
        st.rerun()


# --------------------------------------------------------------------------
# Daily Digest newsletter orchestration -- the ONE place that ties
# together build_digest_content (data_engine.py, no plotting dependency)
# with the Featured Stock's chart image, since chart-building
# (render_price_volume_chart) and static PNG export (kaleido) both live
# only here. data_engine.py's send_daily_digest_email/build_digest_
# content never import plotly/kaleido themselves -- they just accept a
# `chart_cid` reference string and `inline_images` bytes dict, both
# supplied by this file.
# --------------------------------------------------------------------------

# The chart images for the digest are built by charts.py's
# build_featured_stock_chart_bytes / build_analyst_chart_images /
# build_swing_chart_image (imported below) -- pure Plotly + kaleido, no
# Streamlit dependency, so the exact same functions produce the exact same
# images whether triggered from this UI ("Send Test Email Now"/Preview) or
# from digest/send.py's headless scheduled path (launchd, no Streamlit
# process at all). See charts.py's own module docstring.
# --------------------------------------------------------------------------


# run_daily_digest_send / run_digests_for_all_users: moved to digest/core.py so digest/send.py's headless
# launchd path (no Streamlit process) can call the exact same orchestration this UI does -- see that
# module's docstring. Imported below (top of file) as `from digest.core import ...`.


def render_email_digest_section():
    """📧 EMAIL DIGEST: in-app config for the single global edition's
    schedule/content-selection -- SMTP server credentials (SMTP_HOST/
    PORT/USER/PASSWORD) remain environment secrets, a real security
    boundary (this table is just app state on disk; an env var requires
    server-level access to change). The trigger-mechanism honesty note is
    surfaced here VERBATIM, not just in code comments, per this feature's
    own explicit instruction -- there is no real background scheduler
    behind this, only an in-process check.

    2026-10-13 single-global-edition rebuild: this is now ONE shared
    publication, not a per-app-user digest. Schedule/subscriber-list
    controls below are admin-only (gated on EFFECTIVE_EMAIL being the
    actual edition-admin account, not just IS_ADMIN -- an admin "viewing
    as" a non-admin user sees that user's own reduced view, same as that
    user would). A non-admin account's own email_config row is left in
    place, unused -- see data_engine._get_edition_admin_email /
    build_digest_edition's docstrings for why it's hidden, not deleted."""
    st.markdown('<div class="smd-section">📧 NEWSLETTER</div>', unsafe_allow_html=True)
    conn = get_conn()
    admin_email = _get_edition_admin_email(conn)
    is_edition_admin = admin_email is not None and EFFECTIVE_EMAIL == admin_email

    if not is_edition_admin:
        st.caption(
            "Smart Money Weekly is a single newsletter with one subscriber list and one admin-controlled "
            "schedule -- per-account newsletter settings no longer apply here."
        )
        all_subs = get_all_subscribers(conn, admin_email) if admin_email else []
        own = next((x for x in all_subs if x["email"].lower() == (EFFECTIVE_EMAIL or "").lower()), None)
        if own and not own["unsubscribed"]:
            st.info(f"**{EFFECTIVE_EMAIL}** is subscribed. Use the \"Manage preferences\" link at the "
                    f"bottom of any digest email to change which sections you receive or to unsubscribe -- "
                    f"no login needed there.")
        elif own:
            st.caption(f"**{EFFECTIVE_EMAIL}** is not currently subscribed to the newsletter.")
        else:
            st.caption("This account isn't on the newsletter's subscriber list.")
        return

    config = get_email_config(conn, admin_email)

    launchd_installed = _launchd_job_installed()
    if launchd_installed:
        st.success(
            "Scheduled sends run via launchd, every :00 and :30, independent of whether this dashboard is "
            "open — no browser tab or 'app running' requirement. (This app's own browser-open catch-up "
            "check still runs too, as a second path, but the launchd job is what makes delivery reliable.)",
            icon="✅",
        )
    else:
        st.warning(
            "Scheduled sends currently depend on this dashboard being open in a connected browser tab around "
            "your send time — there is no background scheduler running. See SETTINGS → Global settings → "
            "Digest scheduler (admin) to install the launchd job that removes this dependency.",
            icon="⚠️",
        )

    # Whether the SERVER can send email is deployment configuration (SMTP
    # host/credentials) -- an admin-only concern shown in the ADMIN tab, never
    # here. A regular user only learns whether sending is available, not how
    # it's configured.
    smtp_configured = all(os.environ.get(v) for v in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"))
    if not smtp_configured:
        st.info("Email sending isn't set up on this server yet. Your newsletter settings are saved and will "
                "apply once your administrator enables it.", icon="ℹ️")

    # Part 7: last SCHEDULED send shown separately from last TEST send -- the two can never be conflated
    # since send_daily_digest_email(send_type=) writes to entirely separate columns for each.
    if config["last_send_status"] == "success":
        # Part 3: last_send_error is now ALWAYS a batch-result sentence
        # on a successful send (e.g. "sent to 4 of 5 subscriber(s) — 1
        # failed (a@b.com: <error>)"), not just present on partial
        # failure -- one line covers full success, partial, and (below)
        # total failure consistently.
        batch_note = config["last_send_error"] or "sent"
        st.info(f"✅ Last **scheduled** digest: {batch_note} at {time_ago(config['last_digest_sent_at'])} "
                f"({config['last_digest_sent_at']}).")
    elif config["last_send_status"] == "failed":
        st.error(f"❌ Last scheduled digest attempt failed — {config['last_send_error']}")
    else:
        st.caption("No scheduled digest has gone out yet.")
    if config.get("last_test_sent_at"):
        _test_ok = config.get("last_test_send_status") == "success"
        st.caption(f"{'✅' if _test_ok else '❌'} Last **test** send: "
                   f"{config.get('last_test_send_error') or config.get('last_test_send_status')} at "
                   f"{time_ago(config['last_test_sent_at'])}.")

    # 2026-10-13: last edition's cost, right here where the admin is already looking at send status.
    _last_ed = get_last_digest_edition(conn)
    if _last_ed:
        st.caption(f"📊 Last edition built: **{_last_ed['edition_date']}** — "
                   f"{_last_ed['claude_api_calls']} Claude call(s), ${_last_ed['claude_api_cost']:.4f} — "
                   f"featured **{_last_ed.get('featured_ticker') or '—'}**.")

    enabled = st.toggle("Enable newsletter", value=config["enabled"], key=_uk("digest_enabled_toggle"))

    # ---- Subscribers (this account IS the edition admin at this point -- is_edition_admin gated above) ----
    all_subscribers = get_all_subscribers(conn, admin_email)
    active_count = sum(1 for s in all_subscribers if not s["unsubscribed"])
    st.markdown("**Subscribers**")
    st.caption("Each active row below gets its own INDIVIDUAL email (never a shared To:/Cc:) with its own "
               "unique unsubscribe link and manage-preferences link.")
    st.markdown(f"**Sending to {active_count} active subscriber(s)**")

    acol1, acol2 = st.columns([3, 1])
    new_subscriber = acol1.text_input(
        "Add subscriber", key=_uk("digest_add_subscriber_input"), placeholder="you@example.com",
        label_visibility="collapsed",
    )
    if acol2.button("+ Add Subscriber", key=_uk("digest_add_subscriber_btn"),
                     disabled=not new_subscriber.strip(), width='stretch'):
        _candidate = new_subscriber.strip()
        if not is_valid_email_format(_candidate):
            st.error(f"'{_candidate}' doesn't look like a valid email address — not added.")
        else:
            add_email_subscriber(conn, admin_email, _candidate)
            st.success(f"Added {_candidate}.")
            st.rerun()

    if not all_subscribers:
        st.caption("No subscribers yet -- add one above.")
    for sub in all_subscribers:
        rcol1, rcol2, rcol3, rcol4 = st.columns([3, 1.3, 1.3, 1])
        if sub["unsubscribed"]:
            rcol1.markdown(f"<span style='opacity:0.5;'>{html.escape(sub['email'])}</span>",
                            unsafe_allow_html=True)
            rcol2.markdown(f"<span style='opacity:0.5;color:{COLOR_NEUTRAL};'>Unsubscribed "
                            f"{fmt_date(sub['unsubscribed_at'])}</span>", unsafe_allow_html=True)
            rcol3.caption(f"joined {fmt_date(sub['subscribed_at'])}")
            if rcol4.button("Re-subscribe", key=_uk(f"digest_resub_{sub['email']}"), width='stretch'):
                resubscribe_email(conn, admin_email, sub["email"])
                st.rerun()
        else:
            rcol1.markdown(f"**{html.escape(sub['email'])}**", unsafe_allow_html=True)
            rcol2.markdown(f"<span style='color:{COLOR_BULLISH};'>● Active</span>", unsafe_allow_html=True)
            rcol3.caption(f"joined {fmt_date(sub['subscribed_at'])}")
            if rcol4.button("Remove", key=_uk(f"digest_remove_{sub['email']}"), width='stretch'):
                remove_email_subscriber(conn, admin_email, sub["email"])
                st.rerun()
        # 2026-10-13, Part 1.3: per-subscriber section preferences, admin-editable here (the same thing
        # the subscriber themselves can change via the tokenized "Manage preferences" email link).
        with st.expander(f"Sections for {sub['email']}", expanded=False):
            _sub_sections = set(get_subscriber_sections_included(sub))
            _sec_cols = st.columns(4)
            _chosen = []
            for _i, _key in enumerate(DEFAULT_DIGEST_SECTIONS):
                if _sec_cols[_i % 4].checkbox(
                    DIGEST_SECTION_LABELS[_key], value=_key in _sub_sections,
                    key=_uk(f"subsec_{sub['email']}_{_key}"),
                ):
                    _chosen.append(_key)
            if st.button("Save sections", key=_uk(f"subsec_save_{sub['email']}")):
                set_subscriber_sections_included(conn, sub["unsubscribe_token"], _chosen)
                st.success("Saved.")
                st.rerun()
    c3, c4 = st.columns(2)
    freq_options = ["daily", "weekly"]
    freq_labels = {"daily": "Daily (Mon–Fri)", "weekly": "Weekly"}
    # legacy stored values: 'weekdays_only' is the same thing as Daily now; 'off' is the toggle above
    _stored_freq = "weekly" if config["frequency"] == "weekly" else "daily"
    frequency = c3.selectbox(
        "Frequency", options=freq_options, index=freq_options.index(_stored_freq),
        format_func=lambda k: freq_labels[k], key=_uk("digest_frequency_select"),
        help="Never sent on Saturday or Sunday.",
    )
    _days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekly_day = 0
    if frequency == "weekly":
        weekly_day = _days.index(st.selectbox(
            "Weekly send day", _days, index=config["weekly_day"], key=_uk("digest_weekly_day_select"),
        ))
    try:
        default_time = datetime.strptime(config["send_time_et"] or "07:30", "%H:%M").time()
    except ValueError:
        default_time = dt_time(7, 30)
    if default_time.minute not in (0, 30):
        # Migrate a stale (pre-launchd) value UP to the next half-hour -- the headless scheduler (digest/
        # send.py) only ever checks at :00 and :30, so anything off that grid would silently wait up to ~29
        # extra minutes every single day. Rounds up, never down, so a send never fires earlier than configured.
        # Persisted immediately, so this migration only ever runs once per stale value.
        carry_hours, new_minute = divmod(default_time.minute + (30 - default_time.minute % 30), 60)
        default_time = dt_time((default_time.hour + carry_hours) % 24, new_minute)
        save_email_config(conn, admin_email, send_time_et=default_time.strftime("%H:%M"))
    send_time = c4.time_input(
        "Send time (ET)", value=default_time, step=timedelta(minutes=30), key=_uk("digest_send_time_input"),
        help="Scheduled sends are only checked at :00 and :30 (see the launchd digest scheduler in SETTINGS "
             "-> Global settings), so the picker is limited to those times.",
    )

    c5, c6 = st.columns(2)
    wait_runners = c5.checkbox(
        "Wait for Runners scan to complete", value=config["wait_for_runners"], key=_uk("digest_wait_runners_cb"),
    )
    wait_predictions = c6.checkbox(
        "Wait for Day Predictions to complete", value=config["wait_for_predictions"],
        key=_uk("digest_wait_predictions_cb"),
    )

    st.markdown("**Sections to include**")
    st.caption("This is now the DEFAULT new subscribers are seeded with at signup -- each subscriber's own "
               "choice (above, or via their \"Manage preferences\" link) governs what THEY actually receive.")
    included = set(config["sections_included"])
    section_cols = st.columns(4)
    new_included = []
    for i, key in enumerate(DEFAULT_DIGEST_SECTIONS):
        checked = section_cols[i % 4].checkbox(
            DIGEST_SECTION_LABELS[key], value=key in included, key=_uk(f"digest_section_cb_{key}"),
        )
        if checked:
            new_included.append(key)

    if st.button("Save Email Digest settings", key=_uk("digest_save_settings_btn")):
        _save_kwargs = dict(
            enabled=enabled, frequency=frequency, weekly_day=weekly_day, send_time_et=send_time.strftime("%H:%M"),
            wait_for_runners=wait_runners, wait_for_predictions=wait_predictions, sections_included=new_included,
        )
        save_email_config(conn, admin_email, **_save_kwargs)
        # Part 6: keep the admin's healthchecks.io dead-man's-switch schedule in lock-step with whatever was
        # just saved -- a no-op (logged, not shown here) if HEALTHCHECKS_API_KEY isn't configured.
        healthchecks.sync_check_schedule(conn, admin_email)
        st.success("Email Digest settings saved.")
        st.rerun()

    ready, reasons = is_daily_digest_ready(conn, admin_email)
    st.caption(f"Readiness right now: {'✅ ready' if ready else '⏳ not ready — ' + ' '.join(reasons)}")

    with st.expander("⏰ Reliable wake schedule (optional, needs sudo)"):
        st.caption(
            "If this Mac can sleep, the launchd digest job (see SETTINGS → Global settings → Digest "
            "scheduler) can't run while it's asleep. This wakes the machine shortly before your send time so "
            "it's awake for the tick. Copy the command below into Terminal yourself -- it needs sudo, so "
            "this app can never run it for you, and it doesn't follow this UI automatically: re-copy it "
            "any time you change Send time above."
        )
        _cmd, _local_time, _tz_name = pmset_wake_command_for(send_time.strftime("%H:%M"))
        st.code(_cmd, language="bash")
        st.caption(f"({send_time.strftime('%H:%M')} ET → {_local_time} {_tz_name} on this machine, 10 minutes "
                   f"before send time.)")
        st.warning(
            "This wake WILL NOT happen if: the lid is closed AND the Mac is running on battery (Apple "
            "disables scheduled wake on battery with the lid closed -- Power Nap/scheduled-wake needs either "
            "the lid open, or the Mac plugged into AC power). Plug in and/or leave the lid open (or use an "
            "external display so 'open' isn't required) for this to actually fire overnight.",
            icon="⚠️",
        )

    with st.expander("Digest settings"):
        _ov = get_featured_override(conn, admin_email)
        _auto_label = "Auto (rule-based)"
        _pickable = sorted(set(watchlist) | set(get_todays_runner_tickers(conn, admin_email) or []))
        _options = [_auto_label] + _pickable
        _pick = st.selectbox(
            "Featured stock for next digest", _options,
            index=_options.index(_ov["ticker"]) if (_ov and _ov["ticker"] in _options) else 0,
            key=_uk("digest_override_select"),
        )
        # Defaults to "just the next digest": a forgotten sticky pick would quietly pin the newsletter to a stale
        # name for weeks.
        _mode_labels = ["Just the next digest", "Until I change it"]
        _mode_label = st.radio("Applies to", _mode_labels, index=1 if (_ov and _ov["mode"] == "sticky") else 0,
                               horizontal=True, key=_uk("digest_override_mode"))
        if _ov:
            st.caption(f"Next digest will feature: **{_ov['ticker']}** "
                       f"({'manual, one-time' if _ov['mode'] == 'once' else 'manual, until changed'}"
                       f"{', lapses ' + _ov['expires_at'] if _ov.get('expires_at') else ''}).")
        else:
            st.caption("Next digest will feature: **auto-selected**.")
        _c1, _c2 = st.columns(2)
        if _c1.button("Apply", key=_uk("digest_override_apply")):
            if _pick == _auto_label:
                clear_featured_override(conn, admin_email)
            else:
                set_featured_override(conn, admin_email, _pick, "once" if _mode_label == _mode_labels[0] else "sticky")
            st.rerun()
        if _c2.button("Clear override", key=_uk("digest_override_clear"), disabled=_ov is None):
            clear_featured_override(conn, admin_email)
            st.rerun()
        st.caption("A one-time pick is used by the next scheduled digest only (a test send doesn't use it up) and "
                   "lapses after 7 days if none goes out; an 'until I change it' pick lapses after 14 days. "
                   "Takes effect on the NEXT rebuild -- today's already-built edition (if any) keeps its "
                   "current featured stock until you Rebuild.")

        # 2026-10-14: today's edition (if one already exists) was built BEFORE this override was applied,
        # so it still shows the old featured stock -- surface that explicitly instead of letting it look
        # like Apply silently did nothing. Recomputed on every render (not just right after Apply), so it
        # stays accurate across reruns/page loads too.
        _today_ed_for_ov = get_digest_edition(conn)
        if _ov and _today_ed_for_ov and _today_ed_for_ov.get("featured_ticker") != _ov["ticker"]:
            st.info(f"Today's edition already features **{_today_ed_for_ov.get('featured_ticker') or '—'}**.")
            if st.button(f"🔁 Rebuild now with {_ov['ticker']}", key=_uk("digest_override_rebuild_now")):
                st.session_state[_uk("_digest_rebuild_confirm_open")] = True
                st.rerun()
            st.caption(f"Or leave it — {_ov['ticker']} will apply to the next edition.")

    bcol1, bcol2, bcol3 = st.columns(3)
    if bcol1.button("👁️ Preview Today's Digest", key=_uk("digest_preview_btn")):
        st.session_state[_uk("_digest_preview_open")] = True
    if bcol2.button(
        "📨 Send Test Email Now", key=_uk("digest_send_test_btn"),
        disabled=not (smtp_configured and active_count),
        help="Sends today's STORED edition (building it first, at most one real Claude call, if it doesn't "
             "exist yet) to every active recipient over real SMTP, bypassing the schedule/duplicate-send "
             "guard. Never triggers a second API call if today's edition already exists." if smtp_configured
             else "Set SMTP_HOST/SMTP_USER/SMTP_PASSWORD first.",
    ):
        with st.spinner("Loading today's edition (building it if it doesn't exist yet)..."):
            result = run_daily_digest_send(conn, admin_email, force=True)
        if result.get("sent"):
            st.success(f"Test email sent to: {', '.join(result.get('recipients_sent') or [])}.")
            if result.get("recipients_failed"):
                st.warning(f"Failed for: {', '.join(result['recipients_failed'])}.")
        else:
            st.error(f"Send failed: {result.get('reason')}")
        st.rerun()
    if bcol3.button("🔁 Rebuild today's edition", key=_uk("digest_rebuild_btn"),
                    help="Forces a fresh build -- a new, real, billed Claude call for the featured stock, "
                         "even if today's edition already exists. Replaces it for EVERYONE, including "
                         "anyone who already received today's send."):
        st.session_state[_uk("_digest_rebuild_confirm_open")] = True

    if st.session_state.get(_uk("_digest_rebuild_confirm_open")):
        with st.expander("⚠️ Confirm rebuild", expanded=True):
            _existing_ed = get_digest_edition(conn)
            if _existing_ed:
                st.warning(
                    f"Today's edition already exists (built {time_ago(_existing_ed['built_at'])}, featured "
                    f"**{_existing_ed.get('featured_ticker') or '—'}**, ${_existing_ed['claude_api_cost']:.4f} "
                    f"so far). Rebuilding makes a NEW, real, billed Claude call and replaces it for every "
                    f"subscriber, including anyone who already received today's send."
                )
            else:
                st.info("No edition exists for today yet -- this will build the first one (at most one "
                        "real, billed Claude call).")
            _rcol1, _rcol2 = st.columns(2)
            if _rcol1.button("Continue — rebuild now", key=_uk("digest_rebuild_confirm_btn")):
                with st.spinner("Rebuilding today's edition..."):
                    # consume_override=True: a rebuild is the admin deliberately re-picking right now, so a
                    # one-time override used by THIS rebuild is spent by it -- it must not also apply to
                    # tomorrow's build. See build_digest_edition's docstring (2026-10-14).
                    build_digest_edition(conn, force=True, consume_override=True)
                st.session_state[_uk("_digest_rebuild_confirm_open")] = False
                st.success("Edition rebuilt.")
                st.rerun()
            if _rcol2.button("Cancel", key=_uk("digest_rebuild_cancel_btn")):
                st.session_state[_uk("_digest_rebuild_confirm_open")] = False
                st.rerun()

    def _build_substack_payload(edition):
        """Shared by "Export for Substack" and "Send to Substack" (2026-10-14 fidelity rebuild): re-renders
        a FRESH, chart-complete html_full for this edition via render_edition_for_recipient -- the SAME
        zero-API-call, cached-featured_stock re-render real sends use -- with real chart_cid/analyst_cids/
        swing_cids wired in via the SAME charts.py calls the real email makes. Doesn't read
        digest_editions.html_full directly: that stored string is built headless inside build_digest_edition
        (data_engine.py has no plotly dependency) and so never contains the chart <img> tags the real email
        embeds -- confirmed live (0 "cid:" occurrences in a real stored edition's html_full). Returns
        (substack_content, images); images is {key: png_bytes} in the SAME order the email renders them,
        which is also the order export_substack_zip/send_edition_to_substack number/insert them in."""
        featured_ticker = edition.get("featured_ticker")
        featured_stock = (edition.get("sections") or {}).get("featured_stock")
        email_chart_images, chart_cid = {}, None
        if featured_ticker and featured_stock:
            chart_bytes, chart_cid = build_featured_stock_chart_bytes(featured_ticker, conn)
            if chart_bytes and chart_cid:
                email_chart_images[chart_cid] = chart_bytes
            email_chart_images.update(build_analyst_chart_images(featured_ticker, featured_stock, conn))
            email_chart_images.update(build_swing_chart_image(featured_stock))
        fresh = render_edition_for_recipient(conn, edition, DEFAULT_DIGEST_SECTIONS, chart_cid=chart_cid)
        # build_substack_markdown takes the fresh HTML as an explicit argument (2026-10-14 hardening) --
        # never reads edition["html_full"] (the build-time-only stored string) itself.
        content = build_substack_markdown(conn, edition, fresh["html_body"])
        images = render_substack_image_slots(content["image_slots"], email_chart_images)
        return content, images

    # ---- Send to Substack (draft only) -- 2026-10-13, Part 2 ----
    st.markdown("**Substack**")
    _last_sub_post = get_last_substack_post(conn)
    if _last_sub_post:
        st.caption(f"Last Substack post: {time_ago(_last_sub_post['created_at'])} "
                   f"({_last_sub_post['status']}{' — ' + _last_sub_post['error'] if _last_sub_post.get('error') else ''}).")
    else:
        st.caption("No Substack post attempted yet.")

    # 2026-10-14: a read-only auth check, separate from draft creation -- isolates whether a 401 is coming
    # from the cookie itself (fails at "construction", before any draft code runs) vs. something scoped to
    # the drafts endpoint specifically (passes construction/get_user_id, fails at "drafts_read"). Never
    # creates/touches a real draft, never logs or displays the cookie value.
    if st.button("🔌 Test Substack connection (read-only)", key=_uk("digest_substack_test_btn")):
        with st.spinner("Checking Substack auth..."):
            _conn_test = test_substack_connection()
        if _conn_test["ok"]:
            st.success(f"✅ Signed in (user_id={_conn_test['user_id']}). Resolved publication: "
                       f"{_conn_test['resolved_publication_url']}.")
            if _conn_test["resolved_publication_url"] and "monotrading" not in _conn_test["resolved_publication_url"]:
                st.warning("This doesn't look like monotrading.substack.com -- check SUBSTACK_PUBLICATION_URL "
                           "and that the signed-in account owns that publication.")
        elif _conn_test["stage"] == "not_configured":
            st.error(_conn_test["error"])
        elif _conn_test["stage"] == "construction":
            st.error(f"❌ Auth failed before any draft-specific code ran ({_conn_test['error']}). The cookie "
                      f"itself is being rejected -- this is a sign-in problem, not specific to draft creation.")
        else:
            st.warning(f"⚠️ Signed in OK (user_id={_conn_test.get('user_id')}, publication "
                        f"{_conn_test.get('resolved_publication_url')}), but a read-only check at stage "
                        f"'{_conn_test['stage']}' still failed: {_conn_test['error']}. Narrower than a dead "
                        f"cookie -- likely a permissions/scope gap on that publication, not the sign-in itself.")
        # Never shows the cookie value itself -- only its shape -- but this is usually enough to tell
        # "wrong cookie pasted" apart from "right cookie, still rejected."
        if "cookie_names" in _conn_test:
            st.caption(f"Cookie names found: {', '.join(_conn_test['cookie_names']) or '(none)'} · "
                       f"substack.sid: {_conn_test['sid_shape']} (length {_conn_test['sid_value_length']}).")
            if "substack.lli" not in _conn_test["cookie_names"]:
                st.caption("⚠️ No `substack.lli` cookie found -- Substack sessions normally need it alongside "
                           "substack.sid. Add `substack.lli=1` to SUBSTACK_COOKIE if DevTools shows it.")

    _today_sub_post = get_substack_post_for_edition(conn, today_et())
    scol1, scol2 = st.columns(2)
    _substack_btn_label = "📤 Already sent to Substack — send again" if _today_sub_post else "📤 Send to Substack (draft)"
    if scol1.button(_substack_btn_label, key=_uk("digest_substack_btn")):
        st.session_state[_uk("_digest_substack_confirm_open")] = True
    if _today_sub_post and scol1.button("Open existing draft", key=_uk("digest_substack_open_btn")):
        if _today_sub_post.get("draft_url"):
            st.markdown(f"[Open draft in Substack]({_today_sub_post['draft_url']})")
        else:
            st.caption("No draft URL recorded for this post.")

    # Always available, independent of python-substack/API/cookie (Part 2.5) -- built from the already-
    # built edition's cached data, never a rebuild, never a new Claude call (only cheap, free chart/
    # headless-browser rendering, same as Preview's own chart-building).
    _export_edition = get_digest_edition(conn)
    if _export_edition:
        with st.spinner("Rendering charts/tables for export (headless browser)..."):
            _export_content, _export_images = _build_substack_payload(_export_edition)
        _export_zip = export_substack_zip(_export_content["markdown"], _export_images)
        scol2.download_button(
            "⬇️ Export for Substack (.zip)", data=_export_zip,
            file_name=f"substack_{_export_edition['edition_date']}.zip", mime="application/zip",
            key=_uk("digest_substack_export_btn"),
            help="Markdown + numbered PNGs (every section, same order/content as the email) for pasting "
                 "into Substack by hand -- works even if the unofficial API/cookie isn't set up.",
        )
    else:
        scol2.caption("Build today's edition first (Preview or Send Test) to export.")

    if st.session_state.get(_uk("_digest_substack_confirm_open")):
        with st.expander("⚠️ Confirm Send to Substack", expanded=True):
            st.warning(
                "This creates a DRAFT in Substack — it is never published automatically. Uses today's "
                "already-built edition (no rebuild, no new Claude call). You'll edit wording and publish "
                "manually in Substack."
            )
            if _today_sub_post:
                st.error(
                    f"This edition already has a draft ({time_ago(_today_sub_post['created_at'])}) — "
                    f"continuing will create ANOTHER draft, not update the existing one."
                )
            _sscol1, _sscol2 = st.columns(2)
            if _sscol1.button("Continue — create draft", key=_uk("digest_substack_confirm_btn")):
                with st.spinner("Rendering charts/tables and creating the Substack draft..."):
                    _sub_edition = get_or_build_todays_edition(conn)
                    _sub_content, _sub_images = _build_substack_payload(_sub_edition)
                    _result = send_edition_to_substack(conn, _sub_edition, _sub_content["markdown"], _sub_images)
                st.session_state[_uk("_digest_substack_confirm_open")] = False
                if _result.get("ok"):
                    st.success(f"Draft created: {_result.get('draft_url') or _result.get('draft_id')}")
                else:
                    st.error(f"Substack send failed: {_result.get('error')}")
                st.rerun()
            if _sscol2.button("Cancel", key=_uk("digest_substack_cancel_btn")):
                st.session_state[_uk("_digest_substack_confirm_open")] = False
                st.rerun()

    with st.expander("Substack history"):
        _posts = get_substack_posts(conn)
        if not _posts:
            st.caption("No Substack posts yet.")
        else:
            st.dataframe(pd.DataFrame([{
                "Edition": p["edition_date"], "Status": p["status"],
                "Created": fmt_date(p["created_at"]),
                "Draft": p["draft_url"] or "—", "Error": p["error"] or "",
            } for p in _posts]), width='stretch', hide_index=True)

    if st.session_state.get(_uk("_digest_preview_open")):
        with st.expander("👁️ Today's Digest Preview", expanded=True):
            with st.spinner("Loading today's edition (building it if it doesn't exist yet)..."):
                _edition = get_or_build_todays_edition(conn)
                _featured_stock = (_edition.get("sections") or {}).get("featured_stock")
                _featured_ticker = _edition.get("featured_ticker")
                _preview_images, _chart_cid = {}, None
                if _featured_ticker and _featured_stock:
                    _chart_bytes, _chart_cid = build_featured_stock_chart_bytes(_featured_ticker, conn)
                    if _chart_bytes and _chart_cid:
                        _preview_images[_chart_cid] = _chart_bytes
                    _preview_images.update(build_analyst_chart_images(_featured_ticker, _featured_stock, conn))
                    _preview_images.update(build_swing_chart_image(_featured_stock))
                # Fixed (2026-10-14): this used to read _edition["html_full"] directly -- the STORED HTML,
                # rendered headless at build time with the TEMPLATE CODE as it existed that moment, and with
                # chart_cid=None always (data_engine.py has no plotly dependency, so the stored edition never
                # had a real featured-chart <img> tag to begin with). That meant Preview silently went stale
                # the instant either the digest template changed (confirmed live: today's Macro Calendar
                # columns/banners never appeared in Preview until this fix) OR any underlying market data
                # changed (a CPI actual landing, say) -- while Send Test/the real scheduled send/Substack
                # export were ALREADY correct, since they all go through render_edition_for_recipient, a
                # zero-API-call (digest_render_guard-protected) re-render from the edition's own stored
                # content_json (sections/featured_stock -- the EXPENSIVE, Claude-costed part) against
                # whatever the CURRENT template code and CURRENT live data say. Routing Preview through the
                # same function makes it structurally impossible for this to happen again: nothing left reads
                # digest_editions.html_full for display anywhere in this app -- that column is now only ever
                # written as a build-time record, never read back as "the truth."
                _rendered = render_edition_for_recipient(conn, _edition, DEFAULT_DIGEST_SECTIONS, chart_cid=_chart_cid)
                preview_html = _rendered["html_body"]
            _preview_ov = get_featured_override(conn, admin_email)
            _featured_line = f"Featured: {_edition.get('featured_ticker') or '—'} ({_edition.get('featured_source') or 'auto'})"
            if _preview_ov and _preview_ov["ticker"] != _edition.get("featured_ticker"):
                _featured_line += f" — {_preview_ov['ticker']} pending, applies on next rebuild"
            st.caption(f"Subject: {_edition['subject']}  ·  edition {_edition['edition_date']}  ·  "
                       f"built {time_ago(_edition['built_at'])}  ·  "
                       f"{_edition['claude_api_calls']} Claude call(s), ${_edition['claude_api_cost']:.4f}  ·  "
                       f"rendered live from this data just now")
            st.caption(_featured_line)
            # The real email references its charts via `cid:` MIME attachments
            # (email-transport-only, unresolvable inside a sandboxed web
            # iframe) -- swapped here, PREVIEW ONLY, for base64 data: URIs so
            # this in-app preview can display the same images the email will.
            import base64
            for _cid, _png in _preview_images.items():
                data_uri = f"data:image/png;base64,{base64.b64encode(_png).decode()}"
                preview_html = preview_html.replace(f"cid:{_cid}", data_uri)
            # A genuine sandboxed iframe (not st.markdown) since html_body
            # is a full standalone <html><body> document -- exactly the
            # bytes the real email client would render, not a fragment
            # embedded into this page's own DOM.
            st.components.v1.html(preview_html, height=700, scrolling=True)


_TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,10}$")
MAX_WATCHLIST_SIZE = 50  # every ticker costs real shared-API calls in full_refresh


def render_user_watchlist_section():
    st.markdown('<div class="smd-section">WATCHLIST</div>', unsafe_allow_html=True)
    st.caption("Comma-separated tickers. Saved to your account — persists across restarts and sign-ins.")
    wcol1, wcol2 = st.columns([3, 1])
    with wcol1:
        wl_text = st.text_area(
            "Tickers", value=", ".join(watchlist),
            label_visibility="collapsed", height=80, key=_uk("wl_textarea"),
        )
    with wcol2:
        if st.button("Save", width='stretch', key=_uk("wl_save")):
            parsed = [t.strip().upper() for t in wl_text.split(",") if t.strip()]
            bad = [t for t in parsed if not _TICKER_RE.match(t)]
            if bad:
                st.error(f"Not valid ticker symbol(s): {', '.join(bad)}")
            elif len(set(parsed)) > MAX_WATCHLIST_SIZE and not IS_ADMIN:
                st.error(f"Watchlists are limited to {MAX_WATCHLIST_SIZE} tickers.")
            else:
                save_watchlist(get_conn(), EFFECTIVE_EMAIL, parsed or DEFAULT_STARTER_WATCHLIST)
                st.rerun()
        if st.button("Reset to default", width='stretch', key=_uk("wl_reset")):
            save_watchlist(get_conn(), EFFECTIVE_EMAIL, DEFAULT_STARTER_WATCHLIST)
            st.rerun()


def render_user_entry_zones_section():
    st.markdown('<div class="smd-section">ENTRY ZONES</div>', unsafe_allow_html=True)
    st.caption("Auto-derived for each ticker on your watchlist (see the 📍 WATCHLIST SIGNALS tab for the full "
               "reasoning). Recompute here to force a fresh derivation for all of them.")
    conn = get_conn()
    zones = {z["ticker"]: z for z in get_entry_zones(conn, EFFECTIVE_EMAIL)}
    rows = []
    for t in watchlist:
        z = zones.get(t)
        if z is None:
            rows.append({"Ticker": t, "Entry zone": "—", "Status": "not computed yet", "Target": "—", "Computed": "—"})
            continue
        rng = f"${z['entry_low']:.2f}-${z['entry_high']:.2f}" if z.get("entry_low") is not None else "—"
        tgt = f"${z['implied_target_price']:.2f}" if z.get("implied_target_price") is not None else "—"
        rows.append({"Ticker": t, "Entry zone": rng, "Status": (z.get("last_zone_status") or "—").replace("_", " "),
                     "Target": tgt, "Computed": fmt_date(z.get("computed_at")) if z.get("computed_at") else "—"})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    if st.button("🔄 Recompute all entry zones", key=_uk("settings_recompute_all_zones")):
        with st.spinner("Recomputing..."):
            for t in watchlist:
                refresh_entry_zone(conn, EFFECTIVE_EMAIL, t, force_refresh=True)
        st.rerun()


def _on_settings_refresh_change():
    value = st.session_state[_uk("settings_refresh_interval")]
    save_runners_refresh_interval(get_conn(), EFFECTIVE_EMAIL, value)
    # Keep the RUNNERS tab's own widget (same underlying preference) in
    # step, so it doesn't later "correct" the saved value back.
    st.session_state[_uk("runners_refresh_interval")] = value
    st.session_state[_uk("_runners_refresh_interval_saved")] = value


def render_user_refresh_and_chart_prefs_section():
    conn = get_conn()
    st.markdown('<div class="smd-section">REFRESH INTERVALS</div>', unsafe_allow_html=True)
    saved_interval = load_runners_refresh_interval(conn, EFFECTIVE_EMAIL)
    if _uk("settings_refresh_interval") not in st.session_state:
        st.session_state[_uk("settings_refresh_interval")] = saved_interval
    st.selectbox(
        "Auto-refresh interval (RUNNERS Day Predictions + background signals scan)",
        RUNNERS_REFRESH_INTERVAL_OPTIONS, key=_uk("settings_refresh_interval"), on_change=_on_settings_refresh_change,
        help="'On demand' only updates when you interact with the page. Saved to your account.",
    )

    st.markdown('<div class="smd-section">TIMEFRAME & CHART DEFAULTS</div>', unsafe_allow_html=True)
    st.caption("The interval / lookback / refresh a price chart starts with (FUNDAMENTALS, TICKER DEEP-DIVE). "
               "You can still change them on any chart.")
    cur = load_chart_defaults(conn, EFFECTIVE_EMAIL)
    ccol1, ccol2, ccol3 = st.columns(3)
    new_interval = ccol1.selectbox("Default interval", INTERVAL_OPTIONS, index=INTERVAL_OPTIONS.index(cur["interval"]),
                                   key=_uk("settings_chart_interval"))
    new_lookback = ccol2.selectbox("Default lookback", LOOKBACK_OPTIONS, index=LOOKBACK_OPTIONS.index(cur["lookback"]),
                                   key=_uk("settings_chart_lookback"))
    new_refresh = ccol3.selectbox("Default refresh", CHART_REFRESH_OPTIONS,
                                  index=CHART_REFRESH_OPTIONS.index(cur["refresh"]), key=_uk("settings_chart_refresh"))
    if st.button("Save chart defaults", key=_uk("settings_chart_save")):
        save_chart_defaults(conn, EFFECTIVE_EMAIL, new_interval, new_lookback, new_refresh)
        # Drop live chart widget state so the new defaults apply immediately.
        for _k in [k for k in st.session_state.keys()
                   if k.split("::")[0].endswith(("_interval", "_lookback", "_refresh"))
                   and k.startswith(("fund_", "dd_"))]:
            del st.session_state[_k]
        st.success("Chart defaults saved.")
        st.rerun()


def render_settings_tab():
    """SETTINGS -- PER-USER scope only: this user's watchlist, entry zones,
    newsletter, refresh intervals and chart defaults. Deliberately contains
    NOTHING global: no data-source registry, API keys, SMTP, firm tiers or
    other users' data (those live in the admin-only ADMIN tab, which is
    not even rendered for role='user'). For an admin using "View as user",
    this tab shows -- and edits -- THAT user's settings (EFFECTIVE_EMAIL)."""
    if EFFECTIVE_EMAIL != ACTOR_EMAIL:
        st.warning(f"⚠️ You are viewing and editing **{EFFECTIVE_EMAIL}**'s settings as an administrator.", icon="👁️")
    render_user_watchlist_section()
    render_user_entry_zones_section()
    render_email_digest_section()
    render_user_refresh_and_chart_prefs_section()
    if IS_ADMIN:
        # Admins ALSO get every original (global) setting, in this same tab, below the per-user ones.
        # Rendered only here (not also in the ADMIN tab) so widget keys can't collide.
        st.markdown("<hr style='margin:28px 0 8px 0;border-color:#00d4ff55;'>", unsafe_allow_html=True)
        st.markdown('<div class="smd-section">⚙️ ADMIN — GLOBAL SETTINGS (shared by all users)</div>',
                    unsafe_allow_html=True)
        st.caption("Data-source registry, news feeds, SMTP/API keys, firm tiers, reconciliation, backtests and "
                   "calibration. Visible to admins only; these apply to everyone and are not affected by "
                   "'View as user'.")
        render_admin_global_settings()


def render_source_registry_section():
    """ADMIN ONLY. The data-source health registry (global, shared)."""
    st.markdown('<div class="smd-section">SOURCE REGISTRY</div>', unsafe_allow_html=True)
    health = get_health()
    registry = get_full_registry()

    if st.button("🔄 Recheck All Sources", key="health_recheck_all"):
        get_health(force=True)
        get_full_registry(force=True)
        st.rerun()

    st.markdown("**Active fetchers** (what ⟳ REFRESH DATA touches)")
    fcols = st.columns(len(_STATUS_ORDER))
    for i, key in enumerate(_STATUS_ORDER):
        info = health.get(key, {})
        dot, color, label = STATUS_STYLE.get(info.get("status"), STATUS_STYLE["down"])
        with fcols[i]:
            st.markdown(
                f"**{key.upper().replace('_', ' ')}**<br>"
                f"<span style='color:{color};'>{dot} {label}</span>",
                unsafe_allow_html=True,
            )
            if info.get("error"):
                hint = FIX_HINTS.get(key)
                extra = f" — {hint}" if (info.get("status") == "not_configured" and hint) else ""
                st.caption(f"{info['error']}{extra}")
            fallback = info.get("fallback")
            if fallback:
                fdot, fcolor, flabel = STATUS_STYLE.get(fallback.get("status"), STATUS_STYLE["down"])
                fb_line = f"fallback ({fallback.get('source')}): {fdot} {flabel}"
                if fallback.get("error"):
                    fb_line += f" — {fallback['error']}"
                st.caption(f"↳ {fb_line}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Full source registry**")
    reg_rows = []
    for row in registry["sources"]:
        status = row.get("status")
        if status == "not_applicable":
            status_label = "○ N/A — reference only"
        else:
            dot, _color, label = STATUS_STYLE.get(status, STATUS_STYLE["down"])
            status_label = f"{dot} {label}"
        reg_rows.append({
            "Source": row["key"],
            "Purpose": row["purpose"],
            "Status": status_label,
            "Reason": row.get("error") or "",
            "Env var": row.get("env_var") or "—",
            "Endpoint": row.get("endpoint") or "—",
        })
    st.dataframe(pd.DataFrame(reg_rows), width='stretch', hide_index=True)
    st.caption(f"Checked at {registry.get('checked_at', '—')}")


def _env_status(name):
    return "✅ set" if os.environ.get(name) else "❌ not set"


def render_smtp_and_api_keys_section():
    """ADMIN ONLY. SMTP configuration + API-key status (presence only --
    secret VALUES are never displayed anywhere in the app) + the global
    sender identity."""
    conn = get_conn()
    st.markdown('<div class="smd-section">SMTP & SENDER</div>', unsafe_allow_html=True)
    smtp_rows = [{"Setting": v, "Status": _env_status(v)} for v in
                 ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD")]
    st.dataframe(pd.DataFrame(smtp_rows), width='stretch', hide_index=True)
    if all(os.environ.get(v) for v in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD")):
        st.success("✅ SMTP configured", icon="✅")
    else:
        st.warning("⚠️ SMTP not configured — set SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD environment "
                   "variables. No user's digest can be sent until then.", icon="⚠️")
    _email_from_env = os.environ.get("EMAIL_FROM")
    if _email_from_env:
        st.caption(f"From address: **{_email_from_env}** (set via the EMAIL_FROM environment variable — "
                   f"overrides the field below until that variable is unset).")
    else:
        stored_from = get_app_setting(conn, "from_email") or DEFAULT_DIGEST_FROM_EMAIL
        new_from = st.text_input(
            "Sender (From) address — must be a verified sender/domain identity in your SMTP provider. "
            "Applies to every user's digest.",
            value=stored_from, key="admin_from_email_input", placeholder="digest@yourdomain.com",
        )
        new_name = st.text_input(
            "Sender display name — shown in the inbox as “Name <address>”.",
            value=get_app_setting(conn, "from_name") or de.DEFAULT_DIGEST_FROM_NAME, key="admin_from_name_input",
        )
        if st.button("Save sender address", key="admin_from_email_save"):
            set_app_setting(conn, "from_email", new_from.strip() or DEFAULT_DIGEST_FROM_EMAIL)
            set_app_setting(conn, "from_name", new_name.strip() or de.DEFAULT_DIGEST_FROM_NAME)
            st.success("Sender saved.")
            st.rerun()

    st.markdown('<div class="smd-section">API KEYS</div>', unsafe_allow_html=True)
    st.caption("Presence only — key values are never shown. Set them in the server environment (setup_env.sh).")
    key_rows = [{"Variable": v, "Used for": purpose, "Status": _env_status(v)} for v, purpose in (
        ("ANTHROPIC_API_KEY", "AI Briefing (paid, on-demand; one shared cache for all users)"),
        ("QUIVER_API_KEY", "Congressional trades via QuiverQuant (paid)"),
        ("FRED_API_KEY", "Fed funds rate / macro calendar"),
        ("OPENAI_API_KEY", "Optional"),
        ("BOOTSTRAP_ADMIN_EMAIL", "First-admin gate (only consulted while the users table is empty)"),
        ("APP_PUBLIC_URL", "Base URL used in emailed unsubscribe links"),
        ("HEALTHCHECKS_API_KEY", "Digest dead-man's-switch (healthchecks.io) -- see Digest scheduler below"),
    )]
    st.dataframe(pd.DataFrame(key_rows), width='stretch', hide_index=True)


def _launchd_job_installed():
    """True if the headless digest launchd job is currently loaded (`launchctl print`/`list`), False if not
    OR if launchctl itself isn't available (e.g. not macOS, or this process can't reach launchd) -- fails
    soft either way, never raises, since this is purely informational."""
    import subprocess
    try:
        result = subprocess.run(
            ["launchctl", "list", "com.smartmoney.digest"], capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def render_digest_scheduler_section():
    """ADMIN ONLY. Install status + instructions for the headless launchd digest scheduler (digest/send.py,
    `python -m digest.send`) -- this is a one-time, per-machine setup step (unlike Send time/Frequency,
    which every user controls for themselves and which take effect immediately with no plist changes)."""
    st.markdown('<div class="smd-section">DIGEST SCHEDULER (launchd)</div>', unsafe_allow_html=True)
    st.caption(
        "Runs the daily digest headlessly (`python -m digest.send`, no Streamlit/browser needed) every :00 "
        "and :30, all day. Each user's own Send time/Frequency/'Wait for...' settings (in their own "
        "SETTINGS → Newsletter) decide what actually goes out on any given tick -- nothing about timing "
        "lives in the scheduler itself, so this only needs to be installed ONCE per machine, never touched "
        "again after that."
    )
    if _launchd_job_installed():
        st.success("✅ Installed and loaded (`com.smartmoney.digest`).", icon="✅")
    else:
        st.warning("⚠️ Not installed — scheduled sends currently depend on a browser tab being open.", icon="⚠️")
    st.markdown("**Install / reinstall** (run in Terminal, from the project directory):")
    st.code("./digest/install_launchd.sh", language="bash")
    st.markdown("**Uninstall:**")
    st.code("./digest/uninstall_launchd.sh", language="bash")
    st.markdown("**Verify it's loaded:**")
    st.code("launchctl list | grep smartmoney", language="bash")
    st.markdown("**Watch the logs:**")
    st.code(f"tail -f {os.getcwd()}/digest/logs/digest.out.log", language="bash")
    st.caption(
        "The wake-schedule (`pmset`) command needed for this to fire even while the Mac would otherwise "
        "sleep is shown per-user in SETTINGS → Newsletter → 'Reliable wake schedule' (it needs sudo, so it "
        "can never be run automatically). If different users have different Send times, use the EARLIEST "
        "enabled one for the machine's wake schedule -- pmset only supports one repeating wake time at "
        "once, so the machine must be awake for the earliest of everyone's ticks."
    )


def render_admin_global_settings():
    """ADMIN ONLY. Everything global/shared that used to sit in SETTINGS."""
    render_source_registry_section()
    render_news_feeds_section()
    render_smtp_and_api_keys_section()
    render_digest_scheduler_section()
    render_firm_tiers_section()

    st.markdown('<div class="smd-section">EARNINGS SIMULATOR TRACK RECORD</div>', unsafe_allow_html=True)
    if st.button("🔄 Reconcile Predictions", key="reconcile_predictions_btn"):
        result = reconcile_earnings_predictions(get_conn())
        st.success(
            f"Checked {result['checked_count']} pending final prediction(s), reconciled "
            f"{result['reconciled_count']}."
        )
        st.rerun()
    render_simulator_track_record_panel()

    st.markdown('<div class="smd-section">DAY PREDICTION TRACK RECORD</div>', unsafe_allow_html=True)
    if st.button("🔄 Reconcile Today's Predictions", key="reconcile_day_predictions_btn"):
        result = reconcile_day_predictions(get_conn())
        # Part 4: calibration is part of the same reconciliation job, not
        # a separate manual step -- runs immediately after every manual
        # reconcile too.
        calibrate_day_prediction_model(get_conn())
        st.success(
            f"Checked {result['checked_count']} pending prior-session prediction(s), reconciled "
            f"{result['reconciled_count']}."
        )
        st.rerun()
    render_day_prediction_track_record_panel()
    render_day_prediction_calibration_panel()
    render_backtested_tickers_registry()


def render_day_prediction_calibration_panel():
    """Part 4 -- honest display of calibrate_day_prediction_model's
    current state: real N, real mean error against the stated 0-3% target
    (never implying precision the sample size doesn't support), the
    active drift/vol scaling factors, and the run-by-run history so it's
    visible whether/how the model's assumptions have actually shifted."""
    conn = get_conn()
    track = get_day_prediction_track_record(conn)
    calibration = get_active_day_prediction_calibration(conn)
    st.caption(
        "Calibration loop: after every reconciliation, calibrate_day_prediction_model() checks the reconciled "
        "history for a systematic magnitude or direction bias and nudges the model's drift/volatility scaling "
        "to correct it — a real adjustment, not just more data collection. Hitting a consistent 0-3% same-day "
        "error on a single name is a genuinely hard target even for professional intraday models; this loop "
        "is what the model is working toward, not a guarantee, and progress is shown honestly below."
    )
    if not track["n"]:
        st.info("Current average error: no reconciled sessions yet (target: 0-3%). Calibration has nothing to work from yet.")
        return
    within_target = track["mean_abs_error_pct"] is not None and track["mean_abs_error_pct"] <= 3.0
    st.markdown(
        f"**Current average error: {track['mean_abs_error_pct']:.2f}% over {track['n']} reconciled session(s) "
        f"(target: 0-3%)** — {'within target.' if within_target else 'still above target.'}"
    )
    if track["n"] < DAY_CALIBRATION_MIN_SAMPLES:
        st.caption(
            f"Only {track['n']} reconciled session(s) — below the {DAY_CALIBRATION_MIN_SAMPLES}-sample floor "
            f"calibrate_day_prediction_model() requires before adjusting anything. This number will move around "
            f"a lot session-to-session until there's real volume behind it; treat it as a direction, not a "
            f"precise readout, until then."
        )
    ccol1, ccol2 = st.columns(2)
    ccol1.metric("Active drift scale", f"{calibration['drift_scale']:.2f}", help="1.00 = no adjustment (default).")
    ccol2.metric("Active volatility scale", f"{calibration['vol_scale']:.2f}", help="1.00 = no adjustment (default).")

    history = get_day_prediction_calibration_history(conn, limit=10)
    if history:
        with st.expander(f"Calibration run history ({len(history)} most recent)"):
            hist_df = pd.DataFrame([{
                "Run": h["calibrated_at"], "N": h["n_samples"],
                "Mean error": f"{h['mean_abs_error_pct']:.2f}%" if h["mean_abs_error_pct"] is not None else "—",
                "Drift scale": f"{h['drift_scale']:.2f}", "Vol scale": f"{h['vol_scale']:.2f}",
                "Note": h["note"],
            } for h in history])
            st.dataframe(hist_df, width='stretch', hide_index=True)


def render_backtested_tickers_registry():
    """Part 6 -- every ticker that's ever been backtested (latest run
    per ticker), plus a per-row "Re-run Backtest" button to keep
    calibration current as the trailing lookback_days window rolls
    forward. Mode is shown per row for reference but is pooled system-
    wide, not actually per-ticker -- stated explicitly so this table
    doesn't imply something the model doesn't do."""
    conn = get_conn()
    st.markdown('<div class="smd-section">BACKTESTED TICKERS</div>', unsafe_allow_html=True)
    summary = get_backtested_tickers_summary(conn)
    if not summary:
        st.caption(
            "No tickers backtested yet — run one from the TICKER DEEP-DIVE tab's DAY PREDICTION BACKTEST "
            "section (search any ticker, click \"🔬 Run Backtest\"), or it'll happen automatically the first "
            "time a new ticker enters RUNNERS."
        )
        return
    mode, pooled_n = get_current_day_prediction_mode(conn)
    mode_label = {"raw_pattern": "Mode 1", "bayesian": "Mode 2", "trained_model": "Mode 3"}.get(mode, mode)
    st.caption(
        f"Mode is pooled system-wide (currently **{mode_label}**, {pooled_n} total reconciled sessions across "
        f"all tickers) — not selected per ticker; shown per row below for reference only."
    )
    hcol1, hcol2, hcol3, hcol4, hcol5, hcol6 = st.columns([1, 1.6, 1.3, 1.1, 1, 1.1])
    for label, col in zip(
        ["Ticker", "Last run", "Sessions found", "Avg |error|", "Mode", ""],
        [hcol1, hcol2, hcol3, hcol4, hcol5, hcol6],
    ):
        col.markdown(f"**{label}**")
    for row in summary:
        rcol1, rcol2, rcol3, rcol4, rcol5, rcol6 = st.columns([1, 1.6, 1.3, 1.1, 1, 1.1])
        rcol1.markdown(row["ticker"])
        rcol2.caption(row["run_at"][:19].replace("T", " "))
        rcol3.caption(f"{row['qualifying_sessions_found']} / {row['lookback_days']}d")
        rcol4.caption(f"{row['mean_abs_error_pct']:.2f}%" if row["mean_abs_error_pct"] is not None else "—")
        rcol5.caption(mode_label)
        if rcol6.button("🔄 Re-run", key=f"rerun_backtest_{row['ticker']}"):
            with st.spinner(f"Re-running backtest for {row['ticker']}..."):
                backtest_day_predictions(row["ticker"], conn)
            st.rerun()


def render_day_prediction_track_record_panel():
    """Part 5/6 -- mirrors render_simulator_track_record_panel exactly,
    for RUNNERS' Day Prediction feature: overall accuracy + mean |error%|
    on the final target AND mean path |error%| (the intraday SHAPE
    reconciliation, not just the end-of-day call) across every reconciled
    day_predictions row."""
    track = get_day_prediction_track_record(get_conn())
    if not track["n"]:
        st.caption(
            "No reconciled Day Predictions yet — accuracy populates once a session ends and reconciliation "
            "runs (manually via the button above, or automatically once per day). This is Part 3's case 2: a "
            "genuinely thin PREDICTION track record, distinct from missing raw price/volatility data (which "
            "is now backfilled synchronously per ticker, in seconds, not something that takes days to resolve)."
        )
        return
    st.caption(
        "Mode 1 (raw pattern) runs until 10+ predictions reconcile system-wide; Mode 2 (statistical) from "
        "there; Mode 3 (trained model) only once it demonstrably beats Mode 2 on held-out data -- the exact "
        "same thresholds as the Earnings Simulator, applied to daily sessions instead."
    )
    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("Directional accuracy", f"{track['accuracy']:.0%}", f"{track['n']} reconciled")
    tcol2.metric(
        "Mean target error", f"{track['mean_abs_error_pct']:.2f}%" if track["mean_abs_error_pct"] is not None else "—",
    )
    tcol3.metric(
        "Mean path error", f"{track['mean_path_abs_error_pct']:.2f}%"
        if track["mean_path_abs_error_pct"] is not None else "—",
        help="Mean |error%| between the simulated intraday path and real logged snapshots -- whether the "
             "PATH SHAPE was realistic, not just whether the final direction call was right.",
    )


def render_simulator_track_record_panel():
    """Part 12 -- honest track-record display: overall accuracy, broken
    down by mode and (where there's enough data) by ticker, across every
    reconciled earnings_predictions row. Shared between SETTINGS and the
    EARNINGS SIMULATOR tab so there's one definition of "the track
    record," not two potentially-different views of the same numbers."""
    track = get_simulator_track_record(get_conn())
    if not track["overall"]:
        st.caption(
            "No reconciled predictions yet — accuracy populates once earnings dates pass and "
            "reconciliation runs (manually via the button above, or automatically once per day)."
        )
        return
    st.caption(
        "This is a heuristic engine with a real feedback loop, not genuine ML yet -- the trained model "
        "(Mode 3) only activates once it demonstrably beats the Bayesian estimate (Mode 2) on held-out "
        "data. Accuracy below improves as this reconciliation log grows."
    )
    o = track["overall"]
    st.metric("Overall accuracy", f"{o['accuracy']:.0%}", f"{o['correct']}/{o['n']} reconciled predictions")
    tcol1, tcol2 = st.columns(2)
    if track["by_mode"]:
        with tcol1:
            st.markdown("**By mode**")
            st.dataframe(
                pd.DataFrame([
                    {"Mode": m, "Accuracy": f"{v['accuracy']:.0%}", "N": v["n"]}
                    for m, v in track["by_mode"].items()
                ]),
                width='stretch', hide_index=True,
            )
    if track["by_ticker"]:
        with tcol2:
            st.markdown("**By ticker** (2+ reconciled predictions)")
            st.dataframe(
                pd.DataFrame([
                    {"Ticker": t, "Accuracy": f"{v['accuracy']:.0%}", "N": v["n"]}
                    for t, v in track["by_ticker"].items()
                ]),
                width='stretch', hide_index=True,
            )


def render_scenario_matrix_table(rows):
    """Custom HTML table for the SCENARIO BREAKDOWN section -- st.dataframe
    (a canvas-based grid) can't wrap long text within a cell, so a full
    Reasoning sentence was getting silently truncated with no way to read
    the rest. table-layout:fixed + explicit column-width percentages +
    white-space:normal/word-wrap:break-word on every cell means each row
    simply grows tall enough to fit its full Reasoning text instead of
    cutting it off; the outer overflow-x:auto div is a safety net only,
    not the fix itself."""
    header_cols = [
        ("EPS", "7%"), ("Revenue", "8%"), ("Named Catalyst", "17%"), ("Probability", "8%"),
        ("Direction", "9%"), ("Magnitude", "11%"), ("Reasoning", "40%"),
    ]
    thead = "".join(
        f'<th style="width:{w};text-align:left;padding:8px 10px;font-size:11px;'
        f'text-transform:uppercase;letter-spacing:0.04em;color:{TEXT_MUTED};'
        f'border-bottom:1px solid rgba(255,255,255,0.14);">{html.escape(label)}</th>'
        for label, w in header_cols
    )
    body_rows = []
    for r in rows:
        cells = [
            r["eps"], r["revenue"], r["catalyst"], f"{r['probability']:.0%}",
            r["expected_direction"], r["expected_magnitude"], r["reasoning"],
        ]
        tds = "".join(
            f'<td style="padding:8px 10px;font-size:13px;color:{TEXT_PRIMARY};vertical-align:top;'
            f'white-space:normal;word-wrap:break-word;overflow-wrap:break-word;'
            f'border-bottom:1px solid rgba(255,255,255,0.07);">{html_safe_snippet(str(cell))}</td>'
            for cell in cells
        )
        body_rows.append(f"<tr>{tds}</tr>")
    st.markdown(
        f'<div style="overflow-x:auto;width:100%;">'
        f'<table style="width:100%;table-layout:fixed;border-collapse:collapse;">'
        f'<thead><tr>{thead}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def render_earnings_simulator_tab():
    """EARNINGS SIMULATOR tab (Parts 1-12). Gated on a recent (<4h) AI
    Briefing (Part 1) -- the whole engine consumes that Briefing's own
    synthesis as primary evidence (Part 2), combined with live market-
    pricing signals (Part 3) that need no gating, through a two-phase
    probability engine (Part 4) that's honest about how thin the evidence
    still is. Never runs anything on page load beyond the gate check --
    every real computation below is already fast/free (no LLM call
    anywhere in this tab; the AI Briefing itself was already generated,
    on request, elsewhere)."""
    st.markdown('<div class="smd-section">SEARCH ANY TICKER</div>', unsafe_allow_html=True)
    sim_ticker = st.text_input(
        "Ticker", value=st.session_state.get("sim_ticker", ""), label_visibility="collapsed",
        placeholder="e.g. PYPL, MU, STX", key="sim_ticker_input",
    ).strip().upper()

    if not sim_ticker:
        st.caption("Enter a ticker to run the simulator.")
        return
    st.session_state["sim_ticker"] = sim_ticker
    conn = get_conn()

    brief = get_cached_ai_brief(conn, sim_ticker, max_age_hours=AI_BRIEF_CACHE_HOURS)
    if not brief:
        st.warning(
            f"Run the AI Briefing for {sim_ticker} first — the simulator uses its full synthesis "
            f"(catalysts, institutional read, news context) as its primary input."
        )
        st.caption(
            "Go to the **TICKER DEEP-DIVE** tab, search this ticker, and click '🧠 Generate AI Briefing'. "
            "Come back here once it's done (cached for 4 hours)."
        )
        if st.button(f"Pre-fill {sim_ticker} on TICKER DEEP-DIVE", key="sim_prefill_dd"):
            st.session_state["dd_ticker"] = sim_ticker
            st.info("Ticker pre-filled — click the TICKER DEEP-DIVE tab above.")
        return

    age_h = None
    fetched_dt = pd.to_datetime(brief.get("_fetched_at"), utc=True, errors="coerce")
    if pd.notna(fetched_dt):
        age_h = (pd.Timestamp.now(tz="UTC") - fetched_dt).total_seconds() / 3600
    st.caption(
        f"Using AI Briefing generated {f'{age_h:.1f}h ago' if age_h is not None else 'recently'} "
        f"(fresh within the {AI_BRIEF_CACHE_HOURS}h window)."
    )

    # Part 6.1 -- prominent earnings-date label at the very top, before
    # anything else renders, so it's never ambiguous when the event
    # actually is relative to any chart/table further down the page.
    # Pulled straight from the Briefing's own "Next earnings" verdict --
    # no separate fetch.
    next_verdict_preview = next(
        (v for v in (brief.get("verdicts") or []) if "earnings" in (v.get("horizon") or "").lower()), None
    )
    earnings_date_preview = (next_verdict_preview or {}).get("target_date")
    if earnings_date_preview:
        days_txt = ""
        try:
            days_out = (pd.Timestamp(earnings_date_preview).date() - date.today()).days
            days_txt = f" ({days_out} days from today)" if days_out >= 0 else " (already passed)"
        except (TypeError, ValueError):
            pass
        st.markdown(
            f"<div style='font-size:16px;font-weight:700;color:{ACCENT};margin:2px 0 12px 0;'>"
            f"📅 Earnings Date: {md_safe(earnings_date_preview)}{days_txt}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No confirmed next-earnings date found in the current Briefing.")

    with st.spinner(f"Running probability engine for {sim_ticker}..."):
        prob = estimate_earnings_probability(sim_ticker, conn)
        strategy = recommend_earnings_strategy(sim_ticker, conn, probability_result=prob)

    mode = prob["mode"]
    mode_badge = {
        "raw_pattern": ("RAW PATTERN — thin evidence", COLOR_RETAIL),
        "bayesian": ("BAYESIAN ESTIMATE — still building evidence", COLOR_INSTITUTIONAL),
        "trained_model": ("TRAINED MODEL — validated", COLOR_BULLISH),
    }.get(mode, (mode, COLOR_NEUTRAL))
    magnitude = prob.get("magnitude_estimate_pct")
    market = prob.get("market_pricing") or {}
    spot = market.get("current_price")

    # ------------------------------------------------------------------
    # BLOCK A — "What the numbers say" (Part 1): unchanged Mode 1/2/3
    # gating logic, just relabeled. Always renders.
    # ------------------------------------------------------------------
    st.markdown('<div class="smd-section">BLOCK A — WHAT THE NUMBERS SAY (QUANTITATIVE)</div>',
               unsafe_allow_html=True)
    st.markdown(interp_badge_html(mode_badge[0], mode_badge[1]), unsafe_allow_html=True)
    st.caption(
        f"Magnitude estimate: ±{magnitude:.1f}% (from current IV pricing — high confidence, live market "
        f"estimate)." if magnitude is not None else "No live IV-based magnitude estimate available."
    )

    if mode == "raw_pattern":
        st.info(md_safe(prob["sentence"]))
        st.caption(md_safe(prob["note"]))
        rc = prob["raw_counts"]
        st.dataframe(pd.DataFrame([{
            "Beats on record": rc["beats"], "Beats followed by UP move": rc["beats_followed_by_up_move"],
            "Total tracked quarters": rc["total_tracked_quarters"],
            "Pooled reconciled predictions (system-wide)": rc["pooled_reconciled_predictions"],
        }]), width='stretch', hide_index=True)
    else:
        if mode == "bayesian":
            n_pooled = conn.execute(
                "SELECT COUNT(*) FROM earnings_predictions WHERE prediction_correct IS NOT NULL"
            ).fetchone()[0]
            st.caption(f"Bayesian estimate, still building evidence ({n_pooled} pooled outcomes system-wide).")
        elif mode == "trained_model" and prob.get("model_meta"):
            held_out = prob["model_meta"]["held_out_accuracy"]
            st.caption(f"Trained model, validated at {held_out:.0%} held-out accuracy.")
        st.plotly_chart(
            render_earnings_probability_chart(
                prob["prob_down"], prob["prob_flat"], prob["prob_up"], magnitude, spot,
            ),
            width='stretch', key=f"sim_prob_chart_{sim_ticker}",
        )
        st.markdown(
            f"Directional lean: **{prob['prob_up']:.0%} up / {prob['prob_down']:.0%} down** "
            f"(confidence: **{prob['confidence_level']}**)"
        )
        if prob.get("reasoning"):
            with st.expander("Why this number — reasoning", expanded=False):
                for r in prob["reasoning"]:
                    st.markdown(f"- {md_safe(r)}")

    # ------------------------------------------------------------------
    # BLOCK B — "What the current setup says" (Parts 1-2, the bug fix):
    # ALWAYS renders once a Briefing exists, independent of Block A's
    # mode. Pulled straight from prob["qualitative_read"], which
    # estimate_earnings_probability() now computes unconditionally
    # BEFORE branching on Mode 1/2/3 -- this is the fix for the bug
    # where this synthesis was computed but never reached the page.
    # ------------------------------------------------------------------
    st.markdown('<div class="smd-section">BLOCK B — WHAT THE CURRENT SETUP SAYS (QUALITATIVE)</div>',
               unsafe_allow_html=True)
    qual = prob.get("qualitative_read")
    if not qual:
        st.caption("No qualitative read available from the Briefing.")
    else:
        lean_color = (
            COLOR_BULLISH if qual["bias"] == "BULLISH" else COLOR_BEARISH if qual["bias"] == "BEARISH"
            else COLOR_NEUTRAL
        )
        st.markdown(
            f"**Directional lean: <span style='color:{lean_color};'>{qual['lean_phrase']}</span> "
            f"({qual['confidence']} confidence)**",
            unsafe_allow_html=True,
        )
        st.markdown(md_safe(qual["reasoning_text"]))
        if qual.get("key_risk"):
            st.caption(f"Key risk (per the Briefing): {md_safe(qual['key_risk'])}")
        st.caption(
            "This is the AI Briefing's own synthesized verdict, restated with the specific inputs that "
            "drove it — not a new probability number."
        )

    # ------------------------------------------------------------------
    # SCENARIO BREAKDOWN (Parts 1-3 of the scenario-matrix fix): the full
    # EPS x Revenue x (optional) Catalyst combinatorial matrix, always
    # rendered once a Briefing exists -- this is the mechanism that
    # surfaces a "beats everything but the named catalyst hasn't landed"
    # trap as its own explicit row, instead of it disappearing into one
    # blended probability. Block A's/Mode 2's prob_up/down/flat above are
    # DERIVED from summing these same rows (Part 4) -- never a separate,
    # potentially-inconsistent number.
    # ------------------------------------------------------------------
    st.markdown('<div class="smd-section">SCENARIO BREAKDOWN</div>', unsafe_allow_html=True)
    scenario_matrix = prob.get("scenario_matrix")
    if not scenario_matrix or not scenario_matrix.get("rows"):
        st.caption("No scenario breakdown available.")
    else:
        if scenario_matrix.get("named_catalyst"):
            st.caption(
                f"Named catalyst (from the AI Briefing's CATALYSTS TO WATCH): "
                f"**{md_safe(scenario_matrix['named_catalyst'])}** — split into its own Achieved/Delayed rows below."
            )
        else:
            st.caption(
                "No specific product/technology catalyst named in the current Briefing — showing EPS × "
                "Revenue combinations only (Catalyst column is N/A, not fabricated)."
            )
        st.caption(scenario_matrix["sample_size_note"])
        # Part 1 of the magnitude-convention fix: the table's magnitudes
        # use a FIXED historical convention (~5-9% directional, 0-3% flat),
        # never the ticker's raw, volatile IV-implied move -- shown here as
        # a clearly separate, explicitly-labeled figure so both are visible
        # without conflating "the scenario we're testing" with "what the
        # market is currently pricing."
        conv_col, mkt_col = st.columns(2)
        conv_col.info(f"📐 Scenario convention: {md_safe(scenario_matrix['scenario_convention_note'])}")
        implied = scenario_matrix.get("market_implied_move_pct")
        atm_iv = scenario_matrix.get("atm_iv_pct")
        if implied is not None:
            iv_txt = f" (from current {atm_iv:.0f}% IV)" if atm_iv is not None else ""
            bigger_txt = (
                " — the market is pricing a much bigger swing than a typical earnings reaction, itself "
                "a signal worth noting" if atm_iv and atm_iv >= 80 else ""
            )
            mkt_col.warning(f"📊 Market-implied move: ±{implied:.1f}%{iv_txt}{md_safe(bigger_txt)}")
        else:
            mkt_col.caption("Market-implied move: unavailable.")
        render_scenario_matrix_table(scenario_matrix["rows"])
        total_pct = sum(r["probability"] for r in scenario_matrix["rows"])
        st.caption(f"Probabilities sum to {total_pct:.0%}.")

    st.markdown('<div class="smd-section">OPTIONS STRATEGY</div>', unsafe_allow_html=True)
    st.caption(md_safe(strategy["note"]))
    if strategy.get("caution"):
        st.warning(md_safe(strategy["caution"]))

    candidates_raw = strategy["candidates"]
    all_candidates = (
        candidates_raw if isinstance(candidates_raw, list)
        else candidates_raw.get("calls", []) + candidates_raw.get("puts", [])
    )
    if not all_candidates:
        st.caption("No candidates matched the delta/liquidity filters for this ticker's chain right now.")
    else:
        badge_defs = [
            ("delta", DELTA_BADGE_COLORS, None), ("iv", IV_BADGE_COLORS, None),
            ("vol_oi", None, VOL_OI_PROFILES), ("dte", None, None),
            ("theta", None, THETA_PCT_PROFILES), ("spread", None, SPREAD_PROFILES),
        ]
        for i, c in enumerate(all_candidates):
            type_color = COLOR_BULLISH if c["type"] == "call" else COLOR_BEARISH
            badges_html = []
            for key, color_map, tier_profiles in badge_defs:
                b = c["badges"].get(key) or {}
                label = b.get("label")
                if color_map is not None:
                    color = color_map.get(label, COLOR_NEUTRAL)
                elif tier_profiles is not None:
                    color = _tier_escalation_color(label, tier_profiles)
                else:
                    color = COLOR_NEUTRAL
                badges_html.append(interp_badge_html(label, color, b.get("note")))

            analysis = analyze_earnings_contract(
                sim_ticker, conn, c, market=market, earnings_date=strategy.get("earnings_date"),
                scenario_matrix=scenario_matrix,
            )

            # Raw Greeks alongside the interpretive badges above -- badges
            # answer "what kind of trade is this," these numbers answer
            # "exactly how much." Same bs_greeks() values already stored on
            # options_flow and used to assign the delta/theta badges above
            # (see _earnings_candidate_contracts), not recomputed here.
            # Theta/vega are per-share in storage (standard Black-Scholes
            # convention) -- x100 to show the standard per-contract dollar
            # figures a trader actually thinks in, matching how delta/gamma
            # are conventionally shown per-share (undecorated 0.XX/0.XXXX)
            # but theta/vega per-contract ($X.XX).
            delta_txt = f"{c['delta']:.2f}" if c.get("delta") is not None else "—"
            gamma_txt = f"{c['gamma']:.4f}" if c.get("gamma") is not None else "—"
            theta_txt = f"-${abs(c['theta']) * 100:.2f}/day" if c.get("theta") is not None else "—"
            vega_txt = f"${c['vega'] * 100:.2f}" if c.get("vega") is not None else "—"
            greeks_html = (
                f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:14px;font-size:12px;'
                f'color:{TEXT_SECONDARY};">'
                f'<span>Delta: <b style="color:{TEXT_PRIMARY};">{delta_txt}</b></span>'
                f'<span>Gamma: <b style="color:{TEXT_PRIMARY};">{gamma_txt}</b></span>'
                f'<span>Theta: <b style="color:{TEXT_PRIMARY};">{theta_txt}</b></span>'
                f'<span>Vega: <b style="color:{TEXT_PRIMARY};">{vega_txt}</b></span>'
                f'</div>'
            )

            st.markdown(md_safe(f"""
            <div class="smcard">
              <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <div style="font-size:15px;font-weight:700;color:{TEXT_PRIMARY};">
                  <span style="color:{type_color};">{c['type'].upper()}</span>
                  ${c['strike']:g} &mdash; {html.escape(str(c['expiration']))}
                </div>
                <div style="font-size:12px;color:{TEXT_MUTED};">{c.get('dte', '—')} DTE</div>
              </div>
              <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px;">{"".join(badges_html)}</div>
              {greeks_html}
            </div>
            """), unsafe_allow_html=True)

            if not analysis:
                st.caption("Not enough data to simulate this contract (missing price/IV/DTE).")
                continue

            # Part 9 order: header stats -> P/L chart -> reasoning -> IV
            # crush transparency -> heatmap (expandable).
            pop = analysis["probability_of_profit"]
            max_profit_txt = "Uncapped" if c["type"] == "call" else f"${c['strike'] * 100:,.0f} (if stock → $0)"
            scol1, scol2, scol3, scol4, scol5 = st.columns(5)
            scol1.metric("Net Debit", f"${analysis['net_debit']:,.0f}")
            scol2.metric("Max Loss", f"${analysis['max_loss']:,.0f}")
            scol3.metric("Max Profit", max_profit_txt)
            scol4.metric("Chance of Profit", f"{pop:.0%}" if pop is not None else "—")
            scol5.metric("Breakeven", f"${analysis['breakeven']:.2f}" if analysis["breakeven"] else "—")

            pl_fig = render_pl_curve_chart(analysis["pl_curve"], analysis["pl_curve_earnings"], spot)
            if pl_fig is not None:
                st.plotly_chart(pl_fig, width='stretch', key=f"sim_plcurve_{sim_ticker}_{i}")

            st.markdown(f"*{md_safe(analysis['reasoning'])}*")

            crush = analysis["iv_crush"]
            entry_iv, post_iv = c.get("iv_pct"), analysis["contract_pl"].get("post_earnings_iv_pct")
            st.caption(
                f"Pre-earnings IV: {entry_iv:.1f}% → Modeled post-earnings IV: ~{post_iv:.0f}% "
                + (f"(based on {crush['n_events']} real historical IV-crush event(s) for this ticker)."
                   if crush["source"] == "historical" else
                   "(conservative default reversion — no historical pre/post-earnings IV pattern on record "
                   "for this ticker yet, so this is an estimate, not measured.)")
                if entry_iv is not None and post_iv is not None else "IV crush estimate unavailable."
            )
            decomp_rows = [{
                "Scenario": s["scenario"], "Price move": f"{s['price_move_pct']:+.1f}%",
                "P/L from price move": f"{s['price_component_dollar']:+,.2f}",
                "P/L from IV crush": f"{s['iv_crush_component_dollar']:+,.2f}",
                "Total P/L ($)": f"{s['pl_dollar']:+,.2f}", "Total P/L (%)": f"{s['pl_pct']:+.1f}%",
            } for s in analysis["contract_pl"]["scenarios"]]
            st.dataframe(pd.DataFrame(decomp_rows), width='stretch', hide_index=True)

            with st.expander("Detailed grid — price × date"):
                hm_fig = render_pl_heatmap_chart(analysis["heatmap"])
                if hm_fig is not None:
                    st.plotly_chart(hm_fig, width='stretch', key=f"sim_heatmap_{sim_ticker}_{i}")
                    if analysis["heatmap"].get("earnings_date_in_range"):
                        st.caption(
                            f"Earnings date ({analysis['heatmap']['earnings_date_in_range']}) is included as "
                            f"a column — IV crush applies to that date and every date after it."
                        )
                else:
                    st.caption("Not enough data to build the detailed grid for this contract.")

    st.markdown('<div class="smd-section">LOG THIS PREDICTION</div>', unsafe_allow_html=True)
    st.caption(
        "Logs a timestamped snapshot of this run to earnings_predictions -- every run is logged (the "
        "timeline itself is useful data), but only the FINAL check-in before the earnings date should be "
        "marked final -- that's the one row reconciled against the actual outcome afterward."
    )
    lcol1, lcol2 = st.columns([1, 2])
    is_final = lcol1.checkbox("This is the final pre-earnings check-in", key=f"sim_final_{sim_ticker}")
    if lcol2.button("📝 Log Prediction", key=f"sim_log_{sim_ticker}"):
        log_earnings_prediction(conn, sim_ticker, prob, strategy_result=strategy, is_final_prediction=is_final)
        st.success(f"Logged {'FINAL ' if is_final else ''}prediction for {sim_ticker}.")

    with st.expander("📚 Backfill historical earnings data / view track record"):
        if st.button(f"Backfill historical earnings for {sim_ticker}", key=f"sim_backfill_{sim_ticker}"):
            with st.spinner("Scraping real historical earnings + price reactions..."):
                result = backfill_earnings_history(sim_ticker, conn)
            st.success(result["confidence_note"])
        render_simulator_track_record_panel()


_MACRO_IMPACT_BADGE = {
    "High": ("🔴", COLOR_BEARISH), "Medium": ("🟠", "#e8c547"), "Low": ("⚪", TEXT_SECONDARY),
}


def render_macro_release_banner(conn):
    """Spec item 5a/5b -- rendered once, at the very top of the page (right after the header, above every
    tab). Reads the SAME get_macro_release_banner_state the digest's pre-release callout reads, so the two
    can never disagree. Display only -- the one-time alert EMAIL is sent separately by
    check_and_send_macro_release_alerts (called at the dashboard's existing full-load catch-up, see below),
    not from here; this banner can show a post-release surprise even on a day no email went out for it
    (e.g. an ambiguous-direction release)."""
    state = get_macro_release_banner_state(conn)
    if state.get("pre_release"):
        st.warning(state["pre_release"]["text"])
    if state.get("post_release"):
        post = state["post_release"]
        if post["direction"] == "risk_off":
            st.error(post["text"])
        else:
            st.warning(post["text"])


def render_macro_calendar_tab():
    """MACRO CALENDAR tab (Part 4) -- standalone, ticker-agnostic event
    log: officially scheduled economic/Fed releases only (TradingView's
    unofficial economic-calendar endpoint + ForexFactory fallback for
    forecast/previous, the Fed's published FOMC calendar, and FRED when
    FRED_API_KEY is set -- kept as-is, used only as a cross-check against
    TradingView's actuals, never removed). Deliberately out of scope:
    unscheduled events (political press conferences, ad hoc announcements,
    tariff timing) -- no reliable structured free source exists for those;
    NEWS and the AI Briefing's catalyst detection cover them once they occur
    or are formally announced.

    One wide, chronologically-sorted table for the selected window --
    date/time ET, event name, impact badge, prior/forecast/actual, surprise
    (vs. MACRO_DIRECTION_MAP's per-indicator hot/cold call), CPI market odds
    (get_cpi_bracket_odds, CPI rows only), and watchlist-relevance tags
    (get_watchlist_relevant_tickers), no truncation. (An earlier day-column
    grid version packed 3 st.metric() blocks into 1/5th-page-width columns
    and was unreadably cramped in practice -- replaced.) Any value with no
    real source yet (no forecast landed, FOMC's no-forecast-concept, or no
    matching Polymarket bracket) shows "—" -- never fabricated."""
    st.markdown('<div class="smd-section">MACRO CALENDAR</div>', unsafe_allow_html=True)
    st.caption(
        "Officially scheduled economic/Fed releases only, from TradingView's economic calendar "
        "(forecast/previous/actual, no key needed), ForexFactory (forecast/previous fallback), the "
        "Federal Reserve's published FOMC calendar, and FRED when FRED_API_KEY is set (cross-check only). "
        "Unscheduled events (political press conferences, ad hoc announcements, tariff timing) are covered "
        "by the NEWS tab and the AI Briefing's catalyst detection once they occur or are formally announced "
        "— this calendar has no reliable free source for those."
    )
    conn = get_conn()

    hcol1, hcol2 = st.columns([1, 3])
    if hcol1.button("🔄 Refresh Calendar", key="macro_calendar_refresh"):
        with st.spinner("Fetching TradingView/ForexFactory/FOMC/FRED calendar data..."):
            cached_macro_events(conn, force_refresh=True)
        st.rerun()
    events = cached_macro_events(conn)
    if not events:
        st.warning("No macro events on record yet — click 🔄 Refresh Calendar.")
        return

    fcol1, fcol2, fcol3 = st.columns(3)
    weeks_ahead = fcol1.selectbox("Window", ["This week", "Next 2 weeks", "Next 4 weeks", "Next 8 weeks"], index=2)
    high_only = fcol2.checkbox("High impact only", key="macro_high_only")
    relevant_only = fcol3.checkbox("Relevant to my watchlist only", key="macro_relevant_only")

    n_weeks = {"This week": 1, "Next 2 weeks": 2, "Next 4 weeks": 4, "Next 8 weeks": 8}[weeks_ahead]
    today = pd.Timestamp.now(tz="America/New_York").date()
    window_end = today + timedelta(weeks=n_weeks)

    upcoming = [e for e in events if today.isoformat() <= e["scheduled_date"] <= window_end.isoformat()]
    if high_only:
        upcoming = [e for e in upcoming if e["impact_level"] == "High"]

    # Relevance is computed once per event_type per render (cheap -- pure
    # DB reads of already-cached fundamentals, no live fetch) rather than
    # once per event, since every CPI row shares the same relevant set.
    relevance_cache = {}
    def _relevant_for(event_type):
        if event_type not in relevance_cache:
            relevance_cache[event_type] = get_watchlist_relevant_tickers(conn, event_type, watchlist)
        return relevance_cache[event_type]

    if relevant_only:
        upcoming = [e for e in upcoming if _relevant_for(e["event_type"])]

    if not upcoming:
        st.info("No matching events in this window.")
        return

    # One wide, chronologically-sorted table for the whole window --
    # replaces the original day-column grid (Streamlit's st.metric()
    # blocks truncate hard in a narrow column, and 5 simultaneous
    # columns each with 3 sub-metrics + 3 caption lines was unreadably
    # cramped in practice). A single table shows every field in full,
    # scrolls naturally for a long window, and stays scannable.
    upcoming_sorted = sorted(upcoming, key=lambda e: (e["scheduled_date"], e["scheduled_time_et"] or ""))
    st.caption(
        f"{len(upcoming_sorted)} event(s), {today.isoformat()} to {window_end.isoformat()}. Forecast/Actual come "
        f"from TradingView (ForexFactory fills a missing forecast for the current week only); FOMC rate "
        f"decisions have no forecast concept and always show \"—\" there. Surprise is blank until both a real "
        f"forecast and actual exist for that release. Market odds is CPI-only (real Polymarket bracket markets); "
        f"a ⚠ next to an event name means FRED's own actual disagreed with TradingView's beyond rounding."
    )
    table_rows = []
    for ev in upcoming_sorted:
        dot, _ = _MACRO_IMPACT_BADGE.get(ev["impact_level"], ("⚪", TEXT_SECONDARY))
        relevant = _relevant_for(ev["event_type"])
        # Display-only shortening -- the real event_name (used in hover
        # text, expanders, reconciliation notes, etc.) is never touched,
        # this table cell alone was getting cut off by the long SEP
        # suffix regardless of column width.
        display_name = ev["event_name"].replace(" + Summary of Economic Projections", " (+SEP)")
        if ev.get("fred_mismatch"):
            display_name += " ⚠"

        # Macro Calendar upgrade -- Surprise + Market odds, purely additive columns. surprise_direction is
        # None for anything without BOTH a real forecast and actual (FOMC, and any TradingView/FRED series
        # still awaiting its release) -- shown as "—", never guessed.
        surprise_txt = "—"
        direction = ev.get("surprise_direction")
        if direction and ev.get("surprise") is not None:
            sign = "+" if ev["surprise"] > 0 else ""
            unit_suffix = "pp" if ev.get("unit") == "%" else ""
            if direction == "ambiguous":
                surprise_txt = f"🟡 {sign}{ev['surprise']:g}{unit_suffix}"
            elif direction == "in_line":
                surprise_txt = f"⚪ {sign}{ev['surprise']:g}{unit_suffix} (in line)"
            else:
                dot_color = "🔴" if direction == "risk_off" else "🟢"
                surprise_txt = f"{dot_color} {sign}{ev['surprise']:g}{unit_suffix}"

        odds_txt = "—"
        series_key = ev.get("series_key")
        if series_key and "cpi" in series_key and ev.get("forecast_raw") is not None:
            odds = get_cpi_bracket_odds(conn, series_key, ev["scheduled_date"], ev["forecast_raw"])
            if odds and odds.get("thin_market"):
                odds_txt = "thin market"
            elif odds:
                odds_txt = f"{odds['hotter_pct']:.0f}% hotter / {odds['cooler_pct']:.0f}% cooler"

        table_rows.append({
            "Date": ev["scheduled_date"], "Day": pd.Timestamp(ev["scheduled_date"]).strftime("%a"),
            "Time (ET)": _format_et_time_12h(ev["scheduled_time_et"]) or "—",
            "Event": display_name, "Impact": f"{dot} {ev['impact_level']}",
            "Prior": ev.get("prior_value") or "—", "Forecast": ev.get("forecast_value") or "—",
            "Actual": ev.get("actual_value") or "—", "Surprise": surprise_txt, "Market odds": odds_txt,
            "Relevant to your watchlist": ", ".join(relevant) if relevant else "—",
        })
    # Explicit pixel widths, not "small"/"medium"/"large" -- those still
    # truncated long event names and multi-ticker relevance lists
    # (confirmed live: "Nonfarm Payrolls (Employment Situatio…" and
    # "FOMC Rate Decision + Summary of Econ…" both got cut off). Short
    # columns pinned small so the two text-heavy columns get real room;
    # the table scrolls horizontally rather than truncating if it still
    # doesn't fit.
    st.dataframe(
        pd.DataFrame(table_rows), width='stretch', hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn(width="small"),
            "Day": st.column_config.TextColumn(width="small"),
            "Time (ET)": st.column_config.TextColumn(width="small"),
            "Impact": st.column_config.TextColumn(width="small"),
            "Prior": st.column_config.TextColumn(width="small"),
            "Forecast": st.column_config.TextColumn(width="small"),
            "Actual": st.column_config.TextColumn(width="small"),
            "Surprise": st.column_config.TextColumn(width="small"),
            "Market odds": st.column_config.TextColumn(width=150),
            "Event": st.column_config.TextColumn(width=320),
            "Relevant to your watchlist": st.column_config.TextColumn(width=280),
        },
    )

    with st.expander("About this calendar's relevance tags"):
        st.caption(
            "Simple, defensible rules, not a model: FOMC/rate decisions flag high-beta (>1.5) names, "
            "Financial Services, and REITs; CPI/PCE flag consumer discretionary/staples and rate-sensitive "
            "sectors (Real Estate, Financial Services, Utilities); Jobs Reports (NFP) flag high-beta names "
            "specifically as a broad-market proxy. GDP/Retail Sales/Jobless Claims have no rule defined yet, "
            "so no tickers are flagged for them rather than guessing. Beta/sector come from each ticker's "
            "already-cached fundamentals."
        )


# --------------------------------------------------------------------------
# SWING FORECAST tab
# --------------------------------------------------------------------------

def _swing_trading_day_number(event_date_str):
    """Which trading day of the swing window (1-indexed, counting from
    today) event_date_str falls on -- small local helper, not a data_
    engine function, since it's only ever used for this tab's own
    display formatting (check_earnings_collision computes an equivalent
    day_num for its own earnings_date internally, but that's a separate,
    brand-new function this same build already owns -- no existing
    "working feature" is touched either way)."""
    try:
        ed = pd.Timestamp(event_date_str).date()
    except (TypeError, ValueError):
        return None
    day_num, d = 0, date.today()
    while d < ed:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            day_num += 1
    return day_num


def render_swing_verdict_panel(ticker, conn, forecast, horizon_days):
    st.markdown('<div class="smd-section">OVERALL VERDICT</div>', unsafe_allow_html=True)
    move_pct = (forecast["target_price"] / forecast["current_price"] - 1) * 100
    dot = "🟢" if forecast["predicted_direction"] == "UP" else "🔴" if forecast["predicted_direction"] == "DOWN" else "⚪"

    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    vcol1.metric("Current price", f"${forecast['current_price']:.2f}")
    vcol2.metric(f"Target ({horizon_days}d, {forecast['target_date']})", f"${forecast['target_price']:.2f}", f"{move_pct:+.2f}%")
    vcol3.metric(
        "Confidence band (p10-p90)",
        f"${forecast['band_low']:.2f} – ${forecast['band_high']:.2f}" if forecast.get("band_low") is not None else "—",
    )
    vcol4.metric("Confidence", forecast.get("confidence_level") or "—")

    mode_label = {"raw_pattern": "Mode 1: raw pattern", "bayesian": "Mode 2: Bayesian",
                  "trained_model": "Mode 3: trained model"}.get(forecast["mode"], forecast["mode"])
    backtest_n = forecast.get("backtest_n") or 0
    st.markdown(md_safe(
        f"**{dot} {forecast['predicted_direction']}** — {ticker} target ${forecast['target_price']:.2f} "
        f"({move_pct:+.2f}%) by {forecast['target_date']} ({horizon_days} trading days). {mode_label} "
        f"({backtest_n} reconciled {horizon_days}-day session(s) on record"
        + (f", avg |error| {forecast['backtest_error_pct']:.2f}%" if forecast.get("backtest_error_pct") is not None else "")
        + ")."
    ))

    reasoning = []
    if not forecast.get("briefing_inputs"):
        reasoning.append(
            "No AI Briefing on record for this ticker -- the qualitative-verdict component of the lean above "
            "was skipped (renormalized over technicals + momentum only, see the banner above to generate one)."
        )
    tf = forecast.get("trend_features") or {}
    if tf.get("trend_structure"):
        reasoning.append(f"Technical structure: {tf['trend_structure']}.")
    if tf.get("momentum_10d_pct") is not None:
        reasoning.append(f"10-day momentum: {tf['momentum_10d_pct']:+.2f}%.")

    div_trend = get_divergence_score_trend(ticker, conn)
    if div_trend.get("trend") not in (None, "no_history", "insufficient_history"):
        reasoning.append(f"Smart-money conviction score is {div_trend['trend']} (currently {div_trend.get('current')}).")

    rs_trend = get_relative_strength_trend(ticker, conn)
    if rs_trend.get("trend"):
        reasoning.append(
            f"{rs_trend['trend'].capitalize()} its {rs_trend['benchmark']} benchmark over the last "
            f"{rs_trend['lookback_days']} sessions ({rs_trend['relative_pct']:+.2f} pts relative)."
        )

    activity = get_recency_weighted_activity(ticker, conn)
    activity_parts = []
    if activity.get("insider_net_weighted"):
        activity_parts.append(f"insider activity net {'buying' if activity['insider_net_weighted'] > 0 else 'selling'} (recency-weighted)")
    if activity.get("congress_net_weighted"):
        activity_parts.append(f"congressional activity net {'buying' if activity['congress_net_weighted'] > 0 else 'selling'} (recency-weighted)")
    if activity_parts:
        reasoning.append("Recent " + " and ".join(activity_parts) + ".")

    dp_trend = get_dark_pool_trend(ticker, conn)
    if dp_trend.get("pattern") not in (None, "insufficient_history", "no_clear_trend"):
        reasoning.append(f"Dark pool signal: {dp_trend['pattern']} pattern over the last {dp_trend.get('n')} sessions.")

    # Part 7 -- never let an in-window macro event sit unacknowledged.
    events = forecast.get("active_macro_events") or []
    if events:
        beta = None
        try:
            fund_env = cached_fundamentals(conn, ticker, max_age_hours=12)
            beta = (fund_env.get("data") or {}).get("info", {}).get("beta")
        except Exception:
            beta = None
        for ev in events:
            try:
                relevant = get_watchlist_relevant_tickers(conn, ev["event_type"], [ticker])
            except Exception:
                relevant = []
            beta_note = f", and {ticker}'s beta of {beta:.2f} puts it in the flagged-sensitive group" if (relevant and beta) else ""
            day_num = _swing_trading_day_number(ev["scheduled_date"])
            day_txt = f"Day {day_num}" if day_num else ev["scheduled_date"]
            # Anti-hallucination fix (Part 3): the impact reasoning below is
            # ALWAYS grounded in either a real Polymarket probability or this
            # ticker's own backtested historical reaction -- never a bare
            # directional assertion, via get_macro_event_grounded_reasoning
            # (the same function the MACRO EVENTS table below cites, so the
            # two can never disagree).
            grounded = get_macro_event_grounded_reasoning(ticker, conn, ev, horizon_days)
            reasoning.append(
                f"This {horizon_days}-day window includes a {ev['event_name']} on {day_txt} "
                f"({ev['scheduled_date']}, {ev['impact_level']} impact){beta_note}. {grounded['text']}"
            )

    if reasoning:
        st.markdown("**Reasoning:**")
        for r in reasoning:
            st.caption(f"• {r}")


def render_swing_macro_events_section(ticker, conn, forecast, horizon_days):
    st.markdown('<div class="smd-section">MACRO EVENTS IN THIS WINDOW</div>', unsafe_allow_html=True)
    events = forecast.get("active_macro_events") or []
    if not events:
        st.caption(f"No scheduled economic/Fed events within this {horizon_days}-day window.")
        return

    event_ids = [e["event_id"] for e in events]
    placeholders = ",".join("?" * len(event_ids))
    detail_rows = conn.execute(
        f"SELECT event_id, prior_value, forecast_value FROM macro_events WHERE event_id IN ({placeholders})",
        event_ids,
    ).fetchall()
    detail_map = {r[0]: {"prior": r[1], "forecast": r[2]} for r in detail_rows}

    grounding_label = {"polymarket": "📊 Polymarket", "backtest": "📈 Backtested", "none": "— No data"}
    table_rows = []
    grounded_texts = []
    for ev in events:
        day_num = _swing_trading_day_number(ev["scheduled_date"])
        try:
            relevant = get_watchlist_relevant_tickers(conn, ev["event_type"], [ticker])
        except Exception:
            relevant = []
        det = detail_map.get(ev["event_id"], {})
        # Anti-hallucination fix (Part 3): grounded, not asserted -- see
        # get_macro_event_grounded_reasoning's own docstring. Same call the
        # OVERALL VERDICT reasoning above uses, so the two never disagree.
        grounded = get_macro_event_grounded_reasoning(ticker, conn, ev, horizon_days)
        table_rows.append({
            "Event": ev["event_name"],
            "Date": ev["scheduled_date"],
            "Time (ET)": _format_et_time_12h(ev.get("scheduled_time_et")) or "—",
            "Day of window": f"Day {day_num} of {horizon_days}" if day_num else "—",
            "Impact": ev["impact_level"],
            "Prior": det.get("prior") or "—",
            "Forecast": det.get("forecast") or "—",
            f"{ticker} relevance": "⚠️ Flagged sensitive" if relevant else "—",
            "Impact reasoning grounded in": grounding_label.get(grounded["grounded_in"], "—"),
        })
        grounded_texts.append(f"**{ev['event_name']}** ({ev['scheduled_date']}): {grounded['text']}")
    st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True,
                 key=f"swing_macro_events_{ticker}_{horizon_days}")
    with st.expander("Full impact reasoning per event (grounded)", expanded=False):
        for t in grounded_texts:
            st.caption(t)


def render_swing_price_chart(ticker, conn, forecast, horizon_days):
    hist = get_history(conn, "price_history", ticker, days_back=60)
    if hist is None or hist.empty:
        return
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["close"], mode="lines", line=dict(color=ACCENT, width=2), name="Close",
    ))
    target_color = (
        COLOR_BULLISH if forecast["predicted_direction"] == "UP"
        else COLOR_BEARISH if forecast["predicted_direction"] == "DOWN" else COLOR_NEUTRAL
    )
    target_ts = pd.Timestamp(forecast["target_date"])

    # Bug fix: render the CACHED simulated daily-step path (real drift +
    # random-shock + Brownian-bridge texture, generated once at commit
    # time by log_swing_forecast -- see simulate_multiday_path) instead
    # of a straight 2-point ruler line from today's price to the target.
    committed_path = get_committed_swing_path(conn, ticker, horizon_days, forecast["session_date"])
    if committed_path:
        path_x = [pd.Timestamp(d) for d, _ in committed_path]
        path_y = [p for _, p in committed_path]
    else:
        path_x, path_y = [hist["date"].iloc[-1], target_ts], [forecast["current_price"], forecast["target_price"]]
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y, mode="lines+markers", line=dict(color=target_color, width=2, dash="dash"),
        marker=dict(size=5), name=f"{horizon_days}d projected path",
    ))
    all_y = list(hist["close"]) + path_y
    if forecast.get("band_low") is not None and forecast.get("band_high") is not None:
        fig.add_trace(go.Scatter(
            x=[target_ts, target_ts], y=[forecast["band_low"], forecast["band_high"]],
            mode="lines", line=dict(color=TEXT_MUTED, width=6), opacity=0.35, name="Confidence band (p10-p90)",
        ))
        all_y += [forecast["band_low"], forecast["band_high"]]

    events = forecast.get("active_macro_events") or []
    add_macro_event_markers(fig, events, y_bottom=min(all_y) * 0.99)

    fig.update_yaxes(title_text="Price ($)")
    st.plotly_chart(
        apply_theme(fig, height=380, legend=dict(orientation="h", y=-0.25, bgcolor="rgba(0,0,0,0)")),
        width='stretch', key=f"swing_price_chart_{ticker}_{horizon_days}",
    )
    render_macro_event_expander(events)


def render_swing_strategy_section(ticker, conn, forecast, horizon_days):
    st.markdown('<div class="smd-section">OPTIONS STRATEGY</div>', unsafe_allow_html=True)
    strategy = recommend_swing_strategy(ticker, conn, forecast)
    st.caption(md_safe(strategy["note"]))
    if strategy.get("caution"):
        st.warning(md_safe(strategy["caution"]))

    candidates = strategy["candidates"]["calls"] + strategy["candidates"]["puts"]
    if not candidates:
        st.caption(strategy.get("note") or "No candidates matched the delta/liquidity filters for this ticker's chain right now.")
        return

    market = get_market_pricing_signals(ticker, conn)
    for i, c in enumerate(candidates):
        type_color = COLOR_BULLISH if c["type"] == "call" else COLOR_BEARISH
        badge_defs = [
            ("delta", DELTA_BADGE_COLORS, None), ("iv", IV_BADGE_COLORS, None),
            ("vol_oi", None, VOL_OI_PROFILES), ("dte", None, None),
            ("theta", None, THETA_PCT_PROFILES), ("spread", None, SPREAD_PROFILES),
        ]
        badges_html = []
        for key, color_map, tier_profiles in badge_defs:
            b = c["badges"].get(key) or {}
            label = b.get("label")
            if color_map is not None:
                color = color_map.get(label, COLOR_NEUTRAL)
            elif tier_profiles is not None:
                color = _tier_escalation_color(label, tier_profiles)
            else:
                color = COLOR_NEUTRAL
            badges_html.append(interp_badge_html(label, color, b.get("note")))

        delta_txt = f"{c['delta']:.2f}" if c.get("delta") is not None else "—"
        gamma_txt = f"{c['gamma']:.4f}" if c.get("gamma") is not None else "—"
        theta_txt = f"-${abs(c['theta']) * 100:.2f}/day" if c.get("theta") is not None else "—"
        vega_txt = f"${c['vega'] * 100:.2f}" if c.get("vega") is not None else "—"
        greeks_html = (
            f'<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:14px;font-size:12px;'
            f'color:{TEXT_SECONDARY};">'
            f'<span>Delta: <b style="color:{TEXT_PRIMARY};">{delta_txt}</b></span>'
            f'<span>Gamma: <b style="color:{TEXT_PRIMARY};">{gamma_txt}</b></span>'
            f'<span>Theta: <b style="color:{TEXT_PRIMARY};">{theta_txt}</b></span>'
            f'<span>Vega: <b style="color:{TEXT_PRIMARY};">{vega_txt}</b></span>'
            f'</div>'
        )
        st.markdown(md_safe(f"""
        <div class="smcard">
          <div style="display:flex;justify-content:space-between;align-items:baseline;">
            <div style="font-size:15px;font-weight:700;color:{TEXT_PRIMARY};">
              <span style="color:{type_color};">{c['type'].upper()}</span>
              ${c['strike']:g} &mdash; {html.escape(str(c['expiration']))}
            </div>
            <div style="font-size:12px;color:{TEXT_MUTED};">{c.get('dte', '—')} DTE</div>
          </div>
          <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px;">{"".join(badges_html)}</div>
          {greeks_html}
        </div>
        """), unsafe_allow_html=True)

        analysis = analyze_swing_contract(ticker, conn, c, forecast, market=market)
        if not analysis:
            st.caption("Not enough data to simulate this contract (missing price/IV/DTE).")
            continue

        pop = analysis["probability_of_profit"]
        max_profit_txt = "Uncapped" if c["type"] == "call" else f"${c['strike'] * 100:,.0f} (if stock → $0)"
        scol1, scol2, scol3, scol4, scol5 = st.columns(5)
        scol1.metric("Net Debit", f"${analysis['net_debit']:,.0f}")
        scol2.metric("Max Loss", f"${analysis['max_loss']:,.0f}")
        scol3.metric("Max Profit", max_profit_txt)
        scol4.metric("Chance of Profit", f"{pop:.0%}" if pop is not None else "—")
        scol5.metric("Breakeven", f"${analysis['breakeven']:.2f}" if analysis["breakeven"] else "—")

        pl_fig = render_pl_curve_chart(analysis["pl_curve"], analysis["pl_curve_earnings"], forecast["current_price"])
        if pl_fig is not None:
            st.plotly_chart(pl_fig, width='stretch', key=f"swing_plcurve_{ticker}_{horizon_days}_{i}")

        st.markdown(f"*{md_safe(analysis['reasoning'])}*")

        decomp_rows = [{
            "Scenario": s["scenario"], "Price move": f"{s['price_move_pct']:+.1f}%",
            "Target option price": f"${s['target_option_price']:.2f}",
            "P/L ($)": f"{s['pl_dollar']:+,.2f}", "P/L (%)": f"{s['pl_pct']:+.1f}%",
        } for s in analysis["contract_pl"]["scenarios"]]
        st.dataframe(pd.DataFrame(decomp_rows), width='stretch', hide_index=True,
                     key=f"swing_decomp_{ticker}_{horizon_days}_{i}")

        with st.expander("Detailed grid — price × date"):
            hm_fig = render_pl_heatmap_chart(analysis["heatmap"])
            if hm_fig is not None:
                st.plotly_chart(hm_fig, width='stretch', key=f"swing_heatmap_{ticker}_{horizon_days}_{i}")
            else:
                st.caption("Not enough data to build the detailed grid for this contract.")

    gamma_evo = get_gamma_evolution(ticker, conn, horizon_days)
    expiries = gamma_evo.get("expiries") if isinstance(gamma_evo, dict) else None
    if expiries:
        with st.expander("Gamma exposure by expiry (within this horizon)"):
            gamma_rows = [{
                "Expiration": e["expiration"], "Net gamma exposure": f"{e['net_gamma_exposure']:,.0f}",
                "Top strikes": ", ".join(f"${s['strike']:g} ({s['net_gamma_exposure']:+,.0f})" for s in e["top_strikes"]),
            } for e in expiries]
            st.dataframe(pd.DataFrame(gamma_rows), width='stretch', hide_index=True,
                         key=f"swing_gamma_evo_{ticker}_{horizon_days}")


def render_swing_backtest_report(ticker, conn, horizon_days):
    rows, latest_run = get_swing_backtest_report_data(conn, ticker, horizon_days)
    if not latest_run:
        st.caption("No backtest on record yet for this ticker/horizon.")
        return
    track = get_swing_forecast_track_record(conn, ticker=ticker, horizon_days=horizon_days)
    with st.expander(f"📊 Backtest Report — {ticker} ({horizon_days}d)", expanded=True,
                      key=f"swing_backtest_report_{ticker}_{horizon_days}"):
        st.caption(
            f"Based on {track['n']} reconciled {horizon_days}-day session(s) (backtested + live). "
            f"Last backtest run: {latest_run['run_at'][:19].replace('T', ' ')} UTC, covering "
            f"{latest_run['date_range_start']} to {latest_run['date_range_end']} "
            f"({latest_run['qualifying_sessions_found']} of the last {latest_run['lookback_days']} trading days)."
        )
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Sessions backtested", latest_run["sessions_backtested"])
        scol2.metric("Avg |error|", f"{latest_run['mean_abs_error_pct']:.2f}%"
                     if latest_run["mean_abs_error_pct"] is not None else "—")
        scol3.metric("Directional accuracy", f"{latest_run['direction_accuracy']:.0%}"
                     if latest_run["direction_accuracy"] is not None else "—")

        calibration = get_active_swing_calibration(conn, horizon_days)
        st.caption(
            f"Current calibration bias correction for {horizon_days}-day forecasts: "
            f"drift_scale={calibration['drift_scale']:.2f}, vol_scale={calibration['vol_scale']:.2f} "
            f"(1.00/1.00 = no adjustment)."
        )

        if not rows:
            st.caption("No backtested rows with a resolved error yet.")
            return
        errors = [r["error_pct"] for r in rows if r["error_pct"] is not None]
        if errors:
            hist_fig = go.Figure(data=[go.Histogram(x=errors, marker_color=ACCENT, nbinsx=min(20, max(6, len(errors) // 2)))])
            hist_fig.add_vline(x=0, line=dict(color=TEXT_SECONDARY, width=1, dash="dot"))
            hist_fig.update_xaxes(title_text="error_pct (actual vs. predicted target, %)")
            hist_fig.update_yaxes(title_text="Sessions")
            st.plotly_chart(
                apply_theme(hist_fig, height=280, margin=dict(l=50, r=30, t=20, b=50)),
                width='stretch', key=f"swing_backtest_hist_{ticker}_{horizon_days}",
            )

        ranked = sorted([r for r in rows if r["error_pct"] is not None], key=lambda r: abs(r["error_pct"]))
        best5, worst5 = ranked[:5], list(reversed(ranked[-5:]))

        def _fmt(r):
            return {
                "Date": r["session_date"], "Predicted Target": f"${r['target_price']:.2f}",
                "Actual Close": f"${r['actual_close_price']:.2f}" if r["actual_close_price"] is not None else "—",
                "Error %": f"{r['error_pct']:+.2f}%", "Direction correct": "✅" if r["prediction_correct_direction"] else "❌",
            }

        bcol, wcol = st.columns(2)
        with bcol:
            st.markdown("**Best 5 (lowest |error|)**")
            st.dataframe(pd.DataFrame([_fmt(r) for r in best5]), width='stretch', hide_index=True,
                         key=f"swing_backtest_best5_{ticker}_{horizon_days}")
        with wcol:
            st.markdown("**Worst 5 (highest |error|)**")
            st.dataframe(pd.DataFrame([_fmt(r) for r in worst5]), width='stretch', hide_index=True,
                         key=f"swing_backtest_worst5_{ticker}_{horizon_days}")


def render_swing_forecast_tab():
    st.markdown('<div class="smd-section">SWING FORECAST</div>', unsafe_allow_html=True)
    st.caption(
        "Multi-day (3/5/10 trading session) directional + magnitude forecast for ANY ticker, not scoped "
        "to the watchlist — reuses Day Prediction's and Earnings Simulator's backtest-first calibration, "
        "Mode 1/2/3 honesty system, and macro-event linkage, adapted for a multi-day horizon instead of a "
        "single session or a single earnings event."
    )
    conn = get_conn()

    scol1, scol2 = st.columns([3, 1])
    sw_ticker = scol1.text_input(
        "Ticker", value=st.session_state.get("swing_ticker", ""), label_visibility="collapsed",
        placeholder="e.g. NVDA, MU, STX", key="swing_ticker_input",
    )
    horizon_days = scol2.selectbox(
        "Horizon", SWING_HORIZON_OPTIONS, index=1, key="swing_horizon_select",
        format_func=lambda h: f"{h} trading days",
    )
    if not sw_ticker:
        return
    sw_ticker = sw_ticker.strip().upper()
    st.session_state["swing_ticker"] = sw_ticker

    # Prerequisite (bug fix): the AI Briefing's qualitative synthesis
    # (catalysts, verdicts) is one of the swing lean's three inputs (see
    # _swing_directional_lean's 0.30 weight) -- previously silently
    # skipped with no indication to the user when missing. Surfaced here,
    # not auto-triggered (same paid-API confirmation discipline as
    # TICKER DEEP-DIVE): reuses the EXACT SAME confirmation dialog
    # (_confirm_ai_briefing_dialog, unmodified) via the same
    # ai_brief_pending_ticker session-state flag TICKER DEEP-DIVE's own
    # trigger check already watches for every rerun -- no new dialog or
    # API-call path added, just a second place that can set the flag.
    ai_brief = get_cached_ai_brief(conn, sw_ticker, max_age_hours=AI_BRIEF_CACHE_HOURS)
    if ai_brief:
        fetched_dt = pd.to_datetime(ai_brief.get("_fetched_at"), utc=True, errors="coerce")
        age_label = "recently"
        if pd.notna(fetched_dt):
            age_h = (pd.Timestamp.now(tz="UTC") - fetched_dt).total_seconds() / 3600
            age_label = f"{age_h * 60:.0f}m ago" if age_h < 1 else f"{age_h:.1f}h ago"
        st.caption(f"🧠 Using the AI Briefing generated {age_label} as this forecast's qualitative input.")
    else:
        bcol1, bcol2 = st.columns([4, 1.4])
        bcol1.warning(
            f"No AI Briefing on record for {sw_ticker} yet. It's one of this forecast's three directional-lean "
            f"inputs (technicals, momentum, and the Briefing's own qualitative verdict) — without it the "
            f"forecast still runs, just on technicals/momentum alone. Recommended before relying too heavily "
            f"on the verdict below."
        )
        if bcol2.button("🧠 Generate AI Briefing", width='stretch', key=f"swing_gen_briefing_{sw_ticker}"):
            st.session_state["ai_brief_pending_ticker"] = sw_ticker

    has_daily = conn.execute("SELECT 1 FROM price_history WHERE ticker=? LIMIT 1", (sw_ticker,)).fetchone()
    if not has_daily:
        with st.spinner(f"Preparing {sw_ticker} — downloading price history + running backtests..."):
            readiness = ensure_ticker_data_ready(sw_ticker, conn, bootstrap_swing_horizons=[horizon_days])
    else:
        readiness = ensure_ticker_data_ready(sw_ticker, conn, bootstrap_swing_horizons=[horizon_days])

    if not readiness["ready"]:
        st.error(f"⛔ **{sw_ticker}** — {' '.join(readiness['still_missing']) or 'No reliable price history available.'}")
        return
    if readiness["fetched_now"]:
        st.caption("ℹ️ " + "; ".join(readiness["fetched_now"]))

    forecast = compute_swing_forecast(sw_ticker, conn, horizon_days)
    if not forecast.get("ready"):
        st.info(f"⏳ **{sw_ticker}** — {forecast.get('wait_message', 'Waiting for data.')}")
        return

    log_swing_forecast(conn, sw_ticker, forecast)

    render_swing_verdict_panel(sw_ticker, conn, forecast, horizon_days)
    render_swing_macro_events_section(sw_ticker, conn, forecast, horizon_days)

    ec = forecast.get("earnings_collision") or {}
    if ec.get("collision"):
        st.warning(ec["warning"])

    render_swing_price_chart(sw_ticker, conn, forecast, horizon_days)
    render_swing_strategy_section(sw_ticker, conn, forecast, horizon_days)

    st.markdown('<div class="smd-section">BACKTEST REPORT</div>', unsafe_allow_html=True)
    render_swing_backtest_report(sw_ticker, conn, horizon_days)


# --------------------------------------------------------------------------
# 📍 WATCHLIST SIGNALS -- auto-derived entry zones (Part 1) + synthesized,
# tier-weighted rating signals (Part 3/4). The background scan
# (refresh_all_entry_zones/check_entry_zones/check_rating_changes) runs
# elsewhere (the sidebar-adjacent fragment, Part 6) -- this tab is a pure
# read surface (plus a per-ticker force-recompute button) over the same
# entry_zones/seen_analyst_actions tables that scan writes to. No manual
# entry-zone input anywhere.
# --------------------------------------------------------------------------

_ZONE_STATUS_STYLE = {
    "in_zone": ("IN ZONE", COLOR_BULLISH),
    "near": ("NEAR", COLOR_RETAIL),
    "above": ("ABOVE", COLOR_NEUTRAL),
    "below": ("BELOW", COLOR_NEUTRAL),
    "no_zone": ("NO ZONE", COLOR_NEUTRAL),
    None: ("PENDING", COLOR_NEUTRAL),
}
_ZONE_NEAR_THRESHOLD_PCT = 3.0  # display-only reclassification: 'above'/'below' within this % of the
                                 # zone edge reads as NEAR instead -- last_zone_status itself (DB) is
                                 # untouched, this only changes how it's grouped/colored in this table.

_ANCHOR_SHORT_LABEL = {
    "swing_support": "{touches}-touch support",
    "volume_node": "volume node",
    "current_price_fallback": "⚠ placeholder",
    None: "—",
}


def _zone_display_status(real_status, distance_pct):
    """real_status ('in_zone'/'above'/'below'/'no_zone'/None, from check_entry_zones -- unchanged) plus
    this row's own distance_pct decide the DISPLAY status only ('near' is not a real last_zone_status
    value, purely a dense-overview grouping)."""
    if real_status in ("above", "below") and distance_pct is not None and distance_pct < _ZONE_NEAR_THRESHOLD_PCT:
        return "near"
    return real_status


def _zone_distance_pct(real_status, price, entry_low, entry_high):
    """% from the nearest zone edge -- None when in zone (nothing to measure) or data's missing."""
    if real_status not in ("above", "below") or price is None or not entry_low or not entry_high:
        return None
    if real_status == "above":
        return (price - entry_high) / entry_high * 100.0
    return (entry_low - price) / entry_low * 100.0


def _anchor_short_label(anchor_kind, anchor_touches):
    label = _ANCHOR_SHORT_LABEL.get(anchor_kind, _ANCHOR_SHORT_LABEL[None])
    if anchor_kind == "swing_support":
        label = label.format(touches=anchor_touches if anchor_touches is not None else "?")
    return label


_RATING_SIGNAL_STYLE = {
    "UPGRADE": ("UPGRADE", COLOR_BULLISH),
    "DOWNGRADE": ("DOWNGRADE", COLOR_BEARISH),
    "MIXED": ("MIXED", COLOR_RETAIL),
    "QUIET": ("QUIET", COLOR_NEUTRAL),
}


def _collapse_repeat_reiterations(actions):
    """Display-only collapsing (2026-09-25 fix) for the expanded rating-changes table: repeated
    reiterations from the SAME firm, at the SAME grade (e.g. BTIG reiterating Buy -> Buy 3x), collapse to
    one row -- the most recent occurrence, annotated "(N reiterations since <earliest date>)". Upgrades
    and downgrades are never collapsed, even if a firm somehow repeats one identically, since those are
    real directional events. Never touches synthesize_rating_signal's own actions list, weighted_net, or
    driving_action selection -- this runs on a plain list copy purely for how the table renders."""
    groups = {}
    singles = []
    for a in actions:
        if a["action"] == "reiterated":
            key = (a["firm"], a.get("from_grade"), a.get("to_grade"))
            groups.setdefault(key, []).append(a)
        else:
            singles.append(a)
    collapsed = list(singles)
    for grp in groups.values():
        if len(grp) == 1:
            collapsed.append(grp[0])
        else:
            grp_by_date = sorted(grp, key=lambda a: a["action_date"])
            most_recent = dict(grp_by_date[-1])
            most_recent["_repeat_note"] = f"({len(grp)} reiterations since {fmt_date(grp_by_date[0]['action_date'])})"
            collapsed.append(most_recent)
    return collapsed


def render_watchlist_signals_tab():
    conn = get_conn()

    # "Since your last visit" baseline (Part 4): the marker from the
    # PREVIOUS visit, used to compute this render's summary -- only
    # overwritten to "now" at the very end, after everything below has
    # already used the old value. A first-ever visit (marker is None)
    # treats everything currently on record as new, per load_watchlist_
    # signals_last_viewed's own documented fallback.
    last_viewed = load_watchlist_signals_last_viewed(conn, EFFECTIVE_EMAIL)
    since_baseline = last_viewed or "1970-01-01T00:00:00"
    signals_since = get_watchlist_signals_since(conn, EFFECTIVE_EMAIL, since_baseline, watchlist)
    n_zones, n_actions = len(signals_since["triggered_zones"]), len(signals_since["new_actions"])

    st.markdown('<div class="smd-section">📍 WATCHLIST SIGNALS</div>', unsafe_allow_html=True)
    if last_viewed is None:
        st.caption(f"Welcome — {n_zones} ticker(s) currently in their auto-derived entry zone, {n_actions} "
                   f"rating action(s) on record.")
    else:
        st.caption(f"Since your last visit ({time_ago(last_viewed)}): {n_zones} ticker(s) entered their "
                   f"entry zone, {n_actions} new analyst rating action(s) recorded.")

    # ---- SECTION A: 🎯 Entry Zones (auto-derived -- Part 1; dense-overview rebuild 2026-09-25) ----
    st.markdown('<div class="smd-section">🎯 Entry Zones</div>', unsafe_allow_html=True)
    st.caption(
        "Auto-derived per ticker from technical support structure, its own backtested forward-return "
        "distribution, and its current divergence score -- no manual range entry. Recomputes automatically "
        "in the background (hourly at most); use Recompute all to force a refresh."
    )
    if st.button("🔄 Recompute all", key=_uk("wsig_recompute_all")):
        with st.spinner("Recomputing all entry zones..."):
            refresh_all_entry_zones(conn, EFFECTIVE_EMAIL, watchlist, force_refresh=True)
        st.rerun()

    zones_by_ticker = {z["ticker"]: z for z in get_entry_zones(conn, EFFECTIVE_EMAIL)}
    enriched = []
    for t in watchlist:
        z = zones_by_ticker.get(t)
        price = get_latest_cached_price(conn, t)
        if z is None:
            real_status = None
        elif z.get("entry_low") is None:
            real_status = "no_zone"
        else:
            real_status = z["last_zone_status"]
        distance_pct = _zone_distance_pct(real_status, price, z.get("entry_low") if z else None,
                                           z.get("entry_high") if z else None)
        enriched.append({
            "ticker": t, "zone": z, "price": price, "real_status": real_status,
            "display_status": _zone_display_status(real_status, distance_pct), "distance_pct": distance_pct,
        })

    # --- Summary strip: one line, small cards ---
    ssc1, ssc2, ssc3, ssc4 = st.columns(4)
    ssc1.metric("In zone", sum(1 for e in enriched if e["display_status"] == "in_zone"))
    ssc2.metric("Near zone (<3% away)", sum(1 for e in enriched if e["display_status"] == "near"))
    ssc3.metric("Above zone", sum(1 for e in enriched if e["display_status"] == "above"))
    ssc4.metric("Low-confidence anchors",
                sum(1 for e in enriched if e["zone"] and e["zone"].get("anchor_kind") == "current_price_fallback"))

    # --- Sort: IN ZONE, NEAR, ABOVE (then BELOW, then no-zone/pending -- not explicitly specified, placed
    # last as the least actionable), within each by distance ascending (closer = more urgent first) ---
    _STATUS_SORT_RANK = {"in_zone": 0, "near": 1, "above": 2, "below": 3, "no_zone": 4, None: 5}
    enriched.sort(key=lambda e: (_STATUS_SORT_RANK.get(e["display_status"], 5),
                                  e["distance_pct"] if e["distance_pct"] is not None else 0.0, e["ticker"]))

    ACCURACY_COINFLIP_LOW, ACCURACY_COINFLIP_HIGH = 0.45, 0.55
    display_cols = ["Ticker", "Status", "Price", "Entry zone", "Distance", "90d fwd return", "Anchor",
                     "Accuracy", "Divergence", "Days in zone"]
    rows_data = []
    for e in enriched:
        t, z, price, status = e["ticker"], e["zone"], e["price"], e["display_status"]
        badge_txt, badge_color = _ZONE_STATUS_STYLE.get(status, _ZONE_STATUS_STYLE[None])
        price_txt = f"${price:,.2f}" if price is not None else "—"
        zone_txt = f"${z['entry_low']:,.2f}-${z['entry_high']:,.2f}" if z and z.get("entry_low") is not None else "—"
        distance_txt = f"{e['distance_pct']:.1f}%" if e["distance_pct"] is not None else "—"

        fwd_return = z.get("target_pct") if z else None
        anchor_kind = z.get("anchor_kind") if z else None
        accuracy = z.get("backtest_accuracy") if z else None
        divergence = z.get("divergence_score") if z else None
        divergence_txt = f"{divergence:.0f}" if divergence is not None else "—"

        days_txt = "—"
        if z and status == "in_zone" and z.get("zone_entered_at"):
            try:
                entered = pd.Timestamp(z["zone_entered_at"])
                entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered
                days_txt = str((pd.Timestamp.now(tz="UTC") - entered).days)
            except (TypeError, ValueError):
                days_txt = "—"

        # Contradiction flag (2026-09-25): IN ZONE but the backtested forward return is negative --
        # flagged inline (visible without hover) plus explained once in the caption below the table.
        contradiction = status == "in_zone" and fwd_return is not None and fwd_return < 0
        fwd_return_txt = (f"{fwd_return:+.1f}%" if fwd_return is not None else "—") + (" ⚠" if contradiction else "")

        rows_data.append({
            "Ticker": t, "Status": badge_txt, "Price": price_txt, "Entry zone": zone_txt,
            "Distance": distance_txt, "90d fwd return": fwd_return_txt,
            "Anchor": _anchor_short_label(anchor_kind, z.get("anchor_touches") if z else None),
            "Accuracy": f"{accuracy:.0%}" if accuracy is not None else "—",
            "Divergence": divergence_txt, "Days in zone": days_txt,
            "_status_color": badge_color, "_fwd_return": fwd_return, "_accuracy": accuracy,
            "_is_placeholder": anchor_kind == "current_price_fallback",
        })

    df = pd.DataFrame(rows_data)

    def _style_row(row):
        i = row.name
        styles = {c: "" for c in display_cols}
        styles["Ticker"] = "font-weight:700;font-size:14px;"
        styles["Status"] = f"color:{df.loc[i, '_status_color']};font-weight:700;"
        for c in ("Price", "Entry zone", "Distance", "90d fwd return", "Accuracy", "Divergence", "Days in zone"):
            styles[c] += "font-family:monospace;text-align:right;white-space:nowrap;"
        fwd = df.loc[i, "_fwd_return"]
        if fwd is not None:
            styles["90d fwd return"] += f"color:{COLOR_BULLISH if fwd >= 0 else COLOR_BEARISH};font-weight:700;"
        if df.loc[i, "_is_placeholder"]:
            styles["Anchor"] += f"color:{COLOR_RETAIL};"
        acc = df.loc[i, "_accuracy"]
        if acc is not None and ACCURACY_COINFLIP_LOW <= acc <= ACCURACY_COINFLIP_HIGH:
            styles["Accuracy"] += f"color:{COLOR_NEUTRAL};"
        return [styles[c] for c in display_cols]

    styled = df[display_cols].style.apply(_style_row, axis=1)
    # Explicit height so all watchlist tickers fit without an inner scrollbar (~35px/row + header).
    table_height = 38 + len(df) * 35 + 3
    event = st.dataframe(
        styled, width='stretch', hide_index=True, height=table_height,
        on_select="rerun", selection_mode="single-row", key=_uk("wsig_zone_table"),
        column_config={
            "90d fwd return": st.column_config.TextColumn(
                help="Backtested mean forward return over the next 90 trading days. ⚠ = in entry zone but this is negative."),
            "Anchor": st.column_config.TextColumn(
                help="⚠ placeholder = no real support/volume level found below price -- band is centered on price itself, lowest confidence."),
            "Accuracy": st.column_config.TextColumn(
                help="Swing Forecast's own directional accuracy for this ticker. Greyed between 45-55% -- coin-flip territory, not validation."),
        },
    )
    st.caption("⚠ on 90d fwd return = in entry zone but backtested forward return is negative. "
               "Click a row for detail.")

    # --- Detail panel for the selected row, below the table, one at a time ---
    if event.selection.rows:
        sel_ticker = df.iloc[event.selection.rows[0]]["Ticker"]
        sel = next(e for e in enriched if e["ticker"] == sel_ticker)
        z = sel["zone"]
        st.markdown(f"#### {sel_ticker}")
        dcol1, dcol2 = st.columns([1, 3])
        with dcol1:
            if st.button("🔄 Recompute", key=_uk(f"wsig_detail_recompute_{sel_ticker}")):
                refresh_entry_zone(conn, EFFECTIVE_EMAIL, sel_ticker, force_refresh=True)
                st.rerun()
            st.caption("**Why this zone**")
            st.caption(md_safe((z.get("reasoning") if z else None)
                                or "No entry zone could be derived for this ticker yet."))
        with dcol2:
            if z and z.get("entry_low") is not None:
                hist, _cfg = fetch_price_history(sel_ticker, "1d", "3mo", conn=conn)
                fig = render_entry_zone_mini_chart(hist, z["entry_low"], z["entry_high"], sel["price"], sel_ticker)
                st.plotly_chart(fig, width='stretch', key=_uk(f"wsig_detail_chart_{sel_ticker}"))
            else:
                st.caption("No zone computed for this ticker yet — nothing to chart.")

    # ---- SECTION B: 📈 Rating Changes (synthesized, tier-weighted -- Part 3/4) ----
    st.markdown('<div class="smd-section">📈 Rating Changes</div>', unsafe_allow_html=True)
    st.caption(
        "One synthesized signal per ticker, weighted by the issuing firm's real market-impact tier -- not a "
        "flat list treating every firm equally. Full raw list available per ticker on expand."
    )
    signal_rows = [synthesize_rating_signal(t, conn) for t in watchlist]
    # Real directional/mixed activity sorts above a quiet ticker, then alphabetical. Unchanged -- this is
    # the per-ticker headline ordering, not the expanded table's own sort (see _sort_actions_for_display).
    signal_rows.sort(key=lambda s: (s["signal"] == "QUIET", s["ticker"]))
    for sig in signal_rows:
        badge_txt, badge_color = _RATING_SIGNAL_STYLE.get(sig["signal"], _RATING_SIGNAL_STYLE["QUIET"])
        driving = sig.get("driving_action")
        driving_txt = (
            f"{driving['firm']} ({driving['tier_label']}) {driving['action']} → "
            f"{driving['to_grade'] or '—'} ({fmt_date(driving['action_date'])})"
        ) if driving else "No rating activity on record."
        st.markdown(
            f"**{sig['ticker']}**&nbsp;&nbsp;<span class='chip' style='border-color:{badge_color};"
            f"color:{badge_color};'>{badge_txt}</span>&nbsp;&nbsp;{driving_txt}",
            unsafe_allow_html=True,
        )
        if sig["all_actions"]:
            # Count badge (2026-09-25 fix) on the collapsed header, from the REAL uncollapsed totals -- so
            # it can be skipped without opening it.
            n_up = sum(1 for a in sig["all_actions"] if a["action"] == "upgrade")
            n_down = sum(1 for a in sig["all_actions"] if a["action"] == "downgrade")
            expander_label = f"{sig['n_actions']} actions · {n_up} upgrade{'s' if n_up != 1 else ''} · {n_down} downgrade{'s' if n_down != 1 else ''}"
            with st.expander(expander_label):
                display_actions = _collapse_repeat_reiterations(sig["all_actions"])
                # Tier ascending (Bulge Bracket, Elite Boutique, Smaller/Regional), then date descending
                # within a tier -- display ordering only, never touches synthesize_rating_signal's own
                # driving-action selection or weighted_net above.
                by_date_desc = sorted(display_actions, key=lambda a: a["action_date"], reverse=True)
                display_actions = sorted(by_date_desc, key=lambda a: a["tier"])

                rows = []
                for a in display_actions:
                    grade_txt = (
                        f"{a['from_grade']} → {a['to_grade']}" if a.get("from_grade") and a.get("to_grade")
                        else (a.get("to_grade") or "—")
                    )
                    date_txt = fmt_date(a["action_date"])
                    if a.get("_repeat_note"):
                        date_txt = f"{date_txt}  {a['_repeat_note']}"
                    rows.append({
                        "Firm": a["firm"], "Tier": a["tier_label"], "Grade change": grade_txt,
                        "Action": a["action"], "Date": date_txt,
                        # No real grade change AND not a fresh initiation -- muted (de-emphasized), never
                        # an upgrade/downgrade, which always keep full contrast.
                        "_muted": a["action"] == "reiterated",
                    })
                df = pd.DataFrame(rows)
                muted_flags = df["_muted"]
                styled = df.drop(columns=["_muted"]).style.apply(
                    lambda row: [("color: #6b7684" if muted_flags.iloc[row.name] else "") for _ in row],
                    axis=1,
                )
                st.dataframe(styled, width='stretch', hide_index=True)

    # Resets the "since your last visit" marker to now -- AFTER everything
    # above has already used the old baseline, so this visit's own newly-
    # triggered zones/rating actions are correctly shown as new THIS time,
    # and only stop being flagged starting next visit.
    save_watchlist_signals_last_viewed(conn, EFFECTIVE_EMAIL)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

# EARNINGS SIMULATOR Part 11 -- automatic daily reconciliation. Keyed on
# today's date in session_state (not a bare "has this run" flag), so a
# session left open across midnight still re-runs the next day rather
# than never again. reconcile_earnings_predictions() is cheap/idempotent
# when there's nothing pending, so re-running it once per new session on
# the same day is harmless too.
_today_str = date.today().isoformat()
if st.session_state.get("last_auto_reconcile_date") != _today_str:
    reconcile_earnings_predictions(get_conn())
    # Same pattern, same idempotent-when-nothing-pending guarantee, for
    # RUNNERS' Day Prediction feature (Part 6) -- grades yesterday-or-
    # earlier day_predictions rows against the now-backfilled daily close.
    reconcile_day_predictions(get_conn())
    # Part 4: the calibration loop runs automatically right after
    # reconciliation, same as log_day_prediction_snapshot's same-day
    # close-triggered call -- this is the once-per-session-open fallback
    # for whenever that same-day trigger didn't fire (e.g. the app wasn't
    # open at close and only reconciled on the next load).
    calibrate_day_prediction_model(get_conn())
    # MACRO CALENDAR: force_refresh=True (not just the normal 24h TTL)
    # so a new day always gets a real pull here, regardless of what time
    # yesterday's last fetch happened to land at -- a plain 24h-TTL check
    # could still be "not stale yet" minutes into a new calendar day.
    # Streamlit has no standalone background scheduler, so this is the
    # closest equivalent to "refresh at day start": the first session
    # that touches the app after midnight (local time) triggers it, same
    # pattern as the reconciliation calls just above. For a guaranteed
    # refresh even with the app never opened overnight, an OS-level cron
    # calling `python3 -c "import data_engine as de; "
    # "de.cached_macro_events(de.get_connection(), force_refresh=True)"`
    # would be the real fixed-12:01am equivalent -- outside what a
    # Streamlit script can trigger on its own.
    cached_macro_events(get_conn(), force_refresh=True)
    # SWING FORECAST Part 8 -- same idempotent-when-nothing-pending
    # reconciliation + calibration pattern, additive to the block above.
    # reconcile_swing_forecasts is horizon-agnostic (one pass covers every
    # horizon_days at once); calibration is scoped per horizon_days, so it
    # loops SWING_HORIZON_OPTIONS same as the Backtested Tickers registry
    # would for any other per-horizon setting.
    reconcile_swing_forecasts(get_conn())
    for _h in SWING_HORIZON_OPTIONS:
        calibrate_swing_forecast_model(get_conn(), _h)
    st.session_state["last_auto_reconcile_date"] = _today_str

# ---- Account block (multi-user) ----
st.sidebar.markdown(
    f'<div style="font-size:12px;">Signed in as<br><b>{html.escape(CURRENT_USER["display_name"] or ACTOR_EMAIL)}</b>'
    f'<br><span style="opacity:0.7;">{html.escape(ACTOR_EMAIL)}</span>'
    f'{" &middot; <b>ADMIN</b>" if IS_ADMIN else ""}</div>',
    unsafe_allow_html=True,
)
if st.sidebar.button("Sign out", key="sidebar_signout_btn"):
    st.session_state.pop("_auth_email", None)
    st.logout()

if IS_ADMIN:
    def _on_view_as_change():
        choice = st.session_state.get("_admin_view_as_select")
        st.session_state["view_as_email"] = None if choice in (None, ACTOR_EMAIL) else choice

    _view_as_options = [u["email"] for u in list_users(get_conn())]
    if ACTOR_EMAIL not in _view_as_options:
        _view_as_options.insert(0, ACTOR_EMAIL)
    if st.session_state.get("_admin_view_as_select") not in _view_as_options:
        st.session_state["_admin_view_as_select"] = EFFECTIVE_EMAIL
    st.sidebar.selectbox(
        "👁️ View as user", _view_as_options, key="_admin_view_as_select", on_change=_on_view_as_change,
        format_func=lambda e: f"{e} (me)" if e == ACTOR_EMAIL else e,
        help="Admin only. Switch the data context to any user to see and manage their watchlist, entry "
             "zones and preferences. Your permissions stay admin's.",
    )
    if EFFECTIVE_EMAIL != ACTOR_EMAIL:
        st.sidebar.warning(f"Viewing as {EFFECTIVE_EMAIL}", icon="👁️")
st.sidebar.markdown("<hr style='margin:8px 0;opacity:0.2;'>", unsafe_allow_html=True)

st.sidebar.markdown(
    f'<div style="color:{ACCENT};font-size:16px;font-weight:700;letter-spacing:0.05em;">CONTROL PANEL</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("⟳  REFRESH DATA", width='stretch'):
    with st.status("Refreshing data sources...", expanded=True) as status:
        def _cb(msg):
            status.write(msg)

        full_refresh(watchlist, db_path=DEFAULT_DB_PATH, progress_callback=_cb, force_refresh=True)
        status.update(label="Refresh complete", state="complete")
    st.cache_data.clear()
    get_health(force=True)
    st.rerun()

# --------------------------------------------------------------------------
# WATCHLIST SIGNALS background scan (Part 6): refresh_all_entry_zones()
# (auto-derives every watchlist ticker's zone -- the scheduler-driven
# replacement for the old manual form, itself TTL-gated so a real
# recompute only happens hourly even on a 5-min scan interval) + check_
# entry_zones() (unchanged price-vs-zone status logic) + check_
# rating_changes(). All reuse RUNNERS' own persisted refresh-interval
# setting (load_runners_refresh_interval) and the identical
# st.fragment(run_every=...) mechanism, rather than building a second
# scheduler -- this fragment isn't tied to any one tab's widget, so it
# runs regardless of which tab is currently open (satisfying "on app
# load" too, since a fresh session's first full script run executes this
# unconditionally). No UI here -- the WATCHLIST SIGNALS tab and the daily
# digest email both read the resulting state directly, rather than this
# fragment rendering anything itself.
# --------------------------------------------------------------------------

_signals_run_every = {"On demand": None, "5 min": "5m", "15 min": "15m", "1 hour": "1h"}[
    load_runners_refresh_interval(get_conn(), EFFECTIVE_EMAIL)
]


def _scan_watchlist_signals_fragment():
    _conn = get_conn()
    refresh_all_entry_zones(_conn, EFFECTIVE_EMAIL, watchlist)
    check_entry_zones(_conn, EFFECTIVE_EMAIL)
    check_rating_changes(_conn, watchlist)
    # Email Digest: checked on the SAME timer as everything else in this
    # fragment -- while a browser tab stays open and connected, this is
    # what gives the "retry every N minutes" behavior the digest needs.
    # run_daily_digest_send(force=False) first asks should_send_daily_
    # digest_now() (a cheap pure decision, no AI Briefing call or chart
    # render) and only builds the Featured Stock section + chart + sends
    # when that says ready -- safe to call on every tick. See should_
    # send_daily_digest_now's own docstring for the full honesty note on
    # why this is NOT a real background scheduler.
    try:
        run_digests_for_all_users(_conn)
    except Exception as _digest_err:
        print(f"[run_digests_for_all_users] unexpected error: {_digest_err}")


st.fragment(run_every=_signals_run_every)(_scan_watchlist_signals_fragment)()

# Catch-up path: also checked once on every FULL page load/rerun (not
# just the fragment's own timer above), so a browser opened/reloaded
# late in the day still gets same-day digest delivery on that visit,
# rather than depending on a tab having stayed open and ticking since
# send_time_et. Cheap (a few DB reads when nothing is due, since
# should_send_daily_digest_now's decision gates everything expensive)
# and self-guarded, safe to call unconditionally here.
try:
    run_digests_for_all_users(get_conn())
except Exception as _digest_err:
    print(f"[run_digests_for_all_users] unexpected error (full-load path): {_digest_err}")

# Macro Calendar upgrade (spec item 5b) -- mirrors the catch-up above so a High-impact release's one-time
# alert email still fires even on a day the digest itself is skipped (weekend/already-sent) or when no
# launchd job is configured. Independent try/except -- never blocks page render on a failure here.
try:
    check_and_send_macro_release_alerts(get_conn())
except Exception as _macro_alert_err:
    print(f"[check_and_send_macro_release_alerts] unexpected error (full-load path): {_macro_alert_err}")

_quiver_note = (
    "QUIVER_API_KEY set — congressional trades use QuiverQuant (paid)."
    if os.environ.get("QUIVER_API_KEY")
    else "QUIVER_API_KEY not set — congressional trades fall back to a free, often-unreliable Senate EFD scrape."
)
st.sidebar.markdown(
    f'<div class="smd-note">Sources: yfinance &middot; SEC EDGAR &middot; FINRA ATS &middot; '
    f"QuiverQuant/Senate EFD &middot; Polymarket Gamma API.<br>{_quiver_note}<br>"
    f'Dark pool % falls back to a volume z-score proxy when FINRA is unreachable.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="smd-header">
      <div>
        <div class="smd-title">SMART MONEY INTELLIGENCE</div>
        <div class="smd-subtitle">OPTIONS FLOW &middot; DARK POOLS &middot; CONGRESS &middot; INSIDERS &middot; 13F &middot; POLYMARKET</div>
      </div>
      <div class="smd-updated">LAST REFRESH<br>{last_refreshed()}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_macro_release_banner(get_conn())

_tab_labels = [
    "DIVERGENCE MAP",
    "OPTIONS FLOW",
    "CONGRESS & INSIDER",
    "DARK POOL",
    "POLYMARKET",
    "FUNDAMENTALS",
    "TICKER DEEP-DIVE",
    "NEWS",
    "RUNNERS",
    "SETTINGS",
    "EARNINGS SIMULATOR",
    "MACRO CALENDAR",
    "SWING FORECAST",
    "📍 WATCHLIST SIGNALS",
    "🌪️ VOLATILITY DISLOCATIONS",
]
# The ADMIN tab is only ever CREATED for role='admin' -- for a regular user
# it does not exist at all (not hidden, not disabled), and the code that
# renders it is never reached.
if IS_ADMIN:
    _tab_labels.append("⚙️ ADMIN")
_tabs = st.tabs(_tab_labels)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = _tabs[:15]
tab_admin = _tabs[15] if IS_ADMIN else None

# --------------------------------------------------------------------------
# Tab 1 — DIVERGENCE MAP
# --------------------------------------------------------------------------

with tab1:
    div_df = latest_divergence(watchlist)
    if div_df.empty:
        st.info("No divergence scores yet. Click **⟳ REFRESH DATA** in the sidebar to compute them.")
    else:
        st.plotly_chart(render_divergence_map(div_df, watchlist), width='stretch', key="divergence_map")

        st.markdown('<div class="smd-section">SIGNAL CARDS</div>', unsafe_allow_html=True)
        ordered = div_df.set_index("ticker").reindex(
            [t for t in watchlist if t in div_df["ticker"].values]
        ).reset_index()
        cols = st.columns(4)
        for i, row in ordered.iterrows():
            with cols[i % 4]:
                st.markdown(render_signal_card(row), unsafe_allow_html=True)

        missing = [t for t in watchlist if t not in div_df["ticker"].values]
        if missing:
            st.caption(f"No score yet for: {', '.join(missing)} — refresh to include them.")

# --------------------------------------------------------------------------
# Tab 2 — OPTIONS FLOW
# --------------------------------------------------------------------------

with tab2:
    sel = st.selectbox("Ticker", watchlist, key="opt_ticker")
    f_env = cached_fundamentals(get_conn(), sel, max_age_hours=12)
    f = f_env["data"]
    if f["price_df"].empty:
        st.warning(f"No price data for {sel}.")
    else:
        opt_fig, opt_macro_events = render_price_chart(f["price_df"].tail(120), sel, mode="candlestick")
        st.plotly_chart(opt_fig, width='stretch', key=f"opt_price_chart_{sel}")
        render_macro_event_expander(opt_macro_events)

    latest_fetch_row = q(
        "SELECT MAX(fetch_date) AS d FROM options_flow WHERE ticker=?", (sel,)
    )
    latest_fetch_date = latest_fetch_row["d"].iloc[0] if not latest_fetch_row.empty else None

    exp_options_row = q(
        """SELECT DISTINCT expiration FROM options_flow WHERE ticker=? AND fetch_date=?
           ORDER BY expiration""",
        (sel, latest_fetch_date),
    ) if latest_fetch_date else pd.DataFrame(columns=["expiration"])
    n_cached_expirations = len(exp_options_row)

    cov1, cov2 = st.columns([3, 1.2])
    with cov1:
        if n_cached_expirations:
            st.caption(
                f"{n_cached_expirations} expiration(s) cached for {sel} (default fetch covers "
                f"{DEFAULT_OPTIONS_MAX_EXPIRATIONS} nearest — later LEAPS-dated expirations may be "
                f"missing). Click 'Load ALL expirations' for complete coverage."
            )
    with cov2:
        if cov2.button("🔄 Load ALL expirations", width='stretch', key=f"opt_load_all_{sel}"):
            with st.spinner(f"Pulling every available expiration for {sel}..."):
                cached_options_flow(get_conn(), sel, max_age_hours=0.25, force_refresh=True,
                                    max_expirations=50)
            st.rerun()

    expiry_choices = ["All (unusual activity, near expirations)"] + exp_options_row["expiration"].tolist()
    ec1, ec2 = st.columns([2, 1.4])
    expiry_sel = ec1.selectbox("Expiration", expiry_choices, key=f"opt_expiry_{sel}")
    show_mode = ec2.radio(
        "Show", ["Unusual only", "All contracts"], horizontal=True, key=f"opt_show_mode_{sel}",
        help="Only affects the combined 'All expirations' view below — a single-expiry selection "
             "already shows every contract for that date.",
    )

    if expiry_sel == expiry_choices[0]:
        if show_mode == "Unusual only":
            st.markdown('<div class="smd-section">UNUSUAL OPTIONS ACTIVITY (TODAY)</div>', unsafe_allow_html=True)
            opt_raw = q(
                """SELECT option_type, strike, expiration, volume, open_interest, volume_oi_ratio,
                          implied_volatility, last_price, delta, gamma, theta, vega, bid, ask
                   FROM options_flow WHERE ticker=? AND fetch_date=? AND unusual=1
                   ORDER BY volume DESC LIMIT 40""",
                (sel, latest_fetch_date or date.today().isoformat()),
            )
        else:
            st.markdown('<div class="smd-section">ALL CONTRACTS &mdash; EVERY CACHED EXPIRATION</div>',
                       unsafe_allow_html=True)
            opt_raw_full = q(
                """SELECT option_type, strike, expiration, volume, open_interest, volume_oi_ratio,
                          implied_volatility, last_price, delta, gamma, theta, vega, bid, ask
                   FROM options_flow WHERE ticker=? AND fetch_date=?
                   ORDER BY volume DESC""",
                (sel, latest_fetch_date or date.today().isoformat()),
            )
            SHOW_ALL_CAP = 500
            if len(opt_raw_full) > SHOW_ALL_CAP:
                st.caption(
                    f"Showing the top {SHOW_ALL_CAP} of {len(opt_raw_full)} contracts by volume — "
                    f"select a specific expiration above to see its complete, unlimited chain."
                )
            opt_raw = opt_raw_full.head(SHOW_ALL_CAP)
    else:
        st.markdown(
            f'<div class="smd-section">FULL CHAIN &mdash; {expiry_sel}</div>', unsafe_allow_html=True
        )
        opt_raw = q(
            """SELECT option_type, strike, expiration, volume, open_interest, volume_oi_ratio,
                      implied_volatility, last_price, delta, gamma, theta, vega, bid, ask
               FROM options_flow WHERE ticker=? AND fetch_date=? AND expiration=?
               ORDER BY strike""",
            (sel, latest_fetch_date, expiry_sel),
        )

    if opt_raw.empty:
        st.caption("No options data captured yet for this selection. Click Refresh Data.")
    else:
        call_vol = int(opt_raw.loc[opt_raw["option_type"] == "call", "volume"].sum())
        put_vol = int(opt_raw.loc[opt_raw["option_type"] == "put", "volume"].sum())
        ratio = (call_vol / put_vol) if put_vol else float("nan")
        if pd.isna(ratio):
            ratio_txt, ratio_color = "—", TEXT_PRIMARY
        else:
            ratio_txt = f"{ratio:.2f}"
            ratio_color = COLOR_BULLISH if ratio > 1.5 else COLOR_BEARISH if ratio < 0.67 else TEXT_PRIMARY

        kcol1, kcol2, kcol3 = st.columns(3)
        kcol1.markdown(metric_chip_html("TOTAL CALL VOLUME", f"{call_vol:,}"), unsafe_allow_html=True)
        kcol2.markdown(metric_chip_html("TOTAL PUT VOLUME", f"{put_vol:,}"), unsafe_allow_html=True)
        kcol3.markdown(metric_chip_html("CALL/PUT RATIO", ratio_txt, color=ratio_color), unsafe_allow_html=True)

        st.plotly_chart(render_call_put_split(call_vol, put_vol), width='stretch',
                        key=f"cp_split_{sel}")

        spot = f["snapshot"].get("last_price")
        bias = compute_strike_bias(opt_raw, spot)
        if bias:
            lean_color = (
                COLOR_BULLISH if bias["lean"] == "BULLISH"
                else COLOR_BEARISH if bias["lean"] == "BEARISH"
                else COLOR_NEUTRAL
            )
            st.markdown(
                f"<span class='smd-note'>Skew: {bias['bullish_count']} bullish OTM calls vs "
                f"{bias['bearish_count']} bearish OTM puts &mdash; net "
                f"<span style='color:{lean_color};font-weight:700;'>{bias['lean']}</span> lean</span>",
                unsafe_allow_html=True,
            )

        if expiry_sel != expiry_choices[0]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Calls (left, green) vs puts (right, red) by strike — bar width scaled to the chain's max volume.")
            st.markdown(render_options_chain_table(opt_raw, spot), unsafe_allow_html=True)

        st.caption(
            "Greeks and expected move are calculated from each contract's implied volatility "
            "(Black-Scholes). IV reflects current market pricing and can shift sharply around "
            "earnings or news — treat these as estimates, not guarantees."
        )

        next_earnings_row = q(
            "SELECT next_earnings_date FROM earnings_signal WHERE ticker=? ORDER BY fetched_at DESC LIMIT 1",
            (sel,),
        )
        next_earnings_date = (
            next_earnings_row["next_earnings_date"].iloc[0] if not next_earnings_row.empty else None
        )
        render_options_synthesis_panel(opt_raw, spot=spot, earnings_date=next_earnings_date)

        st.markdown('<div class="smd-section">INTERPRETIVE BADGES</div>', unsafe_allow_html=True)
        st.markdown(render_options_badge_table(opt_raw), unsafe_allow_html=True)

        render_top_unusual_insights(opt_raw)

        st.markdown('<div class="smd-section">GAMMA EXPOSURE BY STRIKE</div>', unsafe_allow_html=True)
        st.caption(
            "Sum of (gamma × open interest × 100) per strike — calls add positive dealer gamma, "
            "puts negative. Concentration near spot approximates a 'gamma wall' where dealer "
            "hedging flow is strongest."
        )
        gamma_fig = render_gamma_exposure_chart(opt_raw, spot)
        if gamma_fig is not None:
            st.plotly_chart(gamma_fig, width='stretch', key=f"gamma_exposure_{sel}_{expiry_sel}")
        else:
            st.caption("No Greeks available for this selection yet — click Refresh Data to re-fetch with Greeks.")

        term_structure = _iv_term_structure(get_conn(), sel)
        if term_structure:
            st.markdown('<div class="smd-section">IV TERM STRUCTURE (ATM IV BY EXPIRY)</div>',
                       unsafe_allow_html=True)
            st.caption(
                "Elevated near-term IV vs. further-out expiries signals the market pricing in a "
                "specific near-term event (earnings, a catalyst, etc.). Background shading follows "
                "the same Low / Moderate / Elevated / Extreme IV tiers as the badges below."
            )
            st.plotly_chart(render_iv_term_structure_chart(term_structure), width='stretch',
                            key=f"iv_term_structure_{sel}")

        with st.expander("Raw table", expanded=(expiry_sel == expiry_choices[0])):
            opt_df = opt_raw.rename(columns={
                "option_type": "Type", "strike": "Strike", "expiration": "Expiry", "volume": "Volume",
                "open_interest": "OI", "volume_oi_ratio": "Vol/OI", "implied_volatility": "IV %",
                "last_price": "Last", "delta": "Delta", "gamma": "Gamma", "theta": "Theta/day",
                "vega": "Vega", "bid": "Bid", "ask": "Ask",
            }).copy()
            # Vol/OI is None whenever open_interest is 0/null (undefined
            # ratio) -- same dtype=object trap as the Greeks columns below
            # (a partly-None column isn't numeric to pandas, so plain
            # .round() throws "NoneType doesn't define __round__"), so it
            # needs the same coerce-first treatment, not a bare .round().
            opt_df["Vol/OI"] = pd.to_numeric(opt_df["Vol/OI"], errors="coerce").round(2)
            opt_df["IV %"] = (pd.to_numeric(opt_df["IV %"], errors="coerce") * 100).round(1)
            for gcol in ["Delta", "Gamma", "Theta/day", "Vega"]:
                # A same-day snapshot fetched before Greeks were wired in
                # (or a degenerate T<=0/sigma<=0 contract) can leave this
                # column all-None -> pandas keeps it as dtype=object,
                # which plain .round() can't handle; coerce first.
                opt_df[gcol] = pd.to_numeric(opt_df[gcol], errors="coerce").round(4)
            opt_df["Spread %"] = opt_df.apply(
                lambda r: compute_spread_pct(r["Bid"], r["Ask"], r["Last"]), axis=1
            )
            st.dataframe(opt_df, width='stretch', hide_index=True)

        st.markdown('<div class="smd-section">📖 WHAT DO THESE BADGES MEAN?</div>', unsafe_allow_html=True)
        render_badge_legend()

# --------------------------------------------------------------------------
# Tab 3 — CONGRESS & INSIDER
# --------------------------------------------------------------------------

with tab3:
    health = get_health()

    c1, c2, c3 = st.columns([1, 1, 2])

    def _count(table, txn_type):
        return int(q(f"SELECT COUNT(*) c FROM {table} WHERE transaction_type=?", (txn_type,))["c"].iloc[0])

    congress_buys, congress_sells = _count("congressional_trades", "BUY"), _count("congressional_trades", "SELL")
    insider_buys, insider_sells = _count("insider_trades", "BUY"), _count("insider_trades", "SELL")

    c1.metric("Congress BUY / SELL", f"{congress_buys} / {congress_sells}")
    c2.metric("Insider BUY / SELL", f"{insider_buys} / {insider_sells}")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        cong_status = health.get("congressional", {})
        dot, color, label = STATUS_STYLE.get(cong_status.get("status"), STATUS_STYLE["down"])
        st.markdown(
            f'<span style="color:{color};">{dot}</span> Congressional feed: '
            f'<span style="color:{color};">{label}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="smd-section">CONGRESSIONAL TRADES (STOCK ACT DISCLOSURES)'
        f'<span>{source_badge(cong_status if cong_status.get("source") else None)}</span></div>',
        unsafe_allow_html=True,
    )
    congress_df = q(
        """SELECT transaction_date AS Date, politician AS Politician, ticker AS Ticker,
                  transaction_type AS Type, amount_range AS Amount
           FROM congressional_trades ORDER BY transaction_date DESC LIMIT 200"""
    )
    if congress_df.empty:
        st.caption(congressional_empty_reason(health))
    else:
        st.dataframe(congress_df, width='stretch', hide_index=True)

    st.markdown('<div class="smd-section">SEC FORM 4 &mdash; INSIDER TRADES</div>', unsafe_allow_html=True)
    insider_df = q(
        """SELECT transaction_date AS Date, ticker AS Ticker, insider_name AS Insider, title AS Title,
                  transaction_type AS Type, shares AS Shares, price AS Price, value AS Value
           FROM insider_trades ORDER BY transaction_date DESC LIMIT 200"""
    )
    if insider_df.empty:
        st.caption("No insider filings captured yet. Click Refresh Data.")
    else:
        st.dataframe(insider_df, width='stretch', hide_index=True)

# --------------------------------------------------------------------------
# Tab 4 — DARK POOL
# --------------------------------------------------------------------------

with tab4:
    dp_df = q(
        f"""SELECT dp.* FROM dark_pool_signals dp
            INNER JOIN (SELECT ticker, MAX(date) md FROM dark_pool_signals
                        WHERE ticker IN ({','.join('?' * len(watchlist))}) GROUP BY ticker) latest
            ON dp.ticker = latest.ticker AND dp.date = latest.md""",
        tuple(watchlist),
    ) if watchlist else pd.DataFrame()

    if dp_df.empty:
        st.info("No dark pool readings yet. Click **⟳ REFRESH DATA** in the sidebar.")
    else:
        dp_lookup = dp_df.set_index("ticker")
        cols = st.columns(4)
        for i, ticker in enumerate([t for t in watchlist if t in dp_lookup.index]):
            row = dp_lookup.loc[ticker]
            with cols[i % 4]:
                st.plotly_chart(
                    render_gauge(row["dark_pool_pct"], ticker, is_proxy=bool(row["is_proxy"])),
                    width='stretch', key=f"dp_gauge_{ticker}",
                )
                st.caption(f"Volume z-score: {row['volume_zscore']:.2f} · signal: {row['signal']}")

# --------------------------------------------------------------------------
# Tab 5 — POLYMARKET
# --------------------------------------------------------------------------

def render_polymarket_section(df, key_prefix, bar_count=15, bar_height=420):
    top = df.dropna(subset=["yes_price"]).head(bar_count).iloc[::-1]
    if not top.empty:
        labels = [q_[:55] + ("…" if len(q_) > 55 else "") for q_ in top["question"]]
        fig = go.Figure(
            go.Bar(
                x=top["yes_price"] * 100, y=labels, orientation="h",
                marker_color=ACCENT,
                text=[f"{v * 100:.0f}%" for v in top["yes_price"]],
                textposition="outside",
                textfont=dict(color=TEXT_SECONDARY),
            )
        )
        fig.update_xaxes(range=[0, 105], title=dict(text="Implied YES probability", font=dict(color=TEXT_MUTED)))
        apply_theme(fig, height=bar_height, margin=dict(l=260, r=40, t=30, b=30))
        st.plotly_chart(fig, width='stretch', key=f"{key_prefix}_bar")

    display_df = df.copy()
    display_df["yes_price"] = (display_df["yes_price"] * 100).round(1)
    display_df.columns = ["Question", "Category", "YES %", "Volume", "Liquidity", "End Date"]
    st.dataframe(display_df, width='stretch', hide_index=True)


with tab5:
    poly_all = q(
        """SELECT question, category, yes_price, volume, liquidity, end_date
           FROM polymarket_events WHERE active=1 ORDER BY volume DESC LIMIT 300"""
    )
    if poly_all.empty:
        st.info("No Polymarket events synced yet. Click **⟳ REFRESH DATA** in the sidebar.")
    else:
        macro_df = filter_by_keywords(poly_all, MACRO_KEYWORDS).sort_values("volume", ascending=False).head(20)

        st.markdown('<div class="smd-section">MACRO SIGNALS</div>', unsafe_allow_html=True)
        st.caption(
            "Markets mentioning macro/geopolitical terms (fed, rate, inflation, china, taiwan, chips, "
            "export, tariff, ...) — filtered for relevance, not sorted by raw volume alone."
        )
        if macro_df.empty:
            st.caption("No data available — click **⟳ REFRESH DATA** to pull macro-relevant markets.")
        else:
            render_polymarket_section(macro_df, key_prefix="macro", bar_height=max(280, 26 * len(macro_df)))

        with st.expander("ALL MARKETS (unfiltered)", expanded=False):
            st.caption("Raw top-volume markets, unfiltered — includes novelty/celebrity/sports markets "
                       "with no bearing on the watchlist.")
            render_polymarket_section(poly_all.head(20), key_prefix="all_markets", bar_height=520)

# --------------------------------------------------------------------------
# Tab 6 — FUNDAMENTALS
# --------------------------------------------------------------------------

with tab6:
    sel6 = st.selectbox("Ticker", watchlist, key="fund_ticker")
    render_timeframe_chart_section(sel6, key_prefix="fund", mode="candlestick")

    st.markdown('<div class="smd-section">WATCHLIST FUNDAMENTALS</div>', unsafe_allow_html=True)
    with st.spinner("Pulling fundamentals..."):
        table = watchlist_fundamentals_table(tuple(watchlist))
    if table.empty:
        st.caption("No fundamentals available.")
    else:
        styled = table.copy()
        styled["Price"] = styled["Price"].map(lambda v: f"${v:.2f}")
        styled["Chg %"] = styled["Chg %"].map(lambda v: f"{v:.2f}%")
        styled["RSI14"] = styled["RSI14"].map(lambda v: fmt_num(v, 1))
        styled["Mkt Cap"] = styled["Mkt Cap"].map(human_money)
        styled["P/E"] = styled["P/E"].map(lambda v: fmt_num(v, 1))
        styled["Beta"] = styled["Beta"].map(lambda v: fmt_num(v, 2))
        st.dataframe(styled, width='stretch', hide_index=True)

# --------------------------------------------------------------------------
# AI Briefing confirmation dialog (Part 2) — the only path that can trigger
# generate_deep_analysis(). Never invoked automatically; only ever opened by
# an explicit click on "Generate AI Briefing" / "Re-run" below.
# --------------------------------------------------------------------------

@st.dialog("Confirm AI Briefing")
def _confirm_ai_briefing_dialog(ticker):
    conn = get_conn()
    cost = estimate_ai_briefing_cost(ticker, conn)
    st.write(md_safe(
        f"This will make 1 call to the Claude API using cached dashboard data for "
        f"**{ticker}** (no new external fetches). Estimated cost: ~${cost['cost_usd']:.2f}. Continue?"
    ))
    st.caption(
        f"Rough estimate from bundled context size: ~{cost['input_tokens']:,} input tokens, "
        f"~{cost['output_tokens']:,} output tokens."
    )
    d1, d2 = st.columns(2)
    if d1.button("Continue", type="primary", width='stretch', key=f"ai_dlg_yes_{ticker}"):
        with st.spinner(f"Generating AI briefing for {ticker}..."):
            errors = st.session_state.setdefault("ai_brief_errors", {})
            try:
                generate_deep_analysis(ticker, conn)
                errors.pop(ticker, None)
            except Exception as e:
                errors[ticker] = str(e)
        st.session_state.pop("ai_brief_pending_ticker", None)
        st.rerun()
    if d2.button("Cancel", width='stretch', key=f"ai_dlg_no_{ticker}"):
        st.session_state.pop("ai_brief_pending_ticker", None)
        st.rerun()


def render_backtest_report(ticker, conn, key_prefix="dd"):
    """Part 4 -- what backtest_day_predictions() actually found for
    `ticker`: summary stats, an error-distribution histogram, directional
    accuracy, and concrete best/worst-5 examples (not just aggregates).
    Renders nothing (silently) if this ticker has never been backtested --
    the caller decides whether that's worth its own message.

    Defined here (before tab7/tab9's module-level code) rather than
    alongside render_day_prediction_panel further down the file --
    dashboard.py executes top-to-bottom as a script, and tab7's DAY
    PREDICTION BACKTEST section calls this directly at module level, so
    it must already be defined by the time Python reaches that line, not
    merely by the time the script finishes.

    `key_prefix` disambiguates every keyed widget below when the SAME
    ticker's report renders from more than one call site in a single
    script run -- e.g. TICKER DEEP-DIVE defaults dd_ticker to a ticker
    that also happens to be showing in RUNNERS' Day Prediction panel
    that same run. Without a distinct prefix per call site, both calls
    would register identical widget keys (e.g. "backtest_hist_STX" from
    both places at once) and Streamlit raises StreamlitDuplicateElementKey
    -- confirmed live for STX. Callers MUST pass a unique prefix per
    distinct call site (not just per ticker)."""
    rows, latest_run = get_backtest_report_data(conn, ticker)
    if not latest_run:
        return
    composition = get_day_prediction_sample_composition(conn, ticker=ticker)
    with st.expander(f"📊 Backtest Report — {ticker}", expanded=True, key=f"backtest_report_{key_prefix}_{ticker}"):
        st.caption(
            f"Based on {composition['backtest_n']} backtested historical session(s) + "
            f"{composition['live_n']} live reconciled session(s). Last backtest run: "
            f"{latest_run['run_at'][:19].replace('T', ' ')} UTC, covering "
            f"{latest_run['date_range_start']} to {latest_run['date_range_end']} "
            f"({latest_run['qualifying_sessions_found']} of the last {latest_run['lookback_days']} trading "
            f"days triggered runner criteria — volume ≥2x 20-day avg or move ≥3%)."
        )
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Qualifying sessions found", latest_run["qualifying_sessions_found"])
        scol2.metric(
            "Avg |error|", f"{latest_run['mean_abs_error_pct']:.2f}%"
            if latest_run["mean_abs_error_pct"] is not None else "—",
        )
        scol3.metric(
            "Directional accuracy", f"{latest_run['direction_accuracy']:.0%}"
            if latest_run["direction_accuracy"] is not None else "—",
        )
        calibration = get_active_day_prediction_calibration(conn)
        st.caption(
            f"Current calibration bias correction applied to new predictions: "
            f"drift_scale={calibration['drift_scale']:.2f}, vol_scale={calibration['vol_scale']:.2f} "
            f"(1.00/1.00 = no adjustment)."
        )

        if not rows:
            st.caption("No backtested rows with a resolved error yet.")
            return

        errors = [r["error_pct"] for r in rows]
        hist_fig = go.Figure(data=[go.Histogram(x=errors, marker_color=ACCENT, nbinsx=min(20, max(6, len(errors) // 2)))])
        hist_fig.add_vline(x=0, line=dict(color=TEXT_SECONDARY, width=1, dash="dot"))
        hist_fig.update_xaxes(title_text="error_pct (actual close vs. predicted target, %)")
        hist_fig.update_yaxes(title_text="Sessions")
        st.plotly_chart(
            apply_theme(hist_fig, height=280, margin=dict(l=50, r=30, t=20, b=50)),
            width='stretch', key=f"backtest_hist_{key_prefix}_{ticker}",
        )

        ranked = sorted(rows, key=lambda r: abs(r["error_pct"]))
        best5, worst5 = ranked[:5], list(reversed(ranked[-5:]))

        def _fmt(r):
            return {
                "Date": r["session_date"], "Predicted Target": f"${r['target_price']:.2f}",
                "Actual Close": f"${r['actual_close_price']:.2f}", "Error %": f"{r['error_pct']:+.2f}%",
                "Direction correct": "✅" if r["prediction_correct_direction"] else "❌",
            }

        bcol, wcol = st.columns(2)
        with bcol:
            st.markdown("**Best 5 (lowest |error|)**")
            st.dataframe(pd.DataFrame([_fmt(r) for r in best5]), width='stretch', hide_index=True,
                         key=f"backtest_best5_{key_prefix}_{ticker}")
        with wcol:
            st.markdown("**Worst 5 (highest |error|)**")
            st.dataframe(pd.DataFrame([_fmt(r) for r in worst5]), width='stretch', hide_index=True,
                         key=f"backtest_worst5_{key_prefix}_{ticker}")


def render_macro_event_note(prediction_row, conn):
    """Part 5.4 -- if this prediction has any active_macro_event_ids
    (populated at commit time by log_day_prediction, see MACRO CALENDAR),
    surface it plainly right under the chart: labeling only, never a
    directional claim ("stock will move because of X") -- that needs
    real reconciled history to learn from, which this labeling is what
    accumulates toward. No-op if the list is empty."""
    event_ids = prediction_row.get("active_macro_event_ids") or []
    if not event_ids:
        return
    events = get_macro_events_by_id(conn, event_ids)
    if not events:
        return
    names = "; ".join(
        f"{e['event_name']} ({e['scheduled_date']}"
        f"{', ' + _format_et_time_12h(e['scheduled_time_et']) if e.get('scheduled_time_et') else ''})"
        for e in events
    )
    st.caption(f"📅 Note: {names} was scheduled within 48h of this session.")


def render_prediction_comparison_table(ticker, conn, committed, session_date, key_prefix):
    """Predicted-vs-actual comparison table for one session, read off the
    committed row's own cached simulated path via _predicted_price_at_time
    -- shared by the live view (today, mid-session) and the last-session
    view (a closed session's full logged history) so they can never
    render this differently. Renders nothing if there are no snapshots or
    no cached path for this session."""
    snaps = get_day_prediction_snapshots(conn, ticker, session_date)
    simulated_path = committed.get("simulated_path") or []
    if not (snaps and simulated_path):
        return
    model_start_time = pd.Timestamp(committed["model_start_time"])
    # Defensive: model_start_time should always be tz-aware (ET) for a
    # real prediction row, but data_engine.log_day_prediction now guards
    # the one known way a naive value could land here (a stale source=
    # 'backtest' row squatting on a live session_date, self-healed
    # there). Belt-and-suspenders guard here too -- .tz_convert(None) on
    # an aware timestamp silently STRIPS tz info rather than raising,
    # which is exactly what caused "Cannot compare tz-naive and
    # tz-aware timestamps" when model_start_time.tz was None.
    if model_start_time.tzinfo is None:
        model_start_time = model_start_time.tz_localize(simulated_path[0][0].tz or "America/New_York")
    table_rows = []
    for s in snaps:
        t = pd.Timestamp(s["snapshot_time"])
        if t.tzinfo is None:
            t = t.tz_localize("UTC").tz_convert(model_start_time.tz)
        predicted_price = _predicted_price_at_time(simulated_path, t)
        actual_price = s["actual_price"]
        delta = actual_price - predicted_price
        table_rows.append({
            "Time": t.strftime("%-I:%M %p"), "Predicted Price": f"${predicted_price:.2f}",
            "Actual Price": f"${actual_price:.2f}", "Delta": f"{delta:+.2f}",
            "Delta %": f"{delta / predicted_price * 100:+.2f}%" if predicted_price else "—",
        })
    st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True,
                 key=f"day_pred_table_{key_prefix}_{ticker}_{session_date}")


def render_last_session_metrics(last):
    """The 4 summary blocks (target price, actual close, error, direction
    call) -- shown for both a real last LIVE session and, as a fallback,
    the most recent BACKTESTED session when no live one exists yet.
    Restored per this request after an earlier pass replaced them with
    the chart alone; both now render together, not one instead of the
    other."""
    move_pct = (
        (last["target_price"] / last["model_start_price"] - 1) * 100
        if last["model_start_price"] else None
    )
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        "Predicted target", f"${last['target_price']:.2f}",
        f"{move_pct:+.2f}%" if move_pct is not None else None,
    )
    if last.get("reconciled_at") and last.get("actual_close_price") is not None:
        mcol2.metric("Actual close", f"${last['actual_close_price']:.2f}")
        mcol3.metric("Error", f"{last['error_pct']:+.2f}%" if last["error_pct"] is not None else "—")
        mcol4.metric("Direction call", "✅ Correct" if last["prediction_correct_direction"] else "❌ Wrong")
    else:
        mcol2.metric("Actual close", "Pending")
        mcol3.metric("Error", "—")
        mcol4.metric("Direction call", "—")


def render_last_session_full(ticker, conn):
    """The most recent LIVE session's full results for `ticker` -- the
    4 summary metric blocks PLUS the same rich chart (real candles,
    projected path, badge) and comparison table the live view renders,
    not either one alone. A routine pre-10am-ET wait for TODAY'S
    prediction shouldn't also hide yesterday's already-complete results.

    Fallback: if this ticker has never had a real LIVE session (e.g. just
    pinned/added and hasn't hit today's buffer window yet), shows the
    most recent BACKTESTED session's metrics instead of nothing -- clearly
    labeled as backtest-based, since a backtest row carries no simulated
    intraday path to chart (confirmed real case: LULU, pinned before its
    first live session committed)."""
    last = get_most_recent_day_prediction(conn, ticker)
    if last:
        st.markdown(f"#### {ticker} — Last session ({last['session_date']}, closed)")
        render_last_session_metrics(last)
        intraday_hist = get_intraday_bars_for_session(conn, ticker, last["session_date"])
        # backtest_error_pct/backtest_n must be the CURRENT track record
        # (same fix as the live chart's own backtest_error_pct bug) --
        # last["backtest_error_pct"] is frozen at whatever it was when
        # this row was committed, and backtest_n has no column on the row
        # at all. Without this, a ticker with hundreds of backtested
        # sessions would still show "insufficient backtest history" on
        # its last-session chart forever, exactly like the earlier bug.
        track = get_day_prediction_track_record(conn, ticker=ticker)
        try:
            fig, chart_macro_events = render_day_prediction_chart(
                ticker, last, intraday_hist if not intraday_hist.empty else None,
                track.get("mean_abs_error_pct"), track.get("n"),
            )
            st.plotly_chart(fig, width='stretch', key=f"last_session_chart_{ticker}_{last['session_date']}")
            render_macro_event_expander(chart_macro_events)
        except RuntimeError as e:
            st.caption(f"⛔ Could not render last session's chart: {e}")
        render_macro_event_note(last, conn)
        render_prediction_comparison_table(ticker, conn, last, last["session_date"], key_prefix="lastsession")
        if last["reconciled_at"] and last["actual_close_price"] is not None:
            outcome = (
                "✅ direction call was correct" if last["prediction_correct_direction"]
                else "❌ direction call was wrong"
            )
            st.caption(md_safe(
                f"Closed at ${last['actual_close_price']:.2f} vs. predicted ${last['target_price']:.2f} "
                f"({last['error_pct']:+.2f}% error) — {outcome}."
            ))
        else:
            st.caption("This session hasn't been reconciled yet — runs automatically right after close.")
        return

    bt_last = get_most_recent_backtest_prediction(conn, ticker)
    if not bt_last:
        st.caption(f"No completed session (live or backtested) on record yet for {ticker}.")
        return
    st.markdown(f"#### {ticker} — No live session yet")
    st.caption(
        f"This ticker hasn't completed a real live session yet — the chart above will populate "
        f"automatically the first time it commits a live prediction (today's session commits at "
        f"10:00 AM ET, once the buffer window passes). Showing the most recent BACKTESTED session "
        f"below instead ({bt_last['session_date']}) as a reference — no chart, since a backtested "
        f"session carries no simulated intraday path to plot, only the metrics."
    )
    render_last_session_metrics(bt_last)
    if bt_last["reconciled_at"] and bt_last["actual_close_price"] is not None:
        outcome = (
            "✅ direction call was correct" if bt_last["prediction_correct_direction"]
            else "❌ direction call was wrong"
        )
        st.caption(md_safe(
            f"(Backtest) Closed at ${bt_last['actual_close_price']:.2f} vs. predicted "
            f"${bt_last['target_price']:.2f} ({bt_last['error_pct']:+.2f}% error) — {outcome}."
        ))


# --------------------------------------------------------------------------
# Tab 7 — TICKER DEEP-DIVE
# --------------------------------------------------------------------------

with tab7:
    st.markdown('<div class="smd-section">SEARCH ANY TICKER</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns([3, 1, 1.3])
    dd_ticker = sc1.text_input(
        "Ticker", value=st.session_state.get("dd_ticker", "STX"),
        label_visibility="collapsed", placeholder="e.g. STX, WDC, MU, LYFT",
    ).strip().upper()
    fetch_live = sc2.button("Fetch live signals", width='stretch')
    gen_briefing_clicked = sc3.button("🧠 Generate AI Briefing", width='stretch')
    if gen_briefing_clicked and dd_ticker:
        st.session_state["ai_brief_pending_ticker"] = dd_ticker

    if st.session_state.get("ai_brief_pending_ticker"):
        _confirm_ai_briefing_dialog(st.session_state["ai_brief_pending_ticker"])

    if dd_ticker:
        st.session_state["dd_ticker"] = dd_ticker
        conn = get_conn()
        health = get_health()

        with st.spinner(f"Pulling live signals for {dd_ticker}...") if fetch_live else nullcontext():
            fund_env = cached_fundamentals(conn, dd_ticker, max_age_hours=12, force_refresh=fetch_live)
            fdd = fund_env["data"]

            if not fdd["price_df"].empty:
                earnings_env = cached_earnings_calendar(conn, dd_ticker, max_age_hours=12, force_refresh=fetch_live)
                targets_env = cached_analyst_targets(conn, dd_ticker, max_age_hours=12, force_refresh=fetch_live)
                buybacks_env = cached_buybacks(conn, dd_ticker, max_age_hours=168, force_refresh=fetch_live)
                news_env = cached_news(conn, dd_ticker, limit=8, max_age_hours=2, force_refresh=fetch_live)
                leaps_env = cached_leaps_candidates(conn, dd_ticker, max_age_hours=1, force_refresh=fetch_live)

                if fetch_live:
                    # Ordering matters: options_flow and earnings_signal must
                    # both be fresh in the DB *before* compute_divergence runs,
                    # since it only ever reads what's already there (it never
                    # fetches live itself) -- computing it first was a real bug,
                    # since it would score off yesterday's (or no) earnings data.
                    cached_options_flow(conn, dd_ticker, max_age_hours=0.25, force_refresh=True)
                    cached_dark_pool(conn, dd_ticker, max_age_hours=1, force_refresh=True)
                    cached_earnings_signal(conn, dd_ticker, max_age_hours=12, force_refresh=True)
                    compute_divergence(dd_ticker, conn)

        if fetch_live:
            st.cache_data.clear()

        if fdd["price_df"].empty:
            st.error(f"No data found for '{dd_ticker}'. Check the symbol.")
        else:
            snap, info = fdd["snapshot"], fdd["info"]
            earnings, targets = earnings_env["data"], targets_env["data"]
            buybacks, news_df = buybacks_env["data"], news_env["data"]

            hc1, hc2, hc3, hc4 = st.columns([2.2, 1, 1, 1.4])
            hc1.markdown(f"### {dd_ticker} &mdash; {info.get('shortName', '')}", unsafe_allow_html=True)
            hc2.metric("Price", f"${snap['last_price']:.2f}", f"{snap['change_pct']:.2f}%")
            hc3.metric("Trend (50/200 SMA)", snap["trend"])

            div_row = q(
                "SELECT * FROM divergence_scores WHERE ticker=? ORDER BY computed_date DESC LIMIT 1",
                (dd_ticker,),
            )
            with hc4:
                st.markdown("<br>", unsafe_allow_html=True)
                if not div_row.empty:
                    r = div_row.iloc[0]
                    st.markdown(badge_html(r["label"], r["score"]), unsafe_allow_html=True)
                else:
                    st.caption("No divergence score yet &mdash; click 'Fetch live signals'.", unsafe_allow_html=True)

            render_timeframe_chart_section(dd_ticker, key_prefix="dd", mode="line")

            # --------------------------------------------------------------
            # AI BRIEFING (Part 2) — synthesizes everything shown lower on
            # the page; never generated automatically, only via the
            # confirmation-gated button/dialog above.
            # --------------------------------------------------------------
            with st.expander("🧠 AI BRIEFING", expanded=True):
                brief_error = st.session_state.get("ai_brief_errors", {}).get(dd_ticker)
                if brief_error:
                    st.error(f"AI briefing failed: {brief_error}")

                ai_brief = get_cached_ai_brief(conn, dd_ticker, max_age_hours=AI_BRIEF_CACHE_HOURS)
                if ai_brief:
                    fetched_dt = pd.to_datetime(ai_brief.get("_fetched_at"), utc=True, errors="coerce")
                    if pd.notna(fetched_dt):
                        age_h = (pd.Timestamp.now(tz="UTC") - fetched_dt).total_seconds() / 3600
                        age_label = f"{age_h * 60:.0f}m ago" if age_h < 1 else f"{age_h:.1f}h ago"
                    else:
                        age_label = "recently"

                    acol1, acol2 = st.columns([4, 1])
                    acol1.caption(f"Last generated: {age_label}")
                    if acol2.button("🔄 Re-run", width='stretch', key=f"ai_rerun_{dd_ticker}"):
                        st.session_state["ai_brief_pending_ticker"] = dd_ticker
                        st.rerun()

                    # Real numeric snapshot -- our own computed values, never
                    # extracted from AI-generated prose. st.metric renders
                    # plain text, so it's immune to the markdown $-as-LaTeX
                    # bug that corrupts numbers embedded in written prose.
                    tech = ai_brief.get("technicals_snapshot") or {}
                    move = ai_brief.get("expected_move_snapshot") or {}
                    if tech or move:
                        # One value per metric -- packing "1D / 5D / 20D" or
                        # "$X / $Y" into a single st.metric overflowed the
                        # narrow column width in a 6-up row (only the first
                        # value stayed visible, the rest got clipped).
                        # Splitting each combined string into its own column
                        # fixes that without shrinking the numbers.
                        mcols = st.columns(9)
                        mcols[0].metric(
                            "Price", f"${tech['current_price']:.2f}" if tech.get("current_price") is not None else "—",
                        )
                        mcols[1].metric(
                            "1D", f"{tech['price_chg_1d_pct']:+.1f}%" if tech.get("price_chg_1d_pct") is not None else "—",
                        )
                        mcols[2].metric(
                            "5D", f"{tech['price_chg_5d_pct']:+.1f}%" if tech.get("price_chg_5d_pct") is not None else "—",
                        )
                        mcols[3].metric(
                            "20D", f"{tech['price_chg_20d_pct']:+.1f}%" if tech.get("price_chg_20d_pct") is not None else "—",
                        )
                        mcols[4].metric("RSI14", fmt_num(tech.get("rsi14"), 1))
                        mcols[5].metric("MACD hist", fmt_num(tech.get("macd_histogram"), 3))
                        mcols[6].metric(
                            "Vol vs 20D avg",
                            f"{tech['volume_vs_20d_avg_pct']:+.0f}%" if tech.get("volume_vs_20d_avg_pct") is not None else "—",
                        )
                        mcols[7].metric(
                            "Exp. move 1wk",
                            f"${move['expected_move_1wk_usd']:.0f}" if move.get("expected_move_1wk_usd") is not None else "—",
                        )
                        mcols[8].metric(
                            "Exp. move 4wk",
                            f"${move['expected_move_4wk_usd']:.0f}" if move.get("expected_move_4wk_usd") is not None else "—",
                        )
                        st.caption(
                            "Greeks and expected move are calculated from each contract's implied "
                            "volatility (Black-Scholes). IV reflects current market pricing and can "
                            "shift sharply around earnings or news — treat these as estimates, not "
                            "guarantees."
                        )

                    ai_brief_header("CURRENT STATE / SETUP")
                    st.markdown(ai_section_or_fallback(ai_brief.get("setup")))

                    ai_brief_header(
                        "TECHNICAL & CATALYST SETUP",
                        subtitle="AI SYNTHESIS — COMBINES TECHNICAL LEVELS WITH NEWS CONTEXT",
                    )
                    st.markdown(ai_section_or_fallback(ai_brief.get("technical_catalyst_setup")))
                    tls = ai_brief.get("technical_levels_snapshot") or {}
                    if tls.get("trend_structure") not in (None, "unknown"):
                        grounding_bits = [f"trend: {tls['trend_structure']}"]
                        if tls.get("key_level_below") is not None:
                            grounding_bits.append(f"support {tls['key_level_below']:,.2f}")
                        if tls.get("key_level_above") is not None:
                            grounding_bits.append(f"resistance {tls['key_level_above']:,.2f}")
                        interval_label = tls.get("interval", DEFAULT_INTERVAL)
                        lookback_label = tls.get("lookback", DEFAULT_LOOKBACK)
                        st.caption(
                            f"Grounded in computed levels ({interval_label}/{lookback_label}): "
                            + " · ".join(grounding_bits)
                        )

                    ai_brief_header("RETAIL & INSTITUTIONAL ANALYSIS")
                    rcol, icol = st.columns(2)
                    with rcol:
                        st.caption("Retail")
                        st.markdown(ai_section_or_fallback(ai_brief.get("retail_analysis")))

                        sentiment_env = cached_retail_sentiment(
                            conn, dd_ticker, max_age_hours=2, force_refresh=fetch_live,
                        )
                        sentiment_posts = sentiment_env["data"] or []
                        # Filters out posts that are just a run of cashtags
                        # with no real commentary (e.g. "$DRAL $KORU $SNDK
                        # $SOXL $WDC") -- pure noise in the raw list, no
                        # actual signal.
                        substantive = [p for p in sentiment_posts if _is_substantive_post_text(p.get("text"))]
                        st_posts = [p for p in substantive if p.get("source") == "stocktwits"]

                        st.markdown(
                            f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                            f'color:{TEXT_SECONDARY};">STOCKTWITS -- TAGGED SENTIMENT</div>',
                            unsafe_allow_html=True,
                        )
                        if st_posts:
                            agg = _aggregate_stocktwits_sentiment(sentiment_posts)
                            ratio_txt = (f"{agg['bull_bear_ratio']}:1" if agg["bull_bear_ratio"] is not None
                                         else "—")
                            st.caption(
                                f"{agg['bullish_count']} bullish · {agg['bearish_count']} bearish · "
                                f"{agg['untagged_count']} untagged (bull/bear ratio: {ratio_txt})"
                            )
                            _SENTIMENT_TAG_COLOR = {"Bullish": COLOR_BULLISH, "Bearish": COLOR_BEARISH}
                            for p in st_posts[:8]:
                                tag = p.get("sentiment")
                                tag_html = (
                                    f'<span style="color:{_SENTIMENT_TAG_COLOR[tag]};font-weight:700;'
                                    f'font-size:10px;margin-right:4px;">{tag.upper()}</span>'
                                    if tag in _SENTIMENT_TAG_COLOR else ""
                                )
                                st.markdown(
                                    f'<div style="font-size:12px;padding:3px 0;border-bottom:1px solid {BORDER};">'
                                    f'{tag_html}{html_safe_snippet(_truncate_post_text(p.get("text")))} '
                                    f'<span style="color:{TEXT_MUTED};font-size:10px;">'
                                    f'{html.escape(p.get("posted_at") or "")}</span></div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No StockTwits posts cached for this ticker yet.")

                        # ApeWisdom is an ATTENTION/MOMENTUM signal, not a
                        # sentiment signal -- an aggregate rank/mention
                        # stat, not a list of individual posts. No per-post
                        # text and no sentiment/polarity score exist in
                        # this data, so neither is fabricated here.
                        st.markdown(
                            f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                            f'color:{TEXT_SECONDARY};">APEWISDOM -- ATTENTION SIGNAL (NOT SENTIMENT)</div>',
                            unsafe_allow_html=True,
                        )
                        ape_env = cached_apewisdom_sentiment(conn, dd_ticker, max_age_hours=2,
                                                              force_refresh=fetch_live)
                        ape = ape_env["data"]
                        if ape:
                            rank_delta = None
                            if ape.get("rank") is not None and ape.get("rank_24h_ago"):
                                rank_delta = ape["rank_24h_ago"] - ape["rank"]  # positive = moved up in rank
                            delta_txt = (f" (was #{ape['rank_24h_ago']})" if rank_delta and rank_delta != 0
                                         else "")
                            acol1, acol2, acol3 = st.columns(3)
                            acol1.metric("Rank", f"#{ape['rank']}" if ape.get("rank") is not None else "—",
                                        delta_txt or None)
                            acol2.metric("Mentions (24h)", ape.get("mentions") if ape.get("mentions") is not None
                                        else "—")
                            attn = ape.get("attention_change_pct")
                            acol3.metric("Attention Δ (24h)", f"{attn:+.0f}%" if attn is not None else "—")
                        else:
                            st.caption(
                                "Not currently ranked on ApeWisdom (no recent Reddit mentions across "
                                "r/wallstreetbets and other finance subreddits)."
                            )
                    with icol:
                        st.caption("Institutional")
                        st.markdown(ai_section_or_fallback(ai_brief.get("institutional_analysis")))

                        thirteenf_env = cached_13f_changes(
                            conn, dd_ticker, max_age_hours=168, force_refresh=fetch_live,
                        )
                        thirteenf_filings = thirteenf_env["data"] or []
                        st.markdown(
                            f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                            f'color:{TEXT_SECONDARY};">13F FILERS</div>', unsafe_allow_html=True,
                        )
                        if thirteenf_filings:
                            for f in thirteenf_filings[:8]:
                                st.markdown(
                                    f'<div style="font-size:12px;padding:3px 0;border-bottom:1px solid {BORDER};">'
                                    f'{html_safe_snippet(f.get("institution") or "Unknown institution")} '
                                    f'<span style="color:{TEXT_MUTED};font-size:10px;">'
                                    f'{html.escape(f.get("filing_date") or "")} &middot; '
                                    f'{html.escape(f.get("form_type") or "")}</span></div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No recent 13F-HR filings found mentioning this ticker (SEC EDGAR).")

                        mb_header_col, mb_btn_col = st.columns([2, 1.3])
                        with mb_header_col:
                            st.markdown(
                                f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                                f'color:{TEXT_SECONDARY};">MARKETBEAT INSTITUTIONAL</div>',
                                unsafe_allow_html=True,
                            )
                        mb_cached = _read_marketbeat_institutional_cache(conn, dd_ticker)
                        with mb_btn_col:
                            if st.button("🏦 Fetch (opens Chrome)", key=f"mb_inst_fetch_{dd_ticker}"):
                                with st.spinner(
                                    "Launching a real Chrome window for MarketBeat -- watch for it "
                                    "to pop up; this can take 10-30s (Cloudflare challenge)...",
                                ):
                                    mb_env = cached_marketbeat_institutional_sentiment(
                                        conn, dd_ticker, force_refresh=True,
                                    )
                                mb_cached = mb_env["data"]
                                if not mb_cached:
                                    st.warning(
                                        "Returned nothing -- check the terminal for "
                                        "[marketbeat_institutional] log lines (ChromeDriver version "
                                        "mismatch, Cloudflare block, or a page layout change are the "
                                        "likely causes)."
                                    )
                        if mb_cached:
                            mcol1, mcol2, mcol3 = st.columns(3)
                            mcol1.metric("Ownership", mb_cached.get("ownership_pct") or "—")
                            buyers = mb_cached.get("buyers")
                            sellers = mb_cached.get("sellers")
                            mcol2.metric(
                                "Buyers / Sellers",
                                f"{buyers if buyers is not None else '—'} / {sellers if sellers is not None else '—'}",
                            )
                            bias = mb_cached.get("net_flow_bias_pct")
                            mcol3.metric("Buy bias", f"{bias}%" if bias is not None else "—")
                            for t in (mb_cached.get("recent_transactions") or [])[:5]:
                                action = t.get("action")
                                action_color = (
                                    COLOR_BULLISH if action == "buy"
                                    else COLOR_BEARISH if action == "sell" else TEXT_SECONDARY
                                )
                                st.markdown(
                                    f'<div style="font-size:12px;padding:3px 0;border-bottom:1px solid {BORDER};">'
                                    f'<span style="color:{action_color};">&#9679;</span> '
                                    f'{html_safe_snippet(t.get("institution") or "—")} '
                                    f'<span style="color:{TEXT_MUTED};font-size:10px;">'
                                    f'{html.escape(t.get("date") or "")} &middot; '
                                    f'{html.escape(t.get("shares") or "—")} shares</span></div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption(
                                "No MarketBeat institutional data cached for this ticker yet -- click "
                                "Fetch above. Requires a real, visible desktop Chrome window; only "
                                "works when this dashboard is running on a machine with an active "
                                "desktop session, never headless/server."
                            )

                    ai_brief_header("NEWS SUMMARY")
                    st.markdown(ai_section_or_fallback(ai_brief.get("news_summary")))

                    ai_brief_header("CATALYSTS TO WATCH")
                    catalysts = ai_brief.get("catalysts") or []
                    if catalysts:
                        ccols = st.columns(min(3, len(catalysts)))
                        for i, cat in enumerate(catalysts):
                            with ccols[i % len(ccols)]:
                                st.markdown(
                                    f"""<div style="border:1px solid {BORDER};border-radius:6px;
                                    padding:10px;margin-bottom:8px;min-height:120px;">
                                    <div style="color:{ACCENT};font-weight:700;font-size:13px;">
                                    {md_safe(cat.get('catalyst', '—'))}</div>
                                    <div style="color:{TEXT_SECONDARY};font-size:11px;margin-top:4px;">
                                    {md_safe(cat.get('expected_timing', '—'))}</div>
                                    <div style="font-size:12px;margin-top:6px;">
                                    {md_safe(cat.get('why_it_matters', ''))}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption(
                            "No specific named catalysts found in this ticker's cached news headlines."
                        )

                    ai_brief_header("EARNINGS TRACK RECORD")
                    st.markdown(ai_section_or_fallback(ai_brief.get("earnings_track_record")))

                    ai_brief_header("VERDICTS")
                    verdicts = ai_brief.get("verdicts") or []
                    if verdicts:
                        vcols = st.columns(len(verdicts))
                        bias_color = {"BULLISH": COLOR_BULLISH, "BEARISH": COLOR_BEARISH, "NEUTRAL": COLOR_NEUTRAL}
                        for i, v in enumerate(verdicts):
                            with vcols[i]:
                                bc = bias_color.get(v.get("bias"), COLOR_NEUTRAL)
                                st.markdown(
                                    f"""<div style="border:1px solid {BORDER};border-radius:6px;padding:10px;">
                                    <div style="font-size:11px;color:{TEXT_SECONDARY};">
                                    {md_safe(v.get('horizon', '—'))} &middot; {md_safe(v.get('target_date', '—'))}</div>
                                    <div style="color:{bc};font-weight:700;font-size:15px;margin-top:4px;">
                                    {md_safe(v.get('bias', '—'))}</div>
                                    <div style="font-size:11px;color:{TEXT_SECONDARY};">
                                    Confidence: {md_safe(v.get('confidence', '—'))}</div>
                                    <div style="font-size:12px;margin-top:6px;">
                                    {md_safe(v.get('key_risk', ''))}</div>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption("No verdicts returned.")
                else:
                    st.caption(
                        "No AI briefing yet. Click '🧠 Generate AI Briefing' above to synthesize "
                        "real fundamentals, technicals, options flow, real earnings track record, "
                        "real retail sentiment (StockTwits/ApeWisdom/Reddit), insider/congressional trades, "
                        "dark pool, and news into a narrative read via the Claude API. Makes one "
                        "paid API call, only on request."
                    )

            st.markdown(
                f'<div class="smd-section">FUNDAMENTALS<span>{source_badge(fund_env)}</span></div>',
                unsafe_allow_html=True,
            )
            fcols = st.columns(5)
            fcols[0].metric("Market Cap", human_money(info.get("marketCap")))
            fcols[1].metric("P/E (TTM)", fmt_num(info.get("trailingPE")))
            fcols[2].metric("Fwd P/E", fmt_num(info.get("forwardPE")))
            fcols[3].metric("Beta", fmt_num(info.get("beta")))
            fcols[4].metric("Rev Growth", fmt_pct(info.get("revenueGrowth")))

            st.markdown(
                f'<div class="smd-section">EARNINGS &amp; ANALYST TARGETS'
                f'<span>{source_badge(targets_env)}</span></div>',
                unsafe_allow_html=True,
            )
            ecol1, ecol2 = st.columns([1, 1.6])
            with ecol1:
                st.metric("Next earnings", fmt_date(earnings.get("next_earnings_date")))
                rec = (targets.get("recommendationKey") or "—").upper().replace("_", " ")
                n_analysts = targets.get("numberOfAnalystOpinions")
                st.metric("Analyst recommendation", rec, f"{n_analysts} analysts" if n_analysts else None)

                ehr_env = cached_earnings_history_real(conn, dd_ticker, max_age_hours=12)
                ehr_rows = [r for r in ehr_env["data"] if r.get("eps_beat_miss_pct") is not None
                            or r.get("revenue_beat_miss_pct") is not None]
                if ehr_rows:
                    latest = ehr_rows[0]
                    eps_pct, rev_pct = latest.get("eps_beat_miss_pct"), latest.get("revenue_beat_miss_pct")
                    eps_txt = (f"EPS {'Beat' if eps_pct >= 0 else 'Miss'} {eps_pct:+.2f}%"
                               if eps_pct is not None else "EPS —")
                    rev_txt = (f"Revenue {'Beat' if rev_pct >= 0 else 'Miss'} {rev_pct:+.2f}%"
                               if rev_pct is not None else "Revenue —")
                    eps_color = COLOR_BULLISH if (eps_pct or 0) >= 0 else COLOR_BEARISH
                    rev_color = COLOR_BULLISH if (rev_pct or 0) >= 0 else COLOR_BEARISH
                    st.markdown(
                        f'<div class="smd-note" style="margin-top:4px;">Latest reported ({latest["earnings_date"]}): '
                        f'<span style="color:{eps_color};font-weight:700;">{eps_txt}</span> / '
                        f'<span style="color:{rev_color};font-weight:700;">{rev_txt}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No real earnings beat/miss data on record for this ticker yet.")

                rec_counts = {k: targets.get(k) for k in ("strongBuy", "buy", "hold", "sell", "strongSell")}
                if any(v for v in rec_counts.values()):
                    st.plotly_chart(render_recommendation_breakdown(rec_counts), width='stretch',
                                    key=f"dd_rec_breakdown_{dd_ticker}")
                    st.caption(
                        f"Breakdown for period {targets.get('rec_period') or '—'} · "
                        f"{sum(v or 0 for v in rec_counts.values())} analysts"
                    )
                else:
                    st.caption("No data available.")
            with ecol2:
                st.markdown("**Analyst Price Targets (next 12 months)**")
                n_analysts_txt = targets.get("numberOfAnalystOpinions")
                median_val = targets.get("targetMedianPrice")
                # Literal "·" instead of the &middot; HTML entity -- st.caption's
                # unsafe_allow_html doesn't decode named entities the way
                # st.markdown does (confirmed live: it rendered the literal
                # text "&middot;" instead of "·"), so the entity+flag
                # combination that works elsewhere via st.markdown doesn't
                # carry over to st.caption. A real Unicode character sidesteps
                # the question entirely -- no HTML processing required at all.
                current_txt = f"Current ${snap['last_price']:,.2f} · " if snap.get("last_price") is not None else ""
                median_txt = f" · median ${median_val:,.2f}" if median_val is not None else ""
                st.caption(
                    f"{current_txt}Aggregated from {n_analysts_txt} analysts via Yahoo Finance{median_txt}."
                    if n_analysts_txt else
                    f"{current_txt}Aggregated via Yahoo Finance (analyst count unavailable){median_txt}."
                )
                has_targets = any(
                    targets.get(k) is not None
                    for k in ("targetLowPrice", "targetMeanPrice", "targetMedianPrice", "targetHighPrice")
                )
                if has_targets:
                    st.plotly_chart(
                        render_price_target_fan(
                            fdd["price_df"], snap["last_price"], targets.get("targetLowPrice"),
                            targets.get("targetMeanPrice"), targets.get("targetHighPrice"),
                        ),
                        width='stretch', key=f"dd_target_range_{dd_ticker}",
                    )
                else:
                    st.caption("No data available.")

            # Full width, not squeezed into ecol2 -- a 5-column table reads
            # far better with the whole panel's width than the ~62% share
            # it had before, and pulling it out of the two-column layout
            # is also what fixes ecol1/ecol2's height mismatch (ecol1's
            # short metrics+chart stack used to finish long before ecol2,
            # which kept growing with this table, leaving a large empty
            # gap under ecol1 the whole time).
            pt_breakdown = fetch_analyst_price_target_breakdown(dd_ticker)
            if not pt_breakdown.empty:
                st.caption("Individual analyst actions with an attached price target (most recent first):")
                show_pt = pt_breakdown.rename(columns={
                    "GradeDate": "Date", "Firm": "Firm", "Action": "Action",
                    "ToGrade": "Rating", "currentPriceTarget": "Price Target",
                }).copy()
                show_pt["Date"] = show_pt["Date"].apply(fmt_date)
                show_pt["Price Target"] = show_pt["Price Target"].map(lambda v: f"${v:,.2f}")
                st.dataframe(show_pt, width='stretch', hide_index=True)
            else:
                st.caption(
                    "Individual analyst names/targets not available via free tier right now "
                    "— showing aggregate only."
                )

            st.markdown("**Earnings History (Last 4 Quarters)**")
            eps_hist = earnings.get("eps_history")
            if eps_hist is not None and not eps_hist.empty:
                eh = eps_hist.copy()
                eh["Earnings Date"] = eh["Earnings Date"].apply(fmt_date)
                st.dataframe(eh, width='stretch', hide_index=True)
            else:
                st.caption("No data available.")

            st.markdown('<div class="smd-section">QUARTERLY FINANCIALS</div>', unsafe_allow_html=True)
            st.caption(
                "Revenue, net income, and net margin per fiscal quarter (yfinance), with the closing "
                "price at that period end and the move over the following 3 trading days — note this is "
                "the price at the fiscal period end, not the later earnings-release date; see EARNINGS "
                "TRACK RECORD in the AI Briefing above for the real announcement-date price reaction."
            )
            qfin_rows = fetch_quarterly_financials_with_price(dd_ticker)
            if qfin_rows:
                qfin_df = pd.DataFrame(qfin_rows)
                show_qfin = qfin_df.rename(columns={
                    "period_end": "Period End", "revenue": "Revenue", "net_income": "Net Income",
                    "margin_pct": "Net Margin", "price_at_period_end": "Price @ Period End",
                    "price_chg_3d_pct": "+3D Price Chg",
                }).copy()
                show_qfin["Revenue"] = show_qfin["Revenue"].map(human_money)
                show_qfin["Net Income"] = show_qfin["Net Income"].map(human_money)
                show_qfin["Net Margin"] = show_qfin["Net Margin"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
                show_qfin["Price @ Period End"] = show_qfin["Price @ Period End"].map(
                    lambda v: f"${v:.2f}" if pd.notna(v) else "—"
                )
                show_qfin["+3D Price Chg"] = show_qfin["+3D Price Chg"].map(
                    lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"
                )
                show_qfin = show_qfin[["Period End", "Revenue", "Net Income", "Net Margin",
                                        "Price @ Period End", "+3D Price Chg"]]
                st.dataframe(show_qfin, width='stretch', hide_index=True)
            else:
                st.caption("No quarterly financials available for this ticker.")

            st.markdown('<div class="smd-section">SMART MONEY SIGNALS (90D)</div>', unsafe_allow_html=True)
            sc_l, sc_r = st.columns(2)
            with sc_l:
                st.caption("Insider (Form 4)")
                idf = q(
                    """SELECT transaction_date AS Date, insider_name AS Insider, title AS Title,
                              transaction_type AS Type, shares AS Shares, value AS Value
                       FROM insider_trades WHERE ticker=? AND transaction_date >= date('now','-90 day')
                       ORDER BY transaction_date DESC LIMIT 10""",
                    (dd_ticker,),
                )
                if not idf.empty:
                    st.dataframe(idf, width='stretch', hide_index=True)
                else:
                    st.caption("No recent insider filings captured.")
            with sc_r:
                cong_status = health.get("congressional", {})
                dot, color, label = STATUS_STYLE.get(cong_status.get("status"), STATUS_STYLE["down"])
                st.markdown(
                    f'Congressional &nbsp;<span style="color:{color};font-size:10px;">{dot} {label}</span>',
                    unsafe_allow_html=True,
                )
                # Per-ticker QuiverQuant lookup (that ticker's full trade
                # history), not just whatever the market-wide "recent" feed
                # happened to capture -- falls back to [] + a clear reason
                # via congressional_empty_reason() when no key is set or the
                # call fails, same as every other empty-state in this app.
                cong_env = cached_congressional_trades_by_ticker(
                    conn, dd_ticker, max_age_hours=24, force_refresh=fetch_live
                )
                cutoff = (pd.Timestamp.now() - pd.Timedelta(days=90)).date().isoformat()
                recent_records = [r for r in cong_env["data"] if (r.get("trade_date") or "") >= cutoff]
                if recent_records:
                    cdf = pd.DataFrame(recent_records)[["trade_date", "senator", "trade_type", "amount"]]
                    cdf.columns = ["Date", "Politician", "Type", "Amount"]
                    st.dataframe(cdf, width='stretch', hide_index=True)
                    st.markdown(source_badge(cong_env), unsafe_allow_html=True)
                else:
                    # Reference SYSTEM STATUS's own health probe instead of
                    # duplicating logic -- shows *why* it's empty (no key /
                    # source down) right where the user is already looking.
                    st.caption(congressional_empty_reason(health))

            with st.expander("Snapshot History (last 10)", expanded=False):
                st.caption(
                    "Every full refresh (or 'Fetch live signals') appends a new row here -- this is what "
                    "lets 'today vs. tomorrow vs. hours before earnings' return different, comparable answers."
                )
                snap_df = q(
                    """SELECT fetched_at AS "Fetched At", snapshot_type AS Type,
                              ROUND(hours_to_earnings, 1) AS "Hrs to Earnings", conviction_score AS Conviction,
                              divergence_label AS Label, smart_call_signals AS "Smart Calls",
                              smart_put_signals AS "Smart Puts"
                       FROM ticker_snapshots WHERE ticker=? ORDER BY fetched_at DESC LIMIT 10""",
                    (dd_ticker,),
                )
                if snap_df.empty:
                    st.caption("No snapshots yet — every refresh from here on will add one.")
                else:
                    st.dataframe(snap_df, width='stretch', hide_index=True)

            st.markdown('<div class="smd-section">OPTIONS FLOW</div>', unsafe_allow_html=True)
            opt_raw_dd = q(
                """SELECT option_type, strike, expiration, volume, open_interest, volume_oi_ratio,
                          implied_volatility, last_price, delta, gamma, theta, vega, bid, ask, unusual
                   FROM options_flow WHERE ticker=? AND fetch_date=? AND unusual=1
                   ORDER BY volume DESC LIMIT 10""",
                (dd_ticker, date.today().isoformat()),
            )
            if not opt_raw_dd.empty:
                render_options_synthesis_panel(
                    opt_raw_dd, spot=snap.get("last_price"), earnings_date=earnings.get("next_earnings_date"),
                )
                odf = opt_raw_dd[
                    ["option_type", "strike", "expiration", "volume", "open_interest", "volume_oi_ratio"]
                ].rename(columns={
                    "option_type": "Type", "strike": "Strike", "expiration": "Expiry", "volume": "Volume",
                    "open_interest": "OI", "volume_oi_ratio": "Vol/OI",
                }).copy()
                # volume_oi_ratio is None whenever open_interest is 0/null
                # (undefined ratio), which leaves the column dtype=object
                # -- plain .round() throws "NoneType doesn't define
                # __round__" on that; coerce first (same fix as the
                # OPTIONS FLOW tab's Raw Table, same underlying data).
                odf["Vol/OI"] = pd.to_numeric(odf["Vol/OI"], errors="coerce").round(2)
                st.dataframe(odf, width='stretch', hide_index=True)
                render_top_unusual_insights(opt_raw_dd, top_n=3)
            else:
                st.caption("No unusual options activity captured yet &mdash; click 'Fetch live signals'.",
                          unsafe_allow_html=True)

            st.markdown(
                f'<div class="smd-section">LEAPS CANDIDATES (STOCK REPLACEMENT)'
                f'<span>{source_badge(leaps_env)}</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Deep-ITM calls, expiry &ge;18mo out, filtered on delta/theta/liquidity rules and ranked "
                "from the borderline (barely-passing) strike deeper ITM.", unsafe_allow_html=True,
            )
            leaps_df = leaps_env["data"]
            if not leaps_df.empty:
                bear_p = leaps_df["bear_price"].iloc[0]
                base_p = leaps_df["base_price"].iloc[0]
                bull_p = leaps_df["bull_price"].iloc[0]
                show = leaps_df[[
                    "contractSymbol", "expiry", "strike", "mid", "delta_est", "breakeven",
                    "option_cost", "bear_roi", "base_roi", "bull_roi",
                ]].copy()
                show.columns = [
                    "Contract", "Expiry", "Strike", "Mid", "Delta", "Breakeven", "Cost (1 ctr)",
                    f"ROI @ ${bear_p:,.0f} (bear)", f"ROI @ ${base_p:,.0f} (base)", f"ROI @ ${bull_p:,.0f} (bull)",
                ]
                show["Strike"] = show["Strike"].map(lambda v: f"${v:.2f}")
                show["Mid"] = show["Mid"].map(lambda v: f"${v:.2f}")
                show["Delta"] = show["Delta"].map(lambda v: f"{v:.3f}")
                show["Breakeven"] = show["Breakeven"].map(lambda v: f"${v:.2f}")
                show["Cost (1 ctr)"] = show["Cost (1 ctr)"].map(lambda v: f"${v:,.0f}")
                for col in show.columns[-3:]:
                    show[col] = show[col].map(lambda v: f"{v:+.1f}%")
                st.dataframe(show, width='stretch', hide_index=True)
            else:
                st.caption("No data available &mdash; no LEAPS expiry &ge;18mo out, or no deep-ITM calls "
                          "passed the stock-replacement rules for this ticker.", unsafe_allow_html=True)

            # --------------------------------------------------------------
            # FIND AN OPTION (Part 8) -- filters today's cached chain by
            # direction/risk/horizon, ranks by liquidity. "Find Matches"
            # always force-refreshes with enough expirations to cover the
            # chosen horizon (see TIME_HORIZON_MAX_EXPIRATIONS) rather than
            # relying on whatever a different tab's narrower default fetch
            # happened to cache today.
            # --------------------------------------------------------------
            st.markdown('<div class="smd-section">FIND AN OPTION</div>', unsafe_allow_html=True)
            st.caption(
                "Filters the option chain by direction, risk profile, and time horizon, then ranks "
                "matches by liquidity (open interest, tightest spread). Greeks and expected move are "
                "Black-Scholes estimates from each contract's own IV — treat as estimates, not guarantees."
            )
            pc1, pc2, pc3, pc4, pc5 = st.columns([1, 1.3, 1.5, 1.3, 1])
            direction_sel = pc1.selectbox("Direction", ["Bullish", "Bearish"], key=f"picker_dir_{dd_ticker}")
            risk_sel = pc2.selectbox("Risk profile", list(RISK_PROFILE_DELTA_RANGES.keys()), index=1,
                                      key=f"picker_risk_{dd_ticker}")
            horizon_sel = pc3.selectbox("Time horizon", list(TIME_HORIZON_DTE_RANGES.keys()),
                                        key=f"picker_horizon_{dd_ticker}")
            budget_sel = pc4.number_input(
                "Max budget/contract ($)", min_value=0, value=0, step=50,
                key=f"picker_budget_{dd_ticker}", help="0 = no budget limit",
            )
            pc5.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            find_clicked = pc5.button("Find Matches", width='stretch', key=f"picker_find_{dd_ticker}")

            picker_key = f"picker_result_{dd_ticker}"
            if find_clicked:
                with st.spinner(f"Pulling {horizon_sel} expirations for {dd_ticker}..."):
                    cached_options_flow(
                        conn, dd_ticker, max_age_hours=0.25, force_refresh=True,
                        max_expirations=TIME_HORIZON_MAX_EXPIRATIONS[horizon_sel],
                    )
                    st.session_state[picker_key] = find_option_candidates(
                        conn, dd_ticker, direction_sel, risk_sel, horizon_sel,
                        max_budget=budget_sel if budget_sel > 0 else None, top_n=5,
                    )

            picker_result = st.session_state.get(picker_key)
            if picker_result is None:
                st.caption("Set your criteria above and click 'Find Matches'.")
            elif not picker_result["candidates"]:
                st.caption(picker_result["note"] or "No matches found.")
            else:
                if picker_result["relaxed"]:
                    st.warning(
                        f"No exact match — relaxed: {', '.join(picker_result['relaxed'])}. "
                        "Results below reflect the relaxed constraints."
                    )
                badge_defs = [
                    ("delta", DELTA_BADGE_COLORS, None), ("iv", IV_BADGE_COLORS, None),
                    ("vol_oi", None, VOL_OI_PROFILES), ("dte", None, None),
                    ("theta", None, THETA_PCT_PROFILES), ("spread", None, SPREAD_PROFILES),
                ]
                for c in picker_result["candidates"]:
                    badges_html = []
                    for key, color_map, tier_profiles in badge_defs:
                        b = c["badges"][key]
                        if color_map is not None:
                            color = color_map.get(b["label"], COLOR_NEUTRAL)
                        elif tier_profiles is not None:
                            color = _tier_escalation_color(b["label"], tier_profiles)
                        else:
                            color = COLOR_NEUTRAL
                        badges_html.append(interp_badge_html(b["label"], color, b["note"]))
                    type_color = COLOR_BULLISH if c["type"] == "call" else COLOR_BEARISH
                    delta_txt = f"{c['delta']:.3f}" if c["delta"] is not None else "—"
                    iv_txt = f"{c['iv_pct']:.1f}%" if c["iv_pct"] is not None else "—"
                    cost_txt = f"${c['cost_per_contract']:,.0f}" if c["cost_per_contract"] is not None else "—"
                    breakeven_txt = f"${c['breakeven']:.2f}" if c["breakeven"] is not None else "—"
                    oi_txt = f"{c['open_interest']:,}" if c["open_interest"] is not None else "—"
                    st.markdown(md_safe(f"""
                    <div class="smcard">
                      <div style="display:flex;justify-content:space-between;align-items:baseline;">
                        <div style="font-size:15px;font-weight:700;color:{TEXT_PRIMARY};">
                          <span style="color:{type_color};">{c['type'].upper()}</span>
                          ${c['strike']:g} &mdash; {html.escape(str(c['expiration']))}
                        </div>
                        <div style="font-size:12px;color:{TEXT_MUTED};">{c['dte']} DTE</div>
                      </div>
                      <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:6px;font-size:12px;
                                  color:{TEXT_SECONDARY};">
                        <div>Delta: <b style="color:{TEXT_PRIMARY};">{delta_txt}</b></div>
                        <div>IV: <b style="color:{TEXT_PRIMARY};">{iv_txt}</b></div>
                        <div>Cost (1 ctr): <b style="color:{TEXT_PRIMARY};">{cost_txt}</b></div>
                        <div>Breakeven: <b style="color:{TEXT_PRIMARY};">{breakeven_txt}</b></div>
                        <div>OI: <b style="color:{TEXT_PRIMARY};">{oi_txt}</b></div>
                      </div>
                      <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px;">
                        {"".join(badges_html)}
                      </div>
                    </div>
                    """), unsafe_allow_html=True)

                st.markdown('<div class="smd-section">📖 WHAT DO THESE BADGES MEAN?</div>',
                           unsafe_allow_html=True)
                render_badge_legend()

            st.markdown(
                f'<div class="smd-section">BUYBACKS<span>{source_badge(buybacks_env)}</span></div>',
                unsafe_allow_html=True,
            )
            if not buybacks["history"].empty:
                bcol1, bcol2 = st.columns([2, 1])
                with bcol1:
                    st.plotly_chart(render_buyback_chart(buybacks["history"]), width='stretch',
                                    key=f"dd_buybacks_{dd_ticker}")
                with bcol2:
                    st.metric("Buyback pace", buybacks["trend"])
            else:
                st.caption("No data available.")

            st.markdown('<div class="smd-section">DARK POOL</div>', unsafe_allow_html=True)
            ddp = q(
                "SELECT * FROM dark_pool_signals WHERE ticker=? ORDER BY date DESC LIMIT 1",
                (dd_ticker,),
            )
            if not ddp.empty:
                row = ddp.iloc[0]
                dpcol1, _ = st.columns([1, 2])
                with dpcol1:
                    st.plotly_chart(
                        render_gauge(row["dark_pool_pct"], dd_ticker, is_proxy=bool(row["is_proxy"])),
                        width='stretch', key=f"dd_gauge_{dd_ticker}",
                    )
            else:
                st.caption("No dark pool reading yet.")

            st.markdown('<div class="smd-section">NEWS</div>', unsafe_allow_html=True)
            if not news_df.empty:
                for _, n in news_df.iterrows():
                    pub = n["publisher"] or "Unknown source"
                    ago = time_ago(n["published_at"])
                    headline = f"[{n['title']}]({n['link']})" if n["link"] else n["title"]
                    st.markdown(
                        f"- {headline} &nbsp;&middot;&nbsp; <span class='smd-note'>{pub} &middot; {ago}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No data available.")

            st.markdown(
                f'<span class="smd-note">More sources: '
                f'<a href="https://www.financialjuice.com/home" target="_blank">Financial Juice</a> &middot; '
                f'<a href="https://www.financialjuice.com/Nasdaq/{dd_ticker}" target="_blank">'
                f'Financial Juice &mdash; {dd_ticker}</a></span>',
                unsafe_allow_html=True,
            )

            related_keywords = keywords_for_ticker(dd_ticker, info.get("sector"))
            st.markdown('<div class="smd-section">RELATED POLYMARKET MARKETS</div>', unsafe_allow_html=True)
            st.caption(f"Filtered by: {', '.join(related_keywords)}")
            poly_all_dd = q(
                """SELECT question, category, yes_price, volume, liquidity, end_date
                   FROM polymarket_events WHERE active=1 ORDER BY volume DESC LIMIT 300"""
            )
            related_df = (
                filter_by_keywords(poly_all_dd, related_keywords).sort_values("volume", ascending=False).head(5)
                if not poly_all_dd.empty else poly_all_dd
            )
            if related_df.empty:
                st.caption("No data available.")
            else:
                show = related_df.copy()
                show["yes_price"] = (show["yes_price"] * 100).round(1)
                show = show[["question", "yes_price", "volume"]]
                show.columns = ["Question", "YES %", "Volume"]
                st.dataframe(show, width='stretch', hide_index=True)

            # Part 5 -- manual backtest control for ANY ticker, not just
            # current RUNNERS/watchlist tickers: dd_ticker's own search box
            # above already lets the user type any symbol, so this reuses
            # it rather than adding a second ticker input.
            st.markdown('<div class="smd-section">📈 DAY PREDICTION BACKTEST</div>', unsafe_allow_html=True)
            st.caption(
                "Simulates the Day Prediction pipeline against real historical daily sessions for this ticker "
                "-- day by day, using only data that would have been available at the time (no lookahead) -- "
                "so a real, sizeable reconciled sample is available immediately instead of waiting weeks for "
                "live RUNNERS sessions to accumulate. See RUNNERS for the live, same-day prediction; this is "
                "historical validation and calibration bootstrap."
            )
            if st.button(f"🔬 Run Backtest for {dd_ticker}", key=f"run_backtest_{dd_ticker}"):
                with st.spinner(
                    f"Backtesting {dd_ticker} against the last {BACKTEST_DEFAULT_LOOKBACK_DAYS} trading days..."
                ):
                    bt_result = backtest_day_predictions(dd_ticker, conn)
                if bt_result.get("ok"):
                    st.success(
                        f"Backtest complete: {bt_result['qualifying_sessions_found']} qualifying session(s) "
                        f"found ({bt_result['sessions_logged']} newly logged, "
                        f"{bt_result['sessions_skipped_existing']} already on record) from "
                        f"{bt_result['date_range_start']} to {bt_result['date_range_end']}. Now available in "
                        f"the RUNNERS Day Prediction flow with real calibrated confidence."
                    )
                else:
                    st.error(bt_result.get("reason", "Backtest failed."))
                st.rerun()
            render_backtest_report(dd_ticker, conn, key_prefix="dd")
            if not get_backtest_report_data(conn, dd_ticker)[1]:
                st.caption(f"{dd_ticker} hasn't been backtested yet — click the button above to run one.")

# --------------------------------------------------------------------------
# Tab 8 — NEWS (watchlist-wide feed; distinct from per-ticker news on
# TICKER DEEP-DIVE)
# --------------------------------------------------------------------------

with tab8:
    st.markdown('<div class="smd-section">WATCHLIST NEWS FEED</div>', unsafe_allow_html=True)
    ncol1, ncol2 = st.columns([3, 1])
    with ncol2:
        st.markdown("<br>", unsafe_allow_html=True)
        force_news = st.button("🔄 Refresh feed", width='stretch', key="news_feed_refresh")

    conn_news = get_conn()
    st.caption("Feeds are managed in the SETTINGS tab → **News Feeds** (add, remove, enable/disable).")

    # Defensive, on top of the real fix (get_conn() is now per-thread,
    # not one globally-shared connection) -- this exact loop is what hit
    # the live crash, so it gets its own explicit belt-and-suspenders
    # guard: one ticker's fetch failing (whatever the cause) surfaces as
    # a soft warning and moves on, never takes down the whole page.
    _news_fetch_errors = []
    with st.spinner("Pulling news across the watchlist...") if force_news else nullcontext():
        for t in watchlist:
            try:
                cached_news(conn_news, t, limit=8, max_age_hours=2, force_refresh=force_news)
            except Exception as e:
                _news_fetch_errors.append((t, str(e)))
    if _news_fetch_errors:
        st.warning(
            "News fetch failed for: " + ", ".join(f"{t} ({err})" for t, err in _news_fetch_errors)
            + " -- other tickers below are unaffected."
        )
    if force_news:
        st.cache_data.clear()

    placeholders = ",".join("?" * len(watchlist))
    news_all = q(
        f"""SELECT ticker, title, publisher, link, published_at, source
            FROM news_cache WHERE ticker IN ({placeholders})
            ORDER BY published_at DESC LIMIT 300""",
        tuple(watchlist),
    ) if watchlist else pd.DataFrame()

    if news_all.empty:
        st.info("No news cached yet for the watchlist. Click '🔄 Refresh feed'.")
    else:
        news_all["published_at"] = pd.to_datetime(news_all["published_at"], utc=True, errors="coerce")
        # Filter options are the currently-enabled feed *names* (the
        # news_sources registry), not article publisher domains -- feed
        # management (add/remove/enable) lives entirely in SETTINGS, this
        # is view-only narrowing of what's already been fetched.
        enabled_feed_names = [s["name"] for s in _read_news_sources(conn_news, enabled_only=True)]
        feed_options = sorted(set(enabled_feed_names) | set(news_all["source"].dropna().unique()))

        fcol1, fcol2, fcol3 = st.columns([2, 1.2, 1.2])
        with fcol1:
            # st.multiselect only applies `default=` the very first time a
            # widget with this `key` is ever rendered in a session -- every
            # add/remove/enable in SETTINGS after that point changes
            # feed_options but NOT the already-initialized widget, which
            # keeps whatever was selected back when it first appeared
            # (confirmed live: adding two new feeds left this stuck
            # showing 0 headlines despite the underlying data being fine).
            # Keying on the actual option set forces a fresh widget --
            # and therefore a fresh, correct default -- exactly when that
            # set changes, instead of silently going stale.
            feed_key = "news_feed_filter_" + "_".join(sorted(feed_options))
            feed_sel = st.multiselect("Feed", feed_options, default=enabled_feed_names or feed_options,
                                       key=feed_key)
        min_d = news_all["published_at"].min()
        max_d = news_all["published_at"].max()
        with fcol2:
            date_from = st.date_input(
                "From", value=(min_d.date() if pd.notna(min_d) else date.today()), key="news_date_from"
            )
        with fcol3:
            date_to = st.date_input(
                "To", value=(max_d.date() if pd.notna(max_d) else date.today()), key="news_date_to"
            )

        filtered = news_all[
            news_all["source"].isin(feed_sel)
            & (news_all["published_at"].dt.date >= date_from)
            & (news_all["published_at"].dt.date <= date_to)
        ]

        st.caption(
            f"{len(filtered)} headlines · {len(watchlist)} tickers · "
            f"chronological, most recent first"
        )

        if filtered.empty:
            st.caption("No headlines match the current filters.")
        else:
            for _, n in filtered.iterrows():
                pub = md_safe(n["publisher"] or "Unknown source")
                feed_name = md_safe(n["source"] or "—")
                ago = time_ago(n["published_at"])
                title = md_safe(n["title"])
                headline = f"[{title}]({n['link']})" if n["link"] else title
                st.markdown(
                    f"""<div style="border:1px solid {BORDER};border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                    <span style="background:{PANEL_ALT};color:{ACCENT};font-size:10px;font-weight:700;
                    padding:2px 8px;border-radius:10px;letter-spacing:0.04em;">{n['ticker']}</span>
                    <div style="margin-top:6px;">{headline}</div>
                    <div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;">{pub} &middot; via {feed_name} &middot; {ago}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

# --------------------------------------------------------------------------
def render_day_prediction_chart(ticker, committed, intraday_hist, backtest_error_pct, backtest_n, height=420):
    """Forecast-cone chart for one RUNNERS ticker: real candles for the
    session so far, a shaded 'MODEL ACTIVE' zone from model-start forward,
    a solid projected path to the committed target, the real price path
    drawn on top (dotted, so prediction vs. reality is easy to read as
    they diverge/converge), a horizontal current-price reference line, a
    dashed FW TARGET line, and a directional badge in the corner."""
    fig = go.Figure()

    model_start_time = pd.Timestamp(committed["model_start_time"])
    model_start_price = committed["model_start_price"]
    target_price = committed["target_price"]
    predicted_direction = committed["predicted_direction"]
    session_date = model_start_time.date()
    market_close = pd.Timestamp(
        year=session_date.year, month=session_date.month, day=session_date.day,
        hour=16, minute=0, tz=model_start_time.tz,
    )

    # Part 6: macro events scheduled on this session's own date -- fetched
    # here, but the actual markers aren't drawn until price data (below)
    # is known, since the hover-dot needs a real y_bottom to sit at.
    macro_events_today = get_macro_events_in_range(get_conn(), session_date, session_date,
                                                    overlay_impact_filter="auto")

    if intraday_hist is not None and not intraday_hist.empty:
        fig.add_trace(go.Candlestick(
            x=intraday_hist.index, open=intraday_hist["Open"], high=intraday_hist["High"],
            low=intraday_hist["Low"], close=intraday_hist["Close"],
            increasing_line_color=COLOR_BULLISH, decreasing_line_color=COLOR_BEARISH,
            increasing_fillcolor=COLOR_BULLISH, decreasing_fillcolor=COLOR_BEARISH,
            name="Session so far", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=intraday_hist.index, y=intraday_hist["Close"], mode="lines",
            line=dict(color=COLOR_RETAIL, width=1.5, dash="dot"), name="Actual price path",
        ))
        current_price = float(intraday_hist["Close"].iloc[-1])
    else:
        current_price = model_start_price

    # "Enough" reconciled history to trust THIS ticker's own backtest error
    # over a conservative default -- 5 is a deliberately low bar (this
    # feature starts with zero history system-wide), just enough to not
    # be a single lucky/unlucky day.
    if backtest_n and backtest_n >= 5 and backtest_error_pct is not None:
        error_txt = f"±{backtest_error_pct:.1f}% backtest error"
    else:
        est_pct = committed.get("magnitude_estimate_pct")
        error_txt = f"±{est_pct:.1f}% (est., insufficient backtest history)" if est_pct is not None else "±est."
    # This chart is shared by the live in-progress view (render_day_
    # prediction_panel) and the closed/reconciled "last session" view
    # (render_last_session_full) -- `committed` carries reconciled_at in
    # both cases (None while the session is still open), so that's the
    # one signal that actually distinguishes them. Previously this said
    # "MODEL ACTIVE" unconditionally, which read as wrong/confusing on an
    # already-closed, already-graded session (e.g. WMT's 2026-08-20 chart
    # still saying "MODEL ACTIVE" hours after close and reconciliation).
    status_label = "SESSION CLOSED" if committed.get("reconciled_at") else "MODEL ACTIVE"
    fig.add_vrect(
        x0=model_start_time, x1=market_close, fillcolor="rgba(0,229,255,0.06)", line_width=0,
        annotation_text=f"{status_label} — {error_txt}", annotation_position="top left",
        annotation_font=dict(size=10, color=ACCENT),
    )

    # The cached simulated path (Part 1/2 of the projected-path fix) --
    # a real drift-adjusted random walk with genuine up/down texture, NOT
    # a straight 2-point line, which implied false precision. Generated
    # once at commit time (log_day_prediction) and read back unchanged
    # here every render, so this line is identical across refreshes.
    #
    # Part 3 (this bug report): log_day_prediction now backfills a missing
    # path onto ANY existing row (not just newly-committed ones), so by
    # the time this renders, committed["simulated_path"] should never be
    # empty for a real ticker. If it somehow still is, that means path
    # generation/caching genuinely failed -- per the explicit instruction
    # not to silently degrade to a straight line implying false precision,
    # this raises instead of drawing the old dotted-line fallback. The
    # caller (render_day_prediction_panel) catches this and shows a loud
    # st.error rather than crashing the whole page.
    simulated_path = committed.get("simulated_path") or []
    if not simulated_path:
        raise RuntimeError(
            f"{ticker}: no cached simulated path available for session "
            f"{model_start_time.date().isoformat()} -- path generation/caching failed. Refusing to render a "
            f"straight-line fallback, which would imply false precision."
        )
    fig.add_trace(go.Scatter(
        x=[p[0] for p in simulated_path], y=[p[1] for p in simulated_path], mode="lines",
        line=dict(color="#e8c547", width=2), name="Projected path (simulated)",
    ))

    # Now that real price data is known, draw the macro event line(s) +
    # hover dot at the true bottom of this chart's combined price range
    # (candles, simulated path, current/target levels) -- intraday, so
    # the line positions at the event's real scheduled_time_et, not just
    # the date.
    price_lows = [p for _, p in simulated_path] + [model_start_price, target_price, current_price]
    price_highs = list(price_lows)
    if intraday_hist is not None and not intraday_hist.empty:
        price_lows.append(float(intraday_hist["Low"].min()))
        price_highs.append(float(intraday_hist["High"].max()))
    y_lo, y_hi = min(price_lows), max(price_highs)
    y_bottom = y_lo - (y_hi - y_lo) * 0.04
    add_macro_event_markers(fig, macro_events_today, intraday=True, chart_tz=model_start_time.tz, y_bottom=y_bottom)

    fig.add_hline(y=current_price, line=dict(color=TEXT_SECONDARY, width=1, dash="dot"),
                  annotation_text=f"Last: ${current_price:.2f}", annotation_position="right",
                  annotation_xshift=8)

    # Part 1 fix: the actual target PRICE was getting cut off at the plot
    # boundary -- a wide annotation ("FW TARGET  $483.18 · EXP -1.67%")
    # positioned flush against the right edge with only ~20px of margin
    # (the default BASE_LAYOUT) simply had nowhere to render. Fixed by
    # widening the right margin below AND shifting the label fully
    # outside the plot area (annotation_xshift) rather than clipping
    # against it.
    move_pct = (target_price / model_start_price - 1) * 100 if model_start_price else 0.0
    fig.add_hline(y=target_price, line=dict(color="#e8c547", width=1.5, dash="dash"),
                  annotation_text=f"FW TARGET: ${target_price:.2f}  ·  EXP {move_pct:+.2f}%",
                  annotation_position="right", annotation_font=dict(color="#e8c547", size=11),
                  annotation_xshift=8)

    badge_color = (
        COLOR_BULLISH if predicted_direction == "UP" else COLOR_BEARISH if predicted_direction == "DOWN"
        else COLOR_NEUTRAL
    )
    badge_label = (
        "BULLISH" if predicted_direction == "UP" else "BEARISH" if predicted_direction == "DOWN" else "NEUTRAL"
    )
    # Part 1 fix (this bug report): the badge (an opaque colored box) and
    # the vrect's own "MODEL ACTIVE -- ±X%..." honesty caption were BOTH
    # anchored top-left (x=0.02, y=0.98 vs. annotation_position="top
    # left"), so the badge rendered directly on top of the caption text,
    # making it unreadable. Moved the badge to the bottom-right corner
    # instead: the vrect caption owns top-left, and the "Last"/"FW
    # TARGET" hline labels float near their own price levels (usually
    # mid-chart, not the extreme bottom edge) rather than a fixed paper
    # corner -- bottom-right is the one spot none of them structurally
    # claim. Verified clear on both WDC and MU.
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.03, xanchor="right", yanchor="bottom",
        text=f"<b>{ticker}  {badge_label}  {move_pct:+.2f}%  →  Target: ${target_price:.2f}</b>",
        showarrow=False, font=dict(size=13, color=PANEL), bgcolor=badge_color, bordercolor=badge_color,
        borderpad=6, borderwidth=1,
    )

    fig.update_xaxes(title_text="Time (ET)")
    fig.update_yaxes(title_text="Price ($)")
    fig.update_layout(xaxis_rangeslider_visible=False)
    # Part 1 fix: the honesty-labeling caption ("MODEL ACTIVE -- ±X% (est.,
    # insufficient backtest history)") was rendering partially off-screen
    # at the very top -- the default t=42/r=20 margin (BASE_LAYOUT) has no
    # room for either that top-left annotation or the FW TARGET/Last
    # right-edge labels. Both fixed the same way: give the figure real
    # margin instead of relying on default spacing meant for a plain chart.
    apply_theme(
        fig, height=height, margin=dict(l=55, r=175, t=75, b=60),
        legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)"),
    )
    return fig, macro_events_today


def render_day_prediction_panel(ticker, conn, buffer_minutes=DAY_PREDICTION_BUFFER_MINUTES_DEFAULT):
    """One RUNNERS ticker's full Day Prediction panel: FIRST calls
    ensure_ticker_data_ready (Part 2 of the data-backfill fix) -- a
    synchronous, blocking prerequisite, every time, so a brand-new ticker
    never dead-ends on 'insufficient data' when the fix is one real fetch
    away (shows a brief spinner only when an actual fetch is needed, not
    on every render). Then computes/commits today's target (once per
    session_date -- log_day_prediction is a no-op after the first ready
    call), logs a live snapshot, renders the forecast-cone chart against
    the STABLE committed target (never the freshly-recomputed one, so the
    target line doesn't drift mid-session as compute_day_target's own
    live inputs keep changing), and the running prediction-vs-actual
    table."""
    # A quick pre-check (pure DB row-count reads, no network) so the
    # spinner only shows when a real fetch is actually about to happen --
    # not on every render once this ticker is already warmed up. Just a
    # display heuristic; ensure_ticker_data_ready below is what actually
    # decides (via should_refetch) whether to fetch.
    has_daily = conn.execute("SELECT 1 FROM price_history WHERE ticker=? LIMIT 1", (ticker,)).fetchone()
    has_intraday = conn.execute(
        "SELECT 1 FROM intraday_price_history WHERE ticker=? LIMIT 1", (ticker,)
    ).fetchone()
    if not (has_daily and has_intraday):
        with st.spinner(f"Preparing {ticker} — downloading price history..."):
            readiness = ensure_ticker_data_ready(ticker, conn)
    else:
        readiness = ensure_ticker_data_ready(ticker, conn)

    if not readiness["ready"]:
        # Part 3, case 1: genuine data unavailability -- a real fetch was
        # just attempted (above) and came back empty. Never "haven't
        # fetched it yet."
        st.error(
            f"⛔ **{ticker}** — {' '.join(readiness['still_missing']) or 'No reliable price history available.'}"
        )
        return

    prediction = compute_day_target(ticker, conn, buffer_minutes=buffer_minutes)
    if not prediction.get("ready"):
        reason = prediction.get("reason")
        icon = "⏳" if reason == "waiting_for_buffer" else "⚠️"
        st.info(f"{icon} **{ticker}** — {prediction.get('wait_message', 'Waiting for session data.')}")
        # This request: the Backtest Report and last-session results don't
        # depend on TODAY'S prediction being ready -- they were previously
        # hidden behind this same early return, which meant a routine
        # pre-10am-ET wait (expected, not a bug) also hid unrelated,
        # already-available historical results. Both render regardless.
        render_last_session_full(ticker, conn)
        render_backtest_report(ticker, conn, key_prefix="runners")
        return
    if readiness["still_missing"]:
        # Intraday history genuinely unavailable (not a hard blocker --
        # compute_day_target already fell back to daily-vol scaling) --
        # surfaced once, distinctly from a real error.
        st.caption(f"ℹ️ {' '.join(readiness['still_missing'])}")

    # Part 3: both log_day_prediction (path generation/backfill) and
    # render_day_prediction_chart (the "no path, no silent straight-line"
    # guard) now raise RuntimeError instead of degrading silently if path
    # caching genuinely fails -- caught here and shown as a loud, explicit
    # error so a real failure is visible instead of hidden, without
    # crashing the whole RUNNERS tab for the other tickers in the panel.
    try:
        log_day_prediction(conn, ticker, prediction)
        committed = get_committed_day_prediction(conn, ticker, prediction["session_date"]) or prediction
        log_day_prediction_snapshot(conn, ticker, prediction["session_date"], prediction["current_price"])
        # backtest_error_pct/backtest_n are passed from the FRESH `prediction`
        # dict, not the frozen `committed` row -- unlike target_price/
        # simulated_path (which must stay frozen, since they're the actual
        # committed prediction being graded), this is just an informational
        # confidence caption and should always reflect the CURRENT track
        # record. Using committed's own value was a real bug: a row
        # committed before backtest_day_predictions existed (or before a
        # ticker had 5+ backtested sessions) captured backtest_error_pct=
        # None permanently, so the chart kept saying "insufficient backtest
        # history" forever afterward even once hundreds of backtested
        # sessions existed -- confirmed live on MU/WDC's 2026-08-18 rows.
        fig, chart_macro_events = render_day_prediction_chart(
            ticker, committed, prediction.get("intraday_bars"), prediction.get("backtest_error_pct"),
            prediction.get("backtest_n"),
        )
    except RuntimeError as e:
        st.error(f"⛔ **{ticker}** — {e}")
        return
    # This request: these 4 blocks previously only showed on the "last
    # session" (closed) view -- the live view had no equivalent, so the
    # panel's own composition visibly changed the moment today's session
    # became ready (right at the model-start buffer), which read as
    # metrics randomly disappearing. render_last_session_metrics already
    # uses .get() defensively for the outcome fields, so it's directly
    # compatible with a live, not-yet-reconciled `committed` row -- it
    # just shows "Pending"/"—" for Actual close/Error/Direction call
    # until the session actually closes and reconciles.
    render_last_session_metrics(committed)
    st.plotly_chart(fig, width='stretch', key=f"day_pred_chart_{ticker}_{prediction['session_date']}")
    render_macro_event_expander(chart_macro_events)
    render_macro_event_note(committed, conn)

    # Live comparison table (Part 4): Predicted Price at each snapshot's
    # timestamp is read off the SAME cached simulated path the chart plots
    # (_predicted_price_at_time interpolates between that path's own
    # adjacent 5-min points, not a fresh straight-line calc from the two
    # endpoints) -- Part 3's requirement that the chart and table can never
    # disagree, since both read the exact same underlying path. Shared
    # with the "last session" view (render_prediction_comparison_table)
    # so live and closed-session tables can never render this differently.
    render_prediction_comparison_table(ticker, conn, committed, prediction["session_date"], key_prefix="live")

    # Part 3, case 2: thin RECONCILED PREDICTION track record (Mode 1) is
    # the one legitimate, expected form of "still building" -- it takes
    # real trading days/weeks to accumulate reconciled outcomes, unlike
    # raw price/volatility data (which Part 2 now backfills synchronously,
    # in seconds). Labeled explicitly so the two are never confused.
    #
    # Historical Backtest addition: the pooled sample now bootstraps fast
    # via backtest_day_predictions, so the mode caption states the
    # backtest-vs-live composition explicitly -- live sessions remain the
    # higher-trust category as they accumulate (real intraday momentum +
    # AI Briefing input a backtest can't reconstruct), even though
    # backtested sessions get the sample size up far faster.
    composition = get_day_prediction_sample_composition(get_conn())
    comp_txt = f"{composition['backtest_n']} backtested + {composition['live_n']} live session(s)"
    mode_txt = {
        "raw_pattern": f"Mode 1: raw pattern ({comp_txt}, pooled system-wide) — still below the 10-sample floor",
        "bayesian": f"Mode 2: Bayesian estimate ({comp_txt}, pooled system-wide)",
        "trained_model": f"Mode 3: trained model ({comp_txt} in the training pool)",
    }.get(committed.get("mode"), committed.get("mode"))
    conf_txt = f", {committed['confidence_level']} confidence" if committed.get("confidence_level") else ""
    st.caption(f"Mode: {mode_txt}{conf_txt}.")
    st.caption(
        "Target derived from this ticker's 1-year daily volatility + today's early momentum + technical/"
        "qualitative context — not a trained intraday pattern model (insufficient intraday history exists yet)."
    )
    render_backtest_report(ticker, conn, key_prefix="runners")


# --------------------------------------------------------------------------
# Tab 9 — RUNNERS: unusual real-time activity scan across the watchlist
# --------------------------------------------------------------------------

with tab9:
    st.markdown('<div class="smd-section">UNUSUAL ACTIVITY SCAN</div>', unsafe_allow_html=True)

    now_et = pd.Timestamp.now(tz="America/New_York")
    is_weekday = now_et.weekday() < 5
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    market_is_open = is_weekday and market_open <= now_et <= market_close
    if not market_is_open:
        st.warning(
            "Market closed — Runners reflects last session's data, not live intraday activity.",
            icon="🌙",
        )

    # Computed by data_engine.compute_and_log_runner_scan -- the ONE implementation this tab and the
    # headless launchd digest send (digest/send.py) both call, so "what counts as a runner today" can never
    # disagree between the two.
    top_rows, runner_tickers, total_flagged = compute_and_log_runner_scan(get_conn(), EFFECTIVE_EMAIL, watchlist)

    if not top_rows:
        st.caption("No unusual volume or price moves detected in the current watchlist right now.")
    else:
        runners_df = pd.DataFrame([
            {"Ticker": r["ticker"], "Price": r["price"], "Chg %": r["chg_pct"], "Volume": r["volume"],
             "Vol vs 20D avg": r["vol_ratio"]}
            for r in top_rows
        ])
        styled = runners_df.copy()
        styled["Price"] = styled["Price"].map(lambda v: f"${v:.2f}")
        styled["Chg %"] = styled["Chg %"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        styled["Volume"] = styled["Volume"].map(lambda v: f"{v:,}")
        styled["Vol vs 20D avg"] = styled["Vol vs 20D avg"].map(lambda v: f"{v:.1f}x" if pd.notna(v) else "—")
        st.caption(
            f"Top {len(runners_df)} of {total_flagged} flagged watchlist ticker(s) shown (RUNNERS caps at "
            f"4) — sorted most unusual first (volume ≥2x 20-day average, or move ≥3% intraday)."
        )
        st.dataframe(styled, width='stretch', hide_index=True)

    # Manually pinned tickers (this request): any watchlist ticker can be
    # added to Day Prediction even if it never triggers the unusual-
    # volume/move criteria above -- the same "not restricted to
    # auto-detected candidates" freedom the EARNINGS SIMULATOR tab already
    # gives via its free-text ticker box, except here it's a pick FROM the
    # watchlist (not any arbitrary symbol), since RUNNERS is explicitly
    # watchlist-scoped everywhere else in this tab. Auto-cleanup: a pin
    # naturally drops off the moment its ticker leaves the watchlist --
    # SETTINGS' "Configure Watchlist" is the only place that list changes,
    # and this re-filters against it on every rerun.
    # Persisted to pinned_runners.json (this request) -- previously
    # session_state-only, so every restart (or even a plain browser
    # refresh) silently lost every pin. Loaded from disk once per
    # session, saved back on every add/remove/auto-cleanup so the file
    # never drifts from what's actually shown.
    if "pinned_runners" not in st.session_state:
        st.session_state.pinned_runners = load_pinned_runners(get_conn(), EFFECTIVE_EMAIL)
    _pinned_before_cleanup = list(st.session_state.pinned_runners)
    st.session_state.pinned_runners = [t for t in st.session_state.pinned_runners if t in watchlist]
    if st.session_state.pinned_runners != _pinned_before_cleanup:
        save_pinned_runners(get_conn(), EFFECTIVE_EMAIL, st.session_state.pinned_runners)

    display_tickers = runner_tickers + [t for t in st.session_state.pinned_runners if t not in runner_tickers]

    # Header + manual-add control render whenever there's a watchlist to
    # pick from -- NOT gated on display_tickers, since the whole point is
    # letting the user bootstrap Day Prediction for a ticker even when zero
    # natural runners exist today.
    if watchlist:
        st.markdown('<div class="smd-section">DAY PREDICTION</div>', unsafe_allow_html=True)
        addable = [t for t in watchlist if t not in display_tickers]
        acol1, acol2 = st.columns([3, 1])
        ticker_to_add = acol1.selectbox(
            "Add a watchlist ticker to Day Prediction (even if not currently flagged as a runner)",
            options=["—"] + addable, index=0, key=_uk("runners_manual_add_select"),
        )
        acol2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if acol2.button("➕ Add", key=_uk("runners_manual_add_btn"), disabled=(ticker_to_add == "—"),
                         width='stretch'):
            st.session_state.pinned_runners.append(ticker_to_add)
            save_pinned_runners(get_conn(), EFFECTIVE_EMAIL, st.session_state.pinned_runners)
            st.rerun()

    if display_tickers:
        # Persisted to runners_refresh_interval.json (this request) --
        # previously session_state-only, so any full browser reload/
        # reconnect (a NEW server-side session, not the same one)
        # silently reset a chosen "5 min" back to "On demand" with zero
        # visible warning -- confirmed as the real cause behind a
        # "stuck, not refreshing every 5 min like I set it" report.
        if _uk("runners_refresh_interval") not in st.session_state:
            st.session_state[_uk("runners_refresh_interval")] = load_runners_refresh_interval(get_conn(), EFFECTIVE_EMAIL)
        rcol1, rcol2 = st.columns([1, 3])
        refresh_choice = rcol1.selectbox(
            "Refresh interval", ["On demand", "5 min", "15 min", "1 hour"],
            key=_uk("runners_refresh_interval"),
            help="'On demand' (default, matching the rest of this dashboard) only re-evaluates on your next "
                 "manual interaction. The timed options auto-refresh this section during market hours. Your "
                 "choice is saved and survives a page reload.",
        )
        if refresh_choice != st.session_state.get(_uk("_runners_refresh_interval_saved")):
            save_runners_refresh_interval(get_conn(), EFFECTIVE_EMAIL, refresh_choice)
            st.session_state[_uk("_runners_refresh_interval_saved")] = refresh_choice
        run_every = {"On demand": None, "5 min": "5m", "15 min": "15m", "1 hour": "1h"}[refresh_choice]
        # Part 2 (prior bug report): the timed refresh options kept firing
        # on their own schedule indefinitely, including hours after the
        # 4pm ET close -- the root cause behind the repeated identical-
        # price comparison rows logged at 6:17/7:27/7:32/7:36/7:39 PM ET.
        # log_day_prediction_snapshot now hard-gates on market hours too
        # (belt-and-suspenders), but there's no reason to keep the
        # fragment's own timer running at all once the session is over.
        if run_every and not market_is_open:
            st.caption(f"⏸️ Market is closed — \"{refresh_choice}\" auto-refresh is paused until the next session.")
            run_every = None

        def _render_day_predictions_fragment(tickers, pinned):
            # This request: a visible "last ran" indicator, since the
            # refresh interval only controls the fragment's OWN timer --
            # it still fully re-executes on any other rerun too (any
            # widget touched anywhere on the page), so "every 5 min" was
            # never a strict guarantee and there was no way to tell when
            # it had actually last run. Captured fresh every time this
            # fragment body executes, by definition.
            st.caption(
                f"🕐 Last updated: {pd.Timestamp.now(tz='America/New_York').strftime('%-I:%M:%S %p')} ET "
                f"(refresh: {refresh_choice})"
            )
            for rt in tickers:
                hcol1, hcol2 = st.columns([5, 1])
                hcol1.markdown(f"#### {rt}" + ("  📌 manually added" if rt in pinned else ""))
                if rt in pinned:
                    if hcol2.button("✕ Remove", key=_uk(f"runners_remove_pin_{rt}")):
                        st.session_state.pinned_runners = [p for p in st.session_state.pinned_runners if p != rt]
                        save_pinned_runners(get_conn(), EFFECTIVE_EMAIL, st.session_state.pinned_runners)
                        st.rerun()
                render_day_prediction_panel(rt, get_conn())

        st.fragment(run_every=run_every)(_render_day_predictions_fragment)(
            display_tickers, st.session_state.pinned_runners
        )
    else:
        st.caption("No unusual watchlist tickers right now, and none manually added yet — use the control above.")

# --------------------------------------------------------------------------
# Tab 10 — SETTINGS (source registry + watchlist config; both formerly in
# the sidebar)
# --------------------------------------------------------------------------

with tab10:
    render_settings_tab()

# --------------------------------------------------------------------------
# Tab 11 — EARNINGS SIMULATOR
# --------------------------------------------------------------------------

with tab11:
    render_earnings_simulator_tab()

# --------------------------------------------------------------------------
# Tab 12 — MACRO CALENDAR
# --------------------------------------------------------------------------

with tab12:
    render_macro_calendar_tab()

# --------------------------------------------------------------------------
# Tab 13 — SWING FORECAST
# --------------------------------------------------------------------------

with tab13:
    render_swing_forecast_tab()

# --------------------------------------------------------------------------
# Tab 14 — 📍 WATCHLIST SIGNALS
# --------------------------------------------------------------------------

with tab14:
    render_watchlist_signals_tab()


# --------------------------------------------------------------------------
# Tab 15 — 🌪️ VOLATILITY DISLOCATIONS
# --------------------------------------------------------------------------
# Stage 3 of the Volatility Dislocations screener. Display-only: reads the
# `scan` table Stage 2 already populated, zero option-chain fetches from
# this tab itself. The only action that touches the network is the
# "Run scan now" button, which launches a DETACHED background process
# (data_engine.start_volatility_scan_on_demand) -- never inside this
# request -- so the rest of the app (Runners, Day Predictions, digest,
# every other tab) stays completely unaffected while a scan runs.

def render_volatility_dislocations_tab():
    conn = get_conn()
    st.markdown('<div class="smd-section">🌪️ VOLATILITY DISLOCATIONS</div>', unsafe_allow_html=True)
    st.caption("Names where the options market is pricing an unusual move relative to its own baseline. "
               "Directionless — use it to pick what to look at, not what to buy.")

    # --- on-demand run trigger + live progress (DB-backed, survives a page refresh) ---
    run = get_latest_volatility_scan_run(conn)
    run_active = run is not None and run["status"] == "running"

    col_run, col_status = st.columns([1, 3])
    with col_run:
        if st.button("▶️ Run scan now", disabled=run_active, width='stretch', key=_uk("vol_run_now")):
            new_id = start_volatility_scan_on_demand()
            if new_id is None:
                st.warning("A scan is already running — only one at a time.")
            st.rerun()
    with col_status:
        if run_active:
            stage_label = {"tier1": "Tier 1 rebuild", "tier2": "Tier 2 rebuild",
                            "scan": "Stage 2 scan"}.get(run.get("stage"), run.get("stage") or "starting")
            done, total = run.get("tickers_done") or 0, run.get("tickers_total") or 0
            elapsed_str = "—"
            started = run.get("started_at")
            if started:
                try:
                    started_dt = datetime.fromisoformat(started)
                    elapsed_s = (datetime.now(timezone.utc) - started_dt).total_seconds()
                    elapsed_str = f"{int(elapsed_s // 60)}m {int(elapsed_s % 60)}s"
                except Exception:
                    pass
            st.info(f"⏳ Running — {stage_label}: {done}/{total} tickers done · elapsed {elapsed_str}")
            st.progress(min(done / total, 1.0) if total else 0.0)
            st.button("🔄 Refresh progress", key=_uk("vol_refresh_progress"))  # any click reruns the script
        elif run is not None and run["status"] == "failed":
            st.error(f"Last on-demand run failed: {run.get('error_message') or 'unknown error'}")
        elif run is not None and run["status"] == "done":
            st.caption(f"Last on-demand run finished {run.get('finished_at')} — "
                       f"core={run.get('core_count')} spike={run.get('spike_count')} "
                       f"scan stored={run.get('scan_stored')} skipped={run.get('scan_skipped')}")

    # --- as-of timestamp + provisional / post-close banner ---
    scan_rows_all = get_scan(conn)  # most recent scan_date on record, every period/tier
    if not scan_rows_all:
        st.info("No scan data yet. Click **▶️ Run scan now** above to build the universe and run the scan.")
        return

    scan_date = scan_rows_all[0]["scan_date"]
    captured_at = scan_rows_all[0]["captured_at"]
    gate_mode = scan_rows_all[0]["gate_mode"] or UNIVERSE_GATE_MODE
    st.markdown(f"**As of:** scan date `{scan_date}` · quotes captured **{captured_at or 'unknown'}**")

    is_post_close = False
    if captured_at:
        try:
            is_post_close = datetime.fromisoformat(captured_at).hour >= 16  # ET, market close
        except Exception:
            pass
    if gate_mode == "provisional" or is_post_close:
        st.warning(
            "⚠️ Provisional — quality gate pending Monday's timing test. "
            f"Scan used quotes from {captured_at or 'an unknown time'}; post-close quotes may be wider "
            "than intraday."
        )

    # --- controls ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        period_label = st.radio("Period", ["Week", "Month"], index=1, horizontal=True, key=_uk("vol_period"))
    with c2:
        tier_label = st.radio("Tier", ["All", "Core", "Spike"], horizontal=True, key=_uk("vol_tier"))
    with c3:
        hide_watchlist = st.checkbox("Hide names on my watchlist", value=True, key=_uk("vol_hide_wl"))
    with c4:
        hide_earnings = st.checkbox("Hide names with earnings in window", value=True, key=_uk("vol_hide_earn"))

    period_key = period_label.lower()
    rows = get_scan(conn, scan_date=scan_date, period=period_key)
    if tier_label != "All":
        rows = [r for r in rows if r["tier"] == tier_label.lower()]
    if hide_watchlist:
        wl_set = set(watchlist)
        rows = [r for r in rows if r["ticker"] not in wl_set]

    # de.now_et() (module-qualified), NOT a bare now_et import -- the RUNNERS tab above (tab9, untouched,
    # per the standing "leave Runners exactly as it is" constraint) reassigns a module-level `now_et`
    # local variable to a pd.Timestamp partway through this same script's sequential execution; a bare
    # `now_et` name here would resolve to THAT by the time this function runs, not the imported function.
    today_d = de.now_et().date()

    def _earnings_in_window(r):
        if not r.get("earnings_date"):
            return False
        try:
            ed = datetime.strptime(str(r["earnings_date"])[:10], "%Y-%m-%d").date()
        except Exception:
            return False
        # This row's OWN days_to_expiry (stored per-row) rather than a flat 7/30-day constant --
        # "inside the window" means before THIS specific expiry, and expiry day-counts now vary within
        # a period (Week 5-10d, Month 25-45d).
        window_days = r.get("days_to_expiry") or (10 if period_key == "week" else 45)
        return 0 <= (ed - today_d).days <= window_days

    # Always split (not just when hide_earnings is on) so the collapsed group below can show exactly what
    # the toggle would hide -- computed once, not silently dropped, per the 2026-09-25 follow-up: rows with
    # earnings inside the window are shown in a separate collapsed "Earnings in window" group, not just
    # discarded.
    earnings_hidden_rows = [r for r in rows if _earnings_in_window(r)]
    if hide_earnings:
        rows = [r for r in rows if r not in earnings_hidden_rows]

    if not rows:
        st.info("No names match the current filters for this period. A ticker with no scan row for this "
                "period (e.g. Week where has_weekly=false) is simply absent, not shown empty.")
        return

    # market_cap_bucket / a cached (never live-fetched) "current" price marker both come from the universe
    # table, joined in here by ticker -- `scan` itself doesn't carry market cap, and Stage 3 must not fetch.
    uni_by_ticker = {r["ticker"]: r for r in get_universe(conn)}

    # --- sort: IV-based (iv_rv_ratio), NOT raw implied_move_pct or move_vs_median_60d (2026-09-25 fix) ---
    # Raw implied move % isn't comparable across tickers even within one Period: Month's window alone spans
    # 25-45 days, so two names' move % differ by sqrt(45/25)=1.3x from calendar effects alone, before any
    # real difference in volatility. atm_iv_pct (annualized) and iv_rv_ratio (implied/realized, both
    # annualized) already divide that scaling out, so ranking uses iv_rv_ratio -- move_vs_median_60d stays
    # visible as an informational column but is no longer the sort key.
    rows = sorted(rows, key=lambda r: (r.get("iv_rv_ratio") is None, -(r.get("iv_rv_ratio") or 0)))
    st.caption("Sorted by IV/RV (implied vs. realized volatility), descending — raw implied move % isn't "
               "comparable across different expiry horizons, so it's shown but not ranked on.")

    def _band(r):
        ref, mv = r.get("reference_close"), r.get("implied_move_pct")
        if ref is None or mv is None or mv <= 0:
            return None, None
        return ref * (1 - mv / 100), ref * (1 + mv / 100)

    def _flags(r):
        out = []
        if r.get("move_vs_median_60d") is not None and r["move_vs_median_60d"] >= VOLATILITY_MOVE_DISLOCATION_FLAG:
            out.append("📈 MOVE")
        if (r.get("option_volume_vs_20d_avg") is not None
                and r["option_volume_vs_20d_avg"] >= VOLATILITY_UNUSUAL_OPTVOL_FLAG):
            out.append("🔥 VOL")
        return " ".join(out) if out else "—"

    def _table_rows(rowset):
        out = []
        for r in rowset:
            lower, upper = _band(r)
            bucket = (uni_by_ticker.get(r["ticker"]) or {}).get("market_cap_bucket")
            expiry_txt = "—"
            if r.get("expiry"):
                days_txt = f" ({r['days_to_expiry']}d)" if r.get("days_to_expiry") is not None else ""
                expiry_txt = f"{r['expiry']}{days_txt}"
            out.append({
                "Watchlist": r["ticker"] in watchlist,  # editable checkbox column -- the row-level action
                "Ticker": r["ticker"],
                "Tier": "🟢 Core" if r["tier"] == "core" else "🟠 Spike",
                "Price": f"${r['reference_close']:,.2f}" if r.get("reference_close") is not None else "—",
                "Expiry": expiry_txt,
                "IV %": f"{r['atm_iv_pct']:.1f}%" if r.get("atm_iv_pct") is not None else "—",
                "Implied Move %": f"{r['implied_move_pct']:.1f}%" if r.get("implied_move_pct") is not None else "—",
                "Band": f"${lower:,.2f}–${upper:,.2f}" if lower is not None else "—",
                "Implied ÷ Realized": (f"{r['iv_rv_ratio']:.1f}×" if r.get("iv_rv_ratio") is not None else "—"),
                "vs Own Median": (f"{r['move_vs_median_60d']:.2f}x" if r.get("move_vs_median_60d") is not None
                                  else "building"),
                "Opt Vol vs 20d": (f"{r['option_volume_vs_20d_avg']:.2f}x"
                                   if r.get("option_volume_vs_20d_avg") is not None else "building"),
                "Earnings": (str(r["earnings_date"])[:10] if r.get("earnings_date") else "—"),
                "Short Float": f"{r['short_float']*100:.1f}%" if r.get("short_float") is not None else "—",
                "Mkt Cap": bucket.capitalize() if bucket else "—",
                "Flags": _flags(r),
            })
        return pd.DataFrame(out)

    _NON_WATCHLIST_COLS = ["Ticker", "Tier", "Price", "Expiry", "IV %", "Implied Move %", "Band",
                            "Implied ÷ Realized", "vs Own Median", "Opt Vol vs 20d", "Earnings",
                            "Short Float", "Mkt Cap", "Flags"]

    def _render_table_with_watchlist_action(rowset, editor_key):
        """st.data_editor with only the "Watchlist" column editable -- checking a box for a ticker not
        already on the watchlist adds it (save_watchlist + rerun); every other column is read-only. This
        replaces the old separate 'Row Detail' expander section's "Add to watchlist" button -- the action
        now lives in the table row itself, per the 2026-09-25 follow-up. Unchecking an already-on-watchlist
        row's box does NOT remove it (this was never a remove action, only additive, same as the button it
        replaces)."""
        df = _table_rows(rowset)
        edited = st.data_editor(
            df, width='stretch', hide_index=True, key=editor_key,
            column_config={
                "Watchlist": st.column_config.CheckboxColumn("⭐ Watchlist", help="Check to add to your watchlist"),
                **{c: st.column_config.Column(disabled=True) for c in _NON_WATCHLIST_COLS},
            },
        )
        newly_checked = [
            t for t, was_on, is_on in zip(edited["Ticker"], df["Watchlist"], edited["Watchlist"])
            if is_on and not was_on
        ]
        if newly_checked:
            save_watchlist(get_conn(), EFFECTIVE_EMAIL, sorted(set(watchlist) | set(newly_checked)))
            st.rerun()

    top_rows = rows[:25]
    _render_table_with_watchlist_action(top_rows, _uk(f"vol_table_top_{period_key}_{tier_label}"))
    if len(rows) > 25:
        with st.expander(f"Show all {len(rows)} names"):
            _render_table_with_watchlist_action(rows, _uk(f"vol_table_all_{period_key}_{tier_label}"))

    st.caption(
        "An ATM straddle implies roughly a one-standard-deviation move; price finishes inside the band "
        "about two times in three — it is not a forecast. Where earnings fall inside the window, event "
        "premium inflates the band and it isn't comparable to a non-earnings period."
    )

    # Separate collapsed group (2026-09-25 follow-up) -- rows the "Hide earnings in window" toggle removed
    # from the main list above are NOT just discarded; they're shown here so it's checkable, not a black box.
    if hide_earnings and earnings_hidden_rows:
        with st.expander(f"📅 Earnings in window ({len(earnings_hidden_rows)}) — hidden from the list above",
                          expanded=False):
            sorted_hidden = sorted(earnings_hidden_rows,
                                    key=lambda r: (r.get("iv_rv_ratio") is None, -(r.get("iv_rv_ratio") or 0)))
            _render_table_with_watchlist_action(sorted_hidden, _uk(f"vol_table_earn_{period_key}_{tier_label}"))


with tab15:
    render_volatility_dislocations_tab()


# --------------------------------------------------------------------------
# Tab 16 -- ⚙️ ADMIN (role='admin' only; the tab does not exist otherwise)
# --------------------------------------------------------------------------

def _set_view_as(email):
    """Button callback: switch the admin's data context (runs before the
    next script pass, so it may set widget-bound keys)."""
    st.session_state["view_as_email"] = None if email == ACTOR_EMAIL else email
    st.session_state["_admin_view_as_select"] = email


def render_admin_users_section():
    conn = get_conn()
    users = list_users(conn)
    st.markdown('<div class="smd-section">USERS</div>', unsafe_allow_html=True)
    _flash = st.session_state.pop("_admin_flash", None)
    if _flash:
        st.success(_flash)
    st.caption("Only accounts listed here can sign in. A valid Google login for any other account is refused.")
    st.dataframe(pd.DataFrame([{
        "Email": u["email"], "Name": u["display_name"] or "", "Role": u["role"],
        "Created": fmt_date(u["created_at"]) if u["created_at"] else "—",
        "Last login": fmt_date(u["last_login_at"]) if u["last_login_at"] else "never",
        "Active": "✅" if u["active"] else "⛔ deactivated",
    } for u in users]), width='stretch', hide_index=True)

    st.markdown("**Add user**")
    with st.form("admin_add_user_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([3, 2, 1.2])
        new_email = c1.text_input("Google account email", placeholder="person@gmail.com")
        new_name = c2.text_input("Display name")
        new_role = c3.selectbox("Role", USER_ROLES)
        if st.form_submit_button("➕ Add user"):
            try:
                created = de.create_user(conn, new_email, new_name, new_role, created_by=ACTOR_EMAIL)
            except ValueError as e:
                st.error(str(e))
            else:
                # Rerun so the tables/rows above (built before this form was
                # processed) include the new user immediately.
                st.session_state["_admin_flash"] = (
                    f"Added {created['email']} as {created['role']} with a starter watchlist. "
                    f"Use 'View as' to set up their watchlist."
                )
                st.rerun()

    st.markdown("**Manage users**")
    for u in users:
        email = u["email"]
        is_self = email == ACTOR_EMAIL
        c1, c2, c3, c4, c5 = st.columns([3, 1.6, 1.4, 1.4, 1.6])
        c1.markdown(f"**{html.escape(email)}**{' (you)' if is_self else ''}"
                    f"{'' if u['active'] else ' — <span style=\"color:#e66767;\">deactivated</span>'}",
                    unsafe_allow_html=True)
        new_role = c2.selectbox("Role", USER_ROLES, index=USER_ROLES.index(u["role"]), key=f"admin_role_{email}",
                                label_visibility="collapsed", disabled=is_self)
        if new_role != u["role"] and c2.button("Apply role", key=f"admin_role_apply_{email}"):
            try:
                de.set_user_role(conn, email, new_role)
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if u["active"]:
            if c3.button("Deactivate", key=f"admin_deact_{email}", disabled=is_self):
                try:
                    de.set_user_active(conn, email, False)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        else:
            if c3.button("Reactivate", key=f"admin_react_{email}"):
                de.set_user_active(conn, email, True)
                st.rerun()
        c4.button("👁️ View as", key=f"admin_viewas_{email}", on_click=_set_view_as, args=(email,),
                  disabled=(email == EFFECTIVE_EMAIL))
        with c5:
            confirm = st.checkbox("confirm delete", key=f"admin_del_confirm_{email}", disabled=is_self)
            if st.button("🗑️ Delete", key=f"admin_del_{email}", disabled=is_self or not confirm):
                try:
                    de.delete_user(conn, email)
                    if st.session_state.get("view_as_email") == email:
                        st.session_state["view_as_email"] = None
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
    st.caption("Deleting a user permanently removes their watchlist, entry zones, preferences and recipients. "
               "Shared market data is untouched. You can't change, deactivate or delete your own account, "
               "and the last active admin is always protected.")


def render_admin_overview_section():
    conn = get_conn()
    st.markdown('<div class="smd-section">ALL USERS OVERVIEW</div>', unsafe_allow_html=True)
    users = list_users(conn)
    summary = []
    for u in users:
        tickers = de.get_saved_watchlist(conn, u["email"])
        zones = get_entry_zones(conn, u["email"])
        cfg = get_email_config(conn, u["email"])
        summary.append({
            "User": u["email"], "Tickers": len(tickers), "Watchlist": ", ".join(tickers) or "—",
            "Zones": sum(1 for z in zones if z["entry_low"] is not None),
            "In zone": sum(1 for z in zones if z["last_zone_status"] == "in_zone"),
            "Digest": ("on" if cfg["enabled"] else "off") + f" · {'weekly' if cfg['frequency'] == 'weekly' else 'daily Mon–Fri'} · {cfg['send_time_et']} ET",
            "Recipients": sum(1 for r in get_all_subscribers(conn, u["email"]) if not r["unsubscribed"]),
        })
    st.dataframe(pd.DataFrame(summary), width='stretch', hide_index=True)
    for u in users:
        with st.expander(f"{u['email']} — entry zones"):
            zones = get_entry_zones(conn, u["email"])
            if not zones:
                st.caption("No entry zones computed yet.")
                continue
            st.dataframe(pd.DataFrame([{
                "Ticker": z["ticker"],
                "Entry zone": f"${z['entry_low']:.2f}-${z['entry_high']:.2f}" if z["entry_low"] is not None else "—",
                "Status": (z["last_zone_status"] or "—").replace("_", " "),
                "Target": f"${z['implied_target_price']:.2f}" if z["implied_target_price"] is not None else "—",
            } for z in zones]), width='stretch', hide_index=True)


def render_admin_tab():
    """ADMIN console. The `ACTOR` here is always a verified admin -- this is
    only called when IS_ADMIN, and IS_ADMIN is re-derived from the DB on
    every script run."""
    if not IS_ADMIN:  # belt-and-braces: never render admin controls for a non-admin
        st.error("Admins only.")
        return
    if EFFECTIVE_EMAIL != ACTOR_EMAIL:
        c1, c2 = st.columns([4, 1])
        c1.info(f"Currently viewing as **{EFFECTIVE_EMAIL}** (their SETTINGS, WATCHLIST SIGNALS and RUNNERS "
                f"now show and edit that user's data).", icon="👁️")
        c2.button("Back to my view", key="admin_back_to_me", on_click=_set_view_as, args=(ACTOR_EMAIL,))
    st.caption("Source registry, news feeds, SMTP/API keys, firm tiers, reconciliation and backtests are in the "
               "SETTINGS tab, below the personal settings.")
    sub_users, sub_overview = st.tabs(["Users", "All users overview"])
    with sub_users:
        render_admin_users_section()
    with sub_overview:
        render_admin_overview_section()


if IS_ADMIN:
    with tab_admin:
        render_admin_tab()
