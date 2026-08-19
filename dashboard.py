"""Streamlit UI for the Smart Money Intelligence Dashboard."""

import html
import json
import os
import re
import time
from contextlib import nullcontext
from datetime import date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
)

# --------------------------------------------------------------------------
# Theme / palette
# --------------------------------------------------------------------------

BG = "#030609"
PANEL = "#080d14"
PANEL_ALT = "#0a121b"
BORDER = "rgba(0,229,255,0.14)"
ACCENT = "#00e5ff"
TEXT_PRIMARY = "#e8f4f8"
TEXT_SECONDARY = "#8b9bab"
TEXT_MUTED = "#4d5866"
GRID = "#12202b"

COLOR_INSTITUTIONAL = "#3987e5"   # blue
COLOR_RETAIL = "#c98500"          # amber
COLOR_BULLISH = "#0ca30c"         # good / green
COLOR_BEARISH = "#e66767"         # critical / red
COLOR_NEUTRAL = "#5a6472"         # muted gray

LABEL_COLORS = {
    "SMART_BULLISH": COLOR_BULLISH,
    "SMART_BEARISH_RETAIL_LONG": COLOR_BEARISH,
    "RETAIL_FRENZY": COLOR_RETAIL,
    "INSTITUTIONAL_ACTIVE": COLOR_INSTITUTIONAL,
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
    max-width: 1400px;
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

@st.cache_resource
def get_conn():
    init_db(DEFAULT_DB_PATH)
    return get_connection(DEFAULT_DB_PATH)


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
    (the plain-English 'why' from a classify_*() call) via the title attr."""
    if not label:
        return ""
    title_attr = f' title="{html.escape(note)}"' if note else ""
    return f'<span class="chip" style="border-color:{color};color:{color};"{title_attr}>{html.escape(label)}</span>'


# --------------------------------------------------------------------------
# Chart helpers
# --------------------------------------------------------------------------

BASE_LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(family="Space Mono, monospace", color=TEXT_SECONDARY, size=12),
    margin=dict(l=45, r=20, t=42, b=30),
)


def apply_theme(fig, height=360, **overrides):
    layout = dict(BASE_LAYOUT)
    layout["height"] = height
    layout.update(overrides)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#1c2a36")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor="#1c2a36")
    return fig


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
    apply_theme(
        fig, height=height,
        title=dict(text=f"{ticker} &mdash; Price", font=dict(color=TEXT_PRIMARY, size=14)),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def render_price_volume_chart(df, ticker, mode="line", short_label="SMA short", long_label="SMA long",
                               height=460, levels=None):
    """Price chart (line or candlestick) with a volume subplot beneath it
    and SMA_short/SMA_long overlays -- the timeframe-aware companion to
    render_price_chart, used wherever fetch_price_history() supplies the
    data (its columns are named SMA_short/SMA_long, not SMA50/SMA200).

    `levels`, if given a detect_technical_levels()-shaped dict, overlays
    dashed horizontal lines for every detected support/resistance level --
    the nearest (key_level_below/key_level_above) drawn bold and labeled,
    the rest faint so the chart stays readable when several levels cluster
    close together."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)

    if mode == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                increasing_line_color=COLOR_BULLISH, decreasing_line_color=COLOR_BEARISH,
                increasing_fillcolor=COLOR_BULLISH, decreasing_fillcolor=COLOR_BEARISH,
                name=ticker,
            ), row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close",
                       line=dict(color=ACCENT, width=2)), row=1, col=1,
        )
    if "SMA_short" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_short"], mode="lines", name=short_label,
                       line=dict(color=COLOR_INSTITUTIONAL, width=1.4, dash="dot")), row=1, col=1,
        )
    if "SMA_long" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["SMA_long"], mode="lines", name=long_label,
                       line=dict(color=COLOR_RETAIL, width=1.4, dash="dot")), row=1, col=1,
        )

    vol_colors = [COLOR_BULLISH if c >= o else COLOR_BEARISH for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)

    if levels:
        # Passing annotation_text=None alongside other annotation_* kwargs
        # still makes plotly build an annotation object (just with no
        # `text` key) -- and plotly.js's own default for a missing
        # annotation text is the literal placeholder string "New text",
        # which is exactly the stray label this used to show for every
        # non-key level. Fix: only pass annotation_* kwargs at all when
        # there's real text to show.
        key_below = levels.get("key_level_below")
        key_above = levels.get("key_level_above")
        for lv in levels.get("support_levels", []):
            is_key = lv["level"] == key_below
            hline_kwargs = dict(
                y=lv["level"], row=1, col=1,
                line=dict(color=COLOR_BULLISH, width=1.8 if is_key else 1, dash="dash"),
                opacity=0.95 if is_key else 0.4,
            )
            if is_key:
                hline_kwargs.update(
                    annotation_text=f"S {lv['level']:,.2f}", annotation_position="bottom right",
                    annotation_font_color=COLOR_BULLISH, annotation_font_size=10,
                )
            fig.add_hline(**hline_kwargs)
        for lv in levels.get("resistance_levels", []):
            is_key = lv["level"] == key_above
            hline_kwargs = dict(
                y=lv["level"], row=1, col=1,
                line=dict(color=COLOR_BEARISH, width=1.8 if is_key else 1, dash="dash"),
                opacity=0.95 if is_key else 0.4,
            )
            if is_key:
                hline_kwargs.update(
                    annotation_text=f"R {lv['level']:,.2f}", annotation_position="top right",
                    annotation_font_color=COLOR_BEARISH, annotation_font_size=10,
                )
            fig.add_hline(**hline_kwargs)

    # TradingView-style continuous candles: without this, plotly's default
    # real-time x-axis renders closed-market time (nights, weekends) as
    # visible dead space between sessions -- gaps between trading days on
    # daily+ charts, and a gap every night on intraday charts. rangebreaks
    # tells plotly to compress those closed periods out entirely so
    # consecutive trading bars sit adjacent with no gap. Detected from the
    # data itself (more than one distinct time-of-day present) rather than
    # a separate parameter, since a daily-bar df has no intraday gaps to
    # hide and applying the hour-pattern break to it would be a no-op at
    # best and wrong if it ever isn't.
    is_intraday = pd.Series(df.index).dt.time.nunique() > 1
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if is_intraday:
        # US market hours (NYSE/NASDAQ, all tickers this app covers are
        # US-listed): 9:30am-4:00pm local exchange time. bounds=[16, 9.5]
        # hides 16:00->24:00 and 0:00->9:30, i.e. everything outside that
        # window -- plotly's own documented pattern for this exact case.
        rangebreaks.append(dict(bounds=[16, 9.5], pattern="hour"))
    fig.update_xaxes(rangebreaks=rangebreaks)

    fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    apply_theme(
        fig, height=height,
        title=dict(text=f"{ticker} &mdash; Price &amp; Volume", font=dict(color=TEXT_PRIMARY, size=14)),
        legend=dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


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
        st.markdown(metric_chip_html(
            "KEY SUPPORT BELOW", f"${below:,.2f}" if below is not None else "—", color=COLOR_BULLISH,
        ), unsafe_allow_html=True)
        if below is not None and levels.get("breakdown_target") is not None:
            st.caption(f"Break below {below:,.2f} → next measured-move target ~{levels['breakdown_target']:,.2f}")
    with kcol2:
        above = levels.get("key_level_above")
        st.markdown(metric_chip_html(
            "KEY RESISTANCE ABOVE", f"${above:,.2f}" if above is not None else "—", color=COLOR_BEARISH,
        ), unsafe_allow_html=True)
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
    interval_key = f"{key_prefix}_interval"
    lookback_key = f"{key_prefix}_lookback"

    def _on_interval_change():
        st.session_state[lookback_key] = _INTERVAL_DEFAULT_LOOKBACK.get(
            st.session_state.get(interval_key), DEFAULT_LOOKBACK,
        )

    tcol1, tcol2, tcol3 = st.columns([1.2, 1.2, 1])
    with tcol1:
        interval = st.selectbox(
            "Interval", INTERVAL_OPTIONS, index=INTERVAL_OPTIONS.index(DEFAULT_INTERVAL),
            key=interval_key, on_change=_on_interval_change,
        )
    with tcol2:
        lookback = st.selectbox(
            "Lookback", LOOKBACK_OPTIONS, index=LOOKBACK_OPTIONS.index(DEFAULT_LOOKBACK),
            key=lookback_key,
        )
    with tcol3:
        refresh_choice = st.selectbox(
            "Refresh", ["On demand", "1 min", "5 min"], key=f"{key_prefix}_refresh",
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
        st.plotly_chart(
            render_price_volume_chart(
                hist, ticker, mode=mode,
                short_label=f"SMA {short_w}", long_label=f"SMA {long_w}", levels=levels,
            ),
            width='stretch', key=f"{key_prefix}_pv_{cache_suffix}",
        )
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


def render_price_target_fan(hist_df, current, low, mean, high, height=380):
    """Historical close price flowing into a forward-projecting fan toward
    High/Average/Low analyst targets -- replaces the old static low..high
    number-line entirely, redesigned directly against real Yahoo Finance
    and TradingView price-target chart references (not a guess at their
    style): a real price history line leads into three dashed projection
    lines from the current price to each target, each ending in a
    colored value+percent label at the right edge.

    Median isn't drawn as a 4th fan line -- both reference charts use
    exactly High/Average/Low, and a 4th line sitting close to Average
    would just crowd the labels without adding a materially different
    read -- but it isn't silently dropped either: the caller shows it in
    the caption text above this chart."""
    fig = go.Figure()
    if hist_df is None or hist_df.empty or current is None:
        apply_theme(fig, height=height)
        return fig

    hist = hist_df["Close"].dropna()
    if hist.empty:
        apply_theme(fig, height=height)
        return fig

    last_close = float(hist.iloc[-1])
    # Plotly's annotation/axis-range serialization chokes on raw pandas
    # Timestamps ("Type is not JSON serializable: Timestamp") even though
    # trace x= arrays accept them fine -- ISO date strings are safe
    # everywhere, so every date used outside a trace's own x= array goes
    # through this conversion.
    last_date_str = hist.index[-1].strftime("%Y-%m-%d")
    forecast_end_str = (hist.index[-1] + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    hist_start_str = hist.index[0].strftime("%Y-%m-%d")

    hist_x = hist.index.strftime("%Y-%m-%d").tolist()
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist.values, mode="lines", name="Historical",
        line=dict(color=ACCENT, width=2), fill="tozeroy", fillcolor="rgba(0,229,255,0.07)",
        hovertemplate="%{x|%b %d, %Y}: $%{y:,.2f}<extra></extra>", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[last_date_str], y=[last_close], mode="markers", showlegend=False,
        marker=dict(color=TEXT_PRIMARY, size=8, line=dict(color=PANEL, width=1.5)),
        hovertemplate=f"Current: ${last_close:,.2f}<extra></extra>",
    ))
    # Full-width reference line so "current" reads as a constant benchmark
    # against the whole chart, not just a single pivot dot at the
    # historical/forecast boundary -- makes it immediate to see where each
    # fan line sits relative to today's price at a glance, including back
    # across the historical portion.
    fig.add_hline(y=last_close, line=dict(color=TEXT_PRIMARY, width=1, dash="dash"), opacity=0.35)

    fan = [("High", high, COLOR_BULLISH), ("Average", mean, ACCENT), ("Low", low, COLOR_BEARISH)]
    for name, val, color in fan:
        if val is None or pd.isna(val):
            continue
        val = float(val)
        pct = (val - last_close) / last_close * 100 if last_close else None
        fig.add_trace(go.Scatter(
            x=[last_date_str, forecast_end_str], y=[last_close, val], mode="lines",
            line=dict(color=color, width=1.6, dash="dot"), showlegend=False, hoverinfo="skip",
        ))
        # Whole-number percent (not one decimal) specifically to keep this
        # short enough not to clip against the right edge -- confirmed by
        # rendering and looking at the actual output: "High $147.39 +139.1%"
        # was wide enough to get cut off even with a wide right margin.
        pct_txt = f" {pct:+.0f}%" if pct is not None else ""
        fig.add_annotation(
            x=forecast_end_str, y=val, xref="x", yref="y", xanchor="left", yanchor="middle", xshift=8,
            text=f"<b>{name}</b> ${val:,.2f}{pct_txt}", showarrow=False,
            font=dict(color=PANEL, size=10), bgcolor=color, bordercolor=color, borderpad=4,
        )

    fig.update_xaxes(range=[hist_start_str, forecast_end_str], showgrid=False)
    fig.update_yaxes(title=dict(text="Price ($)", font=dict(color=TEXT_MUTED)))
    apply_theme(fig, height=height, margin=dict(l=45, r=150, t=15, b=35), showlegend=False,
                hovermode="x unified")
    return fig


def render_recommendation_breakdown(counts, height=110):
    """Strong Buy -> Strong Sell is an ordered diverging scale, not a set of
    independent categories, so this uses opacity steps of the two validated
    bullish/bearish poles (plus the validated neutral gray for Hold) rather
    than picking new unvalidated hues for Buy/Sell."""
    order = [
        ("strongBuy", "Strong Buy", COLOR_BULLISH),
        ("buy", "Buy", "rgba(12,163,12,0.55)"),
        ("hold", "Hold", COLOR_NEUTRAL),
        ("sell", "Sell", "rgba(230,103,103,0.55)"),
        ("strongSell", "Strong Sell", COLOR_BEARISH),
    ]
    total = sum((counts.get(k) or 0) for k, _, _ in order)

    fig = go.Figure()
    if total == 0:
        apply_theme(fig, height=height)
        return fig

    for key, label, color in order:
        n = counts.get(key) or 0
        fig.add_trace(go.Bar(
            x=[n], y=["Analysts"], orientation="h", name=f"{label} ({n})",
            marker_color=color, text=[str(n) if n else ""], textposition="inside",
            insidetextfont=dict(color=PANEL if key in ("strongBuy", "strongSell") else TEXT_PRIMARY, size=11),
            hovertemplate=f"{label}: {n}<extra></extra>",
        ))

    fig.update_layout(barmode="stack")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    apply_theme(fig, height=height, margin=dict(l=10, r=10, t=6, b=6),
                legend=dict(orientation="h", y=-0.45, bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    return fig


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


def render_settings_tab():
    """SETTINGS tab: full source registry (more room than the sidebar ever
    had), watchlist configuration, and a manual recheck-all button. Both
    System Status and Configure Watchlist used to live in the sidebar --
    moved here so all configuration lives in one place."""
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

    render_news_feeds_section()

    st.markdown('<div class="smd-section">WATCHLIST</div>', unsafe_allow_html=True)
    st.caption(
        "Comma-separated tickers. Saved to watchlist.json — persists across restarts "
        "instead of resetting to a hardcoded list."
    )
    wcol1, wcol2 = st.columns([3, 1])
    with wcol1:
        wl_text = st.text_area(
            "Tickers", value=", ".join(st.session_state.watchlist),
            label_visibility="collapsed", height=80, key="wl_textarea",
        )
    with wcol2:
        if st.button("Save", width='stretch', key="wl_save"):
            parsed = [t.strip().upper() for t in wl_text.split(",") if t.strip()]
            st.session_state.watchlist = save_watchlist(parsed or DEFAULT_STARTER_WATCHLIST)
            st.rerun()
        if st.button("Reset to default", width='stretch', key="wl_reset"):
            st.session_state.watchlist = save_watchlist(DEFAULT_STARTER_WATCHLIST)
            st.rerun()

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

            st.markdown(f"""
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
            """, unsafe_allow_html=True)

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


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

watchlist = st.session_state.watchlist

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
    st.session_state["last_auto_reconcile_date"] = _today_str

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
    [
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
    ]
)

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
        st.plotly_chart(render_price_chart(f["price_df"].tail(120), sel, mode="candlestick"),
                        width='stretch', key=f"opt_price_chart_{sel}")

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
    st.write(
        f"This will make 1 call to the Claude API using cached dashboard data for "
        f"**{ticker}** (no new external fetches). Estimated cost: ~${cost['cost_usd']:.2f}. Continue?"
    )
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
            fig = render_day_prediction_chart(
                ticker, last, intraday_hist if not intraday_hist.empty else None,
                track.get("mean_abs_error_pct"), track.get("n"),
            )
            st.plotly_chart(fig, width='stretch', key=f"last_session_chart_{ticker}_{last['session_date']}")
        except RuntimeError as e:
            st.caption(f"⛔ Could not render last session's chart: {e}")
        render_prediction_comparison_table(ticker, conn, last, last["session_date"], key_prefix="lastsession")
        if last["reconciled_at"] and last["actual_close_price"] is not None:
            outcome = (
                "✅ direction call was correct" if last["prediction_correct_direction"]
                else "❌ direction call was wrong"
            )
            st.caption(
                f"Closed at ${last['actual_close_price']:.2f} vs. predicted ${last['target_price']:.2f} "
                f"({last['error_pct']:+.2f}% error) — {outcome}."
            )
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
        st.caption(
            f"(Backtest) Closed at ${bt_last['actual_close_price']:.2f} vs. predicted "
            f"${bt_last['target_price']:.2f} ({bt_last['error_pct']:+.2f}% error) — {outcome}."
        )


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
                    st.markdown(f"""
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
                    """, unsafe_allow_html=True)

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

    with st.spinner("Pulling news across the watchlist...") if force_news else nullcontext():
        for t in watchlist:
            cached_news(conn_news, t, limit=8, max_age_hours=2, force_refresh=force_news)
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
    fig.add_vrect(
        x0=model_start_time, x1=market_close, fillcolor="rgba(0,229,255,0.06)", line_width=0,
        annotation_text=f"MODEL ACTIVE — {error_txt}", annotation_position="top left",
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
    return apply_theme(
        fig, height=height, margin=dict(l=55, r=175, t=75, b=60),
        legend=dict(orientation="h", y=-0.2, bgcolor="rgba(0,0,0,0)"),
    )


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
        fig = render_day_prediction_chart(
            ticker, committed, prediction.get("intraday_bars"), prediction.get("backtest_error_pct"),
            prediction.get("backtest_n"),
        )
    except RuntimeError as e:
        st.error(f"⛔ **{ticker}** — {e}")
        return
    st.plotly_chart(fig, width='stretch', key=f"day_pred_chart_{ticker}_{prediction['session_date']}")

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

    runner_rows = []
    for t in watchlist:
        f_env_r = cached_fundamentals(get_conn(), t, max_age_hours=12)
        pdf = f_env_r["data"]["price_df"]
        if pdf.empty or len(pdf) < 21:
            continue
        latest = pdf.iloc[-1]
        avg_vol_20d = pdf["Volume"].tail(21).iloc[:-1].mean()
        vol_ratio = (latest["Volume"] / avg_vol_20d) if avg_vol_20d else None
        chg_pct = float((pdf["Close"].iloc[-1] / pdf["Close"].iloc[-2] - 1) * 100) if len(pdf) > 1 else None
        unusual_vol = vol_ratio is not None and vol_ratio >= 2.0
        big_move = chg_pct is not None and abs(chg_pct) >= 3.0
        if not (unusual_vol or big_move):
            continue
        unusualness = (vol_ratio or 0) + abs(chg_pct or 0) / 3.0
        runner_rows.append({
            "Ticker": t, "Price": latest["Close"], "Chg %": chg_pct,
            "Volume": int(latest["Volume"]), "Vol vs 20D avg": vol_ratio,
            "_unusualness": unusualness,
        })

    if not runner_rows:
        st.caption("No unusual volume or price moves detected in the current watchlist right now.")
        runner_tickers = []
    else:
        # RUNNERS means the day's top movers, not the full watchlist --
        # capped at 4, sorted most-unusual-first, same ranking as before.
        runners_df_full = pd.DataFrame(runner_rows).sort_values("_unusualness", ascending=False).drop(
            columns="_unusualness"
        )
        runners_df = runners_df_full.head(4)
        runner_tickers = runners_df["Ticker"].tolist()
        styled = runners_df.copy()
        styled["Price"] = styled["Price"].map(lambda v: f"${v:.2f}")
        styled["Chg %"] = styled["Chg %"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        styled["Volume"] = styled["Volume"].map(lambda v: f"{v:,}")
        styled["Vol vs 20D avg"] = styled["Vol vs 20D avg"].map(lambda v: f"{v:.1f}x" if pd.notna(v) else "—")
        st.caption(
            f"Top {len(runners_df)} of {len(runners_df_full)} flagged watchlist ticker(s) shown (RUNNERS caps at "
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
    if "pinned_runners" not in st.session_state:
        st.session_state.pinned_runners = []
    st.session_state.pinned_runners = [t for t in st.session_state.pinned_runners if t in watchlist]

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
            options=["—"] + addable, index=0, key="runners_manual_add_select",
        )
        acol2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if acol2.button("➕ Add", key="runners_manual_add_btn", disabled=(ticker_to_add == "—"),
                         width='stretch'):
            st.session_state.pinned_runners.append(ticker_to_add)
            st.rerun()

    if display_tickers:
        rcol1, rcol2 = st.columns([1, 3])
        refresh_choice = rcol1.selectbox(
            "Refresh interval", ["On demand", "5 min", "15 min", "1 hour"], index=0,
            key="runners_refresh_interval",
            help="'On demand' (default, matching the rest of this dashboard) only re-evaluates on your next "
                 "manual interaction. The timed options auto-refresh this section during market hours.",
        )
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
            for rt in tickers:
                hcol1, hcol2 = st.columns([5, 1])
                hcol1.markdown(f"#### {rt}" + ("  📌 manually added" if rt in pinned else ""))
                if rt in pinned:
                    if hcol2.button("✕ Remove", key=f"runners_remove_pin_{rt}"):
                        st.session_state.pinned_runners = [p for p in st.session_state.pinned_runners if p != rt]
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
