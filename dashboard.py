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
    _compute_technical_levels,
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
            use_container_width=True, key=f"{key_prefix}_pv_{cache_suffix}",
        )
        horizon_label = "Intraday setup" if cfg["horizon"] == "intraday" else "Swing setup"
        st.caption(horizon_label)
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.plotly_chart(render_rsi_chart(hist), use_container_width=True,
                            key=f"{key_prefix}_rsi_{cache_suffix}")
        with rcol2:
            st.plotly_chart(render_macd_chart(hist), use_container_width=True,
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


def render_target_range(current, low, mean, median, high, height=200):
    """Horizontal low/mean/median/high/current range. Labels that would
    collide (any two values within ~8% of the overall span) are staggered
    onto separate vertical tiers with a thin connector back to their true
    marker position, so labels never overlap no matter how close two
    values sit -- e.g. current price landing right on the median."""
    points = [
        ("Low", low, COLOR_BEARISH), ("Median", median, COLOR_INSTITUTIONAL),
        ("Mean", mean, ACCENT), ("High", high, COLOR_BULLISH), ("Current", current, TEXT_PRIMARY),
    ]
    points = [(n, float(x), c) for n, x, c in points if x is not None and pd.notna(x)]

    fig = go.Figure()
    if not points:
        apply_theme(fig, height=height)
        return fig

    xs = [x for _, x, _ in points]
    x_span = (max(xs) - min(xs)) or max(abs(x) for x in xs) or 1.0
    collision_threshold = x_span * 0.08

    points_by_x = sorted(points, key=lambda p: p[1])
    last_x_at_level = {}
    level_by_name = {}
    for name, x, _ in points_by_x:
        level = 0
        while level in last_x_at_level and (x - last_x_at_level[level]) < collision_threshold:
            level += 1
        last_x_at_level[level] = x
        level_by_name[name] = level
    max_level = max(level_by_name.values())

    low_val = next((x for n, x, _ in points if n == "Low"), None)
    high_val = next((x for n, x, _ in points if n == "High"), None)
    if low_val is not None and high_val is not None:
        fig.add_trace(go.Scatter(x=[low_val, high_val], y=[0, 0], mode="lines",
                                  line=dict(color=TEXT_MUTED, width=6), hoverinfo="skip", showlegend=False))

    for name, x, color in points:
        level = level_by_name[name]
        label_y = 0.35 + level * 0.55

        fig.add_trace(go.Scatter(
            x=[x], y=[0], mode="markers", showlegend=False,
            marker=dict(color=color, size=15 if name == "Current" else 11,
                        symbol="diamond" if name == "Current" else "circle",
                        line=dict(color=PANEL, width=1)),
            hovertemplate=f"{name}: $%{{x:,.2f}}<extra></extra>",
        ))
        if level > 0:
            fig.add_trace(go.Scatter(
                x=[x, x], y=[0.1, label_y - 0.18], mode="lines",
                line=dict(color=color, width=1, dash="dot"), showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[x], y=[label_y], mode="text", showlegend=False, hoverinfo="skip",
            text=[f"{name}<br>${x:,.0f}"], textfont=dict(color=color, size=10),
        ))

    fig.update_yaxes(visible=False, range=[-0.6, 0.9 + max_level * 0.55])
    fig.update_xaxes(title=dict(text="Price target ($)", font=dict(color=TEXT_MUTED)))
    apply_theme(fig, height=height, margin=dict(l=30, r=30, t=20, b=35), showlegend=False)
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


# --------------------------------------------------------------------------
# SYSTEM STATUS -- rendered at the very top of the page, above everything
# else. Compact one-row badge strip always visible; a collapsible expander
# underneath carries latency/error/fix-hint detail. Health is probed once
# per session (see get_health) so this never re-hits every source on an
# unrelated rerun -- only the "Recheck" button forces a fresh probe.
# --------------------------------------------------------------------------

_STATUS_ORDER = ["options_flow", "dark_pool", "congressional", "news", "13f"]


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
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)
    st.caption(f"Checked at {registry.get('checked_at', '—')}")

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
        if st.button("Save", use_container_width=True, key="wl_save"):
            parsed = [t.strip().upper() for t in wl_text.split(",") if t.strip()]
            st.session_state.watchlist = save_watchlist(parsed or DEFAULT_STARTER_WATCHLIST)
            st.rerun()
        if st.button("Reset to default", use_container_width=True, key="wl_reset"):
            st.session_state.watchlist = save_watchlist(DEFAULT_STARTER_WATCHLIST)
            st.rerun()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

watchlist = st.session_state.watchlist

st.sidebar.markdown(
    f'<div style="color:{ACCENT};font-size:16px;font-weight:700;letter-spacing:0.05em;">CONTROL PANEL</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("⟳  REFRESH DATA", use_container_width=True):
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
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
        st.plotly_chart(render_divergence_map(div_df, watchlist), use_container_width=True, key="divergence_map")

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
                        use_container_width=True, key=f"opt_price_chart_{sel}")

    latest_fetch_row = q(
        "SELECT MAX(fetch_date) AS d FROM options_flow WHERE ticker=?", (sel,)
    )
    latest_fetch_date = latest_fetch_row["d"].iloc[0] if not latest_fetch_row.empty else None

    exp_options_row = q(
        """SELECT DISTINCT expiration FROM options_flow WHERE ticker=? AND fetch_date=?
           ORDER BY expiration""",
        (sel, latest_fetch_date),
    ) if latest_fetch_date else pd.DataFrame()
    expiry_choices = ["All (unusual activity, near expirations)"] + exp_options_row["expiration"].tolist()
    expiry_sel = st.selectbox("Expiration", expiry_choices, key=f"opt_expiry_{sel}")

    if expiry_sel == expiry_choices[0]:
        st.markdown('<div class="smd-section">UNUSUAL OPTIONS ACTIVITY (TODAY)</div>', unsafe_allow_html=True)
        opt_raw = q(
            """SELECT option_type, strike, expiration, volume, open_interest, volume_oi_ratio,
                      implied_volatility, last_price
               FROM options_flow WHERE ticker=? AND fetch_date=? AND unusual=1
               ORDER BY volume DESC LIMIT 40""",
            (sel, latest_fetch_date or date.today().isoformat()),
        )
    else:
        st.markdown(
            f'<div class="smd-section">FULL CHAIN &mdash; {expiry_sel}</div>', unsafe_allow_html=True
        )
        opt_raw = q(
            """SELECT option_type, strike, expiration, volume, open_interest, volume_oi_ratio,
                      implied_volatility, last_price
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

        st.plotly_chart(render_call_put_split(call_vol, put_vol), use_container_width=True,
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

        with st.expander("Raw table", expanded=(expiry_sel == expiry_choices[0])):
            opt_df = opt_raw.rename(columns={
                "option_type": "Type", "strike": "Strike", "expiration": "Expiry", "volume": "Volume",
                "open_interest": "OI", "volume_oi_ratio": "Vol/OI", "implied_volatility": "IV %",
                "last_price": "Last",
            }).copy()
            opt_df["Vol/OI"] = opt_df["Vol/OI"].round(2)
            opt_df["IV %"] = (opt_df["IV %"] * 100).round(1)
            st.dataframe(opt_df, use_container_width=True, hide_index=True)

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
        st.dataframe(congress_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="smd-section">SEC FORM 4 &mdash; INSIDER TRADES</div>', unsafe_allow_html=True)
    insider_df = q(
        """SELECT transaction_date AS Date, ticker AS Ticker, insider_name AS Insider, title AS Title,
                  transaction_type AS Type, shares AS Shares, price AS Price, value AS Value
           FROM insider_trades ORDER BY transaction_date DESC LIMIT 200"""
    )
    if insider_df.empty:
        st.caption("No insider filings captured yet. Click Refresh Data.")
    else:
        st.dataframe(insider_df, use_container_width=True, hide_index=True)

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
                    use_container_width=True, key=f"dp_gauge_{ticker}",
                )
                st.caption(f"Volume z-score: {row['volume_zscore']:.2f} &middot; signal: {row['signal']}",
                           unsafe_allow_html=True)

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
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_bar")

    display_df = df.copy()
    display_df["yes_price"] = (display_df["yes_price"] * 100).round(1)
    display_df.columns = ["Question", "Category", "YES %", "Volume", "Liquidity", "End Date"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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
        st.dataframe(styled, use_container_width=True, hide_index=True)

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
    if d1.button("Continue", type="primary", use_container_width=True, key=f"ai_dlg_yes_{ticker}"):
        with st.spinner(f"Generating AI briefing for {ticker}..."):
            errors = st.session_state.setdefault("ai_brief_errors", {})
            try:
                generate_deep_analysis(ticker, conn)
                errors.pop(ticker, None)
            except Exception as e:
                errors[ticker] = str(e)
        st.session_state.pop("ai_brief_pending_ticker", None)
        st.rerun()
    if d2.button("Cancel", use_container_width=True, key=f"ai_dlg_no_{ticker}"):
        st.session_state.pop("ai_brief_pending_ticker", None)
        st.rerun()


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
    fetch_live = sc2.button("Fetch live signals", use_container_width=True)
    gen_briefing_clicked = sc3.button("🧠 Generate AI Briefing", use_container_width=True)
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
                    if acol2.button("🔄 Re-run", use_container_width=True, key=f"ai_rerun_{dd_ticker}"):
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
                        reddit_posts = [p for p in substantive if p.get("source") in
                                        ("reddit_script_app", "devvit")]

                        st.markdown(
                            f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                            f'color:{TEXT_SECONDARY};">STOCKTWITS</div>', unsafe_allow_html=True,
                        )
                        if st_posts:
                            for p in st_posts[:8]:
                                st.markdown(
                                    f'<div style="font-size:12px;padding:3px 0;border-bottom:1px solid {BORDER};">'
                                    f'{html_safe_snippet(_truncate_post_text(p.get("text")))} '
                                    f'<span style="color:{TEXT_MUTED};font-size:10px;">'
                                    f'{html.escape(p.get("posted_at") or "")}</span></div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No StockTwits posts cached for this ticker yet.")

                        st.markdown(
                            f'<div style="margin-top:10px;font-size:11px;font-weight:700;'
                            f'color:{TEXT_SECONDARY};">REDDIT</div>', unsafe_allow_html=True,
                        )
                        if reddit_posts:
                            for p in reddit_posts[:8]:
                                st.markdown(
                                    f'<div style="font-size:12px;padding:3px 0;border-bottom:1px solid {BORDER};">'
                                    f'{html_safe_snippet(_truncate_post_text(p.get("text")))} '
                                    f'<span style="color:{TEXT_MUTED};font-size:10px;">'
                                    f'{html.escape(p.get("posted_at") or "")}</span></div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption(
                                "No Reddit posts found (Devvit app not published yet -- "
                                "see SETTINGS tab source registry)."
                            )
                    with icol:
                        st.caption("Institutional")
                        st.markdown(ai_section_or_fallback(ai_brief.get("institutional_analysis")))

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
                        "real retail sentiment (Reddit/StockTwits), insider/congressional trades, "
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
                    st.plotly_chart(render_recommendation_breakdown(rec_counts), use_container_width=True,
                                    key=f"dd_rec_breakdown_{dd_ticker}")
                    st.caption(
                        f"Breakdown for period {targets.get('rec_period') or '—'} &middot; "
                        f"{sum(v or 0 for v in rec_counts.values())} analysts", unsafe_allow_html=True,
                    )
                else:
                    st.caption("No data available.")
            with ecol2:
                st.markdown("**Analyst Price Targets (next 12 months)**")
                n_analysts_txt = targets.get("numberOfAnalystOpinions")
                st.caption(
                    f"Aggregated from {n_analysts_txt} analysts via Yahoo Finance."
                    if n_analysts_txt else "Aggregated via Yahoo Finance (analyst count unavailable)."
                )
                has_targets = any(
                    targets.get(k) is not None
                    for k in ("targetLowPrice", "targetMeanPrice", "targetMedianPrice", "targetHighPrice")
                )
                if has_targets:
                    st.plotly_chart(
                        render_target_range(
                            snap["last_price"], targets.get("targetLowPrice"),
                            targets.get("targetMeanPrice"), targets.get("targetMedianPrice"),
                            targets.get("targetHighPrice"),
                        ),
                        use_container_width=True, key=f"dd_target_range_{dd_ticker}",
                    )
                else:
                    st.caption("No data available.")

                pt_breakdown = fetch_analyst_price_target_breakdown(dd_ticker)
                if not pt_breakdown.empty:
                    st.caption("Individual analyst actions with an attached price target (most recent first):")
                    show_pt = pt_breakdown.rename(columns={
                        "GradeDate": "Date", "Firm": "Firm", "Action": "Action",
                        "ToGrade": "Rating", "currentPriceTarget": "Price Target",
                    }).copy()
                    show_pt["Date"] = show_pt["Date"].apply(fmt_date)
                    show_pt["Price Target"] = show_pt["Price Target"].map(lambda v: f"${v:,.2f}")
                    st.dataframe(show_pt, use_container_width=True, hide_index=True)
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
                st.dataframe(eh, use_container_width=True, hide_index=True)
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
                st.dataframe(show_qfin, use_container_width=True, hide_index=True)
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
                    st.dataframe(idf, use_container_width=True, hide_index=True)
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
                    st.dataframe(cdf, use_container_width=True, hide_index=True)
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
                    st.dataframe(snap_df, use_container_width=True, hide_index=True)

            st.markdown('<div class="smd-section">OPTIONS FLOW</div>', unsafe_allow_html=True)
            odf = q(
                """SELECT option_type AS Type, strike AS Strike, expiration AS Expiry, volume AS Volume,
                          open_interest AS OI, ROUND(volume_oi_ratio, 2) AS "Vol/OI"
                   FROM options_flow WHERE ticker=? AND fetch_date=? AND unusual=1
                   ORDER BY volume DESC LIMIT 10""",
                (dd_ticker, date.today().isoformat()),
            )
            if not odf.empty:
                st.dataframe(odf, use_container_width=True, hide_index=True)
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
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.caption("No data available &mdash; no LEAPS expiry &ge;18mo out, or no deep-ITM calls "
                          "passed the stock-replacement rules for this ticker.", unsafe_allow_html=True)

            st.markdown(
                f'<div class="smd-section">BUYBACKS<span>{source_badge(buybacks_env)}</span></div>',
                unsafe_allow_html=True,
            )
            if not buybacks["history"].empty:
                bcol1, bcol2 = st.columns([2, 1])
                with bcol1:
                    st.plotly_chart(render_buyback_chart(buybacks["history"]), use_container_width=True,
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
                        use_container_width=True, key=f"dd_gauge_{dd_ticker}",
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
                st.dataframe(show, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Tab 8 — NEWS (watchlist-wide feed; distinct from per-ticker news on
# TICKER DEEP-DIVE)
# --------------------------------------------------------------------------

with tab8:
    st.markdown('<div class="smd-section">WATCHLIST NEWS FEED</div>', unsafe_allow_html=True)
    ncol1, ncol2 = st.columns([3, 1])
    with ncol2:
        st.markdown("<br>", unsafe_allow_html=True)
        force_news = st.button("🔄 Refresh feed", use_container_width=True, key="news_feed_refresh")

    conn_news = get_conn()
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
        publishers = sorted(p for p in news_all["publisher"].dropna().unique() if p)

        fcol1, fcol2, fcol3 = st.columns([2, 1.2, 1.2])
        with fcol1:
            pub_sel = st.multiselect("Source", publishers, default=publishers, key="news_pub_filter")
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
            news_all["publisher"].isin(pub_sel)
            & (news_all["published_at"].dt.date >= date_from)
            & (news_all["published_at"].dt.date <= date_to)
        ]

        st.caption(
            f"{len(filtered)} headlines &middot; {len(watchlist)} tickers &middot; "
            f"chronological, most recent first", unsafe_allow_html=True,
        )

        if filtered.empty:
            st.caption("No headlines match the current filters.")
        else:
            for _, n in filtered.iterrows():
                pub = md_safe(n["publisher"] or "Unknown source")
                ago = time_ago(n["published_at"])
                title = md_safe(n["title"])
                headline = f"[{title}]({n['link']})" if n["link"] else title
                st.markdown(
                    f"""<div style="border:1px solid {BORDER};border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                    <span style="background:{PANEL_ALT};color:{ACCENT};font-size:10px;font-weight:700;
                    padding:2px 8px;border-radius:10px;letter-spacing:0.04em;">{n['ticker']}</span>
                    <div style="margin-top:6px;">{headline}</div>
                    <div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;">{pub} &middot; {ago}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

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
    else:
        runners_df = pd.DataFrame(runner_rows).sort_values("_unusualness", ascending=False).drop(
            columns="_unusualness"
        )
        styled = runners_df.copy()
        styled["Price"] = styled["Price"].map(lambda v: f"${v:.2f}")
        styled["Chg %"] = styled["Chg %"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        styled["Volume"] = styled["Volume"].map(lambda v: f"{v:,}")
        styled["Vol vs 20D avg"] = styled["Vol vs 20D avg"].map(lambda v: f"{v:.1f}x" if pd.notna(v) else "—")
        st.caption(f"{len(runners_df)} of {len(watchlist)} watchlist tickers flagged — sorted most unusual first "
                   f"(volume ≥2x 20-day average, or move ≥3% intraday).")
        st.dataframe(styled, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Tab 10 — SETTINGS (source registry + watchlist config; both formerly in
# the sidebar)
# --------------------------------------------------------------------------

with tab10:
    render_settings_tab()
