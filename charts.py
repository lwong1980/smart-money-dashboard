"""Pure chart-building functions: Plotly figures + PNG export, zero Streamlit
dependency.

Extracted verbatim (mechanical move, no logic changes) from dashboard.py so
the daily digest's headless scheduled-send path (digest/send.py, run via
launchd with `python -m digest.send`, no Streamlit process) can produce the
exact same chart images the "Send Test Email Now" button does -- the same
render_* functions, the same theme, the same _build_*_chart_* wrappers.
dashboard.py now imports these from here instead of defining them, so
there's exactly one implementation either path can call.

Anything here that needs a DB connection takes one as an explicit parameter
-- never a module-global "current connection," so this file has zero
dependency on dashboard.py (which is what makes it importable without
triggering dashboard.py's own top-level Streamlit calls)."""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_engine import (
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK,
    _format_et_time_12h,
    cached_fundamentals,
    detect_technical_levels,
    fetch_price_history,
    get_macro_events_in_range,
)

# --------------------------------------------------------------------------
# Theme / palette -- the subset dashboard.py's chart functions need. The
# rest of dashboard.py's palette (BG, PANEL_ALT, BORDER, COLOR_CONFLICTED,
# ...) is presentation-only for HTML/CSS elsewhere and stays defined there;
# dashboard.py imports the names below from here rather than redefining
# them, so there is exactly one definition of each color either file uses.
# --------------------------------------------------------------------------

PANEL = "#080d14"
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


_MACRO_IMPACT_COLORS = {"High": COLOR_BEARISH, "Medium": "#e8c547", "Low": TEXT_SECONDARY}


def add_macro_event_markers(fig, events, intraday=False, chart_tz=None, row=None, col=None,
                             y_bottom=None, marker_row=None):
    """Adds a plain thin, low-opacity (35%) dashed vertical line for
    every macro event whose date falls in the chart's own range -- fully
    automatic (Part 6): every price-chart function calls get_macro_
    events_in_range with its OWN visible date range and passes the
    result straight here, so a new macro event starts appearing on every
    relevant chart with zero per-chart/per-ticker configuration the
    moment it's fetched. Color-coded by impact_level (red/orange High --
    FOMC/CPI/NFP; muted gray Medium/Low). On an intraday chart (Day
    Prediction), positions the line at the event's real scheduled_
    time_et instead of just the date, localized to the chart's own tz
    (chart_tz) so it lines up with the rest of that chart's tz-aware
    x-axis.

    Hover detail: a single small dot marker at the BOTTOM of the chart
    (y_bottom, supplied by the caller from its own price data's range --
    never both top and bottom) shows the full event name/date/time on
    mouseover, via a real Scatter trace with hovertext -- NOT
    annotation_hovertext (a Plotly.js quirk: the Python object shows
    text=None but the browser still renders a placeholder). The line
    itself carries no annotation/label of any kind; the dot is the only
    interactive element, and dashboard.py's render_macro_event_expander
    (not moved here -- it's a Streamlit render, not chart-building) is
    the always-visible fallback in the UI."""
    for ev in events:
        color = _MACRO_IMPACT_COLORS.get(ev["impact_level"], TEXT_SECONDARY)
        time_label = f" {_format_et_time_12h(ev['scheduled_time_et'])}" if ev.get("scheduled_time_et") else ""
        if intraday and ev.get("scheduled_time_et"):
            x = pd.Timestamp(f"{ev['scheduled_date']} {ev['scheduled_time_et']}")
            if chart_tz is not None:
                x = x.tz_localize(chart_tz)
        else:
            x = ev["scheduled_date"]
        vline_kwargs = dict(x=x, line=dict(color=color, width=1, dash="dash"), opacity=0.35)
        if row is not None:
            vline_kwargs["row"] = row
        if col is not None:
            vline_kwargs["col"] = col
        fig.add_vline(**vline_kwargs)

        if y_bottom is not None:
            trace_kwargs = dict(
                x=[x], y=[y_bottom], mode="markers",
                marker=dict(symbol="circle", size=7, color=color, line=dict(width=1, color=PANEL)),
                hovertext=f"{ev['event_name']} — {ev['scheduled_date']}{time_label}",
                hoverinfo="text", showlegend=False,
            )
            if marker_row is not None and col is not None:
                fig.add_trace(go.Scatter(**trace_kwargs), row=marker_row, col=col)
            else:
                fig.add_trace(go.Scatter(**trace_kwargs))
    return fig


def render_price_volume_chart(df, ticker, mode="line", short_label="SMA short", long_label="SMA long",
                               height=460, levels=None, conn=None):
    """Price chart (line or candlestick) with a volume subplot beneath it
    and SMA_short/SMA_long overlays -- the timeframe-aware companion to
    render_price_chart, used wherever fetch_price_history() supplies the
    data (its columns are named SMA_short/SMA_long, not SMA50/SMA200).

    `levels`, if given a detect_technical_levels()-shaped dict, overlays
    dashed horizontal lines for every detected support/resistance level --
    the nearest (key_level_below/key_level_above) drawn bold and labeled,
    the rest faint so the chart stays readable when several levels cluster
    close together.

    `conn`: the macro-event overlay needs a DB connection -- callers pass
    one explicitly (dashboard.py's get_conn(), or the headless digest
    script's own connection). If omitted the macro-event overlay is simply
    skipped rather than guessing at a connection to open."""
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

    macro_events = []
    if not df.empty and conn is not None:
        macro_events = get_macro_events_in_range(conn, df.index.min(), df.index.max(),
                                                  overlay_impact_filter="auto")
        # Dot sits on the VOLUME subplot (row 2), at y=0 -- volume bars
        # already start at 0, so this is the true visual bottom of the
        # whole chart with no padding math needed. The vline itself still
        # spans both subplots (row="all") so it visually connects price
        # to the dot below.
        add_macro_event_markers(fig, macro_events, intraday=is_intraday, chart_tz=df.index.tz, row="all", col=1,
                                 y_bottom=0, marker_row=2)

    fig.update_layout(xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    apply_theme(
        fig, height=height,
        title=dict(text=f"{ticker} &mdash; Price &amp; Volume", font=dict(color=TEXT_PRIMARY, size=14)),
        legend=dict(orientation="h", y=1.06, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig, macro_events


def render_price_target_fan(hist_df, current, low, mean, high, height=380):
    """Historical close price flowing into a forward-projecting fan toward
    High/Average/Low analyst targets -- redesigned directly against real
    Yahoo Finance and TradingView price-target chart references: a real
    price history line leads into three dashed projection lines from the
    current price to each target, each ending in a colored value+percent
    label at the right edge.

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


def render_entry_zone_mini_chart(hist_df, entry_low, entry_high, current_price, ticker, height=260):
    """WATCHLIST SIGNALS dense-overview detail panel (2026-09-25): recent
    close-price line, a shaded band for [entry_low, entry_high], and the
    current price marked. Plain Plotly text throughout -- axis titles,
    annotations -- must NOT be md_safe-escaped: that escaping is a
    markdown-renderer workaround (a literal '$' opening/closing
    Streamlit's LaTeX math mode) and doesn't apply here at all; Plotly
    has no such parsing, so an escaped "\\$" would show a literal
    backslash instead of a dollar sign. Returns an empty themed figure
    (never a fabricated band) if entry_low/entry_high or the price
    history is missing."""
    fig = go.Figure()
    if entry_low is None or entry_high is None or hist_df is None or hist_df.empty or "Close" not in hist_df:
        apply_theme(fig, height=height)
        return fig

    closes = hist_df["Close"].dropna()
    x = closes.index.strftime("%Y-%m-%d").tolist()
    fig.add_trace(go.Scatter(
        x=x, y=closes.values, mode="lines", name="Close", line=dict(color=ACCENT, width=1.6),
        hovertemplate="%{x|%b %d, %Y}: $%{y:,.2f}<extra></extra>", showlegend=False,
    ))
    fig.add_hrect(y0=entry_low, y1=entry_high, fillcolor="rgba(0,229,255,0.12)",
                  line=dict(color=ACCENT, width=1, dash="dot"), layer="below")
    fig.add_annotation(
        x=x[0], y=entry_high, xref="x", yref="y", xanchor="left", yanchor="bottom", showarrow=False,
        text=f"Entry zone ${entry_low:,.2f}–${entry_high:,.2f}", font=dict(size=10, color=ACCENT),
    )
    if current_price is not None:
        fig.add_hline(y=current_price, line=dict(color=TEXT_PRIMARY, width=1, dash="dash"), opacity=0.6)
        fig.add_annotation(
            x=x[-1], y=current_price, xref="x", yref="y", xanchor="right", yanchor="bottom", showarrow=False,
            text=f"Current ${current_price:,.2f}", font=dict(size=10, color=TEXT_PRIMARY),
        )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title=dict(text="Price ($)", font=dict(color=TEXT_MUTED)))
    apply_theme(fig, height=height, margin=dict(l=50, r=15, t=15, b=30), hovermode="x unified")
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


def render_table_image(title, headers, rows, width=760, row_height=26, header_height=34):
    """2026-10-13, "Send to Substack": Substack's editor doesn't handle HTML tables or email CSS, so each
    digest table becomes a PNG via this ONE generic go.Table renderer (not five bespoke ones) -- used for
    Institutional Analysis, Day Predictions, Watchlist Signals, Macro Calendar, and Swing Outlook alike.
    `rows` is a list of row-lists already formatted as plain strings (the SAME formatting the HTML email
    uses, reused from the caller -- this function does no number formatting of its own). Height is derived
    from row count so a 3-row table isn't rendered with the same canvas as a 20-row one. Returns PNG bytes
    via the same render_chart_as_image/kaleido path every other digest image already uses, or None if
    kaleido/Chrome isn't available (never raises)."""
    fig = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in headers], fill_color=PANEL, font=dict(color=ACCENT, size=12),
                    align="left", height=header_height, line_color=GRID),
        cells=dict(values=list(zip(*rows)) if rows else [[] for _ in headers],
                   fill_color=PANEL, font=dict(color=TEXT_PRIMARY, size=11), align="left", height=row_height,
                   line_color=GRID),
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TEXT_PRIMARY, family="Space Mono, monospace")),
        paper_bgcolor=PANEL, margin=dict(l=10, r=10, t=40, b=10),
    )
    height = header_height + row_height * max(len(rows), 1) + 50
    return render_chart_as_image(fig, width=width, height=height)


def render_chart_as_image(fig, width=560, height=380):
    """Converts an existing Plotly figure into static PNG bytes via kaleido
    -- the standard way to embed a real chart in an HTML email, since email
    clients can't run Plotly/JS. Returns None (never raises) if kaleido/
    Chrome isn't available in this environment -- the digest still sends,
    just without the chart image, rather than failing the whole send over
    a missing chart dependency."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception as e:
        print(f"[render_chart_as_image] chart export failed: {e}")
        return None


# --------------------------------------------------------------------------
# Digest-specific wrappers: one PNG (or {} / (None, None)) per section,
# built from the render_* functions above -- called identically by
# dashboard.py's "Send Test Email Now"/Preview path and by digest/send.py's
# headless scheduled path, so the two can never produce different images.
# --------------------------------------------------------------------------

FEATURED_STOCK_CHART_CID = "featured_stock_chart"
SWING_CHART_CID = "swing_outlook_chart"
ANALYST_BREAKDOWN_CID = "analyst_breakdown_chart"
ANALYST_FAN_CID = "analyst_target_fan_chart"


def build_featured_stock_chart_bytes(ticker, conn):
    """Reuses render_price_volume_chart -- the EXACT function Ticker
    Deep-Dive's own price chart uses, SMA/support-resistance overlay
    included -- rather than building a second, digest-only chart. Returns
    (png_bytes_or_None, cid_or_None)."""
    hist, cfg = fetch_price_history(ticker, DEFAULT_INTERVAL, DEFAULT_LOOKBACK, conn=conn)
    if hist.empty:
        return None, None
    levels = detect_technical_levels(ticker, DEFAULT_INTERVAL, DEFAULT_LOOKBACK, conn=conn)
    fig, _ = render_price_volume_chart(
        hist, ticker, short_label="SMA short", long_label="SMA long", height=380, levels=levels, conn=conn,
    )
    png_bytes = render_chart_as_image(fig)
    return (png_bytes, FEATURED_STOCK_CHART_CID) if png_bytes else (None, None)


def build_swing_chart_image(featured_stock):
    """The 10-day projected-path chart for the email as a static PNG (email clients can't run Plotly). Same three
    series, names and colours as the SWING FORECAST tab's chart (Close / {h}d projected path / Confidence band),
    built from data_engine.build_swing_chart_data, which fixes the tab chart's two defects: the projection starts
    AT the last close, and the p10-p90 band is a cone shaded across the whole projected span. Exported at 600px
    wide with the legend below the plot and email-sized fonts. Returns {cid: png_bytes} ({} if there's nothing to
    draw) and records the cid in featured_stock["swing_cids"]."""
    sw = (featured_stock or {}).get("swing_outlook")
    ch = sw.get("chart") if sw else None
    if not ch or not ch.get("path"):
        return {}
    h = ch["horizon_days"]
    target_color = (COLOR_BULLISH if ch["direction"] == "UP" else COLOR_BEARISH if ch["direction"] == "DOWN"
                    else COLOR_NEUTRAL)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ch["hist_dates"], y=ch["hist_close"], mode="lines",
                             line=dict(color=ACCENT, width=2.5), name="Close"))
    fig.add_trace(go.Scatter(x=[d for d, _ in ch["path"]], y=[p for _, p in ch["path"]], mode="lines+markers",
                             line=dict(color=target_color, width=2.5, dash="dash"), marker=dict(size=6),
                             name=f"{h}d projected path"))
    if ch.get("band"):
        b = ch["band"]
        fig.add_trace(go.Scatter(x=b["dates"] + b["dates"][::-1], y=b["high"] + b["low"][::-1], fill="toself",
                                 fillcolor="rgba(139,155,171,0.30)", line=dict(width=0), hoverinfo="skip",
                                 name="Confidence band (p10-p90)"))
    apply_theme(fig, height=360, margin=dict(l=56, r=14, t=10, b=8),
                legend=dict(orientation="h", x=0, y=-0.2, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)",
                            font=dict(size=12)))
    fig.update_layout(font=dict(size=12))
    fig.update_yaxes(title_text="Price ($)", title_font=dict(size=12), tickprefix="$")
    fig.update_xaxes(tickformat="%b %-d", nticks=6)
    png = render_chart_as_image(fig, width=600, height=360)
    if not png:
        return {}
    featured_stock["swing_cids"] = {"chart": SWING_CHART_CID}
    return {SWING_CHART_CID: png}


def build_analyst_chart_images(ticker, featured_stock, conn):
    """Renders the EARNINGS & ANALYST TARGETS charts for the digest with the
    EXACT functions the dashboard panel uses (render_recommendation_
    breakdown, render_price_target_fan) -> PNG via kaleido, same as the price
    chart. Returns {cid: png_bytes} and records which charts exist in
    featured_stock["analyst_cids"] so the email references only images that
    were actually produced (anything missing falls back to HTML/text in
    build_digest_content)."""
    images = {}
    at = (featured_stock or {}).get("analyst_targets")
    if not at:
        return images
    cids = {}
    if at.get("rec_counts"):
        png = render_chart_as_image(render_recommendation_breakdown(at["rec_counts"], height=130),
                                    width=560, height=130)
        if png:
            images[ANALYST_BREAKDOWN_CID], cids["breakdown"] = png, ANALYST_BREAKDOWN_CID
    if at.get("has_targets") and at.get("current_price") is not None:
        try:
            price_df = cached_fundamentals(conn, ticker, max_age_hours=12)["data"]["price_df"]
            fig = render_price_target_fan(price_df, at["current_price"], at.get("target_low"),
                                          at.get("target_mean"), at.get("target_high"), height=340)
            png = render_chart_as_image(fig, width=560, height=340)
        except Exception as e:
            print(f"[build_analyst_chart_images] {ticker}: {e}")
            png = None
        if png:
            images[ANALYST_FAN_CID], cids["fan"] = png, ANALYST_FAN_CID
    featured_stock["analyst_cids"] = cids
    return images
