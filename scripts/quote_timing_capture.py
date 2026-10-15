"""One-off diagnostic capture for the Volatility Dislocations screener's
Stage 1 "quote timing" question: does yfinance's ATM bid/ask look
meaningfully different at different times of the trading day, and is it
usable at all after the close?

NEVER touches smart_money.db / the `universe` table. Pure read + write to
a plain JSON file under quote_timing_results/<date>/<slot>.json. This is
intentionally a standalone script (not wired into data_engine.py's Stage 1
pipeline) since it needs PER-LEG (call and put separately) bid/ask/spread,
which the shared `_atm_option_snapshot()` helper does not expose (it
returns only the combined straddle numbers, which is what Stage 1's real
spread gate and Stage 2's implied-move math actually need). It reuses
`_is_third_friday` from data_engine.py so "nearest monthly expiry >=14
days out" is selected with the exact same rule Stage 1 uses -- everything
else here is new, disposable diagnostic code, not options math.

Usage:
    venv/bin/python3 scripts/quote_timing_capture.py --slot 1030
    venv/bin/python3 scripts/quote_timing_capture.py --slot 1030 --date 2026-09-28

`--slot` is just a label (e.g. "1030", "1230", "1430", "1545", "1630") --
it does not have to match wall-clock time exactly; the script also records
the actual capture timestamp so drift is visible in the output.
"""
import argparse
import json
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf  # noqa: E402

import data_engine as de  # noqa: E402

# 10 original mega-caps + 20 more spread across sectors (financials,
# healthcare, energy, consumer disc/staples, industrials, utilities,
# materials, real estate, communication, more tech) -- all real S&P 500
# names, chosen for sector spread, not cherry-picked for tight spreads.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "WMT",
    "BAC", "GS", "UNH", "LLY", "CVX", "HD", "NKE", "KO", "PG", "CAT",
    "BA", "NEE", "LIN", "PLD", "DIS", "NFLX", "AMD", "CRM", "ORCL", "INTC",
]

MONTHLY_MIN_DAYS_OUT = de.UNIVERSE_MONTHLY_MIN_DAYS_OUT  # 14, same floor Stage 1 uses
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quote_timing_results")


def _leg_quote(df, strike):
    """bid/ask/mid/spread_pct/lastTradeDate for one option leg's row closest to `strike`. None if missing."""
    if df is None or df.empty:
        return None
    row = df[df["strike"] == strike]
    if row.empty:
        return None
    row = row.iloc[0]
    # default=None (not _safe_num's own default=0.0): a 0 bid is a real, meaningful value (worthless/no-bid
    # quote), distinct from a genuinely missing field -- collapsing both to 0.0 would hide that distinction.
    bid, ask = de._safe_num(row.get("bid"), default=None), de._safe_num(row.get("ask"), default=None)
    mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
    spread_pct = ((ask - bid) / mid * 100.0) if (mid and mid > 0) else None
    last_trade = row.get("lastTradeDate")
    last_trade_iso = None
    try:
        if last_trade is not None and last_trade == last_trade:  # not NaT/NaN
            last_trade_iso = last_trade.isoformat()
    except Exception:
        pass
    return {
        "bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct,
        "open_interest": de._safe_num(row.get("openInterest"), default=None),
        "volume": de._safe_num(row.get("volume"), default=None),
        "last_trade_date": last_trade_iso,  # yfinance's LAST TRADE time for this contract --
        # NOT a live NBBO quote timestamp; yfinance doesn't expose one. Disclosed, not fabricated.
    }


def capture_one(ticker):
    """Mirrors _atm_option_snapshot's expiry-selection rule (prefer_monthly=True, min_days_out=14) but
    returns separate call/put legs instead of the combined straddle."""
    tk = yf.Ticker(ticker)
    try:
        expirations = list(tk.options)
    except Exception as e:
        return {"ticker": ticker, "error": f"no options: {e}"}
    if not expirations:
        return {"ticker": ticker, "error": "no expirations listed"}

    today = date.today()

    def in_window(exp):
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
        except ValueError:
            return None
        days_out = (d - today).days
        return d if days_out >= MONTHLY_MIN_DAYS_OUT else None

    target_exp, target_date = None, None
    for exp in expirations:
        d = in_window(exp)
        if d is not None and de._is_third_friday(d):
            target_exp, target_date = exp, d
            break
    if target_exp is None:  # fallback: nearest expiry >=14 days out even if not a 3rd Friday
        for exp in expirations:
            d = in_window(exp)
            if d is not None:
                target_exp, target_date = exp, d
                break
    if target_exp is None:
        return {"ticker": ticker, "error": "no expiry >=14 days out"}

    try:
        price = de._safe_num(tk.fast_info.get("lastPrice"), default=None) if hasattr(tk, "fast_info") else None
    except Exception:
        price = None
    if price is None:
        try:
            hist = tk.history(period="5d")
            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            price = None
    if price is None:
        return {"ticker": ticker, "error": "no current price"}

    try:
        chain = de._yf_call_with_backoff(lambda: tk.option_chain(target_exp), what=f"{ticker} {target_exp} chain")
    except Exception as e:
        return {"ticker": ticker, "error": f"chain fetch failed: {e}"}
    calls, puts = chain.calls, chain.puts
    if calls.empty:
        return {"ticker": ticker, "error": "empty call chain"}
    atm_strike = float(min(calls["strike"], key=lambda x: abs(x - price)))

    return {
        "ticker": ticker, "price": price, "expiry": target_exp,
        "days_out": (target_date - today).days, "atm_strike": atm_strike,
        "call": _leg_quote(calls, atm_strike), "put": _leg_quote(puts, atm_strike),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", required=True, help="Label for this capture, e.g. 1030/1230/1430/1545/1630")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD; defaults to today (ET)")
    args = ap.parse_args()

    as_of = args.date or de.today_et().isoformat()
    captured_at_et = de.now_et().isoformat()

    rows = []
    for t in TICKERS:
        rows.append(capture_one(t))
        print(f"  {t}: {rows[-1].get('error') or (rows[-1]['call'] or {}).get('spread_pct')}")

    out_dir = os.path.join(RESULTS_DIR, as_of)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.slot}.json")
    with open(out_path, "w") as f:
        json.dump({"slot": args.slot, "as_of_date": as_of, "captured_at_et": captured_at_et, "rows": rows}, f, indent=1)
    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"[quote_timing] slot={args.slot} as_of={as_of} captured_at={captured_at_et} "
          f"ok={n_ok}/{len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
