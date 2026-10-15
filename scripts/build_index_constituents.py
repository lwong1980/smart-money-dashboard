"""One-time / occasional (re-)generation of index_constituents.json -- the
candidate pool the Volatility Dislocations screener's Stage 1 universe
rebuild reads from (data_engine.rebuild_universe_tier1/tier2). This
script is NEVER run automatically by the app; it's a manual maintenance
step, same philosophy as watchlist.json -- the JSON file, not a live
scrape, is the source of truth the app actually reads, and it's meant to
be hand-editable afterward.

Run with: python3 scripts/build_index_constituents.py

Sources, both real and confirmed working as of 2026-09-25:
  - S&P 500:    Wikipedia "List of S&P 500 companies"
  - Nasdaq-100: Wikipedia "List of NASDAQ-100 companies"

Russell 1000 was investigated and is NOT included: iShares' IWB holdings
CSV endpoint (the standard free source) returns an HTML bot-wall page
instead of the file when fetched from a script -- confirmed live, same
class of block as the Senate EFD Akamai wall already documented
elsewhere in this project (see CLAUDE.md). No other clean free
scriptable Russell 1000 source was found. If you have access to one (or
a manually-downloaded IWB_holdings.csv from ishares.com in a real
browser), add those tickers to index_constituents.json by hand -- it's
a plain JSON list, editing it directly is expected and safe; this script
will not overwrite manual additions unless you re-run it (in which case
re-add them after)."""
import io
import json
import sys

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0"}
OUT_PATH = "index_constituents.json"


def _normalize(ticker):
    # yfinance's own convention: "BRK.B" -> "BRK-B".
    return str(ticker).strip().upper().replace(".", "-")


def fetch_sp500():
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=15, headers=HEADERS)
    r.raise_for_status()
    table = pd.read_html(io.StringIO(r.text))[0]
    return sorted({_normalize(t) for t in table["Symbol"] if isinstance(t, str)})


def fetch_nasdaq100():
    r = requests.get("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", timeout=15, headers=HEADERS)
    r.raise_for_status()
    for table in pd.read_html(io.StringIO(r.text)):
        if len(table) > 50 and "Ticker" in table.columns:
            return sorted({_normalize(t) for t in table["Ticker"] if isinstance(t, str)})
    raise RuntimeError("Nasdaq-100 constituents table not found on the Wikipedia page -- its layout may have changed.")


def main():
    try:
        sp500 = fetch_sp500()
    except Exception as e:
        print(f"S&P 500 fetch failed: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        nasdaq100 = fetch_nasdaq100()
    except Exception as e:
        print(f"Nasdaq-100 fetch failed: {e}", file=sys.stderr)
        sys.exit(1)

    by_ticker = {}
    for t in sp500:
        by_ticker.setdefault(t, []).append("sp500")
    for t in nasdaq100:
        by_ticker.setdefault(t, []).append("nasdaq100")

    print(f"S&P 500: {len(sp500)}, Nasdaq-100: {len(nasdaq100)}, union: {len(by_ticker)}, "
          f"overlap: {len(set(sp500) & set(nasdaq100))}")

    out = {
        "generated_at": pd.Timestamp.utcnow().date().isoformat(),
        "sources": ["sp500 (Wikipedia)", "nasdaq100 (Wikipedia)"],
        "note": ("Russell 1000 NOT included -- iShares' IWB holdings CSV returns an HTML bot-wall page when "
                 "fetched from a script (confirmed live). Add Russell-1000-only tickers by hand below if you "
                 "have another source."),
        "tickers": sorted(by_ticker.keys()),
        "ticker_sources": by_ticker,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=1)
    print(f"Wrote {OUT_PATH}: {len(out['tickers'])} tickers.")


if __name__ == "__main__":
    main()
