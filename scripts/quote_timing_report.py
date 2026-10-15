"""Reads all 5 quote_timing_capture.py slot files for a date, computes the
median-spread report, and applies the 3 decision rules exactly as
specified (2026-09-25 follow-up instruction, item 6):

  a. If median spread at 15:45 is <=8% AND meaningfully tighter than
     16:30 -> Stage 1/Stage 2 run at 15:45. (The real Stage 1 rebuild is
     triggered separately, at 15:45 itself, by run_stage1_rebuild_1545.py
     -- this script only confirms whether that run should be trusted.)
  b. Elif median spread is <=8% at some midday slot (10:30/12:30/14:30)
     but NOT at 15:45 -> the correct run time is 14:30 instead. Today's
     already-completed 15:45 rebuild is flagged as NOT representative;
     this script does NOT retroactively substitute a fake "14:30 result"
     -- a real 14:30 rerun needs to happen at an actual 14:30 next
     trading day.
  c. Elif median spread for JPM/AAPL/MSFT specifically stays >10% at
     EVERY slot -> yfinance bid/ask isn't fit to gate on. Stop, report,
     do not loosen the spread floor.
  Else: none of the 3 rules cleanly match -- report only, no auto action.

`decision.json`'s `stage1_rebuild_1545_valid` flag: the real 15:45
rebuild (run_stage1_rebuild_1545.py) necessarily writes to the actual
`universe` table BEFORE this report can judge whether that moment's
quotes were any good -- so both rule (b) AND rule (c) mark it `false`
here (rule (a) marks it `true`; no rule matching leaves it `null`/
unknown). The table write itself is never undone by this script; this
flag is the record of whether that write should be trusted.

"Meaningfully tighter than 16:30" is a disclosed, adjustable heuristic
(not from the original instruction's exact wording): a candidate slot's
median must be <=80% of the 16:30 median (>=20% relative reduction).

Never touches smart_money.db.

Usage: venv/bin/python3 scripts/quote_timing_report.py --date 2026-09-28
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_engine as de  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quote_timing_results")
SLOTS = ["1030", "1230", "1430", "1545", "1630"]
MIDDAY_SLOTS = ["1030", "1230", "1430"]
ANCHOR_NAMES = ["JPM", "AAPL", "MSFT"]
ENTER_SPREAD_FLOOR = 8.0
GATE_UNUSABLE_FLOOR = 10.0
MEANINGFULLY_TIGHTER_RATIO = 0.80  # candidate slot must be <=80% of 16:30's median -- disclosed heuristic
FLAG_VARIANCE_RATIO = 2.0


def load_slot(as_of, slot):
    path = os.path.join(RESULTS_DIR, as_of, f"{slot}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def avg_leg_spread(row):
    """Average of call/put spread_pct for one ticker's row; whichever leg(s) are present."""
    if row is None or "error" in row:
        return None
    vals = [row[leg]["spread_pct"] for leg in ("call", "put")
            if row.get(leg) and row[leg].get("spread_pct") is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYY-MM-DD, matches quote_timing_capture.py's --date; "
                                                    "defaults to today (ET)")
    args = ap.parse_args()
    as_of = args.date or de.today_et().isoformat()

    slot_data = {s: load_slot(as_of, s) for s in SLOTS}
    missing = [s for s, d in slot_data.items() if d is None]
    if missing:
        print(f"[quote_timing_report] MISSING capture file(s) for slot(s) {missing} on {as_of} -- "
              f"cannot compute a full report. Available: {[s for s in SLOTS if s not in missing]}")

    # per-slot: {ticker: avg_spread_pct}
    per_slot_spreads = {}
    for s, d in slot_data.items():
        if d is None:
            continue
        per_slot_spreads[s] = {row["ticker"]: avg_leg_spread(row) for row in d["rows"]}

    # Median per slot (across all 30 names, ignoring None)
    median_by_slot = {}
    for s, spreads in per_slot_spreads.items():
        vals = [v for v in spreads.values() if v is not None]
        median_by_slot[s] = statistics.median(vals) if vals else None

    print(f"\n=== Quote timing report: {as_of} ===")
    print("Median avg(call,put) spread % by slot:")
    for s in SLOTS:
        m = median_by_slot.get(s)
        print(f"  {s}: {m:.2f}%" if m is not None else f"  {s}: (no data)")

    # Per-name tightest slot + >2x flag
    all_tickers = sorted({t for spreads in per_slot_spreads.values() for t in spreads})
    print("\nPer-name tightest slot (and >2x variance flag):")
    flagged = []
    for t in all_tickers:
        readings = {s: per_slot_spreads[s].get(t) for s in per_slot_spreads if per_slot_spreads[s].get(t) is not None}
        if not readings:
            print(f"  {t}: no usable readings in any slot")
            continue
        tightest_slot = min(readings, key=readings.get)
        vals = list(readings.values())
        ratio = (max(vals) / min(vals)) if min(vals) > 0 else float("inf")
        flag = ratio > FLAG_VARIANCE_RATIO
        if flag:
            flagged.append((t, ratio, readings))
        print(f"  {t}: tightest={tightest_slot} ({readings[tightest_slot]:.2f}%)"
              f"{'  *** >2x VARIANCE ACROSS SLOTS ***' if flag else ''}")

    if flagged:
        print("\nFlagged names (>2x spread variance across slots):")
        for t, ratio, readings in flagged:
            print(f"  {t}: {ratio:.1f}x -- " + ", ".join(f"{s}={v:.1f}%" for s, v in readings.items()))

    # Anchor names for rule (c)
    anchor_by_slot = {s: {t: per_slot_spreads.get(s, {}).get(t) for t in ANCHOR_NAMES} for s in SLOTS}
    print(f"\nAnchor names ({', '.join(ANCHOR_NAMES)}) spread % by slot:")
    for s in SLOTS:
        row = anchor_by_slot[s]
        print(f"  {s}: " + ", ".join(f"{t}={v:.2f}%" if v is not None else f"{t}=(n/a)" for t, v in row.items()))

    # --- Decision rules ---
    m1545, m1630 = median_by_slot.get("1545"), median_by_slot.get("1630")
    rule_a = (m1545 is not None and m1545 <= ENTER_SPREAD_FLOOR and m1630 is not None
              and m1545 <= MEANINGFULLY_TIGHTER_RATIO * m1630)

    best_midday_slot, best_midday_median = None, None
    for s in MIDDAY_SLOTS:
        m = median_by_slot.get(s)
        if m is not None and (best_midday_median is None or m < best_midday_median):
            best_midday_slot, best_midday_median = s, m
    rule_b = (not rule_a and best_midday_median is not None and best_midday_median <= ENTER_SPREAD_FLOOR
              and (m1545 is None or m1545 > ENTER_SPREAD_FLOOR))

    anchor_all_slots_bad = True
    for s in SLOTS:
        for t in ANCHOR_NAMES:
            v = anchor_by_slot[s].get(t)
            if v is not None and v <= GATE_UNUSABLE_FLOOR:
                anchor_all_slots_bad = False
    rule_c = (not rule_a and not rule_b and anchor_all_slots_bad
              and any(v is not None for s in SLOTS for v in anchor_by_slot[s].values()))

    print("\n=== Decision ===")
    # stage1_rebuild_1545_valid: the 15:45 real rebuild (run_stage1_rebuild_1545.py) writes to the actual
    # `universe` table BEFORE this report exists to judge it -- so whenever the data it was gated on turns
    # out to be untrustworthy (rule b OR rule c), that already-written table state must be flagged invalid
    # here, the same way rule (b) does, rather than silently left looking authoritative.
    if rule_a:
        decision = "run_at_1545"
        stage1_rebuild_1545_valid = True
        print(f"RULE (a) MATCHED: 15:45 median ({m1545:.2f}%) is <={ENTER_SPREAD_FLOOR}% and "
              f"<= {MEANINGFULLY_TIGHTER_RATIO*100:.0f}% of 16:30's median ({m1630:.2f}% -> "
              f"threshold {MEANINGFULLY_TIGHTER_RATIO*m1630:.2f}%).")
        print("-> Stage 1 rebuild + (future) Stage 2 scan run at 15:45 ET going forward.")
        print("-> Today's already-completed 15:45 real rebuild (run_stage1_rebuild_1545.py) is VALID -- "
              "trust its core/spike counts.")
    elif rule_b:
        decision = "run_at_1430_not_1545"
        stage1_rebuild_1545_valid = False
        print(f"RULE (b) MATCHED: median at {best_midday_slot} ({best_midday_median:.2f}%) is "
              f"<={ENTER_SPREAD_FLOOR}%, but 15:45 ({'n/a' if m1545 is None else f'{m1545:.2f}%'}) is not.")
        print("-> Correct run time is 14:30 ET, not 15:45.")
        print("-> Today's already-completed 15:45 real rebuild is FLAGGED AS NOT REPRESENTATIVE (invalid). "
              "The `universe` table it wrote to must NOT be trusted as the final Stage 1 answer. A genuine "
              "14:30 rerun is needed on the next trading day -- this script does not fabricate one "
              "retroactively.")
    elif rule_c:
        decision = "gate_unusable_stop"
        stage1_rebuild_1545_valid = False
        print(f"RULE (c) MATCHED: {', '.join(ANCHOR_NAMES)} spread stayed >{GATE_UNUSABLE_FLOOR}% at "
              "every slot with data.")
        print("-> yfinance's bid/ask is not fit to gate on. STOPPING. Do not loosen the spread floor. "
              "Needs a different quality gate (options volume + OI based) -- your call, not automated.")
        print("-> Today's already-completed 15:45 real rebuild is FLAGGED AS INVALID (same as rule b): it "
              "was gated on a spread measurement this rule just determined isn't usable at all, so the "
              "`universe` table state it wrote is not trustworthy either. Don't treat its core/spike "
              "counts as real until a working quality gate exists.")
    else:
        decision = "ambiguous"
        stage1_rebuild_1545_valid = None
        print("NO RULE CLEANLY MATCHED. Reporting only -- no automatic action taken. "
              "Manual review needed before Stage 1 is rerun again. `stage1_rebuild_1545_valid` left as "
              "null (unknown) -- don't assume the 15:45 rebuild is good without checking the numbers above.")

    out = {
        "as_of_date": as_of, "median_by_slot": median_by_slot, "flagged_names": [t for t, _, _ in flagged],
        "anchor_by_slot": anchor_by_slot, "decision": decision,
        "stage1_rebuild_1545_valid": stage1_rebuild_1545_valid,
        "meaningfully_tighter_ratio": MEANINGFULLY_TIGHTER_RATIO,
    }
    out_path = os.path.join(RESULTS_DIR, as_of, "decision.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
