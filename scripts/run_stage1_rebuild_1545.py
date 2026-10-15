"""The ONE real, production Stage 1 rebuild triggered as part of the
2026-09-28 quote-timing test -- runs at 15:45 ET (the time already
favored earlier in this project: "if after-close quotes are unusable,
Stage 2 runs at 3:45pm ET instead of after close") against the REAL
smart_money.db, via the existing, unmodified rebuild_universe_tier1 /
rebuild_universe_tier2.

Deliberately the ONLY real rebuild scheduled that day -- NOT also run
speculatively at 14:30, since both would write to the same
(as_of_date, ticker) primary key and whichever ran last would silently
win, which could bury a genuinely-better 14:30 result under a
worse 15:45 one. Instead: quote_timing_report.py's rule (b) FLAGS this
run as not representative if the diagnostic data says 14:30 was actually
better, rather than this script guessing which time to trust in advance.
If rule (b) fires, a real 14:30 rerun is needed on the next trading day
-- run this same script's logic again then (manually, or a future
scheduled slot), not backdated.

Usage: venv/bin/python3 scripts/run_stage1_rebuild_1545.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_engine as de  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quote_timing_results")


def main():
    as_of = de.today_et().isoformat()
    de.init_db(de.DEFAULT_DB_PATH)
    conn = de.get_connection(de.DEFAULT_DB_PATH)
    print(f"[run_stage1_rebuild_1545] starting real Tier 1 + Tier 2 rebuild at {de.now_et().isoformat()}")
    t1 = de.rebuild_universe_tier1(conn)
    t2 = de.rebuild_universe_tier2(conn)
    print(f"[run_stage1_rebuild_1545] done at {de.now_et().isoformat()}")
    print(f"  Tier 1: {t1}")
    print(f"  Tier 2: {t2}")

    out_dir = os.path.join(RESULTS_DIR, as_of)
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "stage1_rebuild_1545.json"), "w") as f:
        json.dump({"as_of_date": as_of, "tier1": t1, "tier2": t2}, f, indent=1, default=str)
    conn.close()


if __name__ == "__main__":
    main()
