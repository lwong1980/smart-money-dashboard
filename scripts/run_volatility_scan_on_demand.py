"""The DETACHED background process launched by dashboard.py's "Run scan
now" button (via data_engine.start_volatility_scan_on_demand) -- runs
Tier 1 + Tier 2 rebuild, then the Stage 2 scan, all tagged
run_type='on_demand', updating the volatility_scan_runs progress row the
Streamlit tab polls at every step so progress survives a page refresh.

NEVER run this directly for a scheduled/production pass -- Monday's
15:45 ET job uses run_stage1_rebuild_1545.py, untouched by this file.
This script is only ever invoked by start_volatility_scan_on_demand()
with --run-id pointing at the row it already inserted.

Usage: venv/bin/python3 scripts/run_volatility_scan_on_demand.py --run-id 7
"""
import argparse
import os
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_engine as de  # noqa: E402


def _update(conn, run_id, **fields):
    if not fields:
        return
    set_clause = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE volatility_scan_runs SET {set_clause} WHERE id=?", (*fields.values(), run_id))
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, required=True)
    args = ap.parse_args()
    run_id = args.run_id

    de.init_db(de.DEFAULT_DB_PATH)
    conn = de.get_connection(de.DEFAULT_DB_PATH)

    try:
        # Stage 1 -- Tier 1
        _update(conn, run_id, stage="tier1", pid=os.getpid())

        def _tier1_progress(done, total):
            _update(conn, run_id, tickers_total=total, tickers_done=done)

        t1 = de.rebuild_universe_tier1(conn, run_type="on_demand", progress_callback=_tier1_progress)
        _update(conn, run_id, core_count=t1["core_count"])

        # Stage 1 -- Tier 2
        _update(conn, run_id, stage="tier2", tickers_total=0, tickers_done=0)

        def _tier2_progress(done, total):
            _update(conn, run_id, tickers_total=total, tickers_done=done)

        t2 = de.rebuild_universe_tier2(conn, run_type="on_demand", progress_callback=_tier2_progress)
        _update(conn, run_id, spike_count=t2["spike_count"])

        # Stage 2 -- scan
        _update(conn, run_id, stage="scan", tickers_total=0, tickers_done=0)

        def _scan_progress(done, total):
            _update(conn, run_id, tickers_total=total, tickers_done=done)

        scan_result = de.run_volatility_scan(conn, run_type="on_demand", progress_callback=_scan_progress)

        _update(
            conn, run_id, status="done", finished_at=datetime.now(timezone.utc).isoformat(),
            scan_stored=scan_result["stored"], scan_skipped=scan_result["skipped"],
            scan_failed=scan_result["failed"],
        )
        print(f"[on_demand run {run_id}] done: core={t1['core_count']} spike={t2['spike_count']} "
              f"scan_stored={scan_result['stored']} scan_skipped={scan_result['skipped']} "
              f"scan_failed={scan_result['failed']}")
    except Exception as e:
        traceback.print_exc()
        try:
            _update(conn, run_id, status="failed", finished_at=datetime.now(timezone.utc).isoformat(),
                    error_message=str(e)[:2000])
        except Exception:
            pass
    finally:
        conn.close()


if __name__ == "__main__":
    main()
