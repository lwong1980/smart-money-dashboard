#!/bin/bash
# Wrapper launchd invokes for the 2026-09-28 quote-timing test (item 6 of
# the Volatility Dislocations Stage 1 follow-up). Mirrors
# digest/run_scheduled_digest.sh's pattern: launchd jobs start with a
# near-empty environment, so `source setup_env.sh` puts back whatever env
# vars data_engine.py needs (none of the quote-timing scripts need SMTP/
# ANTHROPIC keys, but this keeps the pattern consistent and costs nothing
# if the file's missing).
#
# Usage (only ever called by launchd, via the plist's ProgramArguments):
#   run_quote_timing_monday.sh 1030   -> lightweight capture only
#   run_quote_timing_monday.sh 1230   -> lightweight capture only
#   run_quote_timing_monday.sh 1430   -> lightweight capture only
#   run_quote_timing_monday.sh 1545   -> lightweight capture, THEN the one
#                                         real Stage 1 rebuild (writes to
#                                         smart_money.db's universe table)
#   run_quote_timing_monday.sh 1630   -> lightweight capture only
#   run_quote_timing_monday.sh report -> reads all 5 captures, computes
#                                         the median-spread report, applies
#                                         decision rules a/b/c
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f setup_env.sh ]; then
    source setup_env.sh
else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) WARNING setup_env.sh not found in $(pwd)." >&2
fi

SLOT="${1:?usage: run_quote_timing_monday.sh <1030|1230|1430|1545|1630|report>}"

case "$SLOT" in
    report)
        ./venv/bin/python3 scripts/quote_timing_report.py
        # Self-cleanup: launched DETACHED (nohup + background + disown), not `exec`'d and not run inline.
        # This job's own launchd label (com.smartmoney.quotetiming.report) is one of the 7 the uninstall
        # script boots out, including itself -- `launchctl bootout` on a label that's still actively
        # running can SIGTERM it, which would kill an inline uninstall loop partway through (leaving some
        # of the other 6 jobs still installed for 2027-09-28). Backgrounding + disowning re-parents the
        # uninstall process away from this script before that happens, so it finishes the full loop
        # regardless of what happens to this job's own process right after.
        nohup ./scripts/uninstall_quote_timing_launchd.sh \
            > ./scripts/quote_timing_logs/self_uninstall.out.log 2>&1 &
        disown
        ;;
    1545)
        ./venv/bin/python3 scripts/quote_timing_capture.py --slot 1545
        exec ./venv/bin/python3 scripts/run_stage1_rebuild_1545.py
        ;;
    1030|1230|1430|1630)
        exec ./venv/bin/python3 scripts/quote_timing_capture.py --slot "$SLOT"
        ;;
    *)
        echo "Unknown slot: $SLOT" >&2
        exit 1
        ;;
esac
