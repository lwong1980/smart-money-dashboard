#!/bin/bash
# Removes the 7 one-shot quote-timing launchd jobs. Run this once
# Monday's (2026-09-28) test is done, so they don't fire again next year.
set -euo pipefail

# "report" is deliberately LAST: when this script is launched (detached)
# by the report job's own final step, it's booting out its own still-
# running label along with the other 6 -- doing that last, after
# everything else is already cleanly removed, minimizes what's lost if
# that self-bootout does anything disruptive to this already-detached
# process.
LABELS=(
    com.smartmoney.quotetiming.1030
    com.smartmoney.quotetiming.1230
    com.smartmoney.quotetiming.1430
    com.smartmoney.quotetiming.1545
    com.smartmoney.quotetiming.1630
    com.smartmoney.quotetiming.keepawake
    com.smartmoney.quotetiming.report
)

UID_NUM="$(id -u)"
DOMAIN="gui/$UID_NUM"

for LABEL in "${LABELS[@]}"; do
    DST="$HOME/Library/LaunchAgents/$LABEL.plist"
    if [ -f "$DST" ]; then
        echo "Removing $LABEL"
        launchctl bootout "$DOMAIN" "$DST" 2>/dev/null || true
        rm -f "$DST"
    fi
done
echo "Done."
