#!/bin/bash
# Installs the 7 one-shot launchd jobs for the 2026-09-28 quote-timing
# test (5 captures + report + keep-awake). Safe to re-run.
#
# Self-cleaning: the report job's own final step launches
# uninstall_quote_timing_launchd.sh (detached), so all 7 -- including
# itself -- are normally gone again by ~14:00-14:05 local Monday, with no
# manual step needed. scripts/uninstall_quote_timing_launchd.sh remains
# here as a manual fallback only (e.g. if Monday's run never fires at
# all, or the self-cleanup step itself fails for some reason).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LABELS=(
    com.smartmoney.quotetiming.1030
    com.smartmoney.quotetiming.1230
    com.smartmoney.quotetiming.1430
    com.smartmoney.quotetiming.1545
    com.smartmoney.quotetiming.1630
    com.smartmoney.quotetiming.keepawake
    com.smartmoney.quotetiming.report
)

mkdir -p "$HOME/Library/LaunchAgents" scripts/quote_timing_logs
UID_NUM="$(id -u)"
DOMAIN="gui/$UID_NUM"

for LABEL in "${LABELS[@]}"; do
    SRC="$(pwd)/scripts/$LABEL.plist"
    DST="$HOME/Library/LaunchAgents/$LABEL.plist"
    echo "Installing $DST"
    cp "$SRC" "$DST"
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
        launchctl bootout "$DOMAIN" "$DST" 2>/dev/null || true
    fi
    launchctl bootstrap "$DOMAIN" "$DST"
    launchctl enable "$DOMAIN/$LABEL"
done

echo
echo "Installed 7 jobs. Verify with:"
echo "  launchctl list | grep quotetiming"
echo
echo "*** ONE MORE STEP, NOT AUTOMATED (needs sudo) ***"
echo "To guarantee the Mac actually WAKES from sleep in time for the first"
echo "capture (caffeinate only prevents sleep once already awake -- it"
echo "can't wake a sleeping machine), run this yourself, once, any time"
echo "before Monday morning:"
echo
echo "  sudo pmset schedule wake \"09/28/26 07:15:00\""
echo
echo "(07:15 local America/Vancouver time = 10:15am ET, 15 min before the"
echo "first 10:30am ET capture. Skip this if the Mac reliably stays awake"
echo "on its own during the day -- it's only needed if it might be asleep"
echo "at 07:15 local Monday.)"
echo
echo "The report step (14:00 local) self-uninstalls all 7 jobs when it's"
echo "done -- no manual cleanup needed. Fallback if that doesn't happen:"
echo "  scripts/uninstall_quote_timing_launchd.sh"
