#!/bin/bash
# Removes the headless digest launchd job. The in-app "Send Test Email
# Now" / Preview / the browser-open catch-up path are all completely
# unaffected -- this only removes the launchd scheduler.
set -euo pipefail
LABEL="com.smartmoney.digest"
PLIST_DST="$HOME/Library/LaunchAgents/com.smartmoney.digest.plist"
UID_NUM="$(id -u)"
DOMAIN="gui/$UID_NUM"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    launchctl bootout "$DOMAIN" "$PLIST_DST" 2>/dev/null || true
    echo "Unloaded $LABEL."
else
    echo "$LABEL is not currently loaded."
fi

if [ -f "$PLIST_DST" ]; then
    rm "$PLIST_DST"
    echo "Removed $PLIST_DST."
fi
