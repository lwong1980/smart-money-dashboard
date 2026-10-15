#!/bin/bash
# Installs (or reinstalls) the launchd job that runs the daily digest
# headlessly, every :00 and :30, with no browser/Streamlit process
# required. Safe to re-run: unloads any existing copy first.
#
# What this does NOT do: nothing about your schedule (Send time,
# Frequency) lives in the plist, so you never need to re-run this after
# changing those in the app -- only after moving the project directory or
# changing which Python/venv it uses.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PLIST_SRC="$(pwd)/digest/com.smartmoney.digest.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.smartmoney.digest.plist"
LABEL="com.smartmoney.digest"

mkdir -p "$HOME/Library/LaunchAgents" digest/logs

echo "Installing $PLIST_DST -> $PLIST_SRC"
cp "$PLIST_SRC" "$PLIST_DST"

# Modern launchctl (macOS 10.11+) syntax: bootout/bootstrap against the
# user's GUI domain (gui/<uid>), NOT the deprecated unload/load, which
# stopped reliably reflecting plist edits on recent macOS.
UID_NUM="$(id -u)"
DOMAIN="gui/$UID_NUM"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    echo "Existing job found -- unloading first (bootout)."
    launchctl bootout "$DOMAIN" "$PLIST_DST" 2>/dev/null || true
fi

echo "Loading job (bootstrap)."
launchctl bootstrap "$DOMAIN" "$PLIST_DST"
launchctl enable "$DOMAIN/$LABEL"

echo
echo "Installed. Verify with:"
echo "  launchctl list | grep smartmoney"
echo
echo "Logs:"
echo "  tail -f $(pwd)/digest/logs/digest.out.log"
echo "  tail -f $(pwd)/digest/logs/digest.err.log"
