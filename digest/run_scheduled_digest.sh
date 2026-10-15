#!/bin/bash
# Wrapper launchd actually invokes. Not the plist itself -- the plist just
# runs THIS script, on a fixed :00/:30 schedule, with zero knowledge of
# send times, frequencies, or credentials. This script's only job is to
# set up the same environment `streamlit run dashboard.py` normally gets
# (project directory, secrets) and then hand off to the real entry point.
#
# `source setup_env.sh` (not `./setup_env.sh`) so SMTP_HOST/SMTP_USER/
# SMTP_PASSWORD/ANTHROPIC_API_KEY/etc. actually land in THIS process's
# environment before Python starts -- launchd jobs otherwise start with a
# near-empty environment (no login shell, no .zshrc/.bash_profile), so
# without this the digest send would fail with "SMTP not fully
# configured" every single time despite working fine from a normal
# terminal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -f setup_env.sh ]; then
    source setup_env.sh
else
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ERROR setup_env.sh not found in $(pwd) -- SMTP/API credentials will be missing." >&2
fi

exec ./venv/bin/python3 -m digest.send
