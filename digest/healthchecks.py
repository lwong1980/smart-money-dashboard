"""Dead-man's-switch integration with healthchecks.io (free tier) -- PER
USER, matching the rest of this app's per-user architecture: each user's
own `email_config` row carries their own check's ping URL, created/updated
via the healthchecks Management API whenever they click "Save Email Digest
settings" so the check's expected cron schedule always matches what's
actually configured in the UI, with no manual healthchecks.io dashboard
work required after the first-ever setup.

Requires HEALTHCHECKS_API_KEY (a project-wide API key -- issuing one is an
account-level action, so it's an env var, not a per-user DB value; see
setup_env.sh). Every function here fails SOFT and fast when that key isn't
set: it logs once and returns None/False, never raises, and never blocks
or delays sending the actual email. A dead-man's switch that can crash the
thing it's watching would be worse than not having one."""
import os

import requests

import data_engine as de

HEALTHCHECKS_API_BASE = "https://healthchecks.io/api/v3"
# 60-minute grace, per spec -- generous enough to absorb the 90-minute readiness-retry window's LATER ticks
# without a false alarm, while still catching "nothing sent all morning."
DEFAULT_GRACE_SECONDS = 3600
_WEEKDAY_TO_CRON = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}  # our Mon=0..Fri=4 -> cron's Mon=1..Fri=5


def _api_key():
    key = os.environ.get("HEALTHCHECKS_API_KEY")
    if not key:
        de.digest_log.info(
            "[HEALTHCHECKS] HEALTHCHECKS_API_KEY not set -- dead-man's-switch integration is disabled "
            "(email sending is unaffected). Set it in the environment to enable."
        )
    return key


def cron_expr_for_config(config):
    """The cron expression matching this user's OWN Frequency + Send time (ET) -- exactly what healthchecks
    should expect a ping around. 'weekdays_only' is treated the same as 'daily' (both = Mon-Fri), matching
    should_send_daily_digest_now's own legacy-value handling."""
    hour, minute = config["send_time_et"].split(":")
    if config["frequency"] == "weekly":
        dow = _WEEKDAY_TO_CRON.get(config["weekly_day"], "1")
        return f"{int(minute)} {int(hour)} * * {dow}"
    return f"{int(minute)} {int(hour)} * * 1-5"


def sync_check_schedule(conn, user_email):
    """Creates (first time) or updates (every later call) this user's healthchecks check so its expected
    schedule matches their CURRENT saved email_config -- called from "Save Email Digest settings", so the UI
    is always the source of truth and no plist/dashboard.io manual edit is ever needed. Stores the check's
    ping URL back onto email_config (as a user_preferences entry -- see get_user_pref/set_user_pref -- rather
    than a new email_config column, since it's an operational detail, not a user-facing setting). Returns the
    ping URL on success, None if disabled/failed (logged, never raised)."""
    key = _api_key()
    if not key:
        return None
    config = de.get_email_config(conn, user_email)
    if not config["enabled"] or config["frequency"] == "off":
        # No point keeping a check armed for a newsletter that's off -- healthchecks would just alert forever.
        # Pausing (not deleting) preserves the check's history/uuid for whenever it's re-enabled.
        uuid = de.get_user_pref(conn, user_email, "healthcheck_uuid")
        if uuid:
            try:
                requests.post(f"{HEALTHCHECKS_API_BASE}/checks/{uuid}/pause", headers={"X-Api-Key": key}, timeout=10)
            except Exception as e:
                de.digest_log.warning("[HEALTHCHECKS] %s: failed to pause check: %s", user_email, e)
        return None
    payload = {
        "name": f"Smart Money digest — {user_email}",
        "tags": "smart-money-digest",
        "unique": ["name"],   # re-POSTing with the same name UPDATES the existing check instead of duplicating it
        "schedule": cron_expr_for_config(config),
        "tz": "America/New_York",
        "grace": DEFAULT_GRACE_SECONDS,
    }
    try:
        resp = requests.post(f"{HEALTHCHECKS_API_BASE}/checks/", json=payload, headers={"X-Api-Key": key}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        de.digest_log.warning("[HEALTHCHECKS] %s: failed to sync check schedule: %s", user_email, e)
        return None
    ping_url, uuid = data.get("ping_url"), data.get("uuid")
    if ping_url:
        de.set_user_pref(conn, user_email, "healthcheck_ping_url", ping_url)
        de.set_user_pref(conn, user_email, "healthcheck_uuid", uuid)
        de.digest_log.info("[HEALTHCHECKS] %s: check schedule synced (%s)", user_email, payload["schedule"])
    return ping_url


def ping_success(conn, user_email):
    """Pings this user's healthchecks check to say a SCHEDULED send just succeeded. Test sends must never call
    this (see digest/send.py -- only the scheduled path does). No-op, logged, if healthchecks isn't configured
    or this user has no synced check yet."""
    if not _api_key():
        return
    ping_url = de.get_user_pref(conn, user_email, "healthcheck_ping_url")
    if not ping_url:
        de.digest_log.info("[HEALTHCHECKS] %s: no synced check yet (save Email Digest settings once to create "
                           "one) -- skipping ping.", user_email)
        return
    try:
        requests.get(ping_url, timeout=10)
    except Exception as e:
        # A failed PING must never fail the send it's reporting on -- the email already went out.
        de.digest_log.warning("[HEALTHCHECKS] %s: ping failed (%s)", user_email, e)


def ping_fail(conn, user_email, message=""):
    """Explicitly reports a failure to healthchecks (POST /fail) -- used only where a scheduled send was
    ATTEMPTED and failed, so healthchecks can alert immediately rather than waiting out the full grace
    period. Never called for 'nothing was due this tick' (that's not a failure)."""
    if not _api_key():
        return
    ping_url = de.get_user_pref(conn, user_email, "healthcheck_ping_url")
    if not ping_url:
        return
    try:
        requests.post(f"{ping_url}/fail", data=message[:1000], timeout=10)
    except Exception as e:
        de.digest_log.warning("[HEALTHCHECKS] %s: fail-ping failed (%s)", user_email, e)
