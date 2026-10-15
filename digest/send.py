"""python -m digest.send -- the standalone, headless scheduled-digest entry
point. Runs with NO Streamlit process: launchd invokes this directly, on
the hour and half-hour (see digest/com.smartmoney.digest.plist and
digest/install_launchd.sh) -- there is no send time anywhere in the plist.

2026-10-13 single-global-edition rebuild: every scheduling decision --
enabled, frequency, send time (ET), wait-for-Runners/wait-for-Day-
Predictions -- is read FRESH from the ADMIN account's own saved
email_config row on every tick via data_engine's existing
should_send_daily_digest_now(). Changing "Send time" in the admin's
Settings panel takes effect on the very next :00/:30 tick, with zero
changes to this script or the plist. Every OTHER active account's own
email_config is read by nothing here anymore -- it no longer drives a
send; see data_engine._get_edition_admin_email / build_digest_edition.

Uses the EXACT SAME get_or_build_todays_edition()/send_daily_digest_email()
(via digest.core.run_daily_digest_send) and SES path as the in-app "Send
Test Email Now" button -- this file adds no digest-building or SMTP logic
of its own, only the single-tick scheduling check + the headless readiness
computation (running today's Runners scan / Day Predictions, for the ADMIN
account only, if no UI is open to do it) + the healthchecks dead-man's-
switch ping.

Exit code 0: tick completed cleanly (whether or not anything was actually
due/sent -- "nothing was due" is success, not failure). Exit code 1: the
tick itself errored (e.g. the DB couldn't be opened, or no active admin
account exists) -- launchd's own StandardErrorPath log captures the
traceback."""
import logging
import sys

import data_engine as de
from digest import healthchecks
from digest.core import ensure_todays_readiness, run_daily_digest_send

LOG = logging.getLogger("digest.send")


def _configure_logging():
    # StreamHandler(stdout): the launchd plist redirects this process's stdout/stderr to timestamped log
    # files (StandardOutPath/StandardErrorPath) -- this formatter adds the timestamp INSIDE each line too,
    # since launchd's own redirect is a raw byte pipe with no timestamping of its own.
    if LOG.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False


# Decision reasons that mean "nothing to do this tick" -- no point logging these as loudly as an actual send.
_QUIET_REASONS = {"disabled", "weekend_skipped", "already_sent_today", "not_weekly_day",
                  "already_sent_this_week", "before_send_time"}


def tick(conn, now=None):
    """One launchd invocation. 2026-10-13: checks the ADMIN account's schedule only -- the single global
    edition has exactly one schedule now, not one per app user (see module docstring). Returns
    (sent_count, error_count), kept as a tuple of counts (0 or 1 each) for compatibility with main()'s
    exit-code logic, which predates this rebuild."""
    now = now if now is not None else de.now_et()
    LOG.info("tick starting (now_et=%s)", now.isoformat())
    sent_count, error_count = 0, 0

    email = de._get_edition_admin_email(conn)
    if not email:
        LOG.error("no active admin account exists -- nothing to send this tick.")
        return sent_count, error_count + 1

    # Macro Calendar upgrade (spec item 5b) -- runs unconditionally, BEFORE the digest's own quiet-reason
    # early return below: a High-impact release can land on a day the digest itself is skipped (weekend,
    # already sent today), and its one-time alert email must still fire independently of digest scheduling.
    # Own try/except -- a failure here must never block the normal digest tick that follows.
    try:
        n_alerts = de.check_and_send_macro_release_alerts(conn, now_et=now)
        if n_alerts:
            LOG.info("sent %d macro release alert(s)", n_alerts)
    except Exception as e:
        LOG.error("check_and_send_macro_release_alerts failed this tick: %s", e)

    try:
        decision = de.should_send_daily_digest_now(conn, email, now_et=now)
        if decision["reason"] in _QUIET_REASONS:
            LOG.info("tick complete: 0 sent, 0 error(s) (%s)", decision["reason"])
            return sent_count, error_count
        LOG.info("%s: %s", email, decision["reason"])

        if decision["reason"] == "waiting_for_readiness":
            # No UI is open to run the Runners scan / Day Predictions -- do it ourselves (real fetches,
            # real computation, same functions the dashboard uses), then re-check. If STILL not ready
            # (e.g. well before market open), this tick just retries at the next :00/:30 -- same
            # 90-minute grace window should_send_daily_digest_now already enforces, unchanged.
            ensure_todays_readiness(conn, email)
            decision = de.should_send_daily_digest_now(conn, email, now_et=now)
            LOG.info("%s: re-checked after headless readiness pass -> %s", email, decision["reason"])

        if not decision["ready"]:
            LOG.info("tick complete: 0 sent, 0 error(s) (not ready: %s)", decision["reason"])
            return sent_count, error_count

        result = run_daily_digest_send(conn, email, force=False)
    except Exception as e:
        LOG.error("%s: unexpected error this tick: %s", email, e)
        LOG.info("tick complete: 0 sent, 1 error(s)")
        return sent_count, error_count + 1

    if result.get("sent"):
        LOG.info("%s: sent -- %s", email, result.get("subject"))
        sent_count += 1
        # Test sends never reach here -- run_daily_digest_send(force=False) always send_type="scheduled".
        healthchecks.ping_success(conn, email)
    else:
        LOG.warning("%s: attempted but not sent -- %s", email, result.get("reason"))
        error_count += 1
        healthchecks.ping_fail(conn, email, message=str(result.get("reason")))

    LOG.info("tick complete: %d sent, %d error(s)", sent_count, error_count)
    return sent_count, error_count


def main():
    _configure_logging()
    try:
        de.init_db()
        conn = de.get_connection()
        sent_count, error_count = tick(conn)
    except Exception:
        LOG.exception("tick failed with an unhandled exception")
        sys.exit(1)
    # A tick with zero sends but also zero errors (nothing was due) is a CLEAN exit -- healthchecks' own
    # cron-schedule + grace period is what decides whether a ping was actually expected around this tick,
    # not this script deciding "nothing sent" means "something's wrong." A tick where something WAS due and
    # failed to send exits non-zero (visible in the log files / `echo $?` when run manually), even though
    # ping_fail() already told healthchecks immediately rather than waiting out the exit-code path.
    sys.exit(1 if error_count else 0)


if __name__ == "__main__":
    main()
