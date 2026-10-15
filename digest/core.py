"""Digest send orchestration -- shared, unchanged logic between the
Streamlit app (dashboard.py imports these two functions rather than
defining them) and the headless launchd path (digest/send.py). Moved here
verbatim from dashboard.py; no behavior changed by the move.

Pure Python + charts.py (Plotly/kaleido, no Streamlit) + data_engine.py --
importable with no Streamlit process running, which is the whole point:
digest/send.py needs this exact orchestration without dragging in
dashboard.py's own top-level Streamlit calls."""
import threading

import charts
import data_engine as de

# One lock per user: two browser tabs (or a fragment tick racing a full
# page load, or -- now -- the launchd job racing an open browser tab) can
# both see "due" for the same user in the same instant -- before either
# has recorded last_sent_date -- and would otherwise both send. A non-
# blocking acquire means the second caller just skips; the first one is
# already handling it. NOTE: this lock is process-local -- it only
# prevents a double-send from two callers in the SAME process (e.g. two
# browser tabs served by one `streamlit run`). It can't prevent a race
# between the launchd process and a separately-running `streamlit run`
# process; email_config.last_sent_date (checked inside
# send_daily_digest_email, read-then-written under the same connection's
# transaction) is what actually prevents a genuine cross-process
# double-send -- this lock is just cheap insurance against the common
# same-process case.
_DIGEST_SEND_LOCKS = {}
_DIGEST_SEND_LOCKS_GUARD = threading.Lock()


def run_daily_digest_send(conn, user_email, force=False, recipient_email=None):
    """The single orchestration entry point for an actual digest send --
    called by the Streamlit app's background scan fragment/full-load
    catch-up (an automatic, schedule-driven send), by SETTINGS' "Send Test
    Email Now" button (force=True, immediate, optionally to a not-yet-
    saved `recipient_email` typed into the input), AND by digest/send.py's
    headless launchd tick (force=False, after it has itself confirmed
    readiness -- see that file).

    2026-10-13 rebuild: `user_email` is now always the ADMIN account (the
    single global edition's owner) -- callers no longer loop per app user.
    Content comes from get_or_build_todays_edition() (data_engine.py) --
    builds today's edition, at most ONE real Claude call, only if it
    doesn't already exist; every other call this same day (another Preview
    click, a retried send, Send to Substack) reads the same stored row
    back. Chart images (charts.py's build_*_chart_* wrappers, Plotly +
    kaleido, no Streamlit, never an API call) are still built here each
    send -- cheap/free, not worth persisting -- from the edition's OWN
    featured_stock dict, so the chart numbers can never disagree with the
    text around it.

    `force=False` (the automatic/headless path) first asks should_send_
    daily_digest_now() -- a pure yes/no decision with no chart/API cost of
    its own -- and does nothing further unless it says ready. force=True is
    ONLY ever "Send Test Email Now" -- see send_type below.

    send_type is derived from `force`, never passed separately, so the
    in-app path and the launchd job can never disagree about which
    bookkeeping columns a given send updates: force=True (the test
    button's only use) -> send_type="test" (last_test_* columns only,
    never touches last_sent_date -- "test sends never count as the daily
    send"). force=False (automatic/headless) -> send_type="scheduled"."""
    incomplete_reasons = None
    send_type = "test" if force else "scheduled"
    if not force:
        decision = de.should_send_daily_digest_now(conn, user_email)
        if not decision["ready"]:
            return {"sent": False, "reason": decision["reason"]}
        incomplete_reasons = decision["incomplete_reasons"]

    edition = de.get_or_build_todays_edition(conn)
    featured_stock = (edition.get("sections") or {}).get("featured_stock")
    featured_ticker = edition.get("featured_ticker")
    chart_cid = None
    inline_images = {}
    # ONE snapshot cache for the whole send: every recipient's own re-render (render_edition_for_recipient,
    # inside send_daily_digest_email's loop) shares it, so nothing in the email can disagree with anything
    # else, and price data isn't re-fetched once per recipient.
    snaps = de.SnapshotCache(conn)
    if featured_ticker and featured_stock:
        chart_bytes, chart_cid = charts.build_featured_stock_chart_bytes(featured_ticker, conn)
        if chart_bytes:
            inline_images[chart_cid] = chart_bytes
        inline_images.update(charts.build_analyst_chart_images(featured_ticker, featured_stock, conn))
        inline_images.update(charts.build_swing_chart_image(featured_stock))

    result = de.send_daily_digest_email(
        conn, user_email, recipient_email=recipient_email, force=True, incomplete_reasons=incomplete_reasons,
        edition=edition, chart_cid=chart_cid,
        inline_images=(inline_images or None), snapshots=snaps,
        # only a scheduled send spends a one-time featured override; "Send Test Email Now" (force=True) never does
        consume_override=not force, send_type=send_type,
    )
    if incomplete_reasons:
        result["sent_incomplete"] = True
        result["incomplete_reasons"] = incomplete_reasons
    return result


def run_digests_for_all_users(conn, force=False):
    """The AUTOMATIC digest path. NAME KEPT for call-site compatibility
    (dashboard.py calls this from several places) -- but since the
    2026-10-13 single-global-edition rebuild, there is only ever ONE
    account this can mean anything for: the admin, whose watchlist/
    Runners/Day Predictions/Watchlist Signals/featured-stock rule the one
    shared edition is built from, and whose email_config (enabled toggle,
    frequency, send time, wait-for flags) is the only schedule that drives
    a real send. Every other active account's OWN email_config is left in
    place, unused -- see data_engine._get_edition_admin_email /
    build_digest_edition's docstrings.

    Returns {admin_email: result_dict}, or {} if no active admin account
    exists (should never happen in practice)."""
    admin_email = de._get_edition_admin_email(conn)
    if not admin_email:
        print("[run_digests_for_all_users] no active admin account exists -- nothing to send.")
        return {}
    with _DIGEST_SEND_LOCKS_GUARD:
        lock = _DIGEST_SEND_LOCKS.setdefault(admin_email, threading.Lock())
    if not lock.acquire(blocking=False):
        return {}
    try:
        result = run_daily_digest_send(conn, admin_email, force=force)
    except Exception as _err:
        print(f"[run_digests_for_all_users] {admin_email}: unexpected error: {_err}")
        result = {"sent": False, "reason": f"error: {_err}"}
    finally:
        lock.release()
    return {admin_email: result}


def ensure_todays_readiness(conn, user_email):
    """Headless equivalent of "have the RUNNERS tab open" (2e): if this
    user's config waits on the Runners scan and/or Day Predictions and
    today's results don't exist yet, COMPUTE them now -- there is no UI
    open to do it, so the launchd job must run the exact same
    computations dashboard.py's RUNNERS tab would, via the exact same
    data_engine.py functions (compute_and_log_runner_scan,
    compute_day_target + log_day_prediction), never a reimplementation.

    This does real work (fetches live data, can make network calls) --
    call it only after should_send_daily_digest_now() says
    "waiting_for_readiness" for this user, not on every tick.

    Returns nothing; callers re-check should_send_daily_digest_now()
    afterward to see whether readiness is now satisfied."""
    config = de.get_email_config(conn, user_email)
    if not (config["wait_for_runners"] or config["wait_for_predictions"]):
        return
    watchlist = de.load_watchlist(conn, user_email)
    if de.get_todays_runner_tickers(conn, user_email) is None:
        de.compute_and_log_runner_scan(conn, user_email, watchlist)
    if not config["wait_for_predictions"]:
        return
    runner_tickers = de.get_todays_runner_tickers(conn, user_email) or []
    pinned = [t for t in de.load_pinned_runners(conn, user_email) if t in watchlist and t not in runner_tickers]
    today_iso = de.today_et().isoformat()
    for t in runner_tickers + pinned:
        existing = conn.execute(
            "SELECT 1 FROM day_predictions WHERE ticker=? AND session_date=? AND COALESCE(source,'live')='live'",
            (t, today_iso),
        ).fetchone()
        if existing:
            continue
        try:
            target = de.compute_day_target(t, conn)
            de.log_day_prediction(conn, t, target)
        except Exception as e:
            de.digest_log.warning("[DIGEST] %s: headless Day Prediction failed for %s (%s)", user_email, t, e)
