"""Authentication gate for the Smart Money dashboard.

Uses Streamlit's NATIVE OIDC support (st.login / st.user / st.logout,
Streamlit >= 1.42, plus the `Authlib` package) with Google as the identity
provider -- Streamlit does the OAuth redirect dance, token exchange and the
signed identity cookie; this module only decides what to do with the
identity it hands back.

Two separate questions are answered here, deliberately kept apart:
  1. AUTHENTICATION -- "is this really the Google account it claims to be?"
     Answered entirely by Google + Streamlit (st.user.is_logged_in).
  2. AUTHORIZATION  -- "is that account allowed to use THIS app?" Answered by
     data_engine.authorize_login against the `users` table (admin-managed
     allowlist; an admin bootstrap gated on BOOTSTRAP_ADMIN_EMAIL).
A valid Google login for an unregistered account is refused, never
auto-created.

`read_identity` is the single seam that touches st.user, so tests can
substitute an identity without any backdoor in the production code path.
"""
import streamlit as st

import data_engine as de

# Streamlit's [auth] section keys for a single (default) OIDC provider.
REQUIRED_AUTH_KEYS = ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"
DEFAULT_REDIRECT_URI = "http://localhost:8501/oauth2callback"

SESSION_AUTH_EMAIL_KEY = "_auth_email"


def missing_auth_config():
    """Names of required [auth] keys absent from .streamlit/secrets.toml
    (empty list == fully configured). A key still holding the example
    file's placeholder text counts as missing, so copying the example
    without filling it in is caught here rather than at Google's error
    page."""
    try:
        section = st.secrets.get("auth", None)
    except Exception:  # no secrets.toml at all
        section = None
    if not section:
        return list(REQUIRED_AUTH_KEYS)
    missing = []
    for key in REQUIRED_AUTH_KEYS:
        value = str(section.get(key, "") or "")
        if not value or value.startswith("REPLACE_ME"):
            missing.append(key)
    return missing


def read_identity():
    """The authenticated Google identity, or {"is_logged_in": False}.

    Fails CLOSED: a token with no `email_verified` claim is treated as
    unverified (Google always sends it).

    NOTE on token expiry: Streamlit does not check the ID token's `exp`
    claim, and this deliberately does not either -- Google ID tokens live
    only 1 hour, so enforcing it would sign everyone out hourly. Access is
    instead revoked immediately by the `active` flag / role, which
    require_login re-reads from the DB on every script run; the session
    itself lasts as long as Streamlit's signed identity cookie.
    """
    try:
        user = st.user
        if not user.get("is_logged_in", False):
            return {"is_logged_in": False}
        return {
            "is_logged_in": True,
            "email": de.normalize_email(user.get("email")),
            "name": user.get("name") or None,
            "email_verified": user.get("email_verified") is True,
        }
    except Exception:
        return {"is_logged_in": False}


def _centered(render):
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        render()


def _render_setup_needed(missing):
    def body():
        st.markdown("### 📡 SMART MONEY INTELLIGENCE")
        st.error("Google sign-in isn't configured yet, so the dashboard is locked.")
        st.markdown(
            "Create `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`) and fill in the "
            f"missing values: **{', '.join(missing)}**.\n\n"
            f"In Google Cloud Console, the OAuth client's **Authorized redirect URI** must be exactly:\n\n"
            f"`{DEFAULT_REDIRECT_URI}`\n\n"
            "Then restart the app. See the example file for the full setup steps."
        )

    _centered(body)


def _render_login_screen():
    def body():
        st.markdown("### 📡 SMART MONEY INTELLIGENCE")
        st.caption("OPTIONS FLOW · DARK POOLS · CONGRESS · INSIDERS · 13F · POLYMARKET")
        st.info("Sign in with your Google account to continue.")
        if st.button("Sign in with Google", type="primary", key="auth_login_btn", width="stretch"):
            st.login()

    _centered(body)


def _render_denied(message, status):
    def body():
        st.markdown("### 📡 SMART MONEY INTELLIGENCE")
        if status == "no_admin_configured":
            st.error(message, icon="🛠️")
        else:
            st.error(message, icon="🚫")
        if st.button("Sign out", key="auth_denied_logout_btn"):
            st.session_state.pop(SESSION_AUTH_EMAIL_KEY, None)
            st.logout()

    _centered(body)


def require_login(conn):
    """Gate the ENTIRE app. Returns the current user dict (email,
    display_name, role, ...) or halts the script with st.stop() after
    rendering the appropriate screen (setup-needed / login / denied). Call
    this before rendering anything else.

    Role/active status are re-read from the DB on EVERY script run, never
    cached in session_state -- so an admin demoting or deactivating someone
    takes effect on their very next interaction, not their next login.
    """
    missing = missing_auth_config()
    if missing:
        _render_setup_needed(missing)
        st.stop()

    identity = read_identity()
    if not identity.get("is_logged_in"):
        st.session_state.pop(SESSION_AUTH_EMAIL_KEY, None)
        _render_login_screen()
        st.stop()

    email = identity["email"]
    if st.session_state.get(SESSION_AUTH_EMAIL_KEY) != email:
        # First run for this session/identity: a real "login event"
        # (updates last_login_at; is the only place the bootstrap-admin
        # gate can fire).
        result = de.authorize_login(
            conn, email, email_verified=identity.get("email_verified", False), display_name=identity.get("name"),
        )
        if result["status"] != "ok":
            _render_denied(result["message"], result["status"])
            st.stop()
        st.session_state[SESSION_AUTH_EMAIL_KEY] = email
        return result["user"]

    user = de.get_user(conn, email)
    if not user or not user["active"]:
        st.session_state.pop(SESSION_AUTH_EMAIL_KEY, None)
        _render_denied(de.DEACTIVATED_MESSAGE if user else de.NOT_REGISTERED_MESSAGE,
                       "deactivated" if user else "not_registered")
        st.stop()
    return user
