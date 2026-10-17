"""
Accounts for the VoxCPM API server: users, login sessions, API keys and credits.

Stdlib only (sqlite3 + hashlib), so it adds no dependencies to the server.
- Web pages authenticate with an HttpOnly session cookie.
- API clients send `Authorization: Bearer <api key>` (works with the OpenAI SDK).
- Every generation costs 1 credit per input character, charged before synthesis
  and refunded if synthesis fails.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

WEB_DIR = Path(__file__).parent.resolve() / "web"
DB_PATH = Path(os.getenv("VOXCPM_DB_PATH", Path(__file__).parent.resolve() / "data" / "accounts.db"))
AUTH_ENABLED = os.getenv("VOXCPM_AUTH", "true").lower() in ("true", "1", "yes")
# "demo" grants purchased credits instantly (no real payment); anything else disables purchases.
PAYMENTS_MODE = os.getenv("VOXCPM_PAYMENTS", "off").lower()
SIGNUP_CREDITS = int(os.getenv("VOXCPM_SIGNUP_CREDITS", "2000"))
ADMIN_EMAILS = {e.strip().lower() for e in os.getenv("VOXCPM_ADMIN_EMAILS", "").split(",") if e.strip()}

SESSION_COOKIE = "voxcpm_session"
SESSION_TTL = 30 * 24 * 3600
PBKDF2_ITERATIONS = 600_000

# One-time credit packs. Credits are a single pool for speech, cloning and streaming.
PLANS = {
    "basic": {"name": "Basic", "credits": 30_000, "price_usd": 2.99, "blurb": "For light personal use"},
    "starter": {"name": "Starter", "credits": 70_000, "price_usd": 6.99, "blurb": "For individuals and side projects"},
    "studio": {"name": "Studio", "credits": 150_000, "price_usd": 11.99, "blurb": "For creators and growing teams", "popular": True},
    "studio_max": {"name": "Studio Max", "credits": 800_000, "price_usd": 49.99, "blurb": "For studios and high-volume apps"},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    credits INTEGER NOT NULL DEFAULT 0 CHECK (credits >= 0),
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    prefix TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    created_at INTEGER NOT NULL,
    last_used_at INTEGER
);
CREATE TABLE IF NOT EXISTS credit_ledger (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    delta INTEGER NOT NULL,
    balance INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_user ON credit_ledger(user_id, id);
"""


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:  # commit on success, rollback on error
            yield conn
    finally:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.executescript(SCHEMA)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PBKDF2_ITERATIONS).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    _, iterations, salt, digest = stored.split("$")
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations)).hex()
    return hmac.compare_digest(candidate, digest)


def public_user(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "credits": row["credits"],
        "is_admin": row["email"] in ADMIN_EMAILS,
        "created_at": row["created_at"],
    }


def adjust_credits(conn: sqlite3.Connection, user_id: int, delta: int, reason: str) -> int:
    """Atomically add (or remove) credits and log it. Returns the new balance; 402 if it would go negative."""
    rows = conn.execute(
        "UPDATE users SET credits = credits + ? WHERE id = ? AND credits + ? >= 0 RETURNING credits",
        (delta, user_id, delta),
    ).fetchall()  # fetchall finishes the statement before the next one runs
    if not rows:
        raise HTTPException(status_code=402, detail="Insufficient credits. Buy more at /pricing.")
    balance = rows[0]["credits"]
    conn.execute(
        "INSERT INTO credit_ledger (user_id, delta, balance, reason, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, delta, balance, reason, int(time.time())),
    )
    return balance


# ----------------- Auth dependencies -----------------

def _session_user(request: Request) -> Optional[sqlite3.Row]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with db() as conn:
        return conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at > ?",
            (_sha256(token), int(time.time())),
        ).fetchone()


def _api_key_user(request: Request) -> Optional[sqlite3.Row]:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    key_hash = _sha256(auth[7:].strip())
    with db() as conn:
        user = conn.execute(
            "SELECT u.* FROM api_keys k JOIN users u ON u.id = k.user_id WHERE k.key_hash = ?", (key_hash,)
        ).fetchone()
        if user is not None:
            conn.execute("UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?", (int(time.time()), key_hash))
        return user


def session_user(request: Request) -> sqlite3.Row:
    """Account pages/endpoints: session cookie only, so an API key cannot mint more keys or buy credits."""
    user = _session_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


def admin_user(user: sqlite3.Row = Depends(session_user)) -> sqlite3.Row:
    if user["email"] not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin only.")
    return user


def api_user(request: Request) -> Optional[sqlite3.Row]:
    """Generation endpoints: API key or session cookie. Returns None when VOXCPM_AUTH=false."""
    if not AUTH_ENABLED:
        return None
    user = _api_key_user(request) or _session_user(request)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key. Send 'Authorization: Bearer <key>' (create one at /dashboard).",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def charge(user: Optional[sqlite3.Row], text: str, endpoint: str) -> int:
    """Charge 1 credit per input character. Returns the amount charged (0 when auth is off)."""
    if user is None:
        return 0
    cost = max(1, len(text.strip()))
    with db() as conn:
        adjust_credits(conn, user["id"], -cost, f"Generation {endpoint} ({cost} chars)")
    return cost


def refund(user: Optional[sqlite3.Row], cost: int, endpoint: str):
    if user is None or cost == 0:
        return
    with db() as conn:
        adjust_credits(conn, user["id"], cost, f"Refund: {endpoint} failed")


# ----------------- Request models -----------------

class SignupRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., min_length=8, max_length=256)
    name: str = Field(default="", max_length=100)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=254)
    password: str = Field(..., max_length=256)


class CreateKeyRequest(BaseModel):
    name: str = Field(default="Default key", min_length=1, max_length=60)


class PurchaseRequest(BaseModel):
    plan: str


class AdjustCreditsRequest(BaseModel):
    delta: int = Field(..., ge=-10_000_000, le=10_000_000)
    reason: str = Field(default="Admin adjustment", min_length=1, max_length=200)


# ----------------- Routes -----------------

router = APIRouter()


def _start_session(request: Request, response: Response, user_id: int):
    token = secrets.token_urlsafe(32)
    now = int(time.time())
    with db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_sha256(token), user_id, now + SESSION_TTL),
        )
    response.set_cookie(
        SESSION_COOKIE, token, max_age=SESSION_TTL, httponly=True, samesite="lax",
        secure=request.url.scheme == "https",
    )


@router.post("/api/auth/signup", tags=["Accounts"])
def signup(req: SignupRequest, request: Request, response: Response):
    email = req.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    with db() as conn:
        try:
            user_id = conn.execute(
                "INSERT INTO users (email, name, password_hash, credits, created_at) VALUES (?, ?, ?, 0, ?)",
                (email, req.name.strip(), hash_password(req.password), int(time.time())),
            ).lastrowid
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")
        if SIGNUP_CREDITS > 0:
            adjust_credits(conn, user_id, SIGNUP_CREDITS, "Free signup credits")
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    _start_session(request, response, user_id)
    return public_user(user)


@router.post("/api/auth/login", tags=["Accounts"])
def login(req: LoginRequest, request: Request, response: Response):
    with db() as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (req.email.strip().lower(),)).fetchone()
    if user is None or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    _start_session(request, response, user["id"])
    return public_user(user)


@router.post("/api/auth/logout", tags=["Accounts"])
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with db() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_sha256(token),))
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@router.get("/api/me", tags=["Accounts"])
def me(user: sqlite3.Row = Depends(session_user)):
    return public_user(user)


@router.get("/api/keys", tags=["Accounts"])
def list_keys(user: sqlite3.Row = Depends(session_user)):
    with db() as conn:
        rows = conn.execute(
            "SELECT id, name, prefix, created_at, last_used_at FROM api_keys WHERE user_id = ? ORDER BY id DESC",
            (user["id"],),
        ).fetchall()
    return {"keys": [dict(r) for r in rows]}


@router.post("/api/keys", tags=["Accounts"])
def create_key(req: CreateKeyRequest, user: sqlite3.Row = Depends(session_user)):
    key = "vox_" + secrets.token_urlsafe(32)
    with db() as conn:
        key_id = conn.execute(
            "INSERT INTO api_keys (user_id, name, prefix, key_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user["id"], req.name.strip(), key[:12], _sha256(key), int(time.time())),
        ).lastrowid
    # The full key is only ever returned here; we store its hash.
    return {"id": key_id, "name": req.name.strip(), "prefix": key[:12], "key": key}


@router.delete("/api/keys/{key_id}", tags=["Accounts"])
def delete_key(key_id: int, user: sqlite3.Row = Depends(session_user)):
    with db() as conn:
        deleted = conn.execute("DELETE FROM api_keys WHERE id = ? AND user_id = ?", (key_id, user["id"])).rowcount
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found.")
    return {"ok": True}


@router.get("/api/credits/history", tags=["Accounts"])
def credit_history(user: sqlite3.Row = Depends(session_user)):
    with db() as conn:
        rows = conn.execute(
            "SELECT delta, balance, reason, created_at FROM credit_ledger WHERE user_id = ? ORDER BY id DESC LIMIT 100",
            (user["id"],),
        ).fetchall()
    return {"credits": user["credits"], "history": [dict(r) for r in rows]}


@router.get("/api/plans", tags=["Accounts"])
def plans():
    return {
        "payments_enabled": PAYMENTS_MODE == "demo",
        "signup_credits": SIGNUP_CREDITS,
        "plans": [{"id": k, **v} for k, v in PLANS.items()],
    }


@router.post("/api/purchase", tags=["Accounts"])
def purchase(req: PurchaseRequest, user: sqlite3.Row = Depends(session_user)):
    plan = PLANS.get(req.plan)
    if plan is None:
        raise HTTPException(status_code=404, detail="Unknown plan.")
    # shortcut: no payment provider yet; "demo" mode grants credits without charging. Plug Stripe/PayWay in here.
    if PAYMENTS_MODE != "demo":
        raise HTTPException(status_code=503, detail="Payments are not configured on this server yet.")
    with db() as conn:
        balance = adjust_credits(conn, user["id"], plan["credits"], f"Purchased {plan['name']} pack (demo)")
    return {"ok": True, "credits": balance}


@router.get("/api/admin/users", tags=["Admin"])
def admin_list_users(_: sqlite3.Row = Depends(admin_user)):
    with db() as conn:
        rows = conn.execute(
            """SELECT u.*, (SELECT COUNT(*) FROM api_keys k WHERE k.user_id = u.id) AS key_count,
                      (SELECT -COALESCE(SUM(delta), 0) FROM credit_ledger l
                       WHERE l.user_id = u.id AND (l.delta < 0 OR l.reason LIKE 'Refund:%')) AS credits_used
               FROM users u ORDER BY u.id DESC"""
        ).fetchall()
    return {"users": [{**public_user(r), "key_count": r["key_count"], "credits_used": r["credits_used"]} for r in rows]}


@router.post("/api/admin/users/{user_id}/credits", tags=["Admin"])
def admin_adjust_credits(user_id: int, req: AdjustCreditsRequest, admin: sqlite3.Row = Depends(admin_user)):
    with db() as conn:
        if conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail="User not found.")
        balance = adjust_credits(conn, user_id, req.delta, f"{req.reason} (by {admin['email']})")
    return {"ok": True, "credits": balance}


# ----------------- Pages -----------------

PAGES = {"/": "index.html", "/login": "auth.html", "/signup": "auth.html", "/dashboard": "dashboard.html", "/pricing": "pricing.html"}

def _page(file: str):
    # No parameters on the handler, so the file name can't be influenced by the request.
    def handler():
        return FileResponse(WEB_DIR / file)
    return handler


for _path, _file in PAGES.items():
    router.add_api_route(_path, _page(_file), methods=["GET"], include_in_schema=False)
