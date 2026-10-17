"""Tests for accounts.py: signup/login, API keys, credit charging, purchases and admin."""

import sys
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accounts  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "DB_PATH", tmp_path / "accounts.db")
    monkeypatch.setattr(accounts, "PBKDF2_ITERATIONS", 1_000)  # keep tests fast
    monkeypatch.setattr(accounts, "SIGNUP_CREDITS", 100)
    monkeypatch.setattr(accounts, "AUTH_ENABLED", True)
    monkeypatch.setattr(accounts, "PAYMENTS_MODE", "off")
    monkeypatch.setattr(accounts, "ADMIN_EMAILS", {"admin@example.com"})
    accounts.init_db()

    app = FastAPI()
    app.include_router(accounts.router)

    @app.post("/gen")
    def gen(body: dict, user=Depends(accounts.api_user)):
        cost = accounts.charge(user, body["text"], "/gen")
        if body.get("fail"):
            accounts.refund(user, cost, "/gen")
            raise HTTPException(status_code=500, detail="boom")
        return {"cost": cost}

    return app


def signup(app, email="user@example.com"):
    client = TestClient(app)
    res = client.post("/api/auth/signup", json={"email": email, "password": "correct horse", "name": "Ann"})
    assert res.status_code == 200, res.text
    return client


def test_signup_login_and_duplicate(app):
    client = signup(app)
    assert client.get("/api/me").json()["credits"] == 100

    assert TestClient(app).post("/api/auth/signup", json={"email": "USER@example.com", "password": "12345678"}).status_code == 409
    assert TestClient(app).post("/api/auth/login", json={"email": "user@example.com", "password": "wrong pass"}).status_code == 401
    assert TestClient(app).post("/api/auth/login", json={"email": "User@Example.com", "password": "correct horse"}).status_code == 200

    client.post("/api/auth/logout")
    assert client.get("/api/me").status_code == 401


def test_api_key_charges_refunds_and_revokes(app):
    client = signup(app)
    key = client.post("/api/keys", json={"name": "ci"}).json()
    assert key["key"].startswith("vox_")
    assert "key" not in client.get("/api/keys").json()["keys"][0]  # full key is never listed again

    bearer = TestClient(app, headers={"Authorization": f"Bearer {key['key']}"})
    assert TestClient(app).post("/gen", json={"text": "hi"}).status_code == 401
    assert bearer.post("/gen", json={"text": "hello"}).json() == {"cost": 5}
    assert client.get("/api/me").json()["credits"] == 95

    # Failed generation is refunded
    assert bearer.post("/gen", json={"text": "abc", "fail": True}).status_code == 500
    assert client.get("/api/me").json()["credits"] == 95

    # Not enough credits: rejected, balance untouched
    assert bearer.post("/gen", json={"text": "x" * 96}).status_code == 402
    assert client.get("/api/me").json()["credits"] == 95

    # An API key can't manage the account
    assert bearer.get("/api/keys").status_code == 401

    history = client.get("/api/credits/history").json()["history"]
    assert [h["delta"] for h in history] == [3, -3, -5, 100]
    assert history[0]["balance"] == 95

    assert client.delete(f"/api/keys/{key['id']}").status_code == 200
    assert bearer.post("/gen", json={"text": "hello"}).status_code == 401


def test_users_cannot_touch_each_others_keys(app):
    alice = signup(app, "alice@example.com")
    bob = signup(app, "bob@example.com")
    key_id = alice.post("/api/keys", json={"name": "a"}).json()["id"]
    assert bob.delete(f"/api/keys/{key_id}").status_code == 404
    assert len(alice.get("/api/keys").json()["keys"]) == 1


def test_purchase_modes(app, monkeypatch):
    client = signup(app)
    assert client.post("/api/purchase", json={"plan": "basic"}).status_code == 503
    monkeypatch.setattr(accounts, "PAYMENTS_MODE", "demo")
    assert client.post("/api/purchase", json={"plan": "nope"}).status_code == 404
    res = client.post("/api/purchase", json={"plan": "basic"})
    assert res.json()["credits"] == 100 + accounts.PLANS["basic"]["credits"]
    assert TestClient(app).post("/api/purchase", json={"plan": "basic"}).status_code == 401


def test_admin_user_management(app):
    user = signup(app, "user@example.com")
    admin = signup(app, "admin@example.com")
    assert user.get("/api/admin/users").status_code == 403

    users = admin.get("/api/admin/users").json()["users"]
    target = next(u for u in users if u["email"] == "user@example.com")
    assert admin.post(f"/api/admin/users/{target['id']}/credits", json={"delta": 50, "reason": "bonus"}).json()["credits"] == 150
    assert admin.post(f"/api/admin/users/{target['id']}/credits", json={"delta": -1000}).status_code == 402
    assert admin.post("/api/admin/users/9999/credits", json={"delta": 1}).status_code == 404
    assert user.get("/api/me").json()["credits"] == 150


def test_auth_disabled_is_free(app, monkeypatch):
    monkeypatch.setattr(accounts, "AUTH_ENABLED", False)
    assert TestClient(app).post("/gen", json={"text": "hello"}).json() == {"cost": 0}
