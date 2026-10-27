from __future__ import annotations

from cloud.auth import SESSION_COOKIE, sha256
from cloud.main import app
from cloud.models import AdminSession


def test_dashboard_redirects_without_login(client):
    for path in ("/", "/history", "/inside", "/rfid", "/device", "/plates"):
        response = client.get(path)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_login_wrong_password(client):
    response = client.post(
        "/auth/login", data={"username": "admin", "password": "wrong-password"}
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/login?error=1"
    assert SESSION_COOKIE not in response.cookies


def test_login_dashboard_and_logout_revokes_session(client):
    login = client.post(
        "/auth/login",
        data={"username": "admin", "password": "correct-horse-battery-staple"},
    )
    assert login.status_code == 303
    token = login.cookies[SESSION_COOKIE]
    csrf = login.cookies["sp_csrf"]
    assert client.get("/").status_code == 200

    logout = client.post("/auth/logout", data={"csrf_token": csrf})
    assert logout.status_code == 303
    with app.state.session_factory() as db:
        row = db.get(AdminSession, sha256(token))
        assert row.revoked_at is not None

    # Ngay cả khi phát lại cookie đã bị đánh cắp, session đã thu hồi vẫn vô hiệu.
    client.cookies.set(SESSION_COOKIE, token)
    assert client.get("/").status_code == 303


def test_logout_rejects_csrf(client):
    client.post(
        "/auth/login",
        data={"username": "admin", "password": "correct-horse-battery-staple"},
    )
    assert client.post("/auth/logout", data={"csrf_token": "wrong"}).status_code == 403


def test_device_token_does_not_grant_admin_access(client):
    response = client.get(
        "/",
        headers={"Authorization": "Bearer test-device-token-at-least-24-characters"},
    )
    assert response.status_code == 303
