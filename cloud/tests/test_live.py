"""Minimal regression tests for authenticated, ephemeral camera frames."""
from cloud.main import app
from cloud.auth import SESSION_COOKIE, create_admin_session

FRAME = b"\xff\xd8test-preview-payload\xff\xd9"
TOKEN = "test-device-token-at-least-24-characters"


def test_live_frame_requires_bearer(client):
    res = client.post("/api/v1/edge/live-frame", content=FRAME,
                      headers={"Content-Type": "image/jpeg", "X-Device-ID": "raspberrypi_01"})
    assert res.status_code == 401


def test_upload_and_admin_only_read(client):
    res = client.post("/api/v1/edge/live-frame", content=FRAME,
        headers={"Content-Type": "image/jpeg", "X-Device-ID": "raspberrypi_01",
                 "Authorization": "Bearer " + TOKEN})
    assert res.status_code == 204
    assert client.get("/api/live/frame.jpg").status_code == 401
    with app.state.session_factory() as db:
        token, csrf = create_admin_session(db, app.state.settings, "admin", None)
    client.cookies.set(SESSION_COOKIE, token)
    assert client.get("/api/live/frame.jpg").content == FRAME
    state = client.get("/api/live/status").json()
    assert state["available"] is True
    assert state["device_id"] == "raspberrypi_01"


def test_reject_invalid_image_and_large_payload(client):
    headers = {"Authorization": "Bearer " + TOKEN, "X-Device-ID": "raspberrypi_01",
               "Content-Type": "image/jpeg"}
    assert client.post("/api/v1/edge/live-frame", headers=headers, content=b"oops").status_code == 422
    assert client.post("/api/v1/edge/live-frame", headers=headers,
                       content=b"\xff\xd8" + b"x" * 200_001 + b"\xff\xd9").status_code == 413
