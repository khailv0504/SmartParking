"""Protocol regression tests using existing cloud/tests/conftest.py fixtures."""
import pytest
from starlette.websockets import WebSocketDisconnect
from cloud.main import app
from cloud.auth import SESSION_COOKIE, create_admin_session

FRAME = b"\xff\xd8ws-frame\xff\xd9"
HEADERS = {
    "Authorization": "Bearer test-device-token-at-least-24-characters",
    "X-Device-ID": "raspberrypi_01",
}


def test_ws_requires_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/edge/live-ws") as ws:
            ws.send_bytes(FRAME)


def test_ws_updates_latest_frame_and_rejects_text(client):
    with client.websocket_connect("/api/v1/edge/live-ws", headers=HEADERS) as ws:
        ws.send_bytes(FRAME)
        ws.send_text("bad")
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
        assert exc.value.code == 1003
    with app.state.session_factory() as db:
        token, _ = create_admin_session(db, app.state.settings, "admin", None)
    client.cookies.set(SESSION_COOKIE, token)
    assert client.get("/api/live/frame.jpg").content == FRAME
    info = client.get("/api/live/status").json()
    assert info["incoming_fps"] >= 0.0


def test_ws_enforces_frame_limit(client):
    with client.websocket_connect("/api/v1/edge/live-ws", headers=HEADERS) as ws:
        ws.send_bytes(b"\xff\xd8" + b"x" * 200_001 + b"\xff\xd9")
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
        assert exc.value.code == 1009
