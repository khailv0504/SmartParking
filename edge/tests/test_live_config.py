"""No physical camera or Cloud required."""
from types import SimpleNamespace
import numpy as np
import pytest
from edge.live_uplink import LiveFrameUplink


class FakeCamera:
    def latest_frame(self):
        return SimpleNamespace(frame=np.zeros((720,1280,3),dtype=np.uint8),frame_id=1)


def test_ws_fps_and_jpeg_encoding():
    url = "https://smartparking-cloud.onrender.com/api/v1/edge/events"
    ws = LiveFrameUplink(FakeCamera(), url, "device-token", "raspberrypi_01", fps=10, transport="ws")
    assert ws.ws_url == "wss://smartparking-cloud.onrender.com/api/v1/edge/live-ws"
    jpeg = ws._encode(FakeCamera().latest_frame().frame)
    assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
    with pytest.raises(ValueError):
        LiveFrameUplink(FakeCamera(), url, "device-token", "raspberrypi_01", fps=10, transport="http")
