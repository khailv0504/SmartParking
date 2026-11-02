"""Low-bandwidth HTTPS preview uplink reusing the camera owned by edge.main.

No second VideoCapture, no second YOLO, no recording on Render.
"""
from __future__ import annotations

import logging
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2

from .alpr_preprocess.sources import CameraSource

LOG = logging.getLogger(__name__)


class LiveFrameUplink:
    def __init__(
        self,
        camera: CameraSource,
        events_url: str,
        token: str,
        device_id: str,
        fps: float = 1.0,
        max_width: int = 640,
        quality: int = 55,
    ) -> None:
        if not (0.2 <= fps <= 2.0):
            raise ValueError("Live FPS must be between 0.2 and 2.0 on Pi 4")
        if not (160 <= max_width <= 960) or not (30 <= quality <= 85):
            raise ValueError("Invalid preview resolution or JPEG quality")
        parsed = urlparse(events_url)
        if parsed.scheme != "https" or not parsed.netloc or not token:
            raise ValueError("Live uplink requires Cloud HTTPS URL and Device Token")
        self.url = parsed._replace(
            path="/api/v1/edge/live-frame", params="", query="", fragment=""
        ).geturl()
        self.camera = camera
        self.token = token
        self.device_id = device_id
        self.interval = 1 / fps
        self.width = max_width
        self.quality = quality
        self._lock = threading.Lock()
        self._last_event: str = ""
        self._event_at: float = 0.0

    def note_event(self, outcome: object) -> None:
        action = str(getattr(outcome, "action", ""))[:16]
        plate = str(getattr(outcome, "plate", "") or "-")[:20]
        with self._lock:
            self._last_event = f"LAST RFID: {action} / PLATE: {plate}"
            self._event_at = time.monotonic()

    def _encode(self, frame) -> bytes:
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(
                frame, (self.width, max(1, round(h * self.width / w))),
                interpolation=cv2.INTER_AREA,
            )
        with self._lock:
            recent = (self._last_event if time.monotonic() - self._event_at < 12 else "")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (20, 25, 32), -1)
        cv2.putText(frame, "SMARTPARKING | LIVE CAMERA", (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        if recent:
            y = frame.shape[0]
            cv2.rectangle(frame, (0, y - 30), (frame.shape[1], y), (20, 25, 32), -1)
            cv2.putText(frame, recent, (8, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError("Cannot encode camera frame")
        return buffer.tobytes()

    def run_forever(self, stop: threading.Event) -> None:
        """Best-effort uploader: failures never block RFID/YOLO/SQLite."""
        previous_sequence = None
        failures = 0
        while not stop.is_set():
            start = time.monotonic()
            try:
                packet = self.camera.latest_frame()
                if packet is not None and packet.frame_id != previous_sequence:
                    previous_sequence = packet.frame_id
                    jpeg = self._encode(packet.frame)
                    if len(jpeg) > 200_000:
                        LOG.warning("Skipping oversized preview frame: %s bytes", len(jpeg))
                    else:
                        req = Request(
                            self.url,
                            data=jpeg,
                            headers={
                                "Content-Type": "image/jpeg",
                                "Authorization": f"Bearer {self.token}",
                                "X-Device-ID": self.device_id,
                            },
                            method="POST",
                        )
                        with urlopen(req, timeout=5) as response:
                            if response.status != 204:
                                raise RuntimeError(f"Unexpected live upload status {response.status}")
                        failures = 0
            except (OSError, URLError, HTTPError, RuntimeError, ValueError) as exc:
                failures += 1
                if failures == 1 or failures % 20 == 0:
                    LOG.warning("Live preview uplink unavailable (attempt %s): %s", failures, exc)
            stop.wait(max(0.05, self.interval - (time.monotonic() - start)))
