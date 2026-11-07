"""Independent low-priority live uploader using the existing shared CameraSource.

Transport 'http' is the backward-compatible 1-2 FPS JPEG preview.
Transport 'ws' streams binary JPEG frames on one secure WebSocket (up to 15 FPS).
No second VideoCapture, no YOLO, no local recording or unbounded queue.
"""
from __future__ import annotations

import json
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
        transport: str = "http",
    ) -> None:
        if transport not in ("http", "ws"):
            raise ValueError("Live transport must be http or ws")
        maximum_fps = 15 if transport == "ws" else 2
        if not (0.2 <= fps <= maximum_fps):
            raise ValueError(f"Live FPS must be between 0.2 and {maximum_fps} for {transport}")
        if not (160 <= max_width <= 960) or not (30 <= quality <= 85):
            raise ValueError("Invalid preview resolution or JPEG quality")
        parsed = urlparse(events_url)
        if parsed.scheme != "https" or not parsed.netloc or not token:
            raise ValueError("Live uplink requires Cloud HTTPS URL and Device Token")
        self.url = parsed._replace(
            path="/api/v1/edge/live-frame", params="", query="", fragment=""
        ).geturl()
        self.ws_url = parsed._replace(
            scheme="wss", path="/api/v1/edge/live-ws", params="", query="", fragment=""
        ).geturl()
        self.camera = camera
        self.token = token
        self.device_id = device_id
        self.transport = transport
        self.interval = 1 / fps
        self.width = max_width
        self.quality = quality
        self._lock = threading.Lock()
        self._last_event: str = ""
        self._event_at: float = 0.0
        self._evidence_preview = None
        self._evidence_at: float = 0.0
        # Ephemeral UI state only. No SQLite event or additional camera capture.
        self._rfid_status: dict | None = None
        self._rfid_version = 0
        self._rfid_at = 0.0

    def note_processing(self, uid: str) -> None:
        # Invoked immediately after RC522 returns a UID, before inference starts.
        with self._lock:
            self._rfid_status = {
                "kind": "rfid_progress", "phase": "PROCESSING",
                "uid": uid[:24], "action": "", "plate": "",
            }
            self._rfid_at = time.monotonic()
            self._rfid_version += 1

    def note_failed(self) -> None:
        with self._lock:
            self._rfid_status = {
                "kind": "rfid_progress", "phase": "ERROR",
                "uid": "", "action": "", "plate": "",
            }
            self._rfid_at = time.monotonic()
            self._rfid_version += 1

    def note_event(self, outcome: object) -> None:
        action = str(getattr(outcome, "action", ""))[:16]
        plate = str(getattr(outcome, "plate", "") or "-")[:20]
        with self._lock:
            self._last_event = f"LAST RFID: {action} / PLATE: {plate}"
            self._event_at = time.monotonic()
            self._rfid_status = {
                "kind": "rfid_progress", "phase": "DONE",
                "uid": str(getattr(outcome, "uid", ""))[:24],
                "action": action, "plate": plate,
            }
            self._rfid_at = self._event_at
            self._rfid_version += 1

    def show_evidence(self, annotated_plate) -> None:
        """PiP snapshot of RFID inference; does not claim per-frame tracking."""
        with self._lock:
            self._evidence_preview = annotated_plate.copy()
            self._evidence_at = time.monotonic()

    def _encode(self, frame) -> bytes:
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, max(1, round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        with self._lock:
            processing = (self._rfid_status is not None
                          and self._rfid_status.get("phase") == "PROCESSING"
                          and time.monotonic() - self._rfid_at < 20)
            recent = ("RFID RECEIVED | RECOGNIZING PLATE..." if processing else
                      self._last_event if time.monotonic() - self._event_at < 12 else "")
            evidence = (self._evidence_preview
                        if time.monotonic() - self._evidence_at < 6 else None)
        if evidence is not None and frame.shape[1] >= 280 and frame.shape[0] >= 200:
            # True OCR boxes on a captured still inset; live video underneath stays
            # smooth and is never labeled as 10 FPS model inference.
            maxw, maxh = min(290, frame.shape[1] // 2), min(155, frame.shape[0] // 2)
            eh, ew = evidence.shape[:2]
            if ew > 0 and eh > 0:
                scale = min(maxw / ew, maxh / eh)
                tile = cv2.resize(evidence, (max(1, round(ew*scale)), max(1, round(eh*scale))),
                                  interpolation=cv2.INTER_AREA)
                th, tw = tile.shape[:2]
                x, y = frame.shape[1]-tw-9, 36
                frame[y:y+th, x:x+tw] = tile
                cv2.rectangle(frame, (x-2,y-2), (x+tw+2,y+th+2), (0,235,0), 2)
                cv2.putText(frame, 'RFID SNAPSHOT', (x, max(10, y-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, .44, (0,235,0), 1, cv2.LINE_AA)
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

    def _next_jpeg(self, last_id: int | None):
        packet = self.camera.latest_frame()
        if packet is None or packet.frame_id == last_id:
            return last_id, None
        jpeg = self._encode(packet.frame)
        if len(jpeg) > 200_000:
            LOG.warning("Skipping oversized preview frame: %s bytes", len(jpeg))
            return packet.frame_id, None
        return packet.frame_id, jpeg

    def _send_http(self, jpeg: bytes) -> None:
        req = Request(
            self.url,
            data=jpeg,
            headers={"Content-Type": "image/jpeg", "Authorization": f"Bearer {self.token}",
                     "X-Device-ID": self.device_id},
            method="POST",
        )
        with urlopen(req, timeout=5) as response:
            if response.status != 204:
                raise RuntimeError(f"Unexpected live upload status {response.status}")

    def run_forever(self, stop: threading.Event) -> None:
        """Best-effort uploader; failure must never block RFID/YOLO/SQLite."""
        if self.transport == "ws":
            self._run_ws(stop)
        else:
            self._run_http(stop)

    def _run_http(self, stop: threading.Event) -> None:
        previous_id = None
        failures = 0
        while not stop.is_set():
            start = time.monotonic()
            try:
                previous_id, jpeg = self._next_jpeg(previous_id)
                if jpeg is not None:
                    self._send_http(jpeg)
                    failures = 0
            except (OSError, URLError, HTTPError, RuntimeError, ValueError) as exc:
                failures += 1
                if failures == 1 or failures % 20 == 0:
                    LOG.warning("Live HTTP unavailable (attempt %s): %s", failures, exc)
            stop.wait(max(0.05, self.interval - (time.monotonic() - start)))

    def _run_ws(self, stop: threading.Event) -> None:
        try:
            from websocket import create_connection
        except ImportError as exc:
            LOG.error("WebSocket transport requires: python -m pip install websocket-client>=1.8")
            return
        failures = 0
        while not stop.is_set():
            ws = None
            try:
                ws = create_connection(
                    self.ws_url,
                    header=[f"Authorization: Bearer {self.token}", f"X-Device-ID: {self.device_id}"],
                    timeout=5,
                    enable_multithread=True,
                )
                failures = 0
                last_id = None
                last_status_version = -1  # Re-send latest status after reconnect.
                sent = 0
                window_start = time.monotonic()
                LOG.info("Live WebSocket connected")
                while not stop.is_set():
                    start = time.monotonic()
                    with self._lock:
                        version = self._rfid_version
                        status = (dict(self._rfid_status)
                                  if self._rfid_status is not None
                                  and time.monotonic() - self._rfid_at < 20 else None)
                    if status is not None and version != last_status_version:
                        # Text messages share the existing authenticated WS;
                        # binary JPEG frames remain backward compatible.
                        ws.send(json.dumps(status, ensure_ascii=True))
                        last_status_version = version
                    last_id, jpeg = self._next_jpeg(last_id)
                    if jpeg is not None:
                        ws.send_binary(jpeg)
                        sent += 1
                    if start - window_start >= 15:
                        LOG.info("Live upload %.1f FPS (%d frames)", sent / (start - window_start), sent)
                        sent = 0
                        window_start = start
                    stop.wait(max(0.005, self.interval - (time.monotonic() - start)))
            except Exception as exc:
                failures += 1
                if failures == 1 or failures % 10 == 0:
                    LOG.warning("Live WebSocket interrupted (attempt %s): %s", failures, exc)
                stop.wait(min(10.0, 2.0 ** min(failures, 3)))
            finally:
                if ws is not None:
                    try:
                        ws.close(timeout=1)
                    except Exception:
                        pass
