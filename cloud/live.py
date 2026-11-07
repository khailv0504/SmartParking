"""Authenticated, bounded live preview from one Raspberry Pi camera.

Legacy HTTPS JPEG upload/read remains available. A WebSocket uplink plus MJPEG
browser stream enables smoother preview without sending inference to Cloud.
A single Uvicorn worker is required because the latest frame is process-local.
"""
from __future__ import annotations

import asyncio
from collections import deque
import json
import re
import secrets
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request, WebSocket, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session, sessionmaker
from starlette.websockets import WebSocketDisconnect

from .auth import resolve_admin_session
from .config import Settings

MAX_FRAME_BYTES = 200_000
FRAME_TTL_SECONDS = 10.0
DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
STREAM_BOUNDARY = "frame"


def is_jpeg(jpeg: bytes) -> bool:
    return len(jpeg) >= 4 and jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")


def valid_device(authorization: str | None, device_id: str | None, token: str) -> bool:
    scheme, _, credential = (authorization or "").partition(" ")
    return bool(
        scheme.lower() == "bearer"
        and credential
        and secrets.compare_digest(credential, token)
        and device_id
        and DEVICE_ID_RE.fullmatch(device_id)
    )


class LiveFrameStore:
    """Single latest frame with condition wakeup, bounded history for rate metrics."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._data: bytes | None = None
        self._device_id: str | None = None
        self._updated_at: str | None = None
        self._received_monotonic = 0.0
        self._sequence = 0
        self._times: deque[float] = deque(maxlen=300)
        self._rfid_progress: dict | None = None
        self._rfid_progress_at = 0.0

    def publish_progress(self, progress: dict, device_id: str) -> None:
        """Transient display state; never a persisted parking event."""
        with self._lock:
            self._rfid_progress = {
                "phase": progress["phase"], "device_id": device_id,
                "uid": progress["uid"], "action": progress["action"],
                "plate": progress["plate"],
            }
            self._rfid_progress_at = time.monotonic()
            self._lock.notify_all()

    def progress_status(self) -> dict | None:
        with self._lock:
            if self._rfid_progress is None:
                return None
            elapsed = time.monotonic() - self._rfid_progress_at
            lifetime = 20 if self._rfid_progress["phase"] == "PROCESSING" else 12
            if elapsed > lifetime:
                return None
            return dict(self._rfid_progress)

    def publish(self, jpeg: bytes, device_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._data = jpeg
            self._device_id = device_id
            self._received_monotonic = now
            self._updated_at = datetime.now(timezone.utc).isoformat()
            self._sequence += 1
            self._times.append(now)
            self._lock.notify_all()

    def snapshot(self) -> bytes | None:
        with self._lock:
            if not self._data or time.monotonic() - self._received_monotonic > FRAME_TTL_SECONDS:
                return None
            return self._data

    def wait_for_next(self, sequence: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        """Wait for a newer image, never queue stale frames per viewer."""
        with self._lock:
            self._lock.wait_for(lambda: self._sequence > sequence, timeout=timeout)
            if not self._data or time.monotonic() - self._received_monotonic > FRAME_TTL_SECONDS:
                return self._sequence, None
            return self._sequence, self._data

    def status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            age = now - self._received_monotonic
            available = self._data is not None and age <= FRAME_TTL_SECONDS
            recent = [t for t in self._times if t >= now - 10.0]
            fps = ((len(recent) - 1) / (recent[-1] - recent[0])) if len(recent) > 1 and recent[-1] > recent[0] else 0.0
            return {
                "available": available,
                "device_id": self._device_id,
                "last_frame_utc": self._updated_at,
                "age_seconds": round(age, 1) if self._data is not None else None,
                "sequence": self._sequence,
                "incoming_fps": round(fps, 1) if available else 0.0,
            }


def build_live_router(
    settings: Settings,
    factory: sessionmaker[Session],
    store: LiveFrameStore,
) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.post("/api/v1/edge/live-frame", status_code=status.HTTP_204_NO_CONTENT)
    async def receive_live_frame(
        request: Request,
        authorization: str | None = Header(default=None),
        device_id: str | None = Header(default=None, alias="X-Device-ID"),
    ) -> Response:
        scheme, _, credential = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not credential or not secrets.compare_digest(credential, settings.device_token):
            raise HTTPException(status_code=401, detail="Invalid device credentials")
        if not device_id or not DEVICE_ID_RE.fullmatch(device_id):
            raise HTTPException(status_code=422, detail="Invalid device identifier")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
            raise HTTPException(status_code=415, detail="JPEG required")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_FRAME_BYTES:
                raise HTTPException(status_code=413, detail="Frame exceeds limit")
        if not is_jpeg(data):
            raise HTTPException(status_code=422, detail="Invalid JPEG framing")
        store.publish(bytes(data), device_id)
        return Response(status_code=204)

    @router.websocket("/api/v1/edge/live-ws")
    async def receive_live_ws(websocket: WebSocket) -> None:
        auth = websocket.headers.get("authorization")
        device_id = websocket.headers.get("x-device-id")
        if not valid_device(auth, device_id, settings.device_token):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                incoming_text = msg.get("text")
                if incoming_text is not None:
                    if len(incoming_text) > 1024:
                        await websocket.close(code=1009)
                        break
                    try:
                        progress = json.loads(incoming_text)
                        if (not isinstance(progress, dict)
                            or progress.get("kind") != "rfid_progress"
                            or progress.get("phase") not in {"PROCESSING", "DONE", "ERROR"}
                            or any(not isinstance(progress.get(field), str)
                                   or len(progress[field]) > 64
                                   for field in ("uid", "action", "plate"))):
                            raise ValueError("Invalid RFID progress")
                    except (ValueError, TypeError):
                        await websocket.close(code=1003)
                        break
                    store.publish_progress(progress, device_id)
                    continue
                jpeg = msg.get("bytes")
                if jpeg is None:
                    await websocket.close(code=1003)
                    break
                if len(jpeg) > MAX_FRAME_BYTES or not is_jpeg(jpeg):
                    await websocket.close(code=1009 if len(jpeg) > MAX_FRAME_BYTES else 1003)
                    break
                store.publish(jpeg, device_id)
        except WebSocketDisconnect:
            pass

    def require_admin(request: Request) -> None:
        try:
            admin = resolve_admin_session(request, factory)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Authentication unavailable") from exc
        if admin is None:
            raise HTTPException(status_code=401, detail="Administrator login required")

    @router.get("/api/live/frame.jpg")
    def latest_frame(request: Request) -> Response:
        require_admin(request)
        jpeg = store.snapshot()
        if jpeg is None:
            return Response(status_code=503, headers={"Cache-Control": "no-store"})
        return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "private, no-store, max-age=0"})

    @router.get("/api/live/status")
    def live_status(request: Request) -> dict:
        require_admin(request)
        info = store.status()
        info["rfid_progress"] = store.progress_status()
        return info

    @router.get("/api/live/stream.mjpg")
    async def live_mjpeg(request: Request) -> StreamingResponse:
        require_admin(request)

        async def stream():
            seq = -1
            next_auth = time.monotonic() + 5.0
            while True:
                if await request.is_disconnected():
                    break
                if time.monotonic() >= next_auth:
                    try:
                        if await asyncio.to_thread(resolve_admin_session, request, factory) is None:
                            break
                    except Exception:
                        break
                    next_auth = time.monotonic() + 5.0
                new_seq, jpeg = await asyncio.to_thread(store.wait_for_next, seq, 1.0)
                if jpeg is None or new_seq == seq:
                    continue
                seq = new_seq
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode("ascii")
                    + b"\r\n\r\n" + jpeg + b"\r\n"
                )

        return StreamingResponse(
            stream(),
            media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
            headers={"Cache-Control": "private, no-store, max-age=0", "X-Accel-Buffering": "no"},
        )

    return router
