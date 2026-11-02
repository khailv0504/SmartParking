"""Ephemeral authenticated camera preview: Edge POSTs JPEG, Admin views latest frame.

This is intentionally NOT video storage or a recording system. A single
Render process hosts the latest frame in RAM; do not use multiple workers.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import Response
from sqlalchemy.orm import Session, sessionmaker

from .auth import resolve_admin_session
from .config import Settings

MAX_FRAME_BYTES = 200_000
FRAME_TTL_SECONDS = 10.0
DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")


class LiveFrameStore:
    """Bounded in-memory latest-frame slot: no DB, disk, or unbounded queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: bytes | None = None
        self._device_id: str | None = None
        self._updated_at: str | None = None
        self._received_monotonic = 0.0
        self._sequence = 0

    def publish(self, jpeg: bytes, device_id: str) -> None:
        with self._lock:
            self._data = jpeg
            self._device_id = device_id
            self._received_monotonic = time.monotonic()
            self._updated_at = datetime.now(timezone.utc).isoformat()
            self._sequence += 1

    def snapshot(self) -> bytes | None:
        with self._lock:
            if not self._data or time.monotonic() - self._received_monotonic > FRAME_TTL_SECONDS:
                return None
            return self._data

    def status(self) -> dict:
        with self._lock:
            age = time.monotonic() - self._received_monotonic
            available = self._data is not None and age <= FRAME_TTL_SECONDS
            return {
                "available": available,
                "device_id": self._device_id,
                "last_frame_utc": self._updated_at,
                "age_seconds": round(age, 1) if self._data is not None else None,
                "sequence": self._sequence,
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
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token or not secrets.compare_digest(
            token, settings.device_token
        ):
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
        if len(data) < 4 or not data.startswith(b"\xff\xd8") or not data.endswith(b"\xff\xd9"):
            raise HTTPException(status_code=422, detail="Invalid JPEG framing")
        store.publish(bytes(data), device_id)
        return Response(status_code=204)

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
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, no-store, max-age=0"},
        )

    @router.get("/api/live/status")
    def live_status(request: Request) -> dict:
        require_admin(request)
        return store.status()

    return router
