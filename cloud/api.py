from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import EdgeEvent, ParkingSession
from .schemas import EdgeEventIn, EventAccepted, EventType, HealthResponse

LOG = logging.getLogger(__name__)


class IdempotencyConflict(Exception):
    pass


class SessionProjectionConflict(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _event_values(
    payload: EdgeEventIn, payload_hash: str, received_at: datetime
) -> dict:
    return {
        "event_id": payload.event_id,
        "device_id": payload.device_id,
        "uid": payload.uid,
        "plate": payload.plate,
        "event_type": payload.event_type.value,
        "reason": payload.reason,
        "session_id": payload.session_id,
        "event_timestamp": payload.timestamp,
        "received_at": received_at,
        "ocr_confidence": payload.ocr_confidence,
        "detection_confidence": payload.detection_confidence,
        "iqa_score": payload.iqa_score,
        "inference_ms": payload.inference_ms,
        "payload_hash": payload_hash,
    }


def _project_session(db: Session, payload: EdgeEventIn, now: datetime) -> None:
    if payload.event_type not in (EventType.ENTRY, EventType.EXIT):
        return
    if not payload.session_id:
        raise SessionProjectionConflict("ENTRY/EXIT event requires session_id")

    row = db.get(ParkingSession, payload.session_id)
    if row is None:
        row = ParkingSession(session_id=payload.session_id, updated_at=now)
        db.add(row)

    if payload.event_type == EventType.ENTRY:
        if row.entry_event_id and row.entry_event_id != payload.event_id:
            raise SessionProjectionConflict(
                "session already has a different ENTRY event"
            )
        row.entry_event_id = payload.event_id
        row.entry_at = payload.timestamp
        # ENTRY là nguồn danh tính chính và hoàn thiện projection exit-first.
        row.uid = payload.uid
        row.plate = payload.plate
    else:
        if row.exit_event_id and row.exit_event_id != payload.event_id:
            raise SessionProjectionConflict(
                "session already has a different EXIT event"
            )
        row.exit_event_id = payload.event_id
        row.exit_at = payload.timestamp
        # Nếu EXIT đến trước, tạm dùng metadata của EXIT cho tới khi ENTRY tới.
        if row.entry_event_id is None:
            row.uid = payload.uid
            row.plate = payload.plate
    row.updated_at = now


def ingest_event(db: Session, payload: EdgeEventIn) -> EventAccepted:
    payload_hash = payload.canonical_hash()
    now = utcnow()
    values = _event_values(payload, payload_hash, now)
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        statement = (
            postgres_insert(EdgeEvent)
            .values(**values)
            .on_conflict_do_nothing(index_elements=[EdgeEvent.event_id])
        )
    elif dialect == "sqlite":
        statement = (
            sqlite_insert(EdgeEvent)
            .values(**values)
            .on_conflict_do_nothing(index_elements=[EdgeEvent.event_id])
        )
    else:  # pragma: no cover - production/test dialects are PostgreSQL/SQLite
        raise RuntimeError(f"Unsupported database dialect: {dialect}")

    # Hàm sở hữu transaction: commit cả event + projection hoặc rollback cả hai.
    # Cách này vẫn đúng nếu caller vừa thực hiện một SELECT và SQLAlchemy đã autobegin.
    try:
        result = db.execute(statement)
        if result.rowcount == 0:
            existing_hash = db.scalar(
                select(EdgeEvent.payload_hash).where(
                    EdgeEvent.event_id == payload.event_id
                )
            )
            if existing_hash == payload_hash:
                db.commit()
                return EventAccepted(
                    event_id=payload.event_id,
                    status="already_processed",
                    duplicate=True,
                )
            LOG.warning(
                "Idempotency conflict event_id=%s device_id=%s",
                payload.event_id,
                payload.device_id,
            )
            raise IdempotencyConflict("event_id already exists with different content")
        _project_session(db, payload, now)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return EventAccepted(event_id=payload.event_id, status="accepted")


def _verify_bearer(authorization: str | None, expected: str) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if (
        scheme.lower() != "bearer"
        or not token
        or not secrets.compare_digest(token, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid device credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def build_api_router(settings: Settings, factory: sessionmaker[Session]) -> APIRouter:
    router = APIRouter()

    @router.get("/api/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        # Không trả DB URL, token, dữ liệu UID/biển số hoặc chi tiết exception.
        return HealthResponse()

    @router.post(
        "/api/v1/edge/events",
        response_model=EventAccepted,
        status_code=status.HTTP_201_CREATED,
        tags=["edge"],
    )
    def receive_event(
        payload: EdgeEventIn,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> EventAccepted:
        _verify_bearer(authorization, settings.device_token)
        if idempotency_key != payload.event_id:
            raise HTTPException(
                status_code=422, detail="Idempotency-Key must match payload event_id"
            )
        try:
            with factory() as db:
                result = ingest_event(db, payload)
            return result
        except (IdempotencyConflict, SessionProjectionConflict) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router
