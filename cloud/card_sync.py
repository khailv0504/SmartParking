"""Idempotent, versioned Edge-authoritative RFID registry snapshots."""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import RFIDCard, RFIDSnapshotState


class CardEntry(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    uid: str = Field(min_length=1, max_length=80)
    enabled: bool
    created_at: datetime

    @field_validator('uid')
    @classmethod
    def normalize_uid(cls, value: str) -> str:
        return value.upper()

    @field_validator('created_at')
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('created_at must be timezone-aware')
        return value.astimezone(timezone.utc)


class CardSnapshotIn(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    device_id: str = Field(min_length=1, max_length=80)
    snapshot_version: int = Field(ge=1, le=9223372036854775807)
    cards: list[CardEntry] = Field(max_length=2000)

    @model_validator(mode='after')
    def unique_uids(self) -> 'CardSnapshotIn':
        uids = [card.uid for card in self.cards]
        if len(uids) != len(set(uids)):
            raise ValueError('Duplicate RFID UIDs in snapshot')
        return self

    def payload_hash(self) -> str:
        cards = sorted(
            (
                {'uid': card.uid, 'enabled': card.enabled,
                 'created_at': card.created_at.isoformat(timespec='microseconds')}
                for card in self.cards
            ), key=lambda value: value['uid']
        )
        canonical = json.dumps(
            {'device_id': self.device_id, 'cards': cards},
            sort_keys=True, ensure_ascii=False, separators=(',', ':'),
        )
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class CardSnapshotAccepted(BaseModel):
    device_id: str
    snapshot_version: int
    status: str
    duplicate: bool
    card_count: int


def _insert_ignore(db: Session, *, device_id: str, now: datetime) -> None:
    dialect = db.get_bind().dialect.name
    insert = pg_insert if dialect == 'postgresql' else sqlite_insert if dialect == 'sqlite' else None
    if insert is None:
        raise RuntimeError('Unsupported SQL dialect')
    statement = insert(RFIDSnapshotState).values(
        device_id=device_id, snapshot_version=0, payload_hash='', received_at=now
    ).on_conflict_do_nothing(index_elements=[RFIDSnapshotState.device_id])
    db.execute(statement)


def ingest_card_snapshot(db: Session, payload: CardSnapshotIn) -> CardSnapshotAccepted:
    now = datetime.now(timezone.utc)
    digest = payload.payload_hash()
    with db.begin():
        _insert_ignore(db, device_id=payload.device_id, now=now)
        # Lock the version row, serializing concurrent requests per device on Postgres.
        state = db.scalar(
            select(RFIDSnapshotState)
            .where(RFIDSnapshotState.device_id == payload.device_id)
            .with_for_update()
        )
        if state is None:
            raise RuntimeError('RFID version row missing')
        if payload.snapshot_version < state.snapshot_version:
            raise ValueError('Stale card snapshot version')
        if payload.snapshot_version == state.snapshot_version:
            if digest != state.payload_hash:
                raise ValueError('Snapshot version conflicts with existing content')
            return CardSnapshotAccepted(
                device_id=payload.device_id,
                snapshot_version=payload.snapshot_version,
                status='already_processed', duplicate=True,
                card_count=len(payload.cards),
            )
        db.execute(delete(RFIDCard).where(RFIDCard.device_id == payload.device_id))
        db.add_all([
            RFIDCard(device_id=payload.device_id, uid=card.uid,
                     enabled=card.enabled, created_at=card.created_at,
                     synced_at=now)
            for card in payload.cards
        ])
        state.snapshot_version = payload.snapshot_version
        state.payload_hash = digest
        state.received_at = now
    return CardSnapshotAccepted(
        device_id=payload.device_id, snapshot_version=payload.snapshot_version,
        status='accepted', duplicate=False, card_count=len(payload.cards),
    )


def build_card_router(settings: Settings, factory: sessionmaker[Session]) -> APIRouter:
    router = APIRouter()

    @router.post('/api/v1/edge/card-snapshots', response_model=CardSnapshotAccepted, tags=['edge'])
    def receive_snapshot(
        payload: CardSnapshotIn,
        authorization: str | None = Header(default=None),
    ) -> CardSnapshotAccepted:
        scheme, _, token = (authorization or '').partition(' ')
        if scheme.lower() != 'bearer' or not token or not secrets.compare_digest(token, settings.device_token):
            raise HTTPException(status_code=401, detail='Invalid device credentials')
        try:
            with factory() as db:
                return ingest_card_snapshot(db, payload)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return router
