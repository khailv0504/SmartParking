"""SQLite local source of truth; event + outbox inserted atomically.

Connection-per-operation simplifies use from controller and sync thread.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    uid TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0,1)),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parking_sessions (
    session_id TEXT PRIMARY KEY,
    uid TEXT NOT NULL REFERENCES cards(uid),
    plate TEXT NOT NULL,
    entry_at TEXT NOT NULL,
    exit_at TEXT,
    CHECK(exit_at IS NULL OR exit_at >= entry_at)
);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_open_uid
    ON parking_sessions(uid) WHERE exit_at IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uniq_open_plate
    ON parking_sessions(plate) WHERE exit_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_entry ON parking_sessions(entry_at DESC);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    uid TEXT,
    plate TEXT,
    event_type TEXT NOT NULL CHECK(event_type IN ('ENTRY','EXIT','DENY','REVIEW','ERROR')),
    reason TEXT,
    session_id TEXT,
    timestamp TEXT NOT NULL,
    ocr_confidence REAL,
    detection_confidence REAL,
    iqa_score REAL,
    inference_ms REAL,
    image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp DESC);
CREATE TABLE IF NOT EXISTS sync_outbox (
    event_id TEXT PRIMARY KEY REFERENCES events(event_id),
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING','SYNCED')),
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at TEXT NOT NULL,
    last_error TEXT,
    synced_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_due ON sync_outbox(status,retry_at);
"""


class Database:
    def __init__(self, path: str | Path, device_id: str = 'raspberrypi_01'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute('PRAGMA foreign_keys=ON')
            conn.execute('PRAGMA busy_timeout=10000')
            conn.execute('PRAGMA journal_mode=WAL')
            yield conn
        finally:
            conn.close()

    def register_card(self, uid: str, enabled: bool = True):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                db.execute('INSERT INTO cards(uid,enabled,created_at) VALUES (?,?,?) '
                           'ON CONFLICT(uid) DO UPDATE SET enabled=excluded.enabled',
                           (uid.upper(), int(enabled), utcnow()))
                db.commit()
            except Exception:
                db.rollback()
                raise

    def card_authorized(self, uid: str) -> bool:
        with self.connect() as db:
            row = db.execute('SELECT enabled FROM cards WHERE uid=?', (uid.upper(),)).fetchone()
            return bool(row and row['enabled'])

    def _insert_event_tx(self, db: sqlite3.Connection, event: dict) -> None:
        cols = ('event_id','device_id','uid','plate','event_type','reason','session_id',
                'timestamp','ocr_confidence','detection_confidence','iqa_score',
                'inference_ms','image_path')
        db.execute(f"INSERT INTO events({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
                   tuple(event.get(c) for c in cols))
        # Only safe relative image id/path is stored; images aren't synced as blobs.
        payload = {k: event.get(k) for k in cols if k != 'image_path'}
        db.execute('INSERT INTO sync_outbox(event_id,payload,retry_at) VALUES (?,?,?)',
                   (event['event_id'],json.dumps(payload,ensure_ascii=False), utcnow()))

    def commit_event(self, event: dict, min_confidence: float = 0.60) -> dict:
        """Make decision inside BEGIN IMMEDIATE; caller must never pre-decide entry/exit.

        event_type is assigned by DB based on current session. Never modify an open
        session on untrusted OCR / mismatched plate / unauthorized UID.
        """
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                exists = db.execute('SELECT event_type, reason FROM events WHERE event_id=?',
                                    (event['event_id'],)).fetchone()
                if exists:
                    db.rollback()
                    return {'type':exists['event_type'],'reason':exists['reason'],'duplicate':True}

                uid = event.get('uid')
                plate = event.get('plate')
                card = db.execute('SELECT enabled FROM cards WHERE uid=?', (uid,)).fetchone() if uid else None
                reason = event.get('reason')
                session_id = None
                action = 'REVIEW'
                if not card or not card['enabled']:
                    action, reason = 'DENY', 'CARD_NOT_AUTHORIZED'
                elif reason:
                    action = 'REVIEW'
                elif not plate or float(event.get('ocr_confidence') or 0.0) < min_confidence:
                    action, reason = 'REVIEW', 'LOW_OCR_CONFIDENCE'
                else:
                    current = db.execute('SELECT * FROM parking_sessions '
                                         'WHERE uid=? AND exit_at IS NULL', (uid,)).fetchone()
                    if current is None:
                        conflict = db.execute('SELECT session_id FROM parking_sessions '
                                              'WHERE plate=? AND exit_at IS NULL', (plate,)).fetchone()
                        if conflict:
                            action, reason = 'REVIEW', 'PLATE_ALREADY_INSIDE'
                        else:
                            session_id = event['event_id']  # stable unique ID
                            db.execute('INSERT INTO parking_sessions(session_id,uid,plate,entry_at) '
                                       'VALUES (?,?,?,?)', (session_id,uid,plate,event['timestamp']))
                            action = 'ENTRY'
                    elif current['plate'] != plate:
                        action, reason = 'REVIEW', 'PLATE_MISMATCH'
                        session_id = current['session_id']
                    else:
                        session_id = current['session_id']
                        db.execute('UPDATE parking_sessions SET exit_at=? WHERE session_id=? AND exit_at IS NULL',
                                   (event['timestamp'],session_id))
                        action = 'EXIT'

                stored = {**event, 'event_type':action, 'reason':reason, 'session_id':session_id}
                self._insert_event_tx(db, stored)
                db.commit()
                return {'type':action,'reason':reason,'session_id':session_id,'event_id':event['event_id']}
            except Exception:
                db.rollback()
                raise

    def record_error(self, event: dict, event_type: str = 'ERROR') -> dict:
        if event_type not in ('ERROR','REVIEW','DENY'):
            raise ValueError('record_error only supports ERROR/REVIEW/DENY')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                existing = db.execute('SELECT event_type FROM events WHERE event_id=?',
                                      (event['event_id'],)).fetchone()
                if existing:
                    db.rollback()
                    return {'type':existing['event_type'],'event_id':event['event_id'],'duplicate':True}
                self._insert_event_tx(db, {**event,'event_type':event_type})
                db.commit()
            except Exception:
                db.rollback()
                raise
        return {'type':event_type,'event_id':event['event_id'],'reason':event.get('reason')}

    def pending(self, limit: int = 20) -> list[tuple[str,dict,int]]:
        with self.connect() as db:
            rows = db.execute('SELECT event_id,payload,attempts FROM sync_outbox '
                              'WHERE status=? AND retry_at<=? ORDER BY retry_at,event_id LIMIT ?',
                              ('PENDING',utcnow(),limit)).fetchall()
            return [(r['event_id'],json.loads(r['payload']),r['attempts']) for r in rows]

    def mark_synced(self, event_id: str) -> None:
        with self.connect() as db:
            db.execute('UPDATE sync_outbox SET status=?,synced_at=?,last_error=NULL WHERE event_id=?',
                       ('SYNCED',utcnow(),event_id))

    def mark_retry(self, event_id: str, attempts: int, error: str) -> None:
        delay = min(300, 2 ** min(attempts, 8))
        next_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(timespec='milliseconds')
        with self.connect() as db:
            db.execute('UPDATE sync_outbox SET attempts=?,last_error=?,retry_at=? WHERE event_id=?',
                       (attempts, str(error)[:400],next_at,event_id))

    def summary(self) -> dict:
        with self.connect() as db:
            return {
                'open_sessions':db.execute('SELECT count(*) FROM parking_sessions WHERE exit_at IS NULL').fetchone()[0],
                'events':db.execute('SELECT count(*) FROM events').fetchone()[0],
                'pending_sync':db.execute("SELECT count(*) FROM sync_outbox WHERE status='PENDING'").fetchone()[0],
            }
