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
CREATE TABLE IF NOT EXISTS card_sync_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    snapshot_version INTEGER NOT NULL CHECK(snapshot_version >= 1),
    synced_version INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at TEXT NOT NULL,
    last_error TEXT
);
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
CREATE TABLE IF NOT EXISTS evidence_outbox (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    kind TEXT NOT NULL CHECK(kind IN ('annotated','plate')),
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING','SYNCED')),
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at TEXT NOT NULL,
    last_error TEXT,
    synced_at TEXT,
    PRIMARY KEY(event_id,kind)
);
CREATE INDEX IF NOT EXISTS idx_evidence_due ON evidence_outbox(status,retry_at);
"""


class Database:
    def __init__(self, path: str | Path, device_id: str = 'raspberrypi_01'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute(
                'INSERT OR IGNORE INTO card_sync_state '
                '(id,snapshot_version,synced_version,retry_at) VALUES (1,1,0,?)',
                (utcnow(),),
            )

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
                # Persist a new revision in the SAME transaction as the card change.
                db.execute(
                    'UPDATE card_sync_state SET snapshot_version=snapshot_version+1, '
                    'attempts=0,retry_at=?,last_error=NULL WHERE id=1',
                    (utcnow(),),
                )
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

    def card_snapshot_due(self) -> tuple[dict, int] | None:
        """Return a consistent snapshot + retry count, or None if not due."""
        with self.connect() as db:
            db.execute('BEGIN')
            try:
                state = db.execute(
                    'SELECT snapshot_version,synced_version,retry_at,attempts '
                    'FROM card_sync_state WHERE id=1'
                ).fetchone()
                if (not state or state['synced_version'] >= state['snapshot_version']
                        or state['retry_at'] > utcnow()):
                    db.commit()
                    return None
                rows = db.execute(
                    'SELECT uid,enabled,created_at FROM cards ORDER BY uid'
                ).fetchall()
                result = {
                    'device_id': self.device_id,
                    'snapshot_version': state['snapshot_version'],
                    'cards': [
                        {'uid': row['uid'], 'enabled': bool(row['enabled']),
                         'created_at': row['created_at']}
                        for row in rows
                    ],
                }
                db.commit()
                return result, state['attempts']
            except Exception:
                db.rollback()
                raise

    def mark_card_snapshot_synced(self, version: int) -> None:
        with self.connect() as db:
            db.execute(
                'UPDATE card_sync_state SET synced_version=?,attempts=0,'
                'last_error=NULL WHERE id=1 AND snapshot_version=? '
                'AND synced_version<?',
                (version, version, version),
            )

    def mark_card_snapshot_retry(self, version: int, attempts: int, error: str) -> None:
        delay = min(300, 2 ** min(attempts, 8))
        next_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(
            timespec='milliseconds'
        )
        with self.connect() as db:
            db.execute(
                'UPDATE card_sync_state SET attempts=?,retry_at=?,last_error=? '
                'WHERE id=1 AND snapshot_version=?',
                (attempts, next_at, str(error)[:400], version),
            )

    def card_sync_status(self) -> dict:
        with self.connect() as db:
            row = db.execute('SELECT * FROM card_sync_state WHERE id=1').fetchone()
            return dict(row)

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

    def enqueue_evidence(self, event_id: str, kind: str, filename: str) -> None:
        if kind not in ('annotated', 'plate'):
            raise ValueError('Unsupported evidence kind')
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO evidence_outbox '
                       '(event_id,kind,filename,retry_at) VALUES (?,?,?,?)',
                       (event_id, kind, filename, utcnow()))

    def pending_evidence(self, limit: int = 3) -> list[tuple[str,str,str,int]]:
        # Do not upload an image before its owning event reached Cloud.
        with self.connect() as db:
            rows = db.execute('SELECT e.event_id,e.kind,e.filename,e.attempts '
                              'FROM evidence_outbox e JOIN sync_outbox s ON s.event_id=e.event_id '
                              'WHERE e.status=? AND e.retry_at<=? AND s.status=? '
                              'ORDER BY e.retry_at LIMIT ?',
                              ('PENDING',utcnow(),'SYNCED',limit)).fetchall()
            return [(r['event_id'],r['kind'],r['filename'],r['attempts']) for r in rows]

    def mark_evidence_synced(self, event_id: str, kind: str) -> None:
        with self.connect() as db:
            db.execute('UPDATE evidence_outbox SET status=?, synced_at=?, last_error=NULL '
                       'WHERE event_id=? AND kind=?', ('SYNCED',utcnow(),event_id,kind))

    def mark_evidence_retry(self, event_id: str, kind: str, attempts: int, error: str) -> None:
        delay = min(300, 2 ** min(attempts, 8))
        retry = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(timespec='milliseconds')
        with self.connect() as db:
            db.execute('UPDATE evidence_outbox SET attempts=?, retry_at=?, last_error=? '
                       'WHERE event_id=? AND kind=?',
                       (attempts,retry,error[:400],event_id,kind))

    def summary(self) -> dict:
        with self.connect() as db:
            return {
                'open_sessions':db.execute('SELECT count(*) FROM parking_sessions WHERE exit_at IS NULL').fetchone()[0],
                'events':db.execute('SELECT count(*) FROM events').fetchone()[0],
                'pending_sync':db.execute("SELECT count(*) FROM sync_outbox WHERE status='PENDING'").fetchone()[0],
            }
