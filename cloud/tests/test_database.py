from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

import cloud.api as api_module
from cloud.api import ingest_event
from cloud.main import app
from cloud.models import EdgeEvent, ParkingSession
from cloud.queries import dashboard_snapshot
from cloud.schemas import EdgeEventIn


def payload(event_id: str, event_type: str, session_id: str | None = "session-1"):
    return EdgeEventIn.model_validate(
        {
            "event_id": event_id,
            "device_id": "raspberrypi_01",
            "uid": "A1B2C3D4",
            "plate": "29AC88688",
            "event_type": event_type,
            "reason": None,
            "session_id": session_id,
            "timestamp": "2026-10-27T01:00:00+00:00",
            "ocr_confidence": 0.9,
            "detection_confidence": 0.8,
            "iqa_score": 0.7,
            "inference_ms": 100,
        }
    )


def test_entry_creates_open_session():
    with app.state.session_factory() as db:
        ingest_event(db, payload("entry-1", "ENTRY"))
        row = db.get(ParkingSession, "session-1")
        assert row.entry_event_id == "entry-1"
        assert row.exit_at is None


def test_exit_closes_correct_session():
    with app.state.session_factory() as db:
        ingest_event(db, payload("entry-1", "ENTRY"))
        exit_payload = payload("exit-1", "EXIT")
        exit_payload.timestamp = datetime(2026, 10, 27, 3, tzinfo=timezone.utc)
        ingest_event(db, exit_payload)
        row = db.get(ParkingSession, "session-1")
        assert row.entry_event_id == "entry-1"
        assert row.exit_event_id == "exit-1"
        assert row.exit_at is not None


def test_exit_before_entry_is_reconciled():
    with app.state.session_factory() as db:
        ingest_event(db, payload("exit-1", "EXIT"))
        first = db.get(ParkingSession, "session-1")
        assert first.entry_at is None
        assert first.exit_event_id == "exit-1"
        ingest_event(db, payload("entry-1", "ENTRY"))
        db.expire_all()
        final = db.get(ParkingSession, "session-1")
        assert final.entry_event_id == "entry-1"
        assert final.exit_event_id == "exit-1"
        assert final.entry_at is not None


def test_duplicate_does_not_duplicate_session():
    with app.state.session_factory() as db:
        item = payload("entry-1", "ENTRY")
        ingest_event(db, item)
        ingest_event(db, item)
        assert db.scalar(select(func.count()).select_from(ParkingSession)) == 1


@pytest.mark.parametrize("event_type", ["REVIEW", "DENY", "ERROR"])
def test_non_session_events_do_not_change_sessions(event_type):
    with app.state.session_factory() as db:
        ingest_event(db, payload(f"event-{event_type}", event_type, None))
        assert db.scalar(select(func.count()).select_from(ParkingSession)) == 0


def test_transaction_rolls_back_event_when_projection_fails(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("simulated projection failure")

    monkeypatch.setattr(api_module, "_project_session", fail)
    with app.state.session_factory() as db:
        with pytest.raises(RuntimeError, match="simulated"):
            ingest_event(db, payload("entry-rollback", "ENTRY"))
    with app.state.session_factory() as db:
        assert db.get(EdgeEvent, "entry-rollback") is None


def test_dashboard_statistics_are_database_backed():
    with app.state.session_factory() as db:
        for index, kind in enumerate(("ENTRY", "REVIEW", "DENY", "ERROR"), start=1):
            item = payload(
                f"event-{index}", kind, "stats-session" if kind == "ENTRY" else None
            )
            item.timestamp = datetime.now(timezone.utc)
            ingest_event(db, item)
    stats = dashboard_snapshot(app.state.session_factory)
    assert stats["inside"] == 1
    assert stats["entry"] == 1
    assert stats["alerts"] == 3
