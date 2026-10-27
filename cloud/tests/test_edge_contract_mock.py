from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from cloud.main import app
from cloud.models import EdgeEvent, ParkingSession
from edge.database import Database
from edge.sync import SyncWorker


def test_real_edge_outbox_payload_reaches_cloud_and_marks_synced(
    tmp_path: Path, client, monkeypatch
):
    edge_db = Database(tmp_path / "edge.db", device_id="raspberrypi_01")
    edge_db.register_card("A1B2C3D4")
    source = {
        "event_id": "33333333-3333-4333-8333-333333333333",
        "device_id": "raspberrypi_01",
        "uid": "A1B2C3D4",
        "plate": "29AC88688",
        "timestamp": "2026-10-27T14:00:00.000+00:00",
        "ocr_confidence": 0.6832,
        "detection_confidence": 0.8749,
        "iqa_score": 0.9571,
        "inference_ms": 1135.31,
        "image_path": "local-only.jpg",
        "reason": None,
    }
    result = edge_db.commit_event(source)
    assert result["type"] == "ENTRY"
    exit_source = {
        **source,
        "event_id": "77777777-7777-4777-8777-777777777777",
        "timestamp": "2026-10-27T14:15:00.000+00:00",
    }
    assert edge_db.commit_event(exit_source)["type"] == "EXIT"
    for index, event_type in enumerate(("DENY", "REVIEW", "ERROR"), start=1):
        edge_db.record_error(
            {
                "event_id": f"edge-{event_type.lower()}-{index}",
                "device_id": "raspberrypi_01",
                "uid": None,
                "timestamp": f"2026-10-27T15:0{index}:00.000+00:00",
                "reason": "SIMULATED",
            },
            event_type,
        )
    worker = SyncWorker(
        edge_db,
        "https://cloud.test/api/v1/edge/events",
        "test-device-token-at-least-24-characters",
    )

    def send(payload):
        assert "image_path" not in payload
        response = client.post(
            "/api/v1/edge/events",
            json=payload,
            headers={
                "Authorization": f"Bearer {worker.token}",
                "Idempotency-Key": payload["event_id"],
            },
        )
        response.raise_for_status()

    monkeypatch.setattr(worker, "send", send)
    assert worker.run_once() == (5, 0)
    assert edge_db.summary()["pending_sync"] == 0
    with app.state.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(EdgeEvent)) == 5
        session = db.get(ParkingSession, source["event_id"])
        assert session.entry_event_id == source["event_id"]
        assert session.exit_event_id == exit_source["event_id"]
        assert session.exit_at is not None
        assert db.scalar(select(func.count()).select_from(ParkingSession)) == 1

    # Gửi lại cùng nội dung vẫn 2xx và Cloud không tạo trùng.
    payload = edge_db.pending()
    assert payload == []
    send(
        {k: v for k, v in source.items() if k != "image_path"}
        | {"event_type": "ENTRY", "session_id": source["event_id"]}
    )
    with app.state.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(EdgeEvent)) == 5
        assert db.scalar(select(func.count()).select_from(ParkingSession)) == 1


def test_edge_keeps_pending_when_cloud_unreachable(tmp_path: Path, monkeypatch):
    edge_db = Database(tmp_path / "edge.db")
    edge_db.record_error(
        {
            "event_id": "offline-event",
            "device_id": "raspberrypi_01",
            "uid": None,
            "timestamp": "2026-10-27T14:00:00+00:00",
            "reason": "SIMULATED",
        }
    )
    worker = SyncWorker(edge_db, "https://cloud.test/api/v1/edge/events", "x" * 24)
    monkeypatch.setattr(
        worker, "send", lambda _payload: (_ for _ in ()).throw(OSError("offline"))
    )
    assert worker.run_once() == (0, 1)
    assert edge_db.summary()["pending_sync"] == 1
