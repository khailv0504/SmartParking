"""Integration thật: chỉ chạy với một PostgreSQL test DB có thể xóa dữ liệu.

Không đặt TEST_POSTGRES_URL thì test SKIPPED, tuyệt đối không báo SQLite mock là
PostgreSQL PASS.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
import uvicorn
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import delete, func, inspect, select

import cloud.api as api_module
from cloud.api import build_api_router, ingest_event
from cloud.config import Settings, get_settings
from cloud.database import make_engine, make_session_factory
from cloud.models import AdminSession, EdgeEvent, ParkingSession
from cloud.schemas import EdgeEventIn
from edge.database import Database
from edge.sync import SyncWorker

POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        not POSTGRES_URL,
        reason="TEST_POSTGRES_URL not configured; real PostgreSQL test skipped",
    ),
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_edge_to_real_postgres_over_http(tmp_path: Path, monkeypatch):
    database_name = urlparse(POSTGRES_URL).path.removeprefix("/")
    assert database_name.endswith("_test"), (
        "Refusing destructive integration setup: TEST_POSTGRES_URL database name "
        "must end with '_test'"
    )
    # Chạy migration thật, không dùng create_all giả lập production.
    previous_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = POSTGRES_URL
    get_settings.cache_clear()
    try:
        command.upgrade(Config("cloud/alembic.ini"), "head")
    finally:
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url
        get_settings.cache_clear()

    settings = Settings(
        DATABASE_URL=POSTGRES_URL,
        SMARTPARKING_DEVICE_TOKEN="postgres-device-token-at-least-24-chars",
        ADMIN_USERNAME="admin",
        ADMIN_PASSWORD_HASH="$argon2id$v=19$m=65536,t=3,p=4$placeholder$placeholder",
        NICEGUI_STORAGE_SECRET="postgres-test-storage-secret-32-chars",
        APP_ENV="test",
    )
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    inspector = inspect(engine)
    assert {"edge_events", "parking_sessions", "admin_sessions"} <= set(
        inspector.get_table_names()
    )
    assert {
        "ix_edge_events_timestamp",
        "ix_edge_events_plate",
        "ix_edge_events_uid",
        "ix_edge_events_type",
        "ix_edge_events_session",
    } <= {item["name"] for item in inspector.get_indexes("edge_events")}
    assert inspector.get_pk_constraint("edge_events")["constrained_columns"] == [
        "event_id"
    ]
    assert "ck_edge_events_type" in {
        item["name"] for item in inspector.get_check_constraints("edge_events")
    }
    parking_uniques = {
        tuple(item["column_names"])
        for item in inspector.get_unique_constraints("parking_sessions")
    }
    assert ("entry_event_id",) in parking_uniques
    assert ("exit_event_id",) in parking_uniques
    with factory() as db, db.begin():
        db.execute(delete(AdminSession))
        db.execute(delete(ParkingSession))
        db.execute(delete(EdgeEvent))

    api = FastAPI()
    api.include_router(build_api_router(settings, factory))
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(api, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started

    edge_db = Database(tmp_path / "edge.db")
    edge_db.register_card("A1B2C3D4")
    event = {
        "event_id": "44444444-4444-4444-8444-444444444444",
        "device_id": "raspberrypi_01",
        "uid": "A1B2C3D4",
        "plate": "29AC88688",
        "timestamp": "2026-10-27T14:00:00+00:00",
        "ocr_confidence": 0.9,
        "detection_confidence": 0.8,
        "iqa_score": 0.7,
        "inference_ms": 123.4,
        "reason": None,
    }
    edge_db.commit_event(event)
    original_payload = edge_db.pending()[0][1]
    worker = SyncWorker(
        edge_db,
        f"http://127.0.0.1:{port}/api/v1/edge/events",
        settings.device_token,
        allow_http_localhost=True,
    )
    try:
        assert worker.run_once() == (1, 0)
        assert edge_db.summary()["pending_sync"] == 0
        with factory() as db:
            assert db.scalar(select(func.count()).select_from(EdgeEvent)) == 1
        worker.send(original_payload)
        with factory() as db:
            assert db.scalar(select(func.count()).select_from(EdgeEvent)) == 1

        # Tạo ENTRY rồi EXIT bằng chính SQLite Edge, nhưng gửi EXIT lên trước.
        edge_db.register_card("BBBB2222")
        second_entry = {
            "event_id": "55555555-5555-4555-8555-555555555555",
            "device_id": "raspberrypi_01",
            "uid": "BBBB2222",
            "plate": "51A12345",
            "timestamp": "2026-10-27T14:10:00+00:00",
            "ocr_confidence": 0.91,
            "detection_confidence": 0.82,
            "iqa_score": 0.73,
            "inference_ms": 110.0,
            "reason": None,
        }
        second_exit = {
            **second_entry,
            "event_id": "66666666-6666-4666-8666-666666666666",
            "timestamp": "2026-10-27T14:20:00+00:00",
        }
        assert edge_db.commit_event(second_entry)["type"] == "ENTRY"
        assert edge_db.commit_event(second_exit)["type"] == "EXIT"
        second_payloads = {
            payload["event_type"]: payload
            for _, payload, _ in edge_db.pending()
            if payload["uid"] == "BBBB2222"
        }
        worker.send(second_payloads["EXIT"])
        worker.send(second_payloads["ENTRY"])
        with factory() as db:
            session = db.get(ParkingSession, second_entry["event_id"])
            assert session.entry_event_id == second_entry["event_id"]
            assert session.exit_event_id == second_exit["event_id"]
            assert session.entry_at is not None
            assert session.exit_at is not None
            assert db.scalar(select(func.count()).select_from(ParkingSession)) == 2

        # DENY/REVIEW/ERROR thật từ record_error không được tạo thêm session.
        for index, event_type in enumerate(("DENY", "REVIEW", "ERROR"), start=1):
            edge_db.record_error(
                {
                    "event_id": f"non-session-{index}",
                    "device_id": "raspberrypi_01",
                    "uid": None,
                    "timestamp": f"2026-10-27T15:0{index}:00+00:00",
                    "reason": "SIMULATED",
                },
                event_type,
            )
        for _, payload, _ in edge_db.pending():
            if payload["event_type"] in {"DENY", "REVIEW", "ERROR"}:
                worker.send(payload)
        with factory() as db:
            assert db.scalar(select(func.count()).select_from(ParkingSession)) == 2

        rollback_payload = EdgeEventIn.model_validate(
            {
                **second_entry,
                "event_id": "rollback-postgres-event",
                "session_id": "rollback-postgres-session",
                "event_type": "ENTRY",
            }
        )

        def fail_projection(*_args, **_kwargs):
            raise RuntimeError("simulated PostgreSQL projection failure")

        monkeypatch.setattr(api_module, "_project_session", fail_projection)
        with factory() as db, pytest.raises(RuntimeError, match="projection failure"):
            ingest_event(db, rollback_payload)
        with factory() as db:
            assert db.get(EdgeEvent, rollback_payload.event_id) is None
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        engine.dispose()
