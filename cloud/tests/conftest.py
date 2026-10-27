from __future__ import annotations

import os
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from sqlalchemy import delete

# cloud.main tạo ASGI app khi import; đặt cấu hình test trước lần import đầu tiên.
TEST_DB = Path("/tmp/smartparking_cloud_pytest.db")
os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{TEST_DB}",
        "SMARTPARKING_DEVICE_TOKEN": "test-device-token-at-least-24-characters",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD_HASH": PasswordHasher().hash("correct-horse-battery-staple"),
        "NICEGUI_STORAGE_SECRET": "test-storage-secret-at-least-32-characters-long",
        "APP_ENV": "test",
    }
)

from cloud.database import Base  # noqa: E402
from cloud.main import app  # noqa: E402
from cloud.models import AdminSession, EdgeEvent, ParkingSession  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def schema():
    Base.metadata.create_all(app.state.engine)
    yield
    app.state.engine.dispose()
    TEST_DB.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def clean_database(schema):
    with app.state.session_factory() as db, db.begin():
        db.execute(delete(AdminSession))
        db.execute(delete(ParkingSession))
        db.execute(delete(EdgeEvent))


@pytest.fixture()
def client():
    with TestClient(app, follow_redirects=False) as test_client:
        yield test_client


@pytest.fixture()
def event_payload():
    return {
        "event_id": "11111111-1111-4111-8111-111111111111",
        "device_id": "raspberrypi_01",
        "uid": "A1B2C3D4",
        "plate": "29AC88688",
        "event_type": "ENTRY",
        "reason": None,
        "session_id": "11111111-1111-4111-8111-111111111111",
        "timestamp": "2026-10-27T14:00:00.000+00:00",
        "ocr_confidence": 0.6832,
        "detection_confidence": 0.8749,
        "iqa_score": 0.9571,
        "inference_ms": 1135.31,
    }


@pytest.fixture()
def device_headers(event_payload):
    return {
        "Authorization": "Bearer test-device-token-at-least-24-characters",
        "Idempotency-Key": event_payload["event_id"],
    }
