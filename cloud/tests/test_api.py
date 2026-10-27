from __future__ import annotations

import pytest
from sqlalchemy import func, select

from cloud.main import app
from cloud.models import EdgeEvent


def post(client, payload, token="test-device-token-at-least-24-characters"):
    return client.post(
        "/api/v1/edge/events",
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": payload["event_id"],
        },
    )


def test_valid_entry(client, event_payload):
    response = post(client, event_payload)
    assert response.status_code == 201
    assert response.json() == {
        "event_id": event_payload["event_id"],
        "status": "accepted",
        "duplicate": False,
    }
    with app.state.session_factory() as db:
        row = db.get(EdgeEvent, event_payload["event_id"])
        assert row.plate == "29AC88688"
        assert (
            row.event_timestamp.utcoffset() is not None
            or row.event_timestamp.tzinfo is None
        )


def test_valid_exit(client, event_payload):
    event_payload.update(
        event_id="22222222-2222-4222-8222-222222222222", event_type="EXIT"
    )
    assert post(client, event_payload).status_code == 201


@pytest.mark.parametrize("event_type", ["DENY", "REVIEW", "ERROR"])
def test_non_session_event_types(client, event_payload, event_type):
    event_payload.update(
        event_type=event_type, session_id=None, plate=None, reason="SIMULATED"
    )
    assert post(client, event_payload).status_code == 201


def test_wrong_and_missing_bearer_token(client, event_payload, device_headers):
    assert post(client, event_payload, "wrong-token").status_code == 401
    assert (
        client.post(
            "/api/v1/edge/events",
            json=event_payload,
            headers={"Idempotency-Key": event_payload["event_id"]},
        ).status_code
        == 401
    )


def test_invalid_payload_and_idempotency_header(client, event_payload, device_headers):
    bad = {**event_payload}
    bad.pop("timestamp")
    assert (
        client.post("/api/v1/edge/events", json=bad, headers=device_headers).status_code
        == 422
    )
    headers = {**device_headers, "Idempotency-Key": "different"}
    assert (
        client.post(
            "/api/v1/edge/events", json=event_payload, headers=headers
        ).status_code
        == 422
    )


def test_duplicate_same_payload_is_2xx_and_not_inserted_twice(client, event_payload):
    assert post(client, event_payload).status_code == 201
    response = post(client, event_payload)
    assert response.status_code == 201
    assert response.json()["duplicate"] is True
    with app.state.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(EdgeEvent)) == 1


def test_duplicate_different_payload_is_conflict(client, event_payload):
    assert post(client, event_payload).status_code == 201
    changed = {**event_payload, "plate": "51A99999"}
    response = post(client, changed)
    assert response.status_code == 409
    assert response.json()["detail"] == "event_id already exists with different content"
    with app.state.session_factory() as db:
        assert db.get(EdgeEvent, event_payload["event_id"]).plate == "29AC88688"


@pytest.mark.parametrize(
    ("field", "value"),
    [("ocr_confidence", -0.1), ("detection_confidence", 1.01), ("iqa_score", 2)],
)
def test_confidence_range(client, event_payload, field, value):
    event_payload[field] = value
    assert post(client, event_payload).status_code == 422


def test_timestamp_requires_timezone(client, event_payload):
    event_payload["timestamp"] = "2026-10-27T14:00:00"
    assert post(client, event_payload).status_code == 422


def test_nullable_edge_fields(client, event_payload):
    event_payload.update(
        event_type="ERROR",
        uid=None,
        plate=None,
        reason=None,
        session_id=None,
        ocr_confidence=None,
        detection_confidence=None,
        iqa_score=None,
        inference_ms=None,
    )
    assert post(client, event_payload).status_code == 201


def test_health_contains_no_secrets(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.text
    assert response.json() == {"status": "ok"}
    assert "test-device-token" not in body
    assert "DATABASE_URL" not in body
