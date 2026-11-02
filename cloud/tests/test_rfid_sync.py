from __future__ import annotations

from sqlalchemy import func, select
from cloud.main import app
from cloud.models import RFIDCard, RFIDSnapshotState
from cloud.queries import rfid_cards


TOKEN = 'test-device-token-at-least-24-characters'


def send(client, version=1, cards=None, token=TOKEN, device_id='raspberrypi_rfid_test'):
    if cards is None:
        cards = [{'uid': 'AA11BB22', 'enabled': True,
                  'created_at': '2026-11-02T10:00:00+07:00'}]
    return client.post(
        '/api/v1/edge/card-snapshots',
        headers={'Authorization': f'Bearer {token}'},
        json={'device_id': device_id, 'snapshot_version': version, 'cards': cards},
    )


def test_card_snapshot_sync_and_dashboard(client):
    response = send(client)
    assert response.status_code == 200, response.text
    assert response.json()['status'] == 'accepted'
    with app.state.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(RFIDCard)) == 1
    rows = rfid_cards(app.state.session_factory)
    assert len(rows) == 1 and rows[0]['status'] == 'Đang kích hoạt'
    assert send(client).json()['duplicate'] is True
    assert send(client, version=2, cards=[
        {'uid': 'AA11BB22', 'enabled': False, 'created_at': '2026-11-02T10:00:00+07:00'},
        {'uid': 'CC33DD44', 'enabled': True, 'created_at': '2026-11-02T10:01:00+07:00'},
    ]).status_code == 200
    rows = rfid_cards(app.state.session_factory)
    assert len(rows) == 2 and rows[0]['status'] == 'Vô hiệu hóa'
    assert send(client, version=3, cards=[]).status_code == 200
    assert rfid_cards(app.state.session_factory) == []


def test_stale_and_conflicting_versions(client):
    assert send(client, version=10).status_code == 200
    assert send(client, version=9).status_code == 409
    assert send(client, version=10, cards=[]).status_code == 409
    with app.state.session_factory() as db:
        state = db.get(RFIDSnapshotState, 'raspberrypi_rfid_test')
        assert state.snapshot_version == 10
        assert db.scalar(select(func.count()).select_from(RFIDCard)) == 1


def test_auth_validation_and_uid_normalization(client):
    assert send(client, token='incorrect').status_code == 401
    assert client.post('/api/v1/edge/card-snapshots', json={}).status_code == 422
    assert send(client, cards=[
        {'uid': 'aa11bb22', 'enabled': True, 'created_at': '2026-11-02T10:00:00+07:00'},
        {'uid': 'AA11BB22', 'enabled': True, 'created_at': '2026-11-02T10:00:00+07:00'},
    ]).status_code == 422
    assert send(client, cards=[{'uid': 'aa11bb22', 'enabled': True,
                                'created_at': '2026-11-02T10:00:00'}]).status_code == 422
