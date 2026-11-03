from __future__ import annotations

from cloud import evidence as evidence_api
from cloud.main import app
from cloud.auth import SESSION_COOKIE, create_admin_session
from cloud.models import EvidenceImage


class MemoryBucket:
    def __init__(self):
        self.objects={}
    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)]=Body
    def get_object(self, Bucket, Key):
        from io import BytesIO
        return {'Body': BytesIO(self.objects[(Bucket, Key)])}


def test_evidence_upload_requires_auth(client, event_payload):
    url=f"/api/v1/edge/evidence/{event_payload['event_id']}/annotated"
    assert client.post(url, content=b'\xff\xd8abc\xff\xd9',
                       headers={'Content-Type':'image/jpeg'}).status_code==401


def test_evidence_private_and_idempotent(client, event_payload, monkeypatch):
    bucket=MemoryBucket()
    monkeypatch.setattr(evidence_api, '_client', lambda settings: bucket)
    event_id=event_payload['event_id']
    assert client.post('/api/v1/edge/events',json=event_payload,
        headers={'Authorization':'Bearer test-device-token-at-least-24-characters',
                 'Idempotency-Key':event_id}).status_code==201
    url=f'/api/v1/edge/evidence/{event_id}/annotated'
    img=b'\xff\xd8data\xff\xd9'
    headers={'Authorization':'Bearer test-device-token-at-least-24-characters',
             'X-Device-ID':event_payload['device_id'],'Content-Type':'image/jpeg'}
    assert client.post(url,content=img,headers=headers).status_code==201
    assert client.post(url,content=img,headers=headers).status_code==201
    assert client.post(url,content=b'\xff\xd8different\xff\xd9',headers=headers).status_code==409
    image_url=f'/api/evidence/{event_id}/annotated.jpg'
    assert client.get(image_url).status_code==401
    with app.state.session_factory() as db:
        token, _=create_admin_session(db,app.state.settings,'admin',None)
    client.cookies.set(SESSION_COOKIE,token)
    result=client.get(image_url)
    assert result.status_code==200
    assert result.content==img
    assert result.headers['content-type']=='image/jpeg'
    with app.state.session_factory() as db:
        assert db.get(EvidenceImage,(event_id,'annotated')) is not None


def test_event_must_exist_before_upload(client):
    eid='22222222-2222-4222-8222-222222222222'
    response=client.post(f'/api/v1/edge/evidence/{eid}/plate',
        headers={'Authorization':'Bearer test-device-token-at-least-24-characters',
                 'X-Device-ID':'raspberrypi_01','Content-Type':'image/jpeg'},
        content=b'\xff\xd8valid\xff\xd9')
    assert response.status_code==409
