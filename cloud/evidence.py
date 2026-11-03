"""Private R2 evidence storage: authenticated upload and authenticated viewing.

Disabled gracefully until R2_* environment is configured on Render.
This API deliberately does not issue public presigned URLs.
"""
from __future__ import annotations

import asyncio
import hashlib
import secrets
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import Response
from sqlalchemy.orm import Session, sessionmaker

from .auth import resolve_admin_session
from .config import Settings
from .models import EdgeEvent, EvidenceImage

MAX_BYTES = 450_000
KINDS = {"annotated", "plate"}


def _uid(s: str) -> str:
    try:
        parsed = UUID(s)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="Invalid event ID") from exc
    if str(parsed) != s:
        raise HTTPException(status_code=422, detail="Invalid event ID")
    return s


def _auth_device(authorization: str | None, expected: str) -> None:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid device credentials")


def _client(settings: Settings):
    if not all((settings.r2_endpoint, settings.r2_bucket,
                settings.r2_access_key_id, settings.r2_secret_access_key)):
        raise HTTPException(status_code=503, detail="Evidence object storage is not configured")
    import boto3
    from botocore.config import Config

    if not settings.r2_endpoint.startswith("https://"):
        raise HTTPException(status_code=503, detail="Evidence object storage endpoint invalid")
    return boto3.client(
        "s3", endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4", connect_timeout=4, read_timeout=8,
                      retries={"max_attempts": 2}),
    )


def build_evidence_router(settings: Settings, factory: sessionmaker[Session]) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.post('/api/v1/edge/evidence/{event_id}/{kind}', status_code=201)
    async def upload_evidence(
        event_id: str, kind: str, request: Request,
        authorization: str | None = Header(default=None),
        device_id: str | None = Header(default=None, alias='X-Device-ID'),
    ) -> dict:
        _auth_device(authorization, settings.device_token)
        _uid(event_id)
        if kind not in KINDS:
            raise HTTPException(status_code=422, detail='Invalid evidence kind')
        if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'image/jpeg':
            raise HTTPException(status_code=415, detail='JPEG required')
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise HTTPException(status_code=413, detail='Evidence image too large')
        if len(data) < 4 or not data.startswith(b'\xff\xd8') or not data.endswith(b'\xff\xd9'):
            raise HTTPException(status_code=422, detail='Invalid JPEG framing')
        payload = bytes(data)
        digest = hashlib.sha256(payload).hexdigest()
        with factory() as db:
            event = db.get(EdgeEvent, event_id)
            if event is None:
                raise HTTPException(status_code=409, detail='Event not synchronized yet')
            if not device_id or event.device_id != device_id:
                raise HTTPException(status_code=403, detail='Device ID does not match event')
            prior = db.get(EvidenceImage, (event_id, kind))
            if prior is not None:
                if prior.sha256 != digest:
                    raise HTTPException(status_code=409, detail='Evidence content conflict')
                return {'status': 'already_uploaded', 'event_id': event_id, 'kind': kind}
        key = f"parking/{device_id}/{event_id}/{kind}.jpg"
        s3 = _client(settings)
        try:
            await asyncio.to_thread(
                s3.put_object, Bucket=settings.r2_bucket, Key=key,
                Body=payload, ContentType='image/jpeg',
                Metadata={'sha256': digest},
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail='Evidence object storage temporarily unavailable') from exc
        try:
            with factory() as db:
                existing = db.get(EvidenceImage, (event_id, kind))
                if existing is None:
                    db.add(EvidenceImage(
                        event_id=event_id, kind=kind, object_key=key,
                        sha256=digest, size_bytes=len(payload),
                        uploaded_at=datetime.now(timezone.utc),
                    ))
                    db.commit()
                elif existing.sha256 != digest:
                    raise HTTPException(status_code=409, detail='Evidence content conflict')
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail='Evidence metadata temporarily unavailable') from exc
        return {'status': 'uploaded', 'event_id': event_id, 'kind': kind}

    @router.get('/api/evidence/{event_id}/{kind}.jpg')
    async def view_evidence(event_id: str, kind: str, request: Request) -> Response:
        try:
            admin = await asyncio.to_thread(resolve_admin_session, request, factory)
        except Exception as exc:
            raise HTTPException(status_code=503, detail='Authentication unavailable') from exc
        if admin is None:
            raise HTTPException(status_code=401, detail='Administrator login required')
        _uid(event_id)
        if kind not in KINDS:
            raise HTTPException(status_code=404, detail='Unknown image type')
        with factory() as db:
            row = db.get(EvidenceImage, (event_id, kind))
            if row is None:
                raise HTTPException(status_code=404, detail='Evidence image not uploaded')
            key = row.object_key
        s3 = _client(settings)
        try:
            obj = await asyncio.to_thread(s3.get_object, Bucket=settings.r2_bucket, Key=key)
            # Cap reads even if object-store content length is malformed.
            def read_image():
                stream = obj['Body']
                try:
                    return stream.read(MAX_BYTES + 1)
                finally:
                    stream.close()
            data = await asyncio.to_thread(read_image)
            if len(data) > MAX_BYTES:
                raise ValueError('Unexpected oversized object')
        except Exception as exc:
            raise HTTPException(status_code=503, detail='Evidence image temporarily unavailable') from exc
        return Response(data, media_type='image/jpeg',
                        headers={'Cache-Control': 'private, no-store, max-age=0'})

    return router
