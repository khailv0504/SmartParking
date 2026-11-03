"""Local crash-safe JPEG spool and an independent sync path for RFID evidence.

All upload attempts happen in the normal Cloud sync thread; no extra camera
and no inference on Cloud. Missing R2 is a RETRY, not an ENTRY/EXIT failure.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import UUID

import cv2

from .evidence_visuals import annotate_scene, crop_evidence

LOG = logging.getLogger(__name__)
MAX_JPEG = 450_000
KINDS = ("annotated", "plate")


def _write_jpeg(path: Path, frame, quality: int = 70) -> None:
    """Atomic rename prevents readers observing a partially-written JPG."""
    if frame.shape[1] > 1024:
        scale = 1024 / frame.shape[1]
        frame = cv2.resize(frame, (1024, max(1, round(frame.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    data = b''
    for q in (quality, 60, 48, 38):
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            raise OSError(f"Cannot JPEG-encode {path.name}")
        data = encoded.tobytes()
        if len(data) <= MAX_JPEG:
            break
    if len(data) > MAX_JPEG:
        raise OSError(f"Evidence {path.name} is too large even at low quality")
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class EvidenceRecorder:
    def __init__(self, db, directory: Path, on_preview=None):
        self.db = db
        self.directory = Path(directory)
        self.on_preview = on_preview

    def save(self, event_id: str, frame, prediction) -> None:
        """Called only after an RFID recognition event; never block the DB decision on failure.

        Successful saves create a SQLite outbox record for each image, after the
        parking event is committed. Retry semantics are idempotent by event/kind.
        """
        UUID(event_id)  # only UUIDs can be used as filenames
        self.directory.mkdir(parents=True, exist_ok=True)
        annotated = annotate_scene(frame, prediction)
        crops = crop_evidence(frame, prediction)
        saved: list[tuple[str, Path]] = []
        for kind, img in (("annotated", annotated), ("plate", crops)):
            if img is None:
                continue
            path = self.directory / f"{event_id}_{kind}.jpg"
            try:
                _write_jpeg(path, img)
                saved.append((kind, path))
            except OSError:
                LOG.exception("Evidence write failed: %s kind=%s", event_id, kind)
        for kind, path in saved:
            self.db.enqueue_evidence(event_id, kind, str(path))
        if crops is not None and self.on_preview is not None:
            self.on_preview(crops)


def upload_pending(db, events_url: str, token: str, max_per_tick: int = 3) -> tuple[int, int]:
    parsed = urlparse(events_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("Evidence sync requires HTTPS")
    origin = parsed._replace(path="", params="", query="", fragment="").geturl().rstrip('/')
    good = bad = 0
    for event_id, kind, filename, attempts in db.pending_evidence(limit=max_per_tick):
        try:
            path = Path(filename)
            if not path.is_file() or path.stat().st_size > MAX_JPEG:
                raise OSError(f"Evidence file missing or oversized: {path.name}")
            data = path.read_bytes()
            if not (data.startswith(b'\xff\xd8') and data.endswith(b'\xff\xd9')):
                raise ValueError("Invalid local JPEG")
            request = Request(
                f"{origin}/api/v1/edge/evidence/{event_id}/{kind}",
                data=data,
                headers={"Content-Type": "image/jpeg", "Authorization": f"Bearer {token}",
                         "X-Device-ID": db.device_id},
                method="POST",
            )
            with urlopen(request, timeout=10) as response:
                if response.status not in (200, 201):
                    raise RuntimeError(f"Unexpected evidence HTTP status {response.status}")
            db.mark_evidence_synced(event_id, kind)
            good += 1
        except (OSError, ValueError, RuntimeError, HTTPError, URLError) as exc:
            db.mark_evidence_retry(event_id, kind, attempts + 1, str(exc)[:400])
            LOG.warning("Evidence upload deferred: event=%s kind=%s error=%s", event_id, kind, type(exc).__name__)
            bad += 1
    return good, bad
