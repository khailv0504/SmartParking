"""At-least-once event sync; cloud MUST dedupe by event_id."""
from __future__ import annotations

import json
import logging
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from .database import Database

LOG = logging.getLogger(__name__)


class SyncWorker:
    def __init__(self, db: Database, url: str, token: str,
                 allow_http_localhost: bool = False):
        parsed = urlparse(url)
        local_dev = parsed.scheme == 'http' and parsed.hostname in ('localhost','127.0.0.1')
        if parsed.scheme != 'https' and not (allow_http_localhost and local_dev):
            raise ValueError('Cloud sync requires HTTPS (except explicitly enabled localhost)')
        if not token:
            raise ValueError('Device token is required')
        self.db = db
        self.url = url
        self.token = token

    def send(self, payload: dict) -> None:
        req = Request(self.url, data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
                      headers={'Content-Type':'application/json','Authorization':f'Bearer {self.token}',
                               'Idempotency-Key':payload['event_id']}, method='POST')
        with urlopen(req,timeout=6) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f'Unexpected HTTP status {response.status}')

    def run_once(self) -> tuple[int,int]:
        success = failed = 0
        for event_id, payload, attempts in self.db.pending():
            try:
                self.send(payload)
                self.db.mark_synced(event_id)
                success += 1
            except (OSError,ValueError,RuntimeError,HTTPError,URLError) as exc:
                self.db.mark_retry(event_id,attempts+1,str(exc))
                LOG.warning('Sync failed for %s: %s',event_id,exc)
                failed += 1
        return success,failed

    def run_forever(self, stop, interval_s: float = 5):
        while not stop.is_set():
            try:
                self.run_once()
            except Exception:
                LOG.exception('Sync worker unexpected error')
            stop.wait(interval_s)
