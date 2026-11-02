"""RFID triggers capture -> YOLO plate -> YOLO characters -> SQLite.

Single-controller/one-lane design. Not a physical gate/barrier controller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import logging
import time
from uuid import uuid4

from .database import Database, utcnow
from .recognition import Recognizer, Recognition

LOG = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = 'IDLE'
    CARD_READ = 'CARD_READ'
    CAPTURING = 'CAPTURING'
    RECOGNIZING = 'RECOGNIZING'
    VALIDATING = 'VALIDATING'
    COMMITTING = 'COMMITTING'
    DONE = 'DONE'
    FAILED = 'FAILED'


@dataclass
class Outcome:
    event_id: str
    state: str
    action: str
    reason: str | None
    uid: str
    plate: str | None
    latency_ms: float
    session_id: str | None = None


class ParkingController:
    def __init__(self, db: Database, selector, recognizer: Recognizer,
                 cooldown_seconds: float = 3.0, min_ocr_conf: float = 0.60):
        self.db = db
        self.selector = selector
        self.recognizer = recognizer
        self.cooldown = cooldown_seconds
        self.min_ocr_conf = min_ocr_conf
        self.state = State.IDLE
        self._last_tap: dict[str,float] = {}

    def process(self, uid: str) -> Outcome:
        """Handle one physical RFID tap. Duplicate scans within cooldown ignored.

        Important: for actual physical passage, a later tap is assumed to alternate
        ENTRY/EXIT for the UID. No gate sensor in current prototype; this cannot
        prove the vehicle actually moved. See README deployment limitations.
        """
        uid = uid.strip().upper()
        if not uid:
            raise ValueError('Empty UID')
        now = time.monotonic()
        last = self._last_tap.get(uid)
        if last is not None and now - last < self.cooldown:
            return Outcome('',State.IDLE.value,'IGNORED','RFID_COOLDOWN',uid,None,0)
        self._last_tap[uid] = now
        event_id = str(uuid4())
        begin = time.perf_counter()
        event = dict(event_id=event_id,device_id=self.db.device_id,uid=uid,
                     plate=None, timestamp=utcnow(),ocr_confidence=None,
                     detection_confidence=None,iqa_score=None,
                     inference_ms=None,image_path=None,reason=None)
        try:
            self.state = State.CARD_READ
            if not self.db.card_authorized(uid):
                event['reason'] = 'CARD_NOT_AUTHORIZED'
                self.state = State.COMMITTING
                result = self.db.record_error(event,'DENY')
                self.state = State.DONE
                return self._finish(result,uid,event,begin)

            self.state = State.CAPTURING
            candidates = self.selector.collect(event_id)
            if not candidates:
                event['reason'] = 'NO_ACCEPTABLE_FRAME'
                self.state = State.COMMITTING
                result = self.db.record_error(event,'REVIEW')
                self.state = State.DONE
                return self._finish(result,uid,event,begin)

            self.state = State.RECOGNIZING
            # Sequential attempts limit RAM. Do not always choose the highest
            # IQA frame as the best OCR result: run OCR over up to K candidates.
            recognized: list[tuple[Recognition,float]] = []
            for frame, iqa_score in candidates:
                prediction = self.recognizer.recognize(frame)
                recognized.append((prediction,iqa_score))
            valid = [(r,q) for r,q in recognized if r.plate and r.confidence >= self.min_ocr_conf]
            self.state = State.VALIDATING
            if not valid:
                best,iq = max(recognized,key=lambda item:item[0].confidence)
                event.update(ocr_confidence=best.confidence,
                             detection_confidence=best.detector_confidence,
                             iqa_score=iq, inference_ms=sum(r.latency_ms for r,_ in recognized),
                             reason=best.reason or 'LOW_OCR_CONFIDENCE')
            else:
                # A disagreement between distinct valid frames is not a reliable event.
                plate_sets = {r.plate for r,_ in valid}
                best,iq = max(valid,key=lambda item:item[0].confidence)
                event.update(plate=best.plate,
                             ocr_confidence=best.confidence,
                             detection_confidence=best.detector_confidence,
                             iqa_score=iq,inference_ms=sum(r.latency_ms for r,_ in recognized))
                if len(plate_sets) > 1:
                    event['reason'] = 'OCR_DISAGREEMENT'
            self.state = State.COMMITTING
            result = self.db.commit_event(event,self.min_ocr_conf)
            self.state = State.DONE
            return self._finish(result,uid,event,begin)
        except Exception as exc:
            self.state = State.FAILED
            LOG.exception('Controller event %s failed',event_id)
            event['reason'] = 'PROCESSING_ERROR'
            try:
                result = self.db.record_error(event,'ERROR')
            except Exception:
                # DB unavailable: caller/service logs exception, no fake success.
                LOG.exception('Cannot persist failing event %s',event_id)
                raise
            return self._finish(result,uid,event,begin)
        finally:
            self.state = State.IDLE
            # Bound cache memory in long-running process.
            if len(self._last_tap) > 2000:
                cutoff = time.monotonic() - max(self.cooldown,10.0)
                self._last_tap = {k:v for k,v in self._last_tap.items() if v >= cutoff}

    def _finish(self, result: dict, uid: str, event: dict, begin: float) -> Outcome:
        return Outcome(event['event_id'],self.state.value,result['type'],result.get('reason'),
                       uid,event.get('plate'),round((time.perf_counter()-begin)*1000,2),
                       result.get('session_id'))
