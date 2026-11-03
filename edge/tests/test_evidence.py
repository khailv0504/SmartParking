from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

import cv2
import numpy as np

from edge.database import Database, utcnow
from edge.evidence import EvidenceRecorder, upload_pending
from edge.evidence_visuals import annotate_scene, GREEN, YELLOW
from edge.recognition import Box, Recognition, OCR_CLASSES


def sample_recognition():
    plate = Box(65, 55, 255, 175, .83, 1)
    c1 = Box(85, 91, 108, 134, .91, OCR_CLASSES.index('2'))
    c2 = Box(115, 91, 140, 134, .85, OCR_CLASSES.index('9'))
    return Recognition('29D266666', .85, .83, .82, ('29D2','66666'), (c1,c2), 24., plate_box=plate)


def test_bbox_colors_in_real_frame_coordinates():
    scene = annotate_scene(np.zeros((240,320,3), np.uint8), sample_recognition())
    assert np.array_equal(scene[175,65], np.array(GREEN, np.uint8))
    assert np.array_equal(scene[134,85], np.array(YELLOW, np.uint8))


def test_evidence_is_spooled_then_uploaded_after_event_synced():
    with tempfile.TemporaryDirectory() as tmp:
        db=Database(Path(tmp)/'parking.db')
        db.register_card('AAAA1111')
        event_id=str(uuid4())
        event={'event_id':event_id, 'device_id':db.device_id, 'uid':'AAAA1111',
               'plate':'29D266666','timestamp':utcnow(),'ocr_confidence':.85}
        assert db.commit_event(event)['type']=='ENTRY'
        preview=[]
        rec=EvidenceRecorder(db, Path(tmp)/'evidence', on_preview=preview.append)
        rec.save(event_id, np.full((240,320,3), 90, np.uint8), sample_recognition())
        assert len(preview)==1
        assert db.pending_evidence()==[]  # parent event has not synced yet
        db.mark_synced(event_id)
        queued=db.pending_evidence()
        assert {row[1] for row in queued}=={'annotated','plate'}
        for _,kind,filename,_ in queued:
            assert Path(filename).is_file()
            assert cv2.imread(filename) is not None
        class Ack:
            status=201
            def __enter__(self): return self
            def __exit__(self,*args): return False
        with patch('edge.evidence.urlopen', return_value=Ack()) as mock:
            ok, failed=upload_pending(db,'https://example.com/api/v1/edge/events','fake-token')
        assert (ok,failed)==(2,0)
        assert mock.call_count==2
        assert db.pending_evidence()==[]
