import sqlite3
import tempfile
import unittest
from pathlib import Path

from edge.controller import ParkingController
from edge.database import Database
from edge.frame_capture import MockFrameSelector
from edge.recognition import MockRecognizer, Recognition


class BrokenSelector:
    def collect(self,event_id):
        raise RuntimeError('Simulated camera disconnection')


class PoorSelector:
    def collect(self,event_id):
        return []


class ConflictingRecognizer:
    def __init__(self):
        self.n=0
    def recognize(self,frame):
        self.n+=1
        plate = '59A12345' if self.n%2 else '59A99999'
        return Recognition(plate,.95,.98,.9,(plate,),(),1.0)


class TwoFrames:
    def collect(self,event_id):
        import numpy as np
        f=np.zeros((100,100,3),dtype=np.uint8)
        return [(f,.8),(f,.9)]


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.db=Database(Path(self.t.name)/'parking.db')
        self.db.register_card('A1B2C3D4')
        self.ctrl=ParkingController(self.db,MockFrameSelector(),MockRecognizer('59A12345'),cooldown_seconds=0)

    def tearDown(self):
        self.t.cleanup()

    def test_entry_exit_and_outbox(self):
        one=self.ctrl.process('A1B2C3D4')
        self.assertEqual(one.action,'ENTRY')
        two=self.ctrl.process('A1B2C3D4')
        self.assertEqual(two.action,'EXIT')
        self.assertEqual(one.session_id,two.session_id)
        self.assertEqual(self.db.summary(),{'open_sessions':0,'events':2,'pending_sync':2})

    def test_unauthorized_card_denied_before_capture(self):
        ctrl=ParkingController(self.db,BrokenSelector(),MockRecognizer(),cooldown_seconds=0)
        result=ctrl.process('BADUID')
        self.assertEqual(result.action,'DENY')
        self.assertEqual(result.reason,'CARD_NOT_AUTHORIZED')

    def test_missing_frames_review(self):
        ctrl=ParkingController(self.db,PoorSelector(),MockRecognizer(),cooldown_seconds=0)
        self.assertEqual(ctrl.process('A1B2C3D4').action,'REVIEW')

    def test_cannot_exit_with_other_plate(self):
        self.assertEqual(self.ctrl.process('A1B2C3D4').action,'ENTRY')
        wrong=ParkingController(self.db,MockFrameSelector(),MockRecognizer('59A99999'),cooldown_seconds=0)
        result=wrong.process('A1B2C3D4')
        self.assertEqual((result.action,result.reason),('REVIEW','PLATE_MISMATCH'))
        self.assertEqual(self.db.summary()['open_sessions'],1)

    def test_camera_failure_recorded(self):
        ctrl=ParkingController(self.db,BrokenSelector(),MockRecognizer(),cooldown_seconds=0)
        self.assertEqual(ctrl.process('A1B2C3D4').action,'ERROR')
        self.assertEqual(self.db.summary()['pending_sync'],1)

    def test_low_confidence_no_session(self):
        ctrl=ParkingController(self.db,MockFrameSelector(),MockRecognizer(confidence=.3),cooldown_seconds=0)
        result=ctrl.process('A1B2C3D4')
        self.assertEqual(result.action,'REVIEW')
        self.assertEqual(self.db.summary()['open_sessions'],0)

    def test_ambiguous_multiple_frames(self):
        ctrl=ParkingController(self.db,TwoFrames(),ConflictingRecognizer(),cooldown_seconds=0)
        result=ctrl.process('A1B2C3D4')
        self.assertEqual((result.action,result.reason),('REVIEW','OCR_DISAGREEMENT'))

    def test_prevent_plate_reuse_with_different_uid(self):
        self.db.register_card('BBBB2222')
        self.assertEqual(self.ctrl.process('A1B2C3D4').action,'ENTRY')
        other=ParkingController(self.db,MockFrameSelector(),MockRecognizer('59A12345'),cooldown_seconds=0)
        result=other.process('BBBB2222')
        self.assertEqual((result.action,result.reason),('REVIEW','PLATE_ALREADY_INSIDE'))

    def test_cooldown_same_card(self):
        ctrl=ParkingController(self.db,MockFrameSelector(),MockRecognizer(),cooldown_seconds=100)
        self.assertEqual(ctrl.process('A1B2C3D4').action,'ENTRY')
        self.assertEqual(ctrl.process('A1B2C3D4').action,'IGNORED')
        self.assertEqual(self.db.summary()['events'],1)

    def test_idempotent_event_commit(self):
        event=dict(event_id='fixed',device_id='pi',uid='A1B2C3D4',plate='59A12345',
                   timestamp='2026-10-27T12:00:00+00:00',ocr_confidence=.9)
        self.assertEqual(self.db.commit_event(event)['type'],'ENTRY')
        self.assertEqual(self.db.commit_event(event)['type'],'ENTRY')
        self.assertEqual(self.db.summary()['events'],1)
        self.assertEqual(self.db.summary()['open_sessions'],1)

    def test_sync_pending_mark(self):
        self.ctrl.process('A1B2C3D4')
        item=self.db.pending()[0]
        self.db.mark_synced(item[0])
        self.assertEqual(self.db.summary()['pending_sync'],0)


if __name__ == '__main__': unittest.main()
