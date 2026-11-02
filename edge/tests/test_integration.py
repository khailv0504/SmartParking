import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from edge.frame_capture import FrameSelector
from edge.recognition import YOLOBoxes
from edge.database import Database
from edge.sync import SyncWorker


class IntegrationTest(unittest.TestCase):
    def test_existing_frame_pipeline_selects_image(self):
        with tempfile.TemporaryDirectory() as d:
            image_path = Path(d)/'input.jpg'
            rng = np.random.default_rng(42)
            image = rng.integers(30,220,(360,640,3),dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path),image))
            selector=FrameSelector(Path(d)/'runs',mock_image=str(image_path))
            best=selector.collect('evt-test')
            self.assertGreaterEqual(len(best),1)
            self.assertEqual(best[0][0].shape,(360,640,3))
            self.assertTrue((Path(d)/'runs'/'evt-test'/'summary.json').exists())

    def test_sync_marks_success_when_api_acks(self):
        with tempfile.TemporaryDirectory() as d:
            db=Database(Path(d)/'p.db')
            db.register_card('AAAA0000')
            db.record_error(dict(event_id='e1',device_id='p1',uid='AAAA0000',
                                 timestamp='2026-10-27T00:00:00+00:00',reason='SIMULATED'))
            worker=SyncWorker(db,'https://example.com/api/v1/edge/events','token')
            with patch.object(worker,'send') as send:
                sent,failed=worker.run_once()
            self.assertEqual((sent,failed),(1,0))
            self.assertEqual(db.summary()['pending_sync'],0)
            self.assertEqual(send.call_args[0][0]['event_id'],'e1')

    def test_sync_retries_without_dropping_event(self):
        with tempfile.TemporaryDirectory() as d:
            db=Database(Path(d)/'p.db')
            db.record_error(dict(event_id='e2',device_id='p1',uid='BAD',
                                 timestamp='2026-10-27T00:00:00+00:00',reason='SIMULATED'))
            worker=SyncWorker(db,'https://example.com/api/v1/edge/events','token')
            with patch.object(worker,'send',side_effect=OSError('offline')):
                sent,failed=worker.run_once()
            self.assertEqual((sent,failed),(0,1))
            self.assertEqual(db.summary()['pending_sync'],1)

    def test_insecure_cloud_url_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            db=Database(Path(d)/'p.db')
            with self.assertRaises(ValueError):
                SyncWorker(db,'http://public.example/api','token')


if __name__=='__main__': unittest.main()
