import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np

from edge.alpr_preprocess.sources import CameraSource
from edge.frame_capture import FrameSelector
from edge.live_uplink import LiveFrameUplink


class FakeCamera:
    description = 'fake-camera'
    def __enter__(self):
        raise AssertionError('FrameSelector must NOT reopen shared camera')
    def __iter__(self):
        from edge.alpr_preprocess.sources import FramePacket
        for i in range(6):
            yield FramePacket(np.full((240, 320, 3), 80 + i, dtype=np.uint8), i + 1, i * .5)


class LiveSharedCameraTests(unittest.TestCase):
    def test_latest_frame_is_copy(self):
        camera = CameraSource(index=0)
        with camera._lock:
            camera._latest = (8, 1.5, np.zeros((24, 32, 3), dtype=np.uint8))
        packet = camera.latest_frame()
        self.assertEqual(packet.frame_id, 8)
        packet.frame[0, 0, 0] = 99
        self.assertEqual(camera._latest[2][0, 0, 0], 0)

    def test_frame_selector_uses_shared_camera(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            selector = FrameSelector(d, shared_camera=FakeCamera(), quality_threshold=0.0)
            # Quality may reject uniform test images; the key invariant is no camera reopen.
            self.assertIsInstance(selector.collect('test-one'), list)

    def test_uplink_jpeg_encoding(self):
        uplink = LiveFrameUplink(CameraSource(index=0),
             'https://smartparking-cloud.onrender.com/api/v1/edge/events',
             'fake-test-token', 'raspberrypi_01', fps=1, max_width=320)
        jpeg = uplink._encode(np.full((480, 640, 3), 90, dtype=np.uint8))
        self.assertTrue(jpeg.startswith(b'\xff\xd8'))
        self.assertTrue(jpeg.endswith(b'\xff\xd9'))
        self.assertLess(len(jpeg), 200_000)
        self.assertEqual(uplink.url, 'https://smartparking-cloud.onrender.com/api/v1/edge/live-frame')


if __name__ == '__main__':
    unittest.main()
