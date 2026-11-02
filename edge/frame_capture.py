"""Adapter around the user's FrameSelectionPipeline (not a rewrite)."""
from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np

from .alpr_preprocess.pipeline import FrameSelectionPipeline, PipelineConfig
from .alpr_preprocess.quality import QualityAssessor, QualityConfig
from .alpr_preprocess.sources import CameraSource, ImageSource


class FrameSelector:
    def __init__(self, output_dir: str | Path, sample_fps: float = 3,
                 capture_seconds: float = 1.5, top_k: int = 3,
                 quality_threshold: float = 0.48, camera_index: int = 0,
                 mock_image: str | None = None,
                 shared_camera: CameraSource | None = None):
        self.output_dir = Path(output_dir)
        self.sample_fps = sample_fps
        self.capture_seconds = capture_seconds
        self.top_k = top_k
        self.quality_threshold = quality_threshold
        self.camera_index = camera_index
        self.mock_image = mock_image
        self.shared_camera = shared_camera

    def collect(self, event_id: str) -> list[tuple[np.ndarray, float]]:
        """Return at most K accepted original frames; rank by full-frame IQA.

        Motion gate disabled: RFID is the event trigger; stationary motorcycles
        must still be processed. RFID authorization gates AI inference; live preview can keep camera open.
        """
        output = self.output_dir / event_id
        pipeline = FrameSelectionPipeline(
            QualityAssessor(QualityConfig(quality_threshold=self.quality_threshold)),
            PipelineConfig(sample_fps=self.sample_fps,
                           max_seconds=self.capture_seconds,
                           top_k=self.top_k,
                           preview=False),
            output_dir=output)
        source = self.shared_camera or (ImageSource(self.mock_image) if self.mock_image else CameraSource(
            index=self.camera_index, width=1280, height=720, fps=30, fourcc='MJPG'))
        # The main process owns the shared camera for its whole lifetime.
        # Do not close/reopen it for each RFID tap.
        with (nullcontext(source) if self.shared_camera is not None else source):
            report = pipeline.run(source, image_mode=bool(self.mock_image))
        frames: list[tuple[np.ndarray,float]] = []
        for item in report['best_frames']:
            if not item['accepted']:
                continue
            img = cv2.imread(str(output / item['file']))
            if img is not None:
                frames.append((img, float(item['score'])))
        return frames


class MockFrameSelector:
    """No camera, no disk IO: safe for unit and smoke testing."""
    def __init__(self, frame: np.ndarray | None = None):
        self.frame = frame if frame is not None else np.zeros((100,200,3),dtype=np.uint8)

    def collect(self, event_id: str) -> list[tuple[np.ndarray,float]]:
        return [(self.frame.copy(),0.9)]
