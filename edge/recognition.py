"""ONNX YOLO detector + YOLO character recognition (NOT CTC OCR).

Output shape assumed from verified files:
- detector.onnx   (1, 6, 8400)   2 classes (names '0' and '1')
- recognizer.onnx (1, 34, 2100)  30 character classes

Both models export Ultralytics detect heads without embedded NMS.
Uses OpenCV DNN to avoid a PyTorch/Ultralytics runtime on Raspberry Pi.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Protocol

import cv2
import numpy as np

from .alpr_preprocess.quality import QualityAssessor, QualityConfig

# Explicit class order comes from recognizer.onnx model metadata.
OCR_CLASSES = "0123456789ABCDEFGHKLMNPSTUVXYZ"
assert len(OCR_CLASSES) == 30
PLATE_CHARS = re.compile(r"^[A-Z0-9]+$")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Recognition:
    plate: str | None
    confidence: float  # heuristic minimum character score, not calibrated probability
    detector_confidence: float
    quality_score: float
    row_texts: tuple[str, ...]
    boxes: tuple[Box, ...]
    latency_ms: float
    reason: str | None = None
    plate_box: Box | None = None


class Recognizer(Protocol):
    def recognize(self, frame: np.ndarray) -> Recognition: ...


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize with padding and preserve aspect ratio; return scale, left, top."""
    if frame is None or frame.size == 0 or size <= 0:
        raise ValueError("Invalid image or input size")
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = min(size, round(w * scale)), min(size, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, scale, left, top


class YOLOBoxes:
    """Generic inference for supplied fixed-shape YOLO detect ONNX graphs."""

    def __init__(self, path: str | Path, size: int, classes: int,
                 conf: float = 0.35, iou: float = 0.45, threads: int = 2):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        self.size = size
        self.classes = classes
        self.conf = conf
        self.iou = iou
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        # OpenCV thread setting is process-wide, set once during model initialization.
        cv2.setNumThreads(max(1, threads))

    def predict(self, frame: np.ndarray) -> list[Box]:
        canvas, scale, pad_x, pad_y = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0,
                                     size=(self.size, self.size), swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        out = np.asarray(self.net.forward())
        if out.ndim != 3 or out.shape[0] != 1 or out.shape[1] != 4 + self.classes:
            raise RuntimeError(
                f"Unexpected YOLO output {out.shape}; expected [1,{4 + self.classes},N]. "
                "Inspect export metadata before changing decoding."
            )
        preds = out[0].T  # (N, 4 + C), xywh center in letterboxed input pixels
        cls_scores = preds[:, 4:]
        class_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(preds)), class_ids]
        keep = np.flatnonzero(scores >= self.conf)
        h, w = frame.shape[:2]
        candidates: list[Box] = []
        boxes_xywh: list[list[int]] = []
        nms_scores: list[float] = []
        for idx in keep:
            cx, cy, bw, bh = preds[idx, :4]
            x1 = max(0, min(w, int(round((cx - bw / 2 - pad_x) / scale))))
            y1 = max(0, min(h, int(round((cy - bh / 2 - pad_y) / scale))))
            x2 = max(0, min(w, int(round((cx + bw / 2 - pad_x) / scale))))
            y2 = max(0, min(h, int(round((cy + bh / 2 - pad_y) / scale))))
            if x2 <= x1 or y2 <= y1:
                continue
            box = Box(x1, y1, x2, y2, float(scores[idx]), int(class_ids[idx]))
            candidates.append(box)
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            nms_scores.append(box.confidence)
        if not candidates:
            return []
        # Class-agnostic NMS avoids two conflicting characters at the same location.
        indices = cv2.dnn.NMSBoxes(boxes_xywh, nms_scores,
                                  score_threshold=self.conf, nms_threshold=self.iou)
        selected = np.asarray(indices).reshape(-1).tolist()
        return sorted((candidates[i] for i in selected),
                      key=lambda b: b.confidence, reverse=True)


def group_text_rows(boxes: list[Box], chars: str = OCR_CLASSES) -> tuple[str, ...]:
    """Order character bounding boxes into 1 or 2 rows by vertical centers.

    Threshold is relative to median detected character height. It is heuristic:
    perspective distortion, severe tilt, or tiny crops need targeted evaluation.
    """
    if not boxes:
        return ()
    med_h = float(np.median([max(1, b.height) for b in boxes]))
    sorted_y = sorted(boxes, key=lambda b: b.cy)
    # A row's center evolves as characters are accumulated.
    rows: list[list[Box]] = []
    centers: list[float] = []
    for box in sorted_y:
        if not rows or abs(box.cy - centers[-1]) > 0.65 * med_h:
            rows.append([box])
            centers.append(box.cy)
        else:
            rows[-1].append(box)
            centers[-1] = float(np.mean([c.cy for c in rows[-1]]))
    if len(rows) > 2:
        # Ambiguous row grouping is unsafe to commit to a parking session.
        return ()
    return tuple("".join(chars[b.class_id] for b in sorted(row, key=lambda x: x.cx))
                 for row in rows)


class YOLOPlateRecognizer:
    def __init__(self, detector_path: str | Path, ocr_path: str | Path,
                 detection_conf: float = 0.35, character_conf: float = 0.35,
                 strict_roi_quality: bool = False, threads: int = 2):
        self.detector = YOLOBoxes(detector_path, 640, 2, detection_conf, threads=threads)
        self.characters = YOLOBoxes(ocr_path, 320, 30, character_conf, threads=threads)
        self.assessor = QualityAssessor(QualityConfig())
        self.strict_roi_quality = strict_roi_quality

    def recognize(self, frame: np.ndarray) -> Recognition:
        start = time.perf_counter()
        plates = self.detector.predict(frame)
        if not plates:
            return Recognition(None, 0, 0, 0, (), (),
                               (time.perf_counter() - start)*1000, "PLATE_NOT_DETECTED")
        # A single lane should have one plate. If several are detected, fail closed
        # rather than silently linking an arbitrary vehicle to the RFID UID.
        if len(plates) > 1:
            return Recognition(None, 0, plates[0].confidence, 0, (), (),
                               (time.perf_counter() - start)*1000, "MULTIPLE_PLATES")
        p = plates[0]
        xpad = round((p.x2 - p.x1) * 0.04)
        ypad = round((p.y2 - p.y1) * 0.04)
        h, w = frame.shape[:2]
        x1, y1 = max(0, p.x1-xpad), max(0, p.y1-ypad)
        x2, y2 = min(w, p.x2+xpad), min(h, p.y2+ypad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return Recognition(None, 0, p.confidence, 0, (), (),
                               (time.perf_counter()-start)*1000, "INVALID_PLATE_ROI", plate_box=p)
        report = self.assessor.assess(crop)
        if self.strict_roi_quality and not report.accepted:
            return Recognition(None, 0, p.confidence, report.quality_score, (), (),
                               (time.perf_counter()-start)*1000, "POOR_ROI_QUALITY", plate_box=p)
        raw_chars = self.characters.predict(crop)
        rows = group_text_rows(raw_chars)
        plate = "".join(rows)
        if not rows or not (7 <= len(plate) <= 10) or not PLATE_CHARS.fullmatch(plate):
            return Recognition(None, 0, p.confidence, report.quality_score, rows,
                               tuple(raw_chars), (time.perf_counter()-start)*1000,
                               "INVALID_OCR_LAYOUT", plate_box=p)
        # The lowest character confidence catches a weak link but isn't a
        # calibrated probability of exact-string correctness.
        confidence = min(c.confidence for c in raw_chars)
        return Recognition(plate, confidence, p.confidence, report.quality_score,
                           rows, tuple(raw_chars), (time.perf_counter()-start)*1000, plate_box=p)


class MockRecognizer:
    def __init__(self, plate: str = "59A12345", confidence: float = 0.95):
        self.plate = plate
        self.confidence = confidence

    def recognize(self, frame: np.ndarray) -> Recognition:
        return Recognition(self.plate, self.confidence, 0.99, 0.8,
                           (self.plate,), (), 1.0)
