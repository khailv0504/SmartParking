"""Evidence-only drawing. Coordinates are in the ORIGINAL camera image.

YOLO inference is triggered by RFID, NOT performed on every preview frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from .recognition import OCR_CLASSES, Recognition

GREEN = (0, 235, 0)   # BGR
YELLOW = (0, 235, 255)
BLACK = (18, 18, 18)


def _label(img: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.6) -> None:
    if not label:
        return
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = max(0, min(x, max(0, w - tw - 5)))
    y = max(th + 6, min(y, h - baseline - 3))
    cv2.rectangle(img, (x, y - th - 5), (min(w - 1, x + tw + 4), min(h - 1, y + baseline + 2)), BLACK, -1)
    cv2.putText(img, label, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)


def annotate_scene(frame: np.ndarray, prediction: Recognition) -> np.ndarray:
    """Draw true plate and per-character boxes, not fabricated OCR labels."""
    out = frame.copy()
    if prediction.plate_box is None:
        return out
    h, w = out.shape[:2]
    p = prediction.plate_box
    cv2.rectangle(out, (max(0, p.x1), max(0, p.y1)),
                  (min(w - 1, p.x2), min(h - 1, p.y2)), GREEN, 3)
    title = f"{prediction.plate or 'REVIEW'} det:{p.confidence:.2f} ocr:{prediction.confidence:.2f}"
    _label(out, title, p.x1, p.y1 - 8, GREEN, 0.7)
    for c in prediction.boxes:
        if not (0 <= c.class_id < len(OCR_CLASSES)):
            continue
        x1, y1 = max(0, c.x1), max(0, c.y1)
        x2, y2 = min(w - 1, c.x2), min(h - 1, c.y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), YELLOW, 2)
        _label(out, OCR_CLASSES[c.class_id], x1, y1 - 2, YELLOW, 0.52)
    return out


def crop_evidence(frame: np.ndarray, prediction: Recognition) -> np.ndarray | None:
    """Annotated crop for PiP and persistent image evidence."""
    p = prediction.plate_box
    if p is None:
        return None
    h, w = frame.shape[:2]
    dx = max(10, round((p.x2 - p.x1) * 0.16))
    dy = max(18, round((p.y2 - p.y1) * 0.24))
    x1, y1 = max(0, p.x1 - dx), max(0, p.y1 - dy)
    x2, y2 = min(w, p.x2 + dx), min(h, p.y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return None
    scene = annotate_scene(frame, prediction)
    return scene[y1:y2, x1:x2].copy()
