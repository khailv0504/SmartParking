"""Temporal sampling và motion/event gate nhẹ cho Edge AI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


class TemporalSampler:
    """Giới hạn số frame ứng viên theo THỜI GIAN, không theo frame index.

    Cách này giữ hành vi gần giống nhau với video 10/25/30 FPS.
    """

    def __init__(self, samples_per_second: float = 2.0) -> None:
        if samples_per_second <= 0:
            raise ValueError("samples_per_second phải > 0")
        self.interval_s = 1.0 / samples_per_second
        self._next_at: Optional[float] = None

    def should_sample(self, timestamp_s: float) -> bool:
        if timestamp_s < 0:
            raise ValueError("timestamp_s phải >= 0")
        if self._next_at is None or timestamp_s + 1e-8 >= self._next_at:
            self._next_at = timestamp_s + self.interval_s
            return True
        return False


@dataclass(frozen=True)
class MotionConfig:
    resize_width: int = 320
    pixel_difference: int = 18
    min_motion_ratio: float = 0.012
    hold_seconds: float = 2.0
    fallback_seconds: float = 3.0


class MotionGate:
    """Chỉ ưu tiên khi chuyển động, nhưng vẫn quét định kỳ khi xe đã dừng.

    Nếu chỉ kiểm tra motion mà không fallback, xe đứng yên trước cổng
    có thể bị bỏ qua hoàn toàn. Không dùng gate này như detector biển số.
    """

    def __init__(self, config: Optional[MotionConfig] = None) -> None:
        self.cfg = config or MotionConfig()
        self._previous: Optional[np.ndarray] = None
        self._last_motion: float = float("-inf")
        self._last_fallback: float = float("-inf")
        self.last_motion_ratio: float = 0.0

    def should_process(self, frame: np.ndarray, timestamp_s: float) -> bool:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        h, w = gray.shape[:2]
        target_w = min(w, self.cfg.resize_width)
        target_h = max(1, round(h * target_w / w))
        small = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        if self._previous is not None:
            delta = cv2.absdiff(small, self._previous)
            ratio = float(np.count_nonzero(delta >= self.cfg.pixel_difference) / delta.size)
            self.last_motion_ratio = ratio
            if ratio >= self.cfg.min_motion_ratio:
                self._last_motion = timestamp_s
        else:
            self.last_motion_ratio = 0.0

        self._previous = small
        in_active_window = timestamp_s - self._last_motion <= self.cfg.hold_seconds
        fallback_due = timestamp_s - self._last_fallback >= self.cfg.fallback_seconds
        if fallback_due:
            self._last_fallback = timestamp_s
        return in_active_window or fallback_due
