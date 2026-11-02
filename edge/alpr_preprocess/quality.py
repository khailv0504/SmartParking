"""Đánh giá chất lượng ảnh KHÔNG dùng mô hình AI, phù hợp thiết bị biên.

Mục đích: nhận biết ảnh quá nhòe, quá tối/sáng, thiếu tương phản,
hoặc có nhiều vùng bão hòa. Điểm IQA chỉ là heuristic, KHÔNG phải
xác suất OCR đúng. Sau YOLO nên đánh giá lại trên vùng ROI biển số.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityConfig:
    # Tất cả phép đo dùng trên ảnh đã thu về tối đa max_width pixel.
    # Đổi max_width -> phải hiệu chỉnh ngưỡng sharpness.
    max_width: int = 640
    min_sharpness: float = 20.0
    min_brightness: float = 30.0
    max_brightness: float = 235.0
    min_contrast: float = 15.0
    max_glare_ratio: float = 0.55
    max_dark_ratio: float = 0.65
    quality_threshold: float = 0.48

    def __post_init__(self) -> None:
        if self.max_width < 32:
            raise ValueError("max_width phải >= 32")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("Khoảng brightness không hợp lệ")
        if not 0 <= self.quality_threshold <= 1:
            raise ValueError("quality_threshold phải trong [0, 1]")
        if self.min_sharpness < 0 or self.min_contrast < 0:
            raise ValueError("min_sharpness và min_contrast phải >= 0")
        if not 0 <= self.max_glare_ratio <= 1 or not 0 <= self.max_dark_ratio <= 1:
            raise ValueError("Tỷ lệ glare/dark phải trong [0, 1]")


@dataclass(frozen=True)
class QualityReport:
    sharpness: float
    brightness: float
    contrast: float
    glare_ratio: float
    dark_ratio: float
    quality_score: float
    accepted: bool
    reasons: tuple[str, ...]
    evaluated_width: int
    evaluated_height: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["reasons"] = ";".join(self.reasons)
        return result


class QualityAssessor:
    def __init__(self, config: Optional[QualityConfig] = None) -> None:
        self.config = config or QualityConfig()

    def assess(
        self,
        frame: np.ndarray,
        roi_xywh: Optional[Tuple[int, int, int, int]] = None,
    ) -> QualityReport:
        """Đo IQA trên frame (toàn ảnh) hoặc vùng ROI (x, y, width, height).

        roi_xywh có thể lấy trực tiếp từ bounding box do detector trả về.
        Không tự cắt làm mất tỷ lệ khung hình của biển số.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame rỗng, không thể đánh giá chất lượng")

        if roi_xywh is not None:
            x, y, w, h = map(int, roi_xywh)
            if w <= 0 or h <= 0:
                raise ValueError("ROI phải có width và height > 0")
            height, width = frame.shape[:2]
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, width), min(y + h, height)
            if x1 >= x2 or y1 >= y2:
                raise ValueError("ROI nằm ngoài ảnh")
            frame = frame[y1:y2, x1:x2]

        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Chỉ hỗ trợ ảnh grayscale, BGR hoặc BGRA")

        h, w = gray.shape
        if w > self.config.max_width:
            target_h = max(1, round(h * self.config.max_width / w))
            gray = cv2.resize(
                gray, (self.config.max_width, target_h),
                interpolation=cv2.INTER_AREA,
            )

        # Laplacian variance: lớn hơn thường sắc nét hơn, nhưng nhạy với nhiễu.
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(lap.var())
        brightness = float(gray.mean())
        contrast = float(gray.std())
        glare_ratio = float(np.count_nonzero(gray >= 245) / gray.size)
        dark_ratio = float(np.count_nonzero(gray <= 20) / gray.size)

        # Normalize để các thành phần có cùng miền [0, 1].
        # Các hằng số 250, 64, 0.2, 0.5 là khởi điểm, cần tuning thực nghiệm.
        sharp_score = float(np.clip(np.log1p(sharpness) / np.log1p(250), 0, 1))
        brightness_score = float(np.clip(1 - abs(brightness - 128) / 128, 0, 1))
        contrast_score = float(np.clip(contrast / 64, 0, 1))
        glare_score = float(np.clip(1 - glare_ratio / 0.20, 0, 1))
        dark_score = float(np.clip(1 - dark_ratio / 0.50, 0, 1))

        score = (
            0.40 * sharp_score
            + 0.15 * brightness_score
            + 0.15 * contrast_score
            + 0.15 * glare_score
            + 0.15 * dark_score
        )

        reasons = []
        cfg = self.config
        if sharpness < cfg.min_sharpness:
            reasons.append("blurry")
        if brightness < cfg.min_brightness:
            reasons.append("too_dark")
        if brightness > cfg.max_brightness:
            reasons.append("too_bright")
        if contrast < cfg.min_contrast:
            reasons.append("low_contrast")
        if glare_ratio > cfg.max_glare_ratio:
            reasons.append("too_much_glare")
        if dark_ratio > cfg.max_dark_ratio:
            reasons.append("too_many_dark_pixels")
        if score < cfg.quality_threshold:
            reasons.append("low_quality_score")

        eh, ew = gray.shape[:2]
        return QualityReport(
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            glare_ratio=glare_ratio,
            dark_ratio=dark_ratio,
            quality_score=round(float(score), 6),
            accepted=(len(reasons) == 0),
            reasons=tuple(reasons),
            evaluated_width=ew,
            evaluated_height=eh,
        )
