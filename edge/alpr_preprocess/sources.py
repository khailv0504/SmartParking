"""Ba nguồn ảnh: image, video file và camera USB trên Linux/Windows."""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_id: int
    timestamp_s: float


class ImageSource:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.description = str(self.path)

    def __enter__(self) -> "ImageSource":
        if not self.path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {self.path}")
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __iter__(self) -> Iterator[FramePacket]:
        # imdecode + fromfile để đọc được đường dẫn tiếng Việt trên Windows.
        data = np.fromfile(str(self.path), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Không đọc được ảnh: {self.path}")
        yield FramePacket(frame, frame_id=0, timestamp_s=0.0)


class VideoSource:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.description = str(self.path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps: float = 0.0

    def __enter__(self) -> "VideoSource":
        if not self.path.is_file():
            raise FileNotFoundError(f"Không tìm thấy video: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not 0 < self.fps < 1000:
            # File không có metadata FPS hợp lệ; timestamp chỉ mang tính tương đối.
            print("[WARN] Không đọc được FPS metadata. Dùng 30 FPS để ước lượng timestamp.")
            self.fps = 30.0
        return self

    def __exit__(self, *_: object) -> None:
        if self.cap is not None:
            self.cap.release()

    def __iter__(self) -> Iterator[FramePacket]:
        assert self.cap is not None
        idx = 0
        last_t = -1.0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            estimate = idx / self.fps
            # Một số codec/backend trả 0 hoặc timestamp không tăng.
            t = pos_ms / 1000 if pos_ms > 0 and pos_ms / 1000 > last_t else estimate
            t = max(t, last_t)
            yield FramePacket(frame, frame_id=idx, timestamp_s=t)
            last_t = t
            idx += 1


class CameraSource:
    """Worker liên tục thu ảnh mới nhất, không xếp queue các frame cũ.

    Lý tưởng cho inference chậm hơn tốc độ camera: 1 frame mới nhất trong RAM.
    Chỉ thread camera được gọi cap.read(); main thread không đọc cap.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30,
        fourcc: str = "MJPG",
        backend: str = "auto",
    ) -> None:
        self.index = index
        self.width, self.height = width, height
        self.requested_fps, self.fourcc = fps, fourcc
        self.backend = backend
        self.description = f"camera:{index}"
        self.cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[tuple[int, float, np.ndarray]] = None
        self._capture_started = 0.0
        self._capture_counter = 0
        self._error: Optional[str] = None

    def __enter__(self) -> "CameraSource":
        if self.backend == "v4l2" or (self.backend == "auto" and sys.platform.startswith("linux")):
            api = cv2.CAP_V4L2
        else:
            api = cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.index, api)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Không mở được camera {self.index}. Kiểm tra /dev/video*, quyền truy cập và --camera-backend."
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        # Không phải backend/driver nào cũng hỗ trợ giảm buffer.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[CAMERA] Đã mở camera={self.index}, thực tế={actual_w}x{actual_h}, FPS báo cáo={actual_fps:.1f}")

        self._capture_started = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread.start()
        return self

    def _capture_loop(self) -> None:
        assert self.cap is not None
        failures = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures >= 30:
                    self._error = "Mất tín hiệu camera (30 lần đọc lỗi liên tiếp)"
                    self._stop.set()
                    break
                time.sleep(0.02)
                continue
            failures = 0
            self._capture_counter += 1
            # Timestamp gần thời điểm cap.read() trả frame, không phải phần cứng sensor.
            timestamp_s = time.monotonic() - self._capture_started
            with self._lock:
                self._latest = (self._capture_counter, timestamp_s, frame)

    def __iter__(self) -> Iterator[FramePacket]:
        last_seen = 0
        while not self._stop.is_set():
            with self._lock:
                latest = self._latest
            if latest is None or latest[0] <= last_seen:
                time.sleep(0.005)
                continue
            seq, t, frame = latest
            last_seen = seq
            yield FramePacket(frame.copy(), frame_id=seq, timestamp_s=t)
        if self._error:
            raise RuntimeError(self._error)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
