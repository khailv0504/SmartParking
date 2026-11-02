"""Điều phối sampling -> IQA -> lưu best frames -> báo cáo benchmark."""
from __future__ import annotations

import csv
import heapq
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .quality import QualityAssessor, QualityReport
from .sampling import MotionGate, TemporalSampler
from .sources import FramePacket

CSV_FIELDS = [
    "frame_id", "timestamp_s", "motion_ratio", "iqa_latency_ms", "sampling_and_iqa_ms",
    "sharpness", "brightness", "contrast", "glare_ratio", "dark_ratio",
    "quality_score", "accepted", "reasons", "evaluated_width", "evaluated_height",
]


@dataclass(frozen=True)
class PipelineConfig:
    sample_fps: float = 2.0
    top_k: int = 3
    max_seconds: float = 0.0
    max_samples: int = 0
    preview: bool = False
    save_all_accepted: bool = False
    verbose: bool = False
    roi_xywh: Optional[tuple[int, int, int, int]] = None


def _write_jpeg(path: Path, frame: np.ndarray) -> None:
    """Dùng imencode + tofile để ghi được đường dẫn có dấu trên Windows."""
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError(f"Không mã hóa được JPEG: {path}")
    encoded.tofile(str(path))


def _annotate(frame: np.ndarray, packet: FramePacket, quality: QualityReport, roi: Optional[tuple]) -> np.ndarray:
    # Chỉ dùng để trực quan, KHÔNG đưa ảnh có overlay trở lại detector/OCR.
    out = frame.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 2)
    color = (0, 180, 0) if quality.accepted else (0, 0, 255)
    lines = [
        f"id={packet.frame_id}  t={packet.timestamp_s:.2f}s",
        f"IQA={quality.quality_score:.3f} {'PASS' if quality.accepted else 'CHECK'}",
        f"sharp={quality.sharpness:.1f} light={quality.brightness:.1f}",
    ]
    for i, line in enumerate(lines):
        y = 30 + 32 * i
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


class FrameSelectionPipeline:
    def __init__(
        self,
        assessor: QualityAssessor,
        config: PipelineConfig,
        output_dir: str | Path,
        motion_gate: Optional[MotionGate] = None,
    ) -> None:
        if config.top_k < 0 or config.max_seconds < 0 or config.max_samples < 0:
            raise ValueError("top_k, max_seconds và max_samples phải >= 0")
        self.assessor = assessor
        self.cfg = config
        self.motion_gate = motion_gate
        self.temporal_sampler = TemporalSampler(config.sample_fps)
        self.output_dir = Path(output_dir)
        # Mỗi phần tử: (accepted, score, frame_id, timestamp, raw_frame, report)
        # Heap chỉ giữ tối đa K frame, KHÔNG giữ video trong RAM.
        self._top: list[tuple[int, float, int, float, np.ndarray, QualityReport]] = []
        self._iqa_latencies: deque[float] = deque(maxlen=2000)
        self._pipeline_latencies: deque[float] = deque(maxlen=2000)

    def _update_top(self, packet: FramePacket, quality: QualityReport) -> None:
        if self.cfg.top_k == 0:
            return
        key = (int(quality.accepted), quality.quality_score, packet.frame_id)
        if len(self._top) >= self.cfg.top_k and key <= self._top[0][:3]:
            return  # Không copy ảnh không cần thiết, giảm memory bandwidth.
        record = (*key, packet.timestamp_s, packet.frame.copy(), quality)
        if len(self._top) < self.cfg.top_k:
            heapq.heappush(self._top, record)
        else:
            heapq.heapreplace(self._top, record)

    @staticmethod
    def _metric(samples: deque[float]) -> dict:
        if not samples:
            return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "mean_ms": round(float(arr.mean()), 3),
        }

    def run(self, source: Any, *, image_mode: bool = False) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_dir = self.output_dir / "best_frames"
        best_dir.mkdir(exist_ok=True)
        # Xóa các best frame từ LẦN CHẠY CŨ để kết quả không bị lẫn.
        for old_file in best_dir.glob("rank_??_frame_*.jpg"):
            old_file.unlink()
        if self.cfg.save_all_accepted:
            (self.output_dir / "accepted_frames").mkdir(exist_ok=True)

        read_count = sampled_count = iqa_count = accepted_count = motion_skips = 0
        start_wall = time.perf_counter()
        first_timestamp: Optional[float] = None
        last_timestamp = 0.0
        stop_by_key = False
        stop_by_interrupt = False
        try:
            with (self.output_dir / "quality_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()

                for packet in source:
                    read_count += 1
                    if first_timestamp is None:
                        first_timestamp = packet.timestamp_s
                    source_elapsed = packet.timestamp_s - first_timestamp
                    last_timestamp = source_elapsed

                    # max_seconds tính theo thời gian video/camera, không phải tốc độ giải mã file.
                    if not image_mode and self.cfg.max_seconds > 0 and source_elapsed > self.cfg.max_seconds:
                        break

                    begin = time.perf_counter()
                    # Ảnh tĩnh luôn được IQA đúng 1 lần, không cần temporal sampling.
                    if not image_mode and not self.temporal_sampler.should_sample(source_elapsed):
                        continue
                    sampled_count += 1

                    motion_ratio = 0.0
                    if self.motion_gate is not None and not image_mode:
                        eligible = self.motion_gate.should_process(packet.frame, source_elapsed)
                        motion_ratio = self.motion_gate.last_motion_ratio
                        if not eligible:
                            motion_skips += 1
                            continue

                    t0 = time.perf_counter()
                    quality = self.assessor.assess(packet.frame, roi_xywh=self.cfg.roi_xywh)
                    iqa_ms = (time.perf_counter() - t0) * 1000.0
                    self._iqa_latencies.append(iqa_ms)

                    iqa_count += 1
                    accepted_count += int(quality.accepted)
                    self._update_top(packet, quality)

                    if self.cfg.save_all_accepted and quality.accepted:
                        filename = f"frame_{packet.frame_id:07d}_q{quality.quality_score:.3f}.jpg"
                        _write_jpeg(self.output_dir / "accepted_frames" / filename, packet.frame)

                    pipeline_ms = (time.perf_counter() - begin) * 1000.0
                    self._pipeline_latencies.append(pipeline_ms)
                    writer.writerow({
                        "frame_id": packet.frame_id,
                        "timestamp_s": round(source_elapsed, 5),
                        "motion_ratio": round(motion_ratio, 6),
                        "iqa_latency_ms": round(iqa_ms, 3),
                        "sampling_and_iqa_ms": round(pipeline_ms, 3),
                        **quality.to_dict(),
                    })

                    if self.cfg.verbose or iqa_count <= 5 or iqa_count % 20 == 0:
                        status = "PASS" if quality.accepted else "REJECT"
                        print(f"[IQA] frame={packet.frame_id:6d} t={source_elapsed:7.2f}s "
                              f"score={quality.quality_score:.3f} {status:6s} "
                              f"blur={quality.sharpness:.1f} ({iqa_ms:.1f}ms)")

                    if self.cfg.preview:
                        annotated = _annotate(packet.frame, packet, quality, self.cfg.roi_xywh)
                        try:
                            cv2.imshow("ALPR Frame Selection - nhan Q de thoat", annotated)
                            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                                stop_by_key = True
                                break
                        except cv2.error as exc:
                            raise RuntimeError(
                                "Không mở được cửa sổ preview. Trên Pi OS Lite dùng chế độ mặc định "
                                "(không --preview), hoặc cài OpenCV có GUI."
                            ) from exc

                    if self.cfg.max_samples and iqa_count >= self.cfg.max_samples:
                        break
        except KeyboardInterrupt:
            # Khi dùng camera 24/7, Ctrl+C vẫn phải xuất best frame và summary.
            stop_by_interrupt = True
            print("\n[INFO] Ctrl+C: đang tổng kết dữ liệu đã thu thập.")
        finally:
            if self.cfg.preview:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        wall_seconds = time.perf_counter() - start_wall
        best_items = sorted(self._top, key=lambda item: item[:3], reverse=True)
        best_results = []
        for rank, (accepted, score, frame_id, timestamp, frame, quality) in enumerate(best_items, start=1):
            path = best_dir / f"rank_{rank:02d}_frame_{frame_id:07d}.jpg"
            _write_jpeg(path, frame)
            best_results.append({
                "rank": rank,
                "frame_id": frame_id,
                "timestamp_s": round(timestamp - (first_timestamp or 0.0), 5),
                "score": score,
                "accepted": bool(accepted),
                "reasons": list(quality.reasons),
                "file": str(path.relative_to(self.output_dir)),
            })

        summary = {
            "source": source.description,
            "source_type": "image" if image_mode else type(source).__name__,
            "elapsed_source_s": round(last_timestamp, 3),
            "elapsed_wall_s": round(wall_seconds, 3),
            "frames_read_by_pipeline": read_count,
            "temporally_sampled": sampled_count,
            "skipped_by_motion_gate": motion_skips,
            "frames_iqa_evaluated": iqa_count,
            "frames_accepted": accepted_count,
            "accepted_ratio": round(accepted_count / iqa_count, 4) if iqa_count else 0.0,
            "iqa_throughput_wall_fps": round(iqa_count / wall_seconds, 3) if wall_seconds else None,
            "iqa_latency": self._metric(self._iqa_latencies),
            "sampling_iqa_latency": self._metric(self._pipeline_latencies),
            "quality_config": asdict(self.assessor.config),
            "pipeline_config": asdict(self.cfg),
            "motion_gate_enabled": self.motion_gate is not None,
            "stop_by_preview_key": stop_by_key,
            "stop_by_keyboard_interrupt": stop_by_interrupt,
            "best_frames": best_results,
            "benchmark_note": (
                "Wall throughput khi đọc video file không phải FPS real-time của camera. "
                "Chưa bao gồm YOLO/OCR. IQA là heuristic, không phải độ tin cậy OCR."
            ),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
