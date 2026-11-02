#!/usr/bin/env python3
"""Chạy Video Sampling + Image Quality Assessment với 3 nguồn đầu vào.

    python main.py --image path/to/plate.jpg
    python main.py --video path/to/parking.mp4 --sample-fps 2
    python main.py --camera 0 --sample-fps 2 --max-seconds 30

Đây là khối TIỀN XỬ LÝ, chưa tích hợp YOLO và OCR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
from pathlib import Path

from .alpr_preprocess.pipeline import FrameSelectionPipeline, PipelineConfig
from .alpr_preprocess.quality import QualityAssessor, QualityConfig
from .alpr_preprocess.sampling import MotionGate, MotionConfig
from .alpr_preprocess.sources import CameraSource, ImageSource, VideoSource


def positive_number(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Giá trị phải lớn hơn 0")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge ALPR - sampling và IQA; chọn đúng 1 loại đầu vào",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--image", type=str, help="Đường dẫn ảnh tĩnh (.jpg, .png,...)")
    sources.add_argument("--video", type=str, help="Đường dẫn video (.mp4, .avi,...)")
    sources.add_argument("--camera", type=int, nargs="?", const=0, metavar="INDEX",
                         help="Index camera USB, thường 0 hoặc 1")

    parser.add_argument("--output", default="runs/frame_selection", help="Thư mục xuất kết quả")
    parser.add_argument("--sample-fps", type=positive_number, default=2.0,
                        help="Số frame ứng viên mỗi giây của video/camera")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Giới hạn số giây từ nguồn video/camera (0 = không giới hạn)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Dừng sau N lần IQA (0 = không giới hạn)")
    parser.add_argument("--top-k", type=int, default=3, help="Số frame tốt nhất cần giữ trong RAM")
    parser.add_argument("--quality-threshold", type=float, default=0.48,
                        help="Điểm IQA tối thiểu để PASS [0,1]")
    parser.add_argument("--iqa-width", type=int, default=640,
                        help="Giới hạn độ rộng ảnh dùng để tính IQA")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Tùy chọn đánh giá ROI cố định: x y width height (demo)")
    parser.add_argument("--motion-gate", action="store_true",
                        help="Dùng frame difference; vẫn quét fallback khi vật thể đứng yên")
    parser.add_argument("--motion-threshold", type=float, default=0.012,
                        help="Tỷ lệ pixel đổi khác đủ để coi là có chuyển động")
    parser.add_argument("--motion-hold", type=float, default=2.0,
                        help="Duy trì trạng thái active sau chuyển động (giây)")
    parser.add_argument("--fallback-seconds", type=positive_number, default=3.0,
                        help="Chu kỳ kiểm tra khi không phát hiện chuyển động")
    parser.add_argument("--width", type=int, default=1280, help="Độ rộng camera mong muốn")
    parser.add_argument("--height", type=int, default=720, help="Độ cao camera mong muốn")
    parser.add_argument("--camera-fps", type=positive_number, default=30.0,
                        help="FPS camera mong muốn (khác sample-fps)")
    parser.add_argument("--fourcc", default="MJPG", help="Pixel format yêu cầu ở camera (4 ký tự)")
    parser.add_argument("--camera-backend", choices=("auto", "v4l2", "any"), default="auto")
    parser.add_argument("--preview", action="store_true", help="Hiện cửa sổ OpenCV, bấm q để dừng")
    parser.add_argument("--save-all-accepted", action="store_true",
                        help="Lưu TẤT CẢ ảnh PASS (có thể tốn dung lượng thẻ nhớ)")
    parser.add_argument("--verbose", action="store_true", help="In thống kê mọi frame được đánh giá")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_seconds < 0 or args.max_samples < 0 or args.top_k < 0:
        raise ValueError("max-seconds, max-samples và top-k phải >= 0")
    if len(args.fourcc) != 4:
        raise ValueError("--fourcc phải có đúng 4 ký tự, ví dụ MJPG hoặc YUYV")
    if not 0 <= args.motion_threshold <= 1 or args.motion_hold < 0:
        raise ValueError("motion-threshold nằm trong [0,1]; motion-hold >= 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Kích thước camera phải > 0")
    if args.preview:
        # Headless OpenCV và Raspberry Pi OS Lite không có highgui/imshow.
        build_info = cv2.getBuildInformation()
        if "GUI:                           NONE" in build_info or "GUI: NONE" in build_info:
            raise RuntimeError("OpenCV hiện tại là bản headless, không hỗ trợ --preview. "
                               "Hãy bỏ --preview hoặc cài opencv-python có GUI.")
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("Không có phiên desktop DISPLAY/WAYLAND_DISPLAY. "
                               "Chạy mặc định qua SSH không cần --preview.")

    qc = QualityConfig(max_width=args.iqa_width, quality_threshold=args.quality_threshold)
    pc = PipelineConfig(
        sample_fps=args.sample_fps,
        max_seconds=args.max_seconds,
        max_samples=args.max_samples,
        top_k=args.top_k,
        preview=args.preview,
        save_all_accepted=args.save_all_accepted,
        verbose=args.verbose,
        roi_xywh=tuple(args.roi) if args.roi else None,
    )
    gate = None
    if args.motion_gate:
        gate = MotionGate(MotionConfig(
            min_motion_ratio=args.motion_threshold,
            hold_seconds=args.motion_hold,
            fallback_seconds=args.fallback_seconds,
        ))

    if args.image is not None:
        source, image_mode = ImageSource(args.image), True
    elif args.video is not None:
        source, image_mode = VideoSource(args.video), False
    else:
        source, image_mode = CameraSource(
            index=args.camera, width=args.width, height=args.height,
            fps=args.camera_fps, fourcc=args.fourcc, backend=args.camera_backend,
        ), False

    if image_mode and args.motion_gate:
        print("[INFO] --motion-gate không áp dụng cho ảnh tĩnh.")

    pipeline = FrameSelectionPipeline(
        assessor=QualityAssessor(qc), config=pc,
        output_dir=Path(args.output), motion_gate=gate if not image_mode else None,
    )
    with source:
        result = pipeline.run(source, image_mode=image_mode)

    brief = {key: result[key] for key in (
        "source", "frames_read_by_pipeline", "temporally_sampled",
        "skipped_by_motion_gate", "frames_iqa_evaluated", "frames_accepted",
        "iqa_latency", "sampling_iqa_latency", "best_frames",
    )}
    print("\n========== FRAME SAMPLING + IQA ==========")
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    print(f"[OUTPUT] {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Đã dừng bằng Ctrl+C")
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
