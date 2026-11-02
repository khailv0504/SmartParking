"""Local MJPEG annotated preview for SmartParking Edge (no Cloud/RFID/SQLite).

Run from the SmartParking repository root:
  python -m edge.live_preview --image img.png
  python -m edge.live_preview --camera 0

The HTTP server listens on 127.0.0.1 by default. Use SSH port forwarding
from your Ubuntu laptop: ssh -L 8080:127.0.0.1:8080 khai@<PI_IP>.

This preview is READ-ONLY and NEVER writes parking events or changes Edge core.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from .alpr_preprocess.sources import CameraSource, ImageSource
from .recognition import OCR_CLASSES, Recognition, YOLOPlateRecognizer

ROOT = Path(__file__).resolve().parent


@dataclass
class SharedPreview:
    """Latest completed annotated image, shared across HTTP clients."""
    jpeg: bytes | None = None
    sequence: int = 0
    info: dict[str, Any] = field(default_factory=lambda: {"status": "starting"})
    closed: bool = False
    condition: threading.Condition = field(default_factory=threading.Condition)

    def publish(self, image: bytes, info: dict[str, Any]) -> None:
        with self.condition:
            self.jpeg = image
            self.info = info
            self.sequence += 1
            self.condition.notify_all()

    def update(self, **info: Any) -> None:
        with self.condition:
            self.info = {**self.info, **info}
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()


def _draw(frame: np.ndarray, rec: Recognition | None, count: int,
          *, character_boxes: bool = False) -> np.ndarray:
    """Draw on the SAME frame that was inferred; never reuse stale boxes."""
    out = frame.copy()
    h, w = out.shape[:2]
    if rec is not None and rec.plate_box is not None:
        b = rec.plate_box
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), (0, 230, 0), 2)
        title = (" / ".join(rec.row_texts) if rec.plate else rec.reason or "REVIEW")
        title = f"{title}  det:{rec.detector_confidence:.2f} ocr:{rec.confidence:.2f}"
        text_y = b.y1 - 9 if b.y1 > 28 else min(h - 12, b.y2 + 22)
        cv2.putText(out, title, (max(5, b.x1), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, title, (max(5, b.x1), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2, cv2.LINE_AA)
        if character_boxes and rec.boxes:
            # Character boxes are relative to the slightly padded plate crop.
            padx = round((b.x2 - b.x1) * 0.04)
            pady = round((b.y2 - b.y1) * 0.04)
            ox, oy = max(0, b.x1 - padx), max(0, b.y1 - pady)
            for char in rec.boxes:
                x1, y1 = ox + char.x1, oy + char.y1
                x2, y2 = ox + char.x2, oy + char.y2
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 225, 255), 1)
                if 0 <= char.class_id < len(OCR_CLASSES):
                    cv2.putText(out, OCR_CLASSES[char.class_id],
                                (x1, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35, (0, 225, 255), 1, cv2.LINE_AA)
    banner = f"SmartParking Edge  |  frame {count}"
    if rec is None:
        banner += "  |  CAMERA ONLY"
    else:
        banner += f"  |  AI {rec.latency_ms:.0f} ms"
        if rec.plate is None:
            banner += f"  |  {rec.reason or 'NO_PLATE'}"
    cv2.rectangle(out, (0, 0), (w, 32), (22, 22, 22), -1)
    cv2.putText(out, banner, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _encode(frame: np.ndarray, max_width: int, quality: int) -> bytes:
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, round(h * max_width / w)),
                           interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Cannot encode preview JPEG")
    return buffer.tobytes()


def make_handler(shared: SharedPreview):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _fmt: str, *_args: object) -> None:
            return  # Avoid printing every browser request / stream reconnect.

        def _response(self, code: int, content_type: str, payload: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                html = b'''<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartParking - Camera LAN</title>
<style>body{background:#101827;color:#e5ecf5;font:16px system-ui;margin:0;padding:24px;}
main{max-width:1100px;margin:auto}h1{font-size:1.5rem}img{display:block;max-width:100%;width:100%;height:auto;background:#253245;border-radius:10px}
pre{white-space:pre-wrap;background:#1b2a3e;border-radius:10px;padding:16px;overflow-wrap:anywhere}
p{color:#aebfd3}</style></head><body><main><h1>SmartParking | Live ALPR Preview</h1>
<p>Frames with bounding boxes are updated when YOLO + OCR inference finishes (not 30 FPS AI).</p>
<img src="/stream.mjpg" alt="Annotated live camera" />
<h2>Status</h2><pre id="status">Starting...</pre>
<script>async function poll(){try{const r=await fetch('/status',{cache:'no-store'});document.getElementById('status').textContent=JSON.stringify(await r.json(),null,2);}catch(e){document.getElementById('status').textContent=String(e)}}poll();setInterval(poll,2000);</script>
</main></body></html>'''
                self._response(200, "text/html; charset=utf-8", html)
            elif path == "/status":
                with shared.condition:
                    payload = json.dumps(shared.info, ensure_ascii=False).encode("utf-8")
                self._response(200, "application/json; charset=utf-8", payload)
            elif path == "/snapshot.jpg":
                with shared.condition:
                    jpeg = shared.jpeg
                if jpeg is None:
                    self._response(503, "text/plain", b"No frame yet")
                else:
                    self._response(200, "image/jpeg", jpeg)
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                last_seq = 0
                try:
                    while True:
                        with shared.condition:
                            shared.condition.wait_for(
                                lambda: shared.sequence != last_seq or shared.closed,
                                timeout=10.0,
                            )
                            if shared.closed:
                                return
                            if shared.sequence == last_seq or shared.jpeg is None:
                                continue
                            last_seq, jpeg = shared.sequence, shared.jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self._response(404, "text/plain", b"Not found")

    return Handler


def inference_worker(shared: SharedPreview, args: argparse.Namespace,
                     stop: threading.Event) -> None:
    try:
        model = None if args.camera_only else YOLOPlateRecognizer(
            args.detector, args.recognizer, threads=args.threads,
        )
        shared.update(status="models_loaded" if model else "camera_only")
        src = ImageSource(args.image) if args.image else CameraSource(
            index=args.camera, width=args.width, height=args.height,
            fps=args.fps, fourcc=args.fourcc,
        )
        with src:
            for packet in src:
                if stop.is_set():
                    return
                start = time.perf_counter()
                result = model.recognize(packet.frame) if model else None
                annotated = _draw(packet.frame, result, packet.frame_id,
                                  character_boxes=args.char_boxes)
                jpeg = _encode(annotated, args.preview_width, args.jpeg_quality)
                shared.publish(jpeg, {
                    "status": "running",
                    "source": src.description,
                    "frame_id": packet.frame_id,
                    "plate": result.plate if result else None,
                    "rows": list(result.row_texts) if result else [],
                    "detector_confidence": result.detector_confidence if result else None,
                    "ocr_min_character_score": result.confidence if result else None,
                    "reason": result.reason if result else None,
                    "inference_ms": round(result.latency_ms, 2) if result else None,
                    "processing_ms": round((time.perf_counter() - start) * 1000, 2),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                })
                if args.image:
                    shared.update(status="image_ready")
                    return
                # The latest-frame capture thread keeps running during inference.
                # Slow down when testing CAMERA ONLY to avoid needless JPEG CPU load.
                elapsed = time.perf_counter() - start
                wait_s = max(0, args.min_interval - elapsed)
                if wait_s and stop.wait(wait_s):
                    return
    except Exception as exc:
        shared.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        shared.update(worker_stopped=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SmartParking LAN MJPEG live ALPR preview")
    p.add_argument("--image", default="", help="Smoke-test with one image instead of camera")
    p.add_argument("--camera", type=int, default=0, help="USB camera index (/dev/video0 => 0)")
    p.add_argument("--camera-only", action="store_true", help="Test capture without AI")
    p.add_argument("--detector", default=str(ROOT / "models" / "detector.onnx"))
    p.add_argument("--recognizer", default=str(ROOT / "models" / "recognizer.onnx"))
    p.add_argument("--threads", type=int, default=2, help="OpenCV CPU threads")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--preview-width", type=int, default=960, help="MJPEG width; lower means less traffic")
    p.add_argument("--jpeg-quality", type=int, default=72)
    p.add_argument("--min-interval", type=float, default=0.2,
                   help="Minimum seconds between displayed frames")
    p.add_argument("--char-boxes", action="store_true", help="Show individual OCR character boxes")
    p.add_argument("--host", default="127.0.0.1", help="Default localhost: use SSH tunnel")
    p.add_argument("--port", type=int, default=8080)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (0 <= args.camera and 32 <= args.preview_width and 1 <= args.jpeg_quality <= 100):
        raise SystemExit("Invalid camera/preview-width/jpeg-quality")
    if args.min_interval < 0 or args.threads < 1:
        raise SystemExit("Invalid min-interval/threads")
    if len(args.fourcc) != 4:
        raise SystemExit("--fourcc must have exactly four characters")
    shared = SharedPreview()
    stop = threading.Event()
    worker = threading.Thread(target=inference_worker, args=(shared, args, stop),
                              name="alpr-preview", daemon=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    server.daemon_threads = True
    worker.start()
    print(f"[PREVIEW] http://{args.host}:{args.port}/")
    print("[PREVIEW] Camera-only / live AI, no RFID, no SQLite, no Cloud")
    print("[PREVIEW] Ctrl+C to stop")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        shared.close()
        server.server_close()
        worker.join(timeout=3)


if __name__ == "__main__":
    main()
