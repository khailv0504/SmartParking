"""Inspect and smoke-test SmartParking's YOLO26 ONNX models without onnxruntime.

Run from the repository root:
    python -m edge.model_healthcheck
    python -m edge.model_healthcheck --image /path/to/test_plate.jpg --repeat 20

OpenCV DNN CPU must support both graphs. Successful synthetic inference only
establishes runtime compatibility; it does not measure license-plate accuracy.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

OCR_CLASSES = "0123456789ABCDEFGHKLMNPSTUVXYZ"
EXPECTED = {
    "detector": {"size": 640, "output": (1, 6, 8400), "classes": ["0", "1"]},
    "recognizer": {"size": 320, "output": (1, 34, 2100), "classes": list(OCR_CLASSES)},
}


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if position >= len(data):
            raise ValueError("Unexpected end of ONNX protobuf")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
        shift += 7
    raise ValueError("Invalid ONNX protobuf varint")


def _fields(data: bytes):
    """Yield protobuf field number, wire type and raw value; no ONNX dependency."""
    p = 0
    while p < len(data):
        tag, p = _read_varint(data, p)
        field, wire = tag >> 3, tag & 7
        if field <= 0:
            raise ValueError("Invalid ONNX protobuf field number")
        if wire == 0:
            value, p = _read_varint(data, p)
        elif wire == 1:
            if p + 8 > len(data):
                raise ValueError("Truncated ONNX protobuf")
            value, p = data[p:p+8], p + 8
        elif wire == 2:
            length, p = _read_varint(data, p)
            if p + length > len(data):
                raise ValueError("Truncated ONNX protobuf length")
            value, p = data[p:p+length], p + length
        elif wire == 5:
            if p + 4 > len(data):
                raise ValueError("Truncated ONNX protobuf")
            value, p = data[p:p+4], p + 4
        else:
            raise ValueError(f"Unsupported ONNX protobuf wire type {wire}")
        yield field, wire, value


def read_metadata(path: str | Path) -> dict[str, str]:
    """Read ONNX ModelProto.metadata_props (field 14) using standard Python."""
    model = Path(path).read_bytes()
    values: dict[str, str] = {}
    for field, wire, value in _fields(model):
        if field != 14 or wire != 2:
            continue
        entry = {}
        for f, w, v in _fields(value):
            if w == 2 and f in (1, 2):
                entry[f] = v.decode("utf-8")
        if 1 in entry and 2 in entry:
            values[entry[1]] = entry[2]
    return values


def verify_model(name: str, path: str | Path, threads: int = 2) -> dict:
    """Validate metadata, inference shape and finite output on a synthetic input."""
    path = Path(path).expanduser().resolve()
    spec = EXPECTED[name]
    meta = read_metadata(path)
    if meta.get("task") != "detect":
        raise ValueError(f"{name}: expected task=detect, got {meta.get('task')!r}")
    if meta.get("end2end") != "False":
        raise ValueError(f"{name}: NMS-free/end2end export is incompatible with current decoder")
    try:
        names = ast.literal_eval(meta["names"])
        sizes = ast.literal_eval(meta["imgsz"])
    except (ValueError, SyntaxError, KeyError) as exc:
        raise ValueError(f"{name}: invalid/missing ONNX metadata: {exc}") from exc
    if not isinstance(names, dict):
        raise ValueError(f"{name}: class metadata should be a dict")
    ordered = [str(names[i]) for i in range(len(spec["classes"]))]
    if ordered != spec["classes"] or len(names) != len(ordered):
        raise ValueError(f"{name}: unexpected class ordering: {ordered}")
    if sizes != [spec["size"], spec["size"]]:
        raise ValueError(f"{name}: unexpected model input metadata: {sizes}")

    # OpenCV 4.13 CPU tested; validate again with the exact Pi OpenCV build.
    cv2.setNumThreads(max(1, threads))
    net = cv2.dnn.readNetFromONNX(str(path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    inp = np.zeros((1, 3, spec["size"], spec["size"]), dtype=np.float32)
    net.setInput(inp)
    tic = time.perf_counter()
    out = np.asarray(net.forward())
    forward_ms = round((time.perf_counter() - tic) * 1000, 2)
    if tuple(out.shape) != spec["output"]:
        raise ValueError(f"{name}: output {out.shape}, expected {spec['output']}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name}: output has NaN or Inf values")
    return {
        "status": "PASS",
        "model": name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input": [1, 3, spec["size"], spec["size"]],
        "output": list(out.shape),
        "class_count": len(ordered),
        "architecture_metadata": meta.get("description", ""),
        "forward_ms_synthetic_first_run": forward_ms,
    }


def main() -> int:
    root = Path(__file__).resolve().parent / "models"
    p = argparse.ArgumentParser(description="SmartParking YOLO26 ONNX compatibility check")
    p.add_argument("--detector", type=Path, default=root / "detector.onnx")
    p.add_argument("--recognizer", type=Path, default=root / "recognizer.onnx")
    p.add_argument("--image", type=Path, help="Optional real-world test image")
    p.add_argument("--repeat", type=int, default=0, help="Repeated end-to-end runs on --image")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.repeat < 0 or args.threads < 1:
        p.error("--repeat >= 0 and --threads >= 1 are required")
    if args.repeat and not args.image:
        p.error("--repeat requires --image")

    try:
        report = {
            "detector": verify_model("detector", args.detector, args.threads),
            "recognizer": verify_model("recognizer", args.recognizer, args.threads),
        }
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f"Cannot load image: {args.image}")
            from edge.recognition import YOLOPlateRecognizer  # local import for standalone preflight
            recognizer = YOLOPlateRecognizer(
                args.detector, args.recognizer, threads=args.threads
            )
            result = recognizer.recognize(frame)
            report["real_image"] = {
                "plate": result.plate,
                "reason": result.reason,
                "latency_ms_first": round(result.latency_ms, 2),
                "character_confidence_heuristic": result.confidence,
            }
            if args.repeat:
                samples = [recognizer.recognize(frame).latency_ms for _ in range(args.repeat)]
                ordered = sorted(samples)
                report["benchmark"] = {
                    "iterations": len(samples),
                    "p50_ms": round(statistics.median(samples), 2),
                    "p95_ms": round(ordered[max(0, int(np.ceil(.95 * len(ordered))) - 1)], 2),
                    "mean_ms": round(statistics.mean(samples), 2),
                }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, IndexError, cv2.error) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
