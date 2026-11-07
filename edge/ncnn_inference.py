"""Native NCNN FP16 inference adapter for Ultralytics YOLO26s detect exports.

Only the inference engine is replaced. The inherited YOLOPlateRecognizer.recognize()
retains the existing ROI, OCR ordering, quality, confidence, and REVIEW behavior.

NCNN is imported lazily: the existing OpenCV/ONNX runtime remains the default.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .alpr_preprocess.quality import QualityAssessor, QualityConfig
from .recognition import Box, OCR_CLASSES, YOLOPlateRecognizer, letterbox

_INPUT_BLOB = "in0"
_OUTPUT_BLOB = "out0"


def _load_model_spec(directory: str | Path, *, size: int, labels: tuple[str, ...]) -> tuple[Path, Path]:
    """Fail closed if the supplied model does not match the shipped YOLO contract."""
    directory = Path(directory).expanduser().resolve()
    param = directory / "model.ncnn.param"
    weights = directory / "model.ncnn.bin"
    metadata = directory / "metadata.yaml"
    for path in (param, weights, metadata):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"NCNN model file missing/empty: {path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("NCNN backend requires PyYAML; install edge/requirements-ncnn.txt") from exc
    data = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid NCNN metadata: {metadata}")
    args = data.get("args")
    names = data.get("names")
    expected_names = list(labels)
    try:
        actual_names = [str(names[i]) for i in range(len(labels))]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid NCNN class indices in {metadata}") from exc
    valid = (
        data.get("task") == "detect"
        and data.get("head") == "Detect"
        and data.get("end2end") is False
        and data.get("batch") == 1
        and data.get("channels") == 3
        and data.get("imgsz") == [size, size]
        and isinstance(args, dict)
        and args.get("quantize") == 16
        and isinstance(names, dict)
        and len(names) == len(labels)
        and actual_names == expected_names
    )
    if not valid:
        raise ValueError(
            f"NCNN export contract mismatch in {metadata}: expected FP16, "
            f"YOLO detect end2end=false, imgsz={size}, classes={expected_names}"
        )

    # The uploaded pnnx graphs use these named tensor blobs. Never guess IDs.
    lines = param.read_text(encoding="utf-8").splitlines()
    if len(lines) < 4 or lines[0].strip() != "7767517":
        raise ValueError(f"Not an NCNN param graph: {param}")
    if not any(line.split()[:2] == ["Input", _INPUT_BLOB] for line in lines[2:]):
        raise ValueError(f"NCNN input blob {_INPUT_BLOB!r} not found in {param}")
    if not any(line.split()[-1] == _OUTPUT_BLOB or _OUTPUT_BLOB in line.split()[4:] for line in lines[2:]):
        raise ValueError(f"NCNN output blob {_OUTPUT_BLOB!r} not found in {param}")

    # Optional pinning of exact binaries shipped with the integration bundle.
    checksum_file = directory / "sha256.json"
    if checksum_file.is_file():
        checksums = json.loads(checksum_file.read_text(encoding="utf-8"))
        for filename, actual_path in ((param.name, param), (weights.name, weights)):
            expected = checksums.get(filename)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"Missing SHA256 for {filename} in {checksum_file}")
            hasher = hashlib.sha256()
            with actual_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest != expected:
                raise ValueError(f"NCNN SHA256 mismatch: {actual_path}")
    return param, weights


def _standardize_output(raw: np.ndarray, *, classes: int, anchors: int) -> np.ndarray:
    """Accept only unambiguous pnnx output layouts; return [4+classes, anchors]."""
    raw = np.asarray(raw)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    expected = (4 + classes, anchors)
    if raw.shape == expected:
        result = raw
    elif raw.shape == expected[::-1]:
        result = raw.T
    else:
        raise RuntimeError(f"NCNN output {raw.shape}; expected {expected} or {expected[::-1]}")
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise RuntimeError("NCNN output has invalid type, NaN, or Inf")
    return result


def _decode_yolo(raw: np.ndarray, frame: np.ndarray, *, size: int, classes: int,
                 conf: float, iou: float) -> list[Box]:
    """Exact same geometric decoding and class-agnostic NMS as YOLOBoxes.predict."""
    # Keep letterbox geometry in sync with edge.recognition.letterbox().
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    pad_x = (size - min(size, round(w * scale))) // 2
    pad_y = (size - min(size, round(h * scale))) // 2
    preds = raw.T
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(preds)), class_ids]
    keep = np.flatnonzero(scores >= conf)
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
    indices = cv2.dnn.NMSBoxes(boxes_xywh, nms_scores,
                               score_threshold=conf, nms_threshold=iou)
    selected = np.asarray(indices).reshape(-1).tolist()
    return sorted((candidates[i] for i in selected),
                  key=lambda b: b.confidence, reverse=True)


class NCNNBoxes:
    """Native NCNN YOLO boxes backend with the same predict() contract as YOLOBoxes."""

    def __init__(self, directory: str | Path, size: int, classes: int,
                 conf: float = 0.35, iou: float = 0.45, threads: int = 2):
        if (size, classes) not in ((640, 2), (320, 30)):
            raise ValueError(f"Unsupported NCNN model contract: size={size}, classes={classes}")
        if threads < 1 or not (0 < conf <= 1) or not (0 < iou <= 1):
            raise ValueError("Invalid NCNN threads or confidence/NMS thresholds")
        labels = ("0", "1") if classes == 2 else tuple(OCR_CLASSES)
        param, weights = _load_model_spec(directory, size=size, labels=labels)
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError("NCNN Python binding missing; install edge/requirements-ncnn.txt") from exc
        self.ncnn = ncnn
        self.net = ncnn.Net()
        cv2.setNumThreads(max(1, threads))
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False  # Raspberry Pi 4: CPU only.
        # Avoid forcing FP16 arithmetic: Pi 4 Cortex-A72 has no native FP16 ALU.
        self.net.opt.use_fp16_arithmetic = False
        if self.net.load_param(str(param)) != 0 or self.net.load_model(str(weights)) != 0:
            raise RuntimeError(f"Failed loading NCNN model from {directory}")
        self.size = size
        self.classes = classes
        self.anchors = 8400 if size == 640 else 2100
        self.conf = conf
        self.iou = iou

    def predict(self, frame: np.ndarray) -> list[Box]:
        canvas, _, _, _ = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1/255.0,
                                     size=(self.size, self.size), swapRB=True, crop=False)
        # Ultralytics' model_ncnn.py constructs ncnn.Mat from float32 CHW.
        image_mat = self.ncnn.Mat(np.ascontiguousarray(blob[0])).clone()
        with self.net.create_extractor() as extractor:
            ret = extractor.input(_INPUT_BLOB, image_mat)
            if isinstance(ret, int) and ret != 0:
                raise RuntimeError(f"NCNN input returned error {ret}")
            status, output_mat = extractor.extract(_OUTPUT_BLOB)
            if status != 0:
                raise RuntimeError(f"NCNN extract returned error {status}")
            raw = np.array(output_mat, copy=True)
        matrix = _standardize_output(raw, classes=self.classes, anchors=self.anchors)
        return _decode_yolo(matrix, frame, size=self.size, classes=self.classes,
                            conf=self.conf, iou=self.iou)


class NCNNPlateRecognizer(YOLOPlateRecognizer):
    """Reuses the unchanged production recognize() logic; swaps only its two nets."""

    def __init__(self, detector_path: str | Path, ocr_path: str | Path,
                 detection_conf: float = 0.35, character_conf: float = 0.35,
                 strict_roi_quality: bool = False, threads: int = 2):
        self.detector = NCNNBoxes(detector_path, 640, 2, detection_conf, threads=threads)
        self.characters = NCNNBoxes(ocr_path, 320, 30, character_conf, threads=threads)
        self.assessor = QualityAssessor(QualityConfig())
        self.strict_roi_quality = strict_roi_quality
