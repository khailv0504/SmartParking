"""NCNN inference contract tests; no ncnn wheel or real RFID hardware required."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from edge.ncnn_inference import (
    NCNNBoxes, NCNNPlateRecognizer, _decode_yolo, _load_model_spec,
    _standardize_output,
)
from edge.recognition import OCR_CLASSES, YOLOBoxes, YOLOPlateRecognizer


def _make_export(tmp_path: Path, *, size=640, classes=2):
    folder = tmp_path / "model"
    folder.mkdir()
    labels = list("01") if classes == 2 else list(OCR_CLASSES)
    data = {
        "task": "detect", "head": "Detect", "end2end": False,
        "batch": 1, "channels": 3, "imgsz": [size, size],
        "args": {"quantize": 16},
        "names": dict(enumerate(labels)),
    }
    (folder / "metadata.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    (folder / "model.ncnn.param").write_text(
        "7767517\n2 2\nInput in0 0 1 in0\nConcat cat 1 1 in0 out0 0=0\n",
        encoding="utf-8",
    )
    (folder / "model.ncnn.bin").write_bytes(b"demo")
    return folder


def test_export_validation_and_incompatible_metadata(tmp_path):
    folder = _make_export(tmp_path)
    param, weights = _load_model_spec(folder, size=640, labels=("0", "1"))
    assert param.exists() and weights.exists()
    meta = yaml.safe_load((folder / "metadata.yaml").read_text())
    meta["end2end"] = True
    (folder / "metadata.yaml").write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match="contract mismatch"):
        _load_model_spec(folder, size=640, labels=("0", "1"))


def test_checksum_failure_closed(tmp_path):
    folder = _make_export(tmp_path)
    (folder / "sha256.json").write_text(json.dumps({
        "model.ncnn.param": "0" * 64, "model.ncnn.bin": "1" * 64
    }))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _load_model_spec(folder, size=640, labels=("0", "1"))


def test_output_layouts_and_invalid_shape():
    raw = np.zeros((6, 8400), dtype=np.float32)
    assert _standardize_output(raw, classes=2, anchors=8400).shape == (6, 8400)
    assert _standardize_output(raw.T, classes=2, anchors=8400).shape == (6, 8400)
    with pytest.raises(RuntimeError, match="output"):
        _standardize_output(np.zeros((6, 2100)), classes=2, anchors=8400)
    raw[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="NaN"):
        _standardize_output(raw, classes=2, anchors=8400)


def test_yolo_decoder_exactly_matches_opencv_version():
    frame = np.zeros((720, 1280, 3), np.uint8)
    raw = np.zeros((6, 3), np.float32)
    raw[:, 0] = [300, 270, 120, 50, .92, .01]
    raw[:, 1] = [302, 270, 120, 50, .88, .02]
    raw[:, 2] = [510, 410, 80, 35, .03, .9]
    class MockDNN:
        def setInput(self, blob):
            assert blob.shape == (1, 3, 640, 640)
        def forward(self):
            return raw[np.newaxis, ...]
    legacy = YOLOBoxes.__new__(YOLOBoxes)
    legacy.size, legacy.classes = 640, 2
    legacy.conf, legacy.iou = .35, .45
    legacy.net = MockDNN()
    expected = legacy.predict(frame)
    actual = _decode_yolo(raw, frame, size=640, classes=2, conf=.35, iou=.45)
    assert actual == expected
    assert len(actual) == 2


def test_native_extractor_matches_contract_without_library(tmp_path, monkeypatch):
    folder = _make_export(tmp_path)
    raw = np.zeros((6, 8400), np.float32)
    raw[:, 0] = [320, 320, 100, 50, .95, .05]
    seen = {"inputs": [], "extracts": []}

    class FakeMat:
        def __init__(self, array):
            assert array.shape == (3, 640, 640)
            assert array.dtype == np.float32
            self.array = array
        def clone(self):
            return self
    class FakeExtractor:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def input(self, name, tensor):
            seen["inputs"].append(name)
            return 0
        def extract(self, name):
            seen["extracts"].append(name)
            return 0, raw
    class FakeNet:
        def __init__(self):
            self.opt = SimpleNamespace(num_threads=None, use_vulkan_compute=None,
                                       use_fp16_arithmetic=None)
        def load_param(self, path):
            assert path.endswith(".param")
            return 0
        def load_model(self, path):
            assert path.endswith(".bin")
            return 0
        def create_extractor(self):
            return FakeExtractor()

    import sys
    monkeypatch.setitem(sys.modules, "ncnn", SimpleNamespace(Net=FakeNet, Mat=FakeMat))
    backend = NCNNBoxes(folder, 640, 2)
    frame = np.full((720, 1280, 3), 127, np.uint8)
    boxes = backend.predict(frame)
    assert len(boxes) == 1
    assert boxes[0].class_id == 0
    assert seen["inputs"] == ["in0"]
    assert seen["extracts"] == ["out0"]
    assert NCNNPlateRecognizer.recognize is YOLOPlateRecognizer.recognize


def test_uploaded_models_contract_if_installed():
    root = Path(__file__).resolve().parents[1] / "models" / "ncnn_yolo26n_fp16"
    for name, size, labels in (
        ("detector", 640, ("0", "1")),
        ("ocr", 320, tuple(OCR_CLASSES)),
    ):
        if not (root / name / "model.ncnn.bin").is_file():
            pytest.skip("NCNN model binaries not installed in repository")
        _load_model_spec(root / name, size=size, labels=labels)
