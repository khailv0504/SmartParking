"""Regression checks for the two packaged YOLO26 models."""
from pathlib import Path

import pytest

from edge.model_healthcheck import OCR_CLASSES, read_metadata, verify_model

ROOT = Path(__file__).resolve().parents[1] / "models"


@pytest.mark.parametrize("name,filename,expected_shape", [
    ("detector", "detector.onnx", [1, 6, 8400]),
    ("recognizer", "recognizer.onnx", [1, 34, 2100]),
])
def test_model_contract(name, filename, expected_shape):
    model = ROOT / filename
    if not model.is_file():
        pytest.skip(f"Model not installed: {model}")
    report = verify_model(name, model)
    assert report["status"] == "PASS"
    assert report["output"] == expected_shape


def test_character_class_mapping():
    model = ROOT / "recognizer.onnx"
    if not model.is_file():
        pytest.skip("Model not installed")
    names = __import__("ast").literal_eval(read_metadata(model)["names"])
    assert "".join(names[i] for i in range(30)) == OCR_CLASSES
