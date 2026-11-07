"""Backend-selection regression: default remains OpenCV DNN, NCNN opt-in only."""
from types import SimpleNamespace
from unittest.mock import patch

from edge.main import ROOT, make_real_recognizer


def test_default_factory_uses_original_onnx_models():
    args = SimpleNamespace(backend='opencv', detector='', recognizer='', strict_roi_quality=False)
    with patch('edge.main.YOLOPlateRecognizer') as opencv:
        result = make_real_recognizer(args)
    opencv.assert_called_once_with(
        ROOT / 'models' / 'detector.onnx',
        ROOT / 'models' / 'recognizer.onnx',
        strict_roi_quality=False,
    )
    assert result is opencv.return_value


def test_ncnn_factory_is_explicit_opt_in():
    args = SimpleNamespace(backend='ncnn', detector='', recognizer='', strict_roi_quality=True)
    with patch('edge.ncnn_inference.NCNNPlateRecognizer') as ncnn:
        result = make_real_recognizer(args)
    ncnn.assert_called_once_with(
        ROOT / 'models' / 'ncnn_yolo26n_fp16' / 'detector',
        ROOT / 'models' / 'ncnn_yolo26n_fp16' / 'ocr',
        strict_roi_quality=True,
    )
    assert result is ncnn.return_value
