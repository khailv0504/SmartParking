# SmartParking YOLO26n NCNN FP16: inference-only benchmark deployment

## Audited source

- Repository: `https://github.com/khailv0504/SmartParking`, commit `ed5f56c36b661ee92343694de19af69474f03ed0`.
- On the original repository, `edge/main.py` selects `YOLOPlateRecognizer`, which calls OpenCV DNN ONNX.
- `edge/controller.py`, RFID, camera acquisition, IQA, SQLite, Evidence, Cloud and the business decisions are NOT modified by this bundle.
- The exact same YOLO26s native-NCNN adapter is reused. Uploaded YOLO26n models have the same input/output interface and 30-character mapping. Different weights means accuracy may differ.

## Model contract (derived from supplied ZIPs; not a measured runtime result)

| | Detector | OCR |
|---|---|---|
| Ultralytics | 8.4.193 YOLO26n | 8.4.193 YOLO26n |
| Input | RGB, CHW, batch 1, 640x640 | RGB, CHW, batch 1, 320x320 |
| Output expected | 6x8400 | 34x2100 |
| Classes | 0,1 | `0123456789ABCDEFGHKLMNPSTUVXYZ` |
| Embedded NMS | false | false |
| Graph blobs | `in0`, `out0` | `in0`, `out0` |
| Precision | FP16 weights (`quantize: 16`) | FP16 weights (`quantize: 16`) |

These NCNN models are NOT INT8. PNNX `.param` output graph matches existing YOLO-style XYWH+scores decoding. Runtime tensor shape and numerical correctness MUST be checked with representative real images on Pi.

## Mode 1: You ALREADY installed the previous YOLO26s SafePatch

The safest path is **weights only**. Do not reapply the YOLO26s patch.

```bash
cd ~/SmartParking-ncnn-test
python3 ~/SmartParking_YOLO26n_NCNN_FP16_Bundle/apply_yolo26n.py "$PWD" --mode overlay
python3 ~/SmartParking_YOLO26n_NCNN_FP16_Bundle/apply_yolo26n.py "$PWD" --mode overlay --apply
```

`overlay` requires the exact SHA256 of the previous audited `edge/ncnn_inference.py`; it adds new YOLO26n models and a benchmark tool, and **does not change `edge/main.py` or any existing YOLO26s model**. The old `--backend ncnn` default remains YOLO26s. To use YOLO26n, specify the explicit model directories:

```bash
.venv/bin/python -m edge.main recognize --backend ncnn \
  --detector edge/models/ncnn_yolo26n_fp16/detector \
  --recognizer edge/models/ncnn_yolo26n_fp16/ocr \
  --image /home/khai/camera_test.jpg
```

## Mode 2: You have a FRESH CLONE of original `main` (no NCNN integration)

```bash
git clone https://github.com/khailv0504/SmartParking.git ~/SmartParking-yolo26n-test
cd ~/SmartParking-yolo26n-test
git checkout ed5f56c36b661ee92343694de19af69474f03ed0
git switch -c feature/yolo26n-ncnn-fp16
python3 ~/SmartParking_YOLO26n_NCNN_FP16_Bundle/apply_yolo26n.py "$PWD" --mode fresh
python3 ~/SmartParking_YOLO26n_NCNN_FP16_Bundle/apply_yolo26n.py "$PWD" --mode fresh --apply
```

Fresh mode adds `--backend ncnn` and sets default NCNN model directories to YOLO26n. **No `--backend` still selects original OpenCV DNN ONNX**. All operational arguments and camera/Cloud behavior are unchanged. Refuses unknown revisions and dirty worktrees.

## Python environment (Pi, Ubuntu; do not reuse live deployment venv)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r edge/requirements.txt -r edge/requirements-ncnn.txt pytest
.venv/bin/python -c 'import ncnn, cv2, numpy, yaml; print("inference deps OK")'
.venv/bin/python -m pytest -q edge/tests/test_ncnn_inference.py edge/tests/test_ncnn_factory.py edge/tests/test_controller.py
```

In overlay mode, the `test_ncnn_factory.py` in the YOLO26s clone expects the YOLO26s default, and that is correct. The new tests for fresh mode are only copied to fresh mode; do not replace previous tests.

## Image-only benchmark (no RFID, camera, database or Cloud)

For YOLO26n:

```bash
.venv/bin/python -m edge.bench_ncnn_static \
  --label YOLO26n_NCNN_FP16 --threads 2 \
  --detector edge/models/ncnn_yolo26n_fp16/detector \
  --recognizer edge/models/ncnn_yolo26n_fp16/ocr \
  --images /home/khai/edge_benchmark/data \
  --warmup 3 --repeat 10 \
  --out /home/khai/edge_benchmark/results/yolo26n_fp16_run1
```

For YOLO26s **if the previous bundle is present**:

```bash
.venv/bin/python -m edge.bench_ncnn_static \
  --label YOLO26s_NCNN_FP16 --threads 2 \
  --detector edge/models/ncnn_fp16/detector \
  --recognizer edge/models/ncnn_fp16/ocr \
  --images /home/khai/edge_benchmark/data \
  --warmup 3 --repeat 10 \
  --out /home/khai/edge_benchmark/results/yolo26s_fp16_run1
```

Benchmarks save a CSV with per-image plate predictions + a JSON with mean, P50, P95, inference-only throughput and peak process RSS. These are **not** full RFID-to-event latency or camera FPS.

To measure plate exact-match accuracy, supply `--ground-truth /path/to/labels.csv` where CSV columns are `image,plate`, with one row per test image. Example:

```
image,plate
vehicle_001.jpg,50CA5874
```

Use exactly the SAME image set, `--warmup`, `--repeat`, `--threads`, power supply and thermal conditions for both variants. Repeat with independent processes and note CPU temperature to avoid thermal bias. Do NOT compare model-only latency to RFID transaction latency. Accuracy comparisons across different checkpoints are pipeline comparisons, not pure runtime comparisons.

## Safety and rollback

- Run still-image preflight before any `run --real` invocation.
- In the `overlay` path, `--backend ncnn` without explicit model paths keeps YOLO26s; pass explicit YOLO26n directories only when you want YOLO26n.
- Use fresh, distinct `--db` for transaction tests and avoid Cloud URL/token to prevent accidental uploads.
- Never start two RFID workers that access the same camera or RFID reader.
- On model errors, stop the test process. Do not silently fall back mid-transaction; this could hide mismatches.
- To rollback to ONNX, run without `--backend` in the original code path; to rollback to YOLO26s NCNN in overlay mode, remove only `--detector` and `--recognizer` flags.
- Original `.onnx` and YOLO26s `.bin/.param` weights are never overwritten by this tool.

## Limits of the included checks

Model ZIP metadata + static PNNX graphs + exact SHA256 were inspected. Neither Pi-native NCNN forward nor full production end-to-end inference has been executed in this bundle-building environment. Actual accuracy and speed are not claimed prior to your Raspberry Pi benchmark.
