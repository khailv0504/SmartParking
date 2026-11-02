# Supplied ONNX models (inspected, not retrained)

| Field | detector.onnx | recognizer.onnx |
|---|---|---|
| Ultralytics task | detect | detect |
| Input | (1,3,640,640) | (1,3,320,320) |
| Output | (1,6,8400) | (1,34,2100) |
| Classes | `0`, `1` | `0123456789ABCDEFGHKLMNPSTUVXYZ` in exact class ID order |
| Export args | `quantize: 8` | `quantize: 8` |
| Export version | Ultralytics 8.4.188 | Ultralytics 8.4.188 |

Recognizer is **character detection**, not text recognition with CTC. Post-processing must order boxes by spatial coordinates and join into a plate string.

Model metadata declares `AGPL-3.0` license; review licensing conditions before distribution of derived software/models.

The exported `quantize: 8` setting alone does not prove an exclusively INT8 runtime. These graphs were loaded and forwarded with OpenCV 4.13 on a host; Raspberry Pi/OpenCV compatibility and plate-level accuracy were **not** verified.
