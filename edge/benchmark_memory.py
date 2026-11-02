import cv2
import os
import resource

from pathlib import Path
from edge.recognition import YOLOPlateRecognizer

ROOT = Path(__file__).resolve().parent
IMAGE = ROOT.parent / "img.png"

def memory_info():
    with open("/proc/meminfo") as f:
        values = {}
        for line in f:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])

    return {
        "available_mb": values["MemAvailable"] / 1024,
        "swap_free_mb": values["SwapFree"] / 1024,
    }

def process_rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0

print("=== BEFORE LOADING MODELS ===")
print(memory_info())

model = YOLOPlateRecognizer(
    ROOT / "models/detector.onnx",
    ROOT / "models/recognizer.onnx",
    threads=2,
)

print("\n=== MODELS LOADED ===")
print("Process RSS:", round(process_rss_mb(), 2), "MiB")
print(memory_info())

image = cv2.imread(str(IMAGE))
if image is None:
    raise RuntimeError(f"Cannot read: {IMAGE}")

result = model.recognize(image)

print("\n=== AFTER INFERENCE ===")
print("Plate:", result.plate)
print("Latency:", round(result.latency_ms, 2), "ms")
print("Process RSS:", round(process_rss_mb(), 2), "MiB")
print(
    "Peak RSS:",
    round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    "MiB",
)
print(memory_info())

input("\nPress Enter to release models and exit...")
