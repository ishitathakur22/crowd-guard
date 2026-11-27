"""Smoke-test the vision stack: can the models actually load and run here?"""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.vision.model_manager import ModelManager, ModelInfo

WEIGHTS = ROOT / "weights"

print("== weights present ==")
for p in sorted(WEIGHTS.glob("*")):
    print(f"  {p.name}: {p.stat().st_size / 2**20:.1f} MB")

print("\n== torch ==")
import torch

print(f"  torch {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  vram: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

print("\n== YOLO load + one inference ==")
try:
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(str(WEIGHTS / "yolov8n.pt"))
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  loaded in {load_ms:.0f} ms")

    frame = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
    t0 = time.perf_counter()
    result = model(frame, classes=[0], verbose=False)
    infer_ms = (time.perf_counter() - t0) * 1000
    n = 0 if result[0].boxes is None else len(result[0].boxes)
    print(f"  inference in {infer_ms:.0f} ms, {n} detections on noise")
    print(f"  boxes.data available: {result[0].boxes is not None}")
except Exception as exc:
    import traceback

    print("  FAILED:", exc)
    traceback.print_exc()

print("\n== MiDaS load ==")
try:
    import cv2

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    print("  MiDaS via torch.hub: OK")
except Exception as exc:
    print(f"  MiDaS via torch.hub FAILED: {type(exc).__name__}: {exc}")
    print("  (the pipeline has a local-weights path; check models.py DepthWorker)")

print("\n== ModelManager status ==")
mm = ModelManager()
try:
    st = mm.get_status()
    print(f"  gpu_available: {st.get('gpu_available')}")
    print(f"  models: {list(st.get('models', {}).keys())}")
except Exception as exc:
    print(f"  status failed: {type(exc).__name__}: {exc}")
