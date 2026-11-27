"""Verify tracking with a real photographic frame (synthetic shapes are not detected)."""

import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from backend.vision.models import YOLOWorker
from backend.vision.pipeline import _w

cache = ROOT / "weights" / ".cache"
cache.mkdir(parents=True, exist_ok=True)
img_path = cache / "bus.jpg"

if not img_path.exists():
    try:
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", img_path)
        print(f"downloaded sample to {img_path}")
    except Exception as exc:
        print("download failed:", exc)

frame = cv2.imread(str(img_path)) if img_path.exists() else None
if frame is None:
    print("no sample image available; cannot verify tracking")
    sys.exit(0)

print(f"frame: {frame.shape}")

worker = YOLOWorker(_w("yolov8n.pt"), "crowd", classes=[0], track=True)
print("loaded:", worker.load())

# Two consecutive "frames": the same image, so ids should persist.
r1 = worker.predict(frame)
r2 = worker.predict(frame)
print(f"\nframe 1: {len(r1)} person detections, tracked: {sum(1 for d in r1 if 'track_id' in d)}")
print(f"frame 2: {len(r2)} person detections, tracked: {sum(1 for d in r2 if 'track_id' in d)}")

if r1 and r2:
    ids1 = {d["track_id"] for d in r1 if "track_id" in d}
    ids2 = {d["track_id"] for d in r2 if "track_id" in d}
    print(f"ids frame 1: {sorted(ids1)}")
    print(f"ids frame 2: {sorted(ids2)}")
    print(f"ids stable across frames: {ids1 == ids2}")
    print(f"sample detection: conf={r1[0]['conf']}, center={r1[0]['center']}, id={r1[0].get('track_id')}")

# Timing on a realistic frame.
import time
worker.predict(frame)
t0 = time.perf_counter()
for _ in range(10):
    worker.predict(frame)
print(f"\nwarm inference + tracking: {(time.perf_counter() - t0) * 100:.1f} ms/frame")
