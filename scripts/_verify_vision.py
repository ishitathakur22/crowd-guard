"""End-to-end verification of the fixed vision path, using real models on synthetic frames."""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from backend.vision.aggregator import VisionAggregator, dedupe, iou
from backend.vision.pipeline import VisionPipeline

print("== aggregator: dedup ==")
a = {"bbox": [0, 0, 100, 200], "class": 0, "conf": 0.9, "task": "crowd"}
b = {"bbox": [2, 2, 101, 201], "class": 0, "conf": 0.7, "task": "flow"}
c = {"bbox": [500, 500, 600, 700], "class": 0, "conf": 0.8, "task": "crowd"}
d = {"bbox": [0, 0, 100, 200], "class": 2, "conf": 0.6, "task": "traffic"}

print(f"  iou(a,b) = {iou(a['bbox'], b['bbox']):.3f}")
merged = dedupe([a, b, c, d])
print(f"  4 detections -> {len(merged)} after dedup (expect 3: a/c plus the car)")
print(f"  survivor keeps highest conf: {merged[0]['conf']} (expect 0.9)")
print(f"  different class is NOT merged: {any(m['class'] == 2 for m in merged)}")

print("\n== aggregator: calibration ==")
agg = VisionAggregator(40, 30)
agg.calibrate_from_corners(1920, 1080)
print(f"  corner fallback calibrated: {agg.is_calibrated}, proper: {agg.is_proper_homography}")
gx, gy = agg.map_to_grid(960, 540)
print(f"  centre pixel (960,540) -> ({gx:.1f}, {gy:.1f}) expect (20.0, 15.0)")

# A real ground-plane calibration: four floor points.
ok = agg.calibrate(
    [[400, 900], [1500, 880], [1750, 300], [180, 320]],
    [[0, 30], [40, 30], [40, 0], [0, 0]],
)
print(f"  proper homography solved: {ok} (proper={agg.is_proper_homography})")
p = agg.map_points_to_grid([[400, 900], [1750, 300]])
print(f"  floor corners map near grid corners: {np.round(p, 1).tolist()}")

print("\n== aggregator: off-plan detections are dropped, not clamped ==")
agg2 = VisionAggregator(40, 30)
agg2.calibrate_from_corners(1920, 1080)
dets = [
    {"bbox": [0, 0, 10, 10], "center": [5, 8], "class": 0, "conf": 0.9, "task": "crowd"},
    {"bbox": [0, 0, 10, 10], "center": [5000, 5000], "class": 0, "conf": 0.9, "task": "crowd"},
]
out = agg2.aggregate(1, dets)
print(f"  2 detections (one far outside) -> {len(out)} agents (expect 1)")
print(f"  no border clamp: {out[0]['x']:.2f}, {out[0]['y']:.2f}")

print("\n== live pipeline: initialize() returns a report ==")
p = VisionPipeline(grid_width=40, grid_height=30)
t0 = time.perf_counter()
report = p.initialize()
print(f"  returned in {(time.perf_counter() - t0):.1f}s")
print(f"  type: {type(report).__name__}, loaded: {report['loaded']}")
print(f"  models: {report['models']}")
print(f"  unavailable: {report['unavailable']}")
print(f"  device: {report['device']}")
assert isinstance(report, dict) and "loaded" in report, "initialize must return a dict"

print("\n== live pipeline: real inference on a synthetic crowd frame ==")
p.use_depth = False
frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
for cx in range(100, 1200, 90):
    for cy in range(300, 660, 70):
        cv2.circle(frame, (cx, cy), 18, (200, 200, 200), -1)
        cv2.rectangle(frame, (cx - 20, cy - 10), (cx + 20, cy + 55), (180, 180, 180), -1)

results = []
for w in p.workers.values():
    if w.loaded:
        results.extend(w.predict(frame))
print(f"  total raw detections from {len(p.workers)} worker(s): {len(results)}")

# Warm inference, then time it.
t0 = time.perf_counter()
for w in p.workers.values():
    if w.loaded:
        w.predict(frame)
print(f"  warm inference across all workers: {(time.perf_counter() - t0) * 1000:.0f} ms")

agents = p.aggregator.aggregate(1, results)
print(f"  after dedup + mapping: {len(agents)} grid agents")
if agents:
    xs = [a["x"] for a in agents]
    ys = [a["y"] for a in agents]
    print(f"  grid x range {min(xs):.1f}..{max(xs):.1f}, y range {min(ys):.1f}..{max(ys):.1f}")
    print(f"  ids stable across frames: ", end="")
    agents2 = p.aggregator.aggregate(2, results)
    print("yes" if [a["id"] for a in agents] == [a["id"] for a in agents2] else "NO")
