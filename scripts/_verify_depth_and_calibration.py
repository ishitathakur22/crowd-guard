"""Verify depth removal and the calibration endpoint."""

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("== depth removed ==")
import backend.vision.models as models
import backend.vision.pipeline as pipeline

print(f"  DepthWorker present: {hasattr(models, 'DepthWorker')}")
print(f"  pipeline imports DepthWorker: {'DepthWorker' in pathlib.Path(pipeline.__file__).read_text(encoding='utf-8')}")

p = pipeline.VisionPipeline(grid_width=40, grid_height=30)
print(f"  pipeline has depth_worker attr: {hasattr(p, 'depth_worker')}")

rep = p.initialize()
print(f"  initialize -> loaded={rep['loaded']}, models={rep['models']}")
print(f"  unavailable: {rep['unavailable']}")
assert "depth" not in rep["models"], "depth must not be reported as loaded"
assert not hasattr(p, "depth_worker"), "the depth worker should be gone"

print("\n== calibration persistence ==")
tmpdir = pathlib.Path(tempfile.mkdtemp())
p.CALIBRATION_FILE = str(tmpdir / "vision_calibration.json")

print(f"  initial status: {p.calibration_status()['proper_homography']}")

img = [[400, 900], [1500, 880], [1750, 300], [180, 320]]
grid = [[0, 30], [40, 30], [40, 0], [0, 0]]

ok = p.save_calibration(img, grid)
print(f"  save_calibration -> {ok}")
assert ok, "a valid four-point calibration must be accepted"
assert pathlib.Path(p.CALIBRATION_FILE).exists(), "calibration must be persisted"

st = p.calibration_status()
print(f"  calibrated: {st['calibrated']}, proper: {st['proper_homography']}, saved: {st['saved_file']}")
assert st["proper_homography"] is True

# Mapped points must land on their known grid coordinates.
mapped = p.aggregator.map_points_to_grid(img)
print(f"  mapped floor corners: {[[round(v, 1) for v in pt] for pt in mapped]}")
for (mx, my), (gx, gy) in zip(mapped, grid):
    assert abs(mx - gx) < 1.0 and abs(my - gy) < 1.0, f"({mx},{my}) != ({gx},{gy})"

print("\n== a fresh pipeline reloads the saved calibration ==")
p2 = pipeline.VisionPipeline(grid_width=40, grid_height=30)
p2.CALIBRATION_FILE = p.CALIBRATION_FILE
ok2 = p2.load_calibration()
print(f"  load_calibration -> {ok2}, proper: {p2.aggregator.is_proper_homography}")
assert ok2 and p2.aggregator.is_proper_homography

print("\n== invalid calibration is rejected, leaving the good one intact ==")
ok3 = p.save_calibration([[0, 0], [1, 1]], [[0, 0], [1, 1]])
print(f"  two-point calibration accepted: {ok3}")
assert ok3 is False, "a calibration with fewer than four points must be rejected"
assert p.aggregator.is_proper_homography, "the previous valid calibration must survive"

print("\nOK - all assertions passed")
