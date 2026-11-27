import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ast

src = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
ast.parse(src)
print("main.py parses OK")
for token in ("/api/vision/calibration", "CalibrationRequest", "calibration_status"):
    print(f"  {token}: {src.count(token)}")

import backend.main as m
print("backend.main imports OK")
routes = [r.path for r in m.app.routes if hasattr(r, "path")]
print(f"  calibration routes: {[r for r in routes if 'calibration' in r]}")

import backend.vision.pipeline as p
print(f"  depth_worker references in pipeline: "
      f"{pathlib.Path(p.__file__).read_text(encoding='utf-8').count('depth_worker')}")
print(f"  load_calibration present: {hasattr(p.VisionPipeline, 'load_calibration')}")
print(f"  save_calibration present: {hasattr(p.VisionPipeline, 'save_calibration')}")
