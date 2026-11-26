import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import backend.simulation as s
    print("import ok")
    e = s.SimulationEngine(seed=1)
    print("engine ok, agents", len(e.agents))
except Exception as exc:
    import traceback

    traceback.print_exc()
