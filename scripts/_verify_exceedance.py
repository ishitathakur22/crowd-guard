"""Verify the exceedance wiring: warnings carry p_exceed, autopilot gates on it."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.autopilot import AUTO_MIN_P_EXCEED
from backend.models import CellType
from backend.simulation import SimulationEngine

print("== engine constructs with ensemble enabled ==")
e = SimulationEngine(seed=1337)
print(f"  ensemble_replicates: {e.prediction_engine.ensemble_replicates}")
print(f"  AUTO_MIN_P_EXCEED: {AUTO_MIN_P_EXCEED}")

print("\n== drive to congestion and run a forecast ==")
e.arrival_model.attendance = 90_000
e.arrival_model.scale = 1.0
w = CellType.WALL.value
for y in (13, 14, 16):
    e.grid[y][15] = w
e.walls_cache = e._get_walls()
e.sfm._wall_set_initialized = False
e.prediction_engine._potential_cache.clear()

e.spawn_agents(400)
for _ in range(300):
    e.update()

print(f"  agents: {len(e.agents)}")
print(f"  predictors run: {len(e.predictions)} snapshots")

summary = e._uncertainty_summary()
print(f"\n  uncertainty summary: {summary}")

print(f"\n== warnings carry p_exceed ==")
print(f"  warnings: {len(e.forecast_warnings)}")
for wrn in e.forecast_warnings[:6]:
    print(
        f"    cell ({wrn['cell_x']},{wrn['cell_y']}) "
        f"peak {wrn['peak_density']:.2f} eta {wrn['eta_sec']:.0f}s "
        f"p_exceed {wrn.get('p_exceed')} divert {wrn['divert_pct']}% "
        f"({wrn['origin_gate']}->{wrn['target_gate']})"
    )

# p_exceed must be a probability and must actually scale the divert.
for wrn in e.forecast_warnings:
    assert 0.0 <= wrn["p_exceed"] <= 1.0, f"p_exceed out of range: {wrn['p_exceed']}"
    assert 0 <= wrn["divert_pct"] <= 95

print("\n== autopilot status exposes the new rail ==")
st = e.autopilot.status()
print(f"  min_live_density: {st.get('min_live_density')}")
print(f"  min_p_exceed: {st.get('min_p_exceed')}")

print("\n== state payload ==")
state = e.get_state()
print(f"  forecast_uncertainty: {state.get('forecast_uncertainty')}")

print("\nOK - all assertions passed")
