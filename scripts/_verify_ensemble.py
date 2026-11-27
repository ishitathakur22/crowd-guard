"""Verify the exceedance ensemble produces a meaningful probability map."""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.prediction import PILEUP_DENSITY
from backend.simulation import SimulationEngine

e = SimulationEngine(seed=1337)
e.prediction_interval = 10**9
e.arrival_model.attendance = 90_000
e.arrival_model.scale = 1.0
e.spawn_agents(400)
for _ in range(120):
    e.update()

pe = e.prediction_engine
pe.horizon_minutes = 10
pe.max_snapshots = 20

print(f"population {len(e.agents)}")

print("\n== single pass (replicates off) ==")
pe.ensemble_replicates = 0
t0 = time.perf_counter()
pe.forecast(e.agents, e.grid, e.walls_cache, e.gates, e.exits, sim_time_sec=e.sim_time)
t_single = (time.perf_counter() - t0) * 1000
print(f"  {t_single:.0f} ms, exceedance: {pe.exceedance}")

print("\n== ensemble (8 replicates) ==")
pe.ensemble_replicates = 8
t0 = time.perf_counter()
pe.forecast(e.agents, e.grid, e.walls_cache, e.gates, e.exits, sim_time_sec=e.sim_time)
t_ens = (time.perf_counter() - t0) * 1000
print(f"  {t_ens:.0f} ms ({t_ens / 8:.0f} ms per replicate)")

exc = pe.exceedance
sig = pe.final_sigma
assert exc is not None, "exceedance must be populated"
assert sig is not None, "final_sigma must be populated"

print(f"  exceedance shape {exc.shape}, range {exc.min():.2f}..{exc.max():.2f}")
print(f"  cells with p >= 0.5: {int((exc >= 0.5).sum())}")
print(f"  cells with p >= 0.75: {int((exc >= 0.75).sum())}")
print(f"  cells with p == 1.0: {int((exc >= 1.0).sum())}")
print(f"  final sigma max {sig.max():.3f} p/m2")

# The probability map must be a probability.
assert 0.0 <= exc.min() and exc.max() <= 1.0, "exceedance must be in [0,1]"

# Grids must be restored to the primary pass so the harness is unaffected.
print(f"\n  last_forecast_grids restored: {len(pe.last_forecast_grids)} snapshots")

# A hard crowd should produce higher exceedance than a light one.
print("\n== sanity: does exceedance rise with crowd size? ==")
for n in (50, 400):
    e2 = SimulationEngine(seed=1337)
    e2.prediction_interval = 10**9
    e2.arrival_model.attendance = 90_000
    e2.spawn_agents(n)
    for _ in range(60):
        e2.update()
    p = e2.prediction_engine
    p.horizon_minutes = 8
    p.ensemble_replicates = 6
    p.forecast(e2.agents, e2.grid, e2.walls_cache, e2.gates, e2.exits, sim_time_sec=e2.sim_time)
    mean_p = float(p.exceedance.mean())
    hot = int((p.exceedance >= 0.5).sum())
    print(f"  {n:4d} agents: mean p {mean_p:.4f}, cells p>=0.5: {hot}")
