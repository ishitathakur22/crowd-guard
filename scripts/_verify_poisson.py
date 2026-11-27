"""Confirm Poisson arrivals have dispersion matching the mean, and that surges occur."""

import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ingress import ArrivalModel
from backend.rng import Rng
from backend.simulation import SimulationEngine, TICK_DURATION_SEC

print("== per-tick arrival dispersion ==")
m = ArrivalModel(archetype="gate", attendance=100_000, scale=1.0,
                 start_hour=17.0, rng=Rng(1337))
draws = [m.draw_arrivals(0.0, TICK_DURATION_SEC) for _ in range(20000)]
mean = statistics.fmean(draws)
var = statistics.fmean([(d - mean) ** 2 for d in draws])
print(f"  mean {mean:.3f}, variance {var:.3f}, ratio {var / mean:.3f}")
print(f"  Poisson expects ratio ~1.0; a deterministic accumulator gives 0.0")
print(f"  observed range {min(draws)}..{max(draws)}")

# How often does a tick exceed 3x the mean? That is the surge that matters.
threshold = 3 * mean
exceed = sum(1 for d in draws if d > threshold)
print(f"  ticks above 3x mean: {exceed} ({100 * exceed / len(draws):.2f}%)")

print("\n== live engine population series ==")
e = SimulationEngine(seed=1337)
e.prediction_interval = 10**9
e.arrival_model.attendance = 55_000
e.arrival_model.scale = 1.0
e.spawn_agents(100)
counts = []
for _ in range(600):
    e.update()
    counts.append(len(e.agents))
print(f"  population min {min(counts)}, max {max(counts)}, "
      f"final {counts[-1]}")
step = [b - a for a, b in zip(counts, counts[1:])]
print(f"  per-tick population change: mean {statistics.fmean(step):+.2f}, "
      f"sd {statistics.pstdev(step):.2f}, "
      f"biggest single-tick rise {max(step)}")
