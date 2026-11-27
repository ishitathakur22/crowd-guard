"""Why does every density sample return no measurement?"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import statistics

from backend.density import local_density
from backend.models import Agent, CellType, Point
from backend.simulation import SimulationEngine

e = SimulationEngine(seed=1337)
e.auto_ingress = False
e.prediction_interval = 10**9
e.grid = [[CellType.EMPTY.value for _ in range(e.width)] for _ in range(e.height)]
e.walls_cache = []
e.sfm._wall_set_initialized = False
e.sfm._wall_set = set()

rng = e.rng
for i in range(60):
    x = rng.uniform(1.0, 6.0)
    y = rng.uniform(1.0, e.height - 1.0)
    e.agents.append(Agent(
        id=e.agent_id_counter,
        pos=Point(x=x, y=y),
        goal=Point(x=e.width - 2.0, y=y),
        path=[],
        desired_speed=1.34,
        radius=0.3,
        mass=80.0,
    ))
    e.agent_id_counter += 1

print(f"spawned {len(e.agents)}")

for tick in range(8):
    before = len(e.agents)
    e.update()
    after = len(e.agents)
    if e.agents:
        speeds = statistics.fmean((a.vel.x**2 + a.vel.y**2) ** 0.5 for a in e.agents)
        xs = [a.pos.x for a in e.agents]
        print(f"tick {tick}: pop {before}->{after}, speed {speeds:.3f}, "
              f"x range {min(xs):.1f}..{max(xs):.1f}")
    else:
        print(f"tick {tick}: pop {before}->{after}  ALL GONE")
        break

if e.agents:
    raw = e.density_analyzer.compute_raw_density(e.agents)
    print(f"raw density sum {raw.sum():.0f}, nonzero cells {int((raw > 0).sum())}")
    ld = local_density(raw, 1)
    occ = raw > 0
    print(f"local density mean over occupied: {ld[occ].mean():.3f}")
else:
    # Why did they vanish? Check the arrival rule.
    print("all agents removed - inspect the goal-arrival test")
    print("agents spawn at x 1..6, goal at x", e.width - 2.0)
