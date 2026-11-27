"""Confirm agents exit before the warmup window ends, and find the right window."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import statistics

from backend.density import local_density
from backend.models import Agent, CellType, Point
from backend.simulation import SimulationEngine

def run(n):
    e = SimulationEngine(seed=1337)
    e.auto_ingress = False
    e.prediction_interval = 10**9
    e.grid = [[CellType.EMPTY.value for _ in range(e.width)] for _ in range(e.height)]
    e.walls_cache = []
    e.sfm._wall_set_initialized = False
    e.sfm._wall_set = set()

    rng = e.rng
    for _ in range(n):
        x = rng.uniform(1.0, 6.0)
        y = rng.uniform(1.0, e.height - 1.0)
        e.agents.append(Agent(
            id=e.agent_id_counter, pos=Point(x=x, y=y),
            goal=Point(x=e.width - 2.0, y=y), path=[],
            desired_speed=1.34, radius=0.3, mass=80.0,
        ))
        e.agent_id_counter += 1

    for tick in range(1, 200):
        e.update()
        if not e.agents:
            return tick, None, None
    speeds = statistics.fmean((a.vel.x**2 + a.vel.y**2)**0.5 for a in e.agents)
    raw = e.density_analyzer.compute_raw_density(e.agents)
    ld = local_density(raw, 1)
    occ = raw > 0
    return None, speeds, float(ld[occ].mean()) if occ.any() else 0.0


for n in (10, 60, 300):
    empty_at, speed, dens = run(n)
    if empty_at:
        print(f"n={n}: venue EMPTY by tick {empty_at} ({empty_at*0.5:.0f}s)")
    else:
        print(f"n={n}: still full at tick 199, speed {speed:.3f}, density {dens:.3f}")
