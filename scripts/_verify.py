"""Verify the restored engine: seeded determinism, plausible speed, physics cost."""

import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.models import Agent, Point
from backend.simulation import SimulationEngine


def state(e):
    return np.array([[a.pos.x, a.pos.y, a.vel.x, a.vel.y] for a in e.agents])


def build(seed, agents=80):
    e = SimulationEngine(seed=seed)
    e.prediction_interval = 10**9
    e.arrival_model.attendance = 8000
    e.arrival_model.scale = 1.0
    e.spawn_agents(agents)
    return e


print("== seeded determinism ==")
a, b = build(4242), build(4242)
for _ in range(40):
    a.update()
    b.update()
print(f"  same seed identical after 40 ticks: {np.array_equal(state(a), state(b))}")

c = build(7)
for _ in range(40):
    c.update()
print(f"  different seed diverges:           {not np.array_equal(state(a), state(c))}")

print("\n== shared-default guard ==")
x = Agent(id=0, pos=Point(x=1.0, y=2.0), goal=Point(x=5.0, y=6.0))
y = Agent(id=1, pos=Point(x=1.0, y=2.0), goal=Point(x=5.0, y=6.0))
x.vel.x = 99.0
print(f"  agents share vel object: {x.vel is y.vel} (must be False)")
print(f"  mutating one leaves other at 0: {y.vel.x == 0.0}")

print("\n== crowd speed (should be near a real pedestrian's 1.3 m/s) ==")
e = SimulationEngine(seed=1337)
e.prediction_interval = 10**9
e.arrival_model.attendance = 55000
e.arrival_model.scale = 1.0
e.spawn_agents(200)
speeds, pops = [], []
for tick in range(500):
    e.update()
    if tick >= 250 and e.agents:
        speeds.append(statistics.fmean((v.vel.x**2 + v.vel.y**2) ** 0.5 for v in e.agents))
        pops.append(len(e.agents))
ms = statistics.fmean(speeds)
print(f"  mean walking speed: {ms:.3f} m/s  ({ms / 1.3:.2f}x real)")
print(f"  mean population:    {statistics.fmean(pops):.0f}")

print("\n== physics cost per tick ==")
for n in (200, 500):
    e = build(1, agents=n)
    e.auto_ingress = False
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        for _ in range(e.physics_substeps):
            e.sfm.update_positions(e.agents, e.walls_cache)
    per_tick = (time.perf_counter() - t0) / reps * 1000
    print(f"  {n} agents: {per_tick:.1f} ms per tick of physics")
