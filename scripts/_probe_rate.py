import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ingress import ArrivalModel
from backend.rng import Rng

m = ArrivalModel(archetype="gate", attendance=4000, scale=1.0, start_hour=3.0, rng=Rng(1))
print("curve[3] =", m.curve[3], " curve[0] =", m.curve[0])
print("rate_at(0) =", m.rate_at(0.0), "people/s")
print("arrivals_in(0, 0.5) =", m.arrivals_in(0.0, 0.5))

total = sum(m.draw_arrivals(t * 0.5, 0.5) for t in range(4000))
print("expected over 4000 ticks =", sum(m.arrivals_in(t * 0.5, 0.5) for t in range(4000)))
print("drawn =", total)

# What attendance is needed for a reliable non-zero result?
for att in (4000, 20000, 60000):
    mm = ArrivalModel(archetype="gate", attendance=att, scale=1.0, start_hour=3.0, rng=Rng(1))
    exp = sum(mm.arrivals_in(t * 0.5, 0.5) for t in range(4000))
    got = sum(mm.draw_arrivals(t * 0.5, 0.5) for t in range(4000))
    print(f"attendance {att}: expected {exp:.2f}, drawn {got}")
