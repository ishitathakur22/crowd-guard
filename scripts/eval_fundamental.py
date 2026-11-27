"""
Fundamental diagram: does the simulated crowd walk like real people?

The speed-density relation `v(ρ)` is the single most-used validation artifact for a
crowd model. It is what "does this physics reproduce human behaviour" reduces to in
practice, and it requires no data acquisition: Weidmann's curve is a published
closed-form fit to empirical pedestrian measurements.

    v(ρ) = v_free · (1 − exp(−γ · (1/ρ − 1/ρ_jam)))

with `v_free = 1.34 m/s`, `γ = 1.913`, `ρ_jam = 5.4 p/m²` (Weidmann 1993, the form
used by the SFPE handbook and by nearly every pedestrian-simulation paper since).

Why this is worth doing even though it may fail
------------------------------------------------
The SFM's repulsion range was set to `B = 0.10` by fitting *one* number: the mean
walking speed in a sparse venue. A single point says nothing about the shape of the
curve, and it is the shape that governs whether a crowd packs and stalls the way real
ones do. Until now every "accuracy" figure in this repository compared the forecast to
the simulator; this compares the simulator to measurements of people.

Usage:
    python scripts/eval_fundamental.py
    python scripts/eval_fundamental.py --points 12 --seed 1337
"""

import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.density import local_density  # noqa: E402
from backend.simulation import SimulationEngine  # noqa: E402

# Weidmann (1993), as used in the SFPE handbook.
V_FREE = 1.34      # m/s, free walking speed
GAMMA = 1.913      # dimensionless shape parameter
RHO_JAM = 5.4      # p/m², density at which movement stops


def weidmann(rho):
    """Empirical walking speed (m/s) at density `rho` (p/m²)."""
    rho = np.maximum(np.asarray(rho, dtype=float), 1e-6)
    # Below ~0.05 p/m² the 1/rho term is enormous and the curve saturates at v_free.
    return V_FREE * (1.0 - np.exp(-GAMMA * (1.0 / rho - 1.0 / RHO_JAM)))


def measure_at_density(spawn_count, seed=1337, ticks=60, warmup=20):
    """
    Free-walking speed and local density for a fixed crowd size.

    Ingress is disabled and walls are removed so this measures walking, not queueing
    at a chokepoint. Agents cross a 40 m venue in roughly 28 s (~57 ticks), so the
    sampling window has to sit *inside* the crossing: a warmup longer than that
    leaves an empty venue and no measurement at all, which is what a 130-tick warmup
    produced before this was diagnosed.

    Warmup is needed only because the SFM relaxes an agent's velocity towards its
    desired value over roughly `tau`; the first ~10 ticks read low.
    """
    engine = SimulationEngine(seed=seed)
    engine.auto_ingress = False
    engine.prediction_interval = 10**9

    # Clear the venue: this measures the fundamental diagram, not the floor plan.
    from backend.models import CellType, Point
    engine.grid = [[CellType.EMPTY.value for _ in range(engine.width)]
                   for _ in range(engine.height)]
    engine.walls_cache = []
    engine.sfm._wall_set_initialized = False
    engine.sfm._wall_set = set()

    # Spawn across the left edge, send everyone to the right edge. Agents are placed
    # directly rather than through a gate, because a gate imposes a throughput limit
    # and this measurement is of free walking, not admission.
    from backend.models import Agent, CellType, Point
    rng = engine.rng
    for i in range(spawn_count):
        x = rng.uniform(1.0, 6.0)
        y = rng.uniform(1.0, engine.height - 1.0)
        goal = Point(x=engine.width - 2.0, y=y)
        engine.agents.append(Agent(
            id=engine.agent_id_counter,
            pos=Point(x=x, y=y),
            goal=goal,
            path=[],
            desired_speed=1.34 + rng.uniform(-0.12, 0.12),
            radius=0.3,
            mass=80.0,
        ))
        engine.agent_id_counter += 1

    speeds, densities = [], []
    for tick in range(ticks):
        engine.update()
        if tick < warmup or not engine.agents:
            continue
        speeds.append(statistics.fmean(
            (a.vel.x ** 2 + a.vel.y ** 2) ** 0.5 for a in engine.agents))
        raw = engine.density_analyzer.compute_raw_density(engine.agents)
        ld = local_density(raw, 1)
        occupied = raw > 0
        if occupied.any():
            # Mean density over the cells people actually occupy - the same quantity
            # Weidmann's measurements report, not a whole-venue average diluted by
            # empty space.
            densities.append(float(ld[occupied].mean()))

    if not speeds or not densities:
        return None
    return {
        "spawn": spawn_count,
        "agents": statistics.fmean([len(engine.agents)]),
        "speed": statistics.fmean(speeds),
        "density": statistics.fmean(densities),
    }


def main():
    ap = argparse.ArgumentParser(description="Fundamental diagram vs Weidmann.")
    ap.add_argument("--points", type=int, nargs="+",
                    default=[10, 30, 60, 120, 200, 300, 450, 600, 800],
                    help="crowd sizes to sample")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = ap.parse_args()

    print("# Fundamental diagram: simulated vs Weidmann (1993)\n")
    print(f"seed {args.seed}, reference v_free={V_FREE} m/s, "
          f"gamma={GAMMA}, rho_jam={RHO_JAM} p/m2\n")

    rows = []
    for n in args.points:
        r = measure_at_density(n, seed=args.seed)
        if r is None:
            print(f"  spawn {n}: no measurement")
            continue
        expected = float(weidmann(r["density"]))
        err = r["speed"] - expected
        rows.append({**r, "weidmann": expected, "error": err})
        print(f"  spawn {n:>4}: density {r['density']:.3f} p/m2, "
              f"sim {r['speed']:.3f} m/s, weidmann {expected:.3f} m/s, "
              f"error {err:+.3f}")

    if not rows:
        sys.exit("no measurements")

    print("\n## Comparison\n")
    print("| density (p/m2) | simulated | Weidmann | error |")
    print("|---|---|---|---|")
    for r in rows:
        print(f"| {r['density']:.3f} | {r['speed']:.3f} | "
              f"{r['weidmann']:.3f} | {r['error']:+.3f} |")

    errors = np.array([r["error"] for r in rows])
    rmse = float(np.sqrt((errors ** 2).mean()))
    mae = float(np.abs(errors).mean())
    bias = float(errors.mean())

    # The error that matters most is at high density: that is where a crowd either
    # packs realistically or does not, and where the forecast's warnings live.
    dense = [r for r in rows if r["density"] >= 1.0]
    dense_err = float(np.abs([r["error"] for r in dense]).mean()) if dense else float("nan")

    print(f"\n**RMSE:** {rmse:.3f} m/s across {len(rows)} points")
    print(f"**MAE:**  {mae:.3f} m/s")
    print(f"**Bias:** {bias:+.3f} m/s "
          f"({'simulation walks faster' if bias > 0 else 'simulation walks slower'} "
          f"than real people)")
    if dense:
        print(f"**Mean error at rho >= 1.0:** {dense_err:.3f} m/s "
              f"over {len(dense)} points")

    if rmse <= 0.10:
        verdict = ("CLOSE - the simulated crowd tracks Weidmann within 0.10 m/s. "
                   "The physics is a credible basis for real-world use.")
    elif rmse <= 0.25:
        verdict = ("PARTIAL - the trend is right but the magnitude is off. Usable "
                   "for relative comparisons, not for absolute evacuation estimates.")
    else:
        verdict = ("NOT VALIDATED - the simulated crowd does not reproduce measured "
                   "pedestrian behaviour. Re-fitting against real trajectories is a "
                   "prerequisite for any real-world claim.")
    print(f"\n**Verdict:** {verdict}")

    path = pathlib.Path(args.out) / "fundamental_diagram.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reference": "Weidmann 1993",
        "v_free": V_FREE, "gamma": GAMMA, "rho_jam": RHO_JAM,
        "seed": args.seed,
        "rmse": rmse, "mae": mae, "bias": bias,
        "dense_error": dense_err,
        "verdict": verdict,
        "points": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
