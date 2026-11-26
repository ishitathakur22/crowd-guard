"""
Record a ground-truth trajectory for forecast scoring.

The forecast claims to say what the venue will look like in N minutes. To check
that claim you need something to check it against, and without a real venue and
real sensors the only reference available is the live simulation itself. So this
runs the live engine forward with a fixed seed, recording the density grid every
tick and the full crowd state at a few cut points.

`scripts/eval_forecast.py` then rewinds to a cut, forecasts from that exact state,
and compares against what the recording says actually happened.

Honest limitation: this measures whether the cheap forecast model reproduces the
expensive Social Force Model. It is internal consistency, not real-world
predictive validity. That is still the right target, because the gap between the
two is a real and fixable source of error. What it cannot tell you is whether the
Social Force Model matches a real crowd - that needs recorded footage with
annotated ground-plane positions.

Usage:
    python scripts/record_trajectory.py
    python scripts/record_trajectory.py --scenario surge --cuts 60 120
"""

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.rng import seed_from_env  # noqa: E402
from backend.simulation import SimulationEngine, TICK_DURATION_SEC  # noqa: E402

# Must match _ForecastState.MAX_PATH: the particle model truncates paths to this.
MAX_PATH = 96

SCENARIOS = {
    # A steady evening build-up. Sized so the venue settles at a few hundred people:
    # enough for real congestion at the central chokepoint, small enough to record
    # thousands of ticks in reasonable time.
    "default": {"attendance": 30_000, "scale": 1.0, "start_hour": 17.0},
    "surge": {"attendance": 55_000, "scale": 1.0, "start_hour": 17.0},
}


def pack_agents(agents):
    """Struct-of-arrays view of the crowd, matching what the forecast consumes."""
    n = len(agents)
    pos = np.zeros((n, 2), dtype=np.float32)
    vel = np.zeros((n, 2), dtype=np.float32)
    goal = np.zeros((n, 2), dtype=np.float32)
    speed = np.zeros(n, dtype=np.float32)
    radius = np.zeros(n, dtype=np.float32)
    path_xy = np.zeros((n, MAX_PATH, 2), dtype=np.float32)
    path_len = np.zeros(n, dtype=np.int32)

    for i, a in enumerate(agents):
        pos[i] = (a.pos.x, a.pos.y)
        vel[i] = (a.vel.x, a.vel.y)
        goal[i] = (a.goal.x, a.goal.y)
        speed[i] = a.desired_speed
        radius[i] = a.radius
        length = min(len(a.path), MAX_PATH)
        for j in range(length):
            path_xy[i, j] = (a.path[j].x, a.path[j].y)
        path_len[i] = length

    return {
        "pos": pos, "vel": vel, "goal": goal, "speed": speed,
        "radius": radius, "path_xy": path_xy, "path_len": path_len,
    }


def main():
    ap = argparse.ArgumentParser(description="Record a ground-truth simulation trajectory.")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="default")
    ap.add_argument("--cuts", type=float, nargs="+", default=[60.0, 120.0, 180.0])
    ap.add_argument("--horizon-minutes", type=float, default=15.0)
    ap.add_argument("--prefill", type=int, default=100)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--attendance", type=int, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else seed_from_env()
    cuts = sorted(float(c) for c in args.cuts)

    scenario = dict(SCENARIOS[args.scenario])
    if args.attendance is not None:
        scenario["attendance"] = args.attendance
    if args.scale is not None:
        scenario["scale"] = args.scale

    # Truth has to outlast the furthest forecast, or the last snapshots have nothing
    # to be scored against.
    needed = max(cuts) + args.horizon_minutes * 60.0 + 2 * TICK_DURATION_SEC
    total_sec = max(needed, (args.minutes or 0) * 60.0)
    total_ticks = int(round(total_sec / TICK_DURATION_SEC))
    cut_ticks = {int(round(c / TICK_DURATION_SEC)): c for c in cuts}

    engine = SimulationEngine(seed=seed)
    # The internal forecast only reads state, but it costs time and this script
    # never uses it - the evaluation does that later.
    engine.prediction_interval = 10**9
    engine.prediction_engine.horizon_minutes = args.horizon_minutes
    engine.arrival_model.attendance = scenario["attendance"]
    engine.arrival_model.scale = scenario["scale"]
    engine.arrival_model.start_hour = scenario["start_hour"]

    print(
        f"recording {total_ticks} ticks ({total_sec / 60:.1f} sim-minutes), "
        f"scenario={args.scenario}, seed={seed}",
        flush=True,
    )
    if args.prefill:
        engine.spawn_agents(args.prefill)

    da = engine.density_analyzer
    density = np.zeros((total_ticks, engine.height, engine.width), dtype=np.float32)
    sim_times = np.zeros(total_ticks, dtype=np.float64)
    counts = np.zeros(total_ticks, dtype=np.int32)
    cuts_out = {}

    wall_start = time.perf_counter()
    for tick in range(total_ticks):
        engine.update()
        density[tick] = da.compute_raw_density(engine.agents)
        sim_times[tick] = engine.sim_time
        counts[tick] = len(engine.agents)

        if tick in cut_ticks:
            for key, arr in pack_agents(engine.agents).items():
                cuts_out[f"cut_{tick}_{key}"] = arr
            print(f"  cut at {cut_ticks[tick]:.0f}s: {len(engine.agents)} agents", flush=True)

        if (tick + 1) % max(1, total_ticks // 8) == 0:
            print(
                f"  {tick + 1}/{total_ticks} ticks, {len(engine.agents)} agents, "
                f"{time.perf_counter() - wall_start:.0f}s wall",
                flush=True,
            )

    wall_sec = time.perf_counter() - wall_start

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trajectory_{args.scenario}"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        density=density, sim_times=sim_times, agent_counts=counts, **cuts_out,
    )

    meta = {
        "scenario": args.scenario,
        "seed": seed,
        "tick_duration_sec": TICK_DURATION_SEC,
        "total_ticks": total_ticks,
        "cuts_sec": cuts,
        "cut_ticks": sorted(cut_ticks),
        "horizon_minutes": args.horizon_minutes,
        "width": engine.width,
        "height": engine.height,
        "arrival_model": {
            "archetype": engine.arrival_model.archetype,
            "attendance": scenario["attendance"],
            "scale": scenario["scale"],
            "start_hour": scenario["start_hour"],
            "weekend": engine.arrival_model.weekend,
        },
        "prediction": {
            "snapshot_interval_sec": engine.prediction_engine.snapshot_interval_sec,
            "sim_dt": engine.prediction_engine.sim_dt,
            "max_snapshots": engine.prediction_engine.max_snapshots,
            "jam_density": engine.prediction_engine.jam_density,
        },
        "population": {
            "min": int(counts.min()) if counts.size else 0,
            "max": int(counts.max()) if counts.size else 0,
            "final": int(counts[-1]) if counts.size else 0,
        },
        "wall_sec": round(wall_sec, 1),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nrecorded {total_ticks} ticks in {wall_sec:.0f}s "
          f"(population {meta['population']['min']}-{meta['population']['max']})")
    print(f"wrote {out_dir / f'{stem}.npz'}")


if __name__ == "__main__":
    main()
