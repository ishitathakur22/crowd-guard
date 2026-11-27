"""
Record a trajectory that actually reaches LoS E, so lead time can be measured.

Every accuracy figure in `benchmarks/README.md` reports lead time and breach
precision/recall as undefined, because the default and surge scenarios never reach
the pile-up threshold (2.17 p/m2). The machinery for scoring those metrics is written
and correct; it has simply never had anything to fire on.

"Bottleneck lead time" - how many seconds of warning the forecast gave before a cell
actually reached crush density - is the number that justifies the whole system. An
accurate forecast that warns nobody is decoration.

This scenario raises the load until the threshold is genuinely crossed, by combining:

  * a high attendance (a sell-out rather than an evening build-up),
  * a tighter chokepoint (the venue's central corridor narrowed further), and
  * a closure, so the remaining route is the only one.

Now feasible because a 1 000-agent crowd costs ~200 ms per tick rather than 6.6 s.

Usage:
    python scripts/record_hotspot.py
    python scripts/record_hotspot.py --attendance 90000 --cuts 60 150
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

from backend.models import CellType  # noqa: E402
from backend.prediction import PILEUP_DENSITY  # noqa: E402
from backend.rng import seed_from_env  # noqa: E402
from backend.simulation import SimulationEngine, TICK_DURATION_SEC  # noqa: E402

MAX_PATH = 96


def pack_agents(agents):
    n = len(agents)
    out = {
        "pos": np.zeros((n, 2), dtype=np.float32),
        "vel": np.zeros((n, 2), dtype=np.float32),
        "goal": np.zeros((n, 2), dtype=np.float32),
        "speed": np.zeros(n, dtype=np.float32),
        "radius": np.zeros(n, dtype=np.float32),
        "path_xy": np.zeros((n, MAX_PATH, 2), dtype=np.float32),
        "path_len": np.zeros(n, dtype=np.int32),
    }
    for i, a in enumerate(agents):
        out["pos"][i] = (a.pos.x, a.pos.y)
        out["vel"][i] = (a.vel.x, a.vel.y)
        out["goal"][i] = (a.goal.x, a.goal.y)
        out["speed"][i] = a.desired_speed
        out["radius"][i] = a.radius
        length = min(len(a.path), MAX_PATH)
        for j in range(length):
            out["path_xy"][i, j] = (a.path[j].x, a.path[j].y)
        out["path_len"][i] = length
    return out


def tighten_venue(engine):
    """
    Narrow the central chokepoint to a single cell.

    The default grid leaves a four-cell gap at rows 13-16 in the wall at column 15.
    Closing three of them forces the whole crowd through one cell, which is what a
    real pinch point looks like and what produces sustained crush density rather than
    a brief spike.
    """
    w = CellType.WALL.value
    for y in (13, 14, 16):
        engine.grid[y][15] = w
    # Re-derive everything that caches the wall layout.
    engine.walls_cache = engine._get_walls()
    engine.sfm._wall_set_initialized = False
    engine.sfm._wall_set = set()
    engine.prediction_engine._potential_cache.clear()
    engine._flow_cache = None
    return engine


def main():
    ap = argparse.ArgumentParser(description="Record a high-density trajectory.")
    ap.add_argument("--attendance", type=int, default=90_000)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cuts", type=float, nargs="+", default=[60.0, 120.0, 180.0])
    ap.add_argument("--horizon-minutes", type=float, default=10.0)
    ap.add_argument("--minutes", type=float, default=14.0)
    ap.add_argument("--prefill", type=int, default=400)
    ap.add_argument("--no-tighten", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else seed_from_env()
    cuts = sorted(float(c) for c in args.cuts)
    total_sec = max(max(cuts) + args.horizon_minutes * 60.0, args.minutes * 60.0)
    total_ticks = int(round(total_sec / TICK_DURATION_SEC))
    cut_ticks = {int(round(c / TICK_DURATION_SEC)): c for c in cuts}

    engine = SimulationEngine(seed=seed)
    engine.prediction_interval = 10**9
    engine.prediction_engine.horizon_minutes = args.horizon_minutes
    engine.arrival_model.attendance = args.attendance
    engine.arrival_model.scale = args.scale
    engine.arrival_model.start_hour = 17.0

    if not args.no_tighten:
        tighten_venue(engine)

    print(
        f"recording {total_ticks} ticks ({total_sec / 60:.1f} sim-min), "
        f"attendance {args.attendance}, tightened={not args.no_tighten}, seed={seed}",
        flush=True,
    )
    if args.prefill:
        engine.spawn_agents(args.prefill)

    da = engine.density_analyzer
    density = np.zeros((total_ticks, engine.height, engine.width), dtype=np.float32)
    counts = np.zeros(total_ticks, dtype=np.int32)
    peak_series = np.zeros(total_ticks, dtype=np.float32)
    cuts_out = {}

    wall_start = time.perf_counter()
    for tick in range(total_ticks):
        engine.update()
        raw = da.compute_raw_density(engine.agents)
        density[tick] = raw
        counts[tick] = len(engine.agents)
        # Track peak *local* density, the quantity the pile-up threshold is defined on.
        peak_series[tick] = float(da.local_density(raw).max())

        if tick in cut_ticks:
            for key, arr in pack_agents(engine.agents).items():
                cuts_out[f"cut_{tick}_{key}"] = arr
            print(
                f"  cut at {cut_ticks[tick]:.0f}s: {len(engine.agents)} agents, "
                f"peak local density {peak_series[tick]:.2f}",
                flush=True,
            )

        if (tick + 1) % max(1, total_ticks // 8) == 0:
            print(
                f"  {tick + 1}/{total_ticks}, {len(engine.agents)} agents, "
                f"peak {peak_series[tick]:.2f}, {time.perf_counter() - wall_start:.0f}s",
                flush=True,
            )

    wall_sec = time.perf_counter() - wall_start
    over = int((peak_series >= PILEUP_DENSITY).sum())

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "trajectory_hotspot"
    np.savez_compressed(
        out_dir / f"{stem}.npz",
        density=density, agent_counts=counts, peak_local=peak_series, **cuts_out,
    )

    meta = {
        "scenario": "hotspot",
        "seed": seed,
        "tick_duration_sec": TICK_DURATION_SEC,
        "total_ticks": total_ticks,
        "cuts_sec": cuts,
        "cut_ticks": sorted(cut_ticks),
        "horizon_minutes": args.horizon_minutes,
        "width": engine.width,
        "height": engine.height,
        "tightened": not args.no_tighten,
        "arrival_model": {
            "archetype": engine.arrival_model.archetype,
            "attendance": args.attendance,
            "scale": args.scale,
            "start_hour": 17.0,
            "weekend": engine.arrival_model.weekend,
        },
        "prediction": {
            "snapshot_interval_sec": engine.prediction_engine.snapshot_interval_sec,
            "sim_dt": engine.prediction_engine.sim_dt,
            "max_snapshots": engine.prediction_engine.max_snapshots,
            "jam_density": engine.prediction_engine.jam_density,
        },
        "population": {
            "min": int(counts.min()), "max": int(counts.max()),
            "final": int(counts[-1]),
        },
        "pileup": {
            "threshold": PILEUP_DENSITY,
            "peak_local_density": round(float(peak_series.max()), 3),
            "ticks_at_or_above_threshold": over,
            "fraction_of_run": round(over / max(total_ticks, 1), 3),
        },
        "wall_sec": round(wall_sec, 1),
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"\nrecorded {total_ticks} ticks in {wall_sec:.0f}s "
        f"(population {meta['population']['min']}-{meta['population']['max']})"
    )
    print(
        f"peak local density {meta['pileup']['peak_local_density']:.2f} p/m2; "
        f"{over} ticks ({100 * meta['pileup']['fraction_of_run']:.1f}%) at or above "
        f"the LoS E threshold of {PILEUP_DENSITY}"
    )
    if over == 0:
        print("  WARNING: threshold never reached - lead time will still be undefined.")
    print(f"wrote {out_dir / f'{stem}.npz'}")


if __name__ == "__main__":
    main()
