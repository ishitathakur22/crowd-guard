"""
Per-stage latency benchmark for the crowd engine.

Why per stage and not just "ticks per second": a single end-to-end number tells you
that something is slow but not what, and the optimisations target specific stages.
Measuring them separately means a change can be judged against the stage it touches
instead of hoping the aggregate moved.

Agent counts are swept because almost every stage is superlinear somewhere: the
Social Force Model is O(n * neighbours), rerouting scales with crowd size, and the
serialised payload grows with n. 50 is a quiet concourse, 1000 is a sell-out.

The engine is built directly, never through `backend.main`, so this pulls in no
FastAPI app, no vision models and no camera.

Usage:
    python scripts/bench.py
    python scripts/bench.py --agents 200 500 --reps 30
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.models import Point  # noqa: E402
from backend.rng import seed_from_env  # noqa: E402
from backend.simulation import SimulationEngine, TICK_DURATION_SEC  # noqa: E402

# Mirrors simulation.REROUTE_COOLDOWN_TICKS. Rerouting is gated per agent on this
# window, so a repeated measurement has to advance the clock or every repeat after
# the first silently measures an empty no-op.
REROUTE_COOLDOWN_TICKS = 10


def _quantile(sorted_samples, q):
    if not sorted_samples:
        return float("nan")
    i = min(len(sorted_samples) - 1, max(0, int(round(q * (len(sorted_samples) - 1)))))
    return sorted_samples[i]


def measure(fn, reps, warmup=2):
    """Time `fn` reps times, discarding `warmup` calls to discount cache warming."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    s = sorted(samples)
    return {
        "reps": reps,
        "mean_ms": round(1000.0 * statistics.fmean(s), 3),
        "p50_ms": round(1000.0 * _quantile(s, 0.50), 3),
        "p95_ms": round(1000.0 * _quantile(s, 0.95), 3),
        "max_ms": round(1000.0 * s[-1], 3),
    }


def _rss_mb():
    try:
        import psutil  # optional, not in requirements

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:
        return None


def bench_one(n_agents, seed, reps):
    engine = SimulationEngine(seed=seed)
    # Hold the population steady so repeats measure the same workload: no new
    # arrivals, and no forecast inside the tick (measured separately below).
    engine.auto_ingress = False
    engine.prediction_interval = 10**9

    t0 = time.perf_counter()
    engine.spawn_agents(n_agents)
    spawn_ms = (time.perf_counter() - t0) * 1000.0

    population = len(engine.agents)
    # Keep total work per row roughly constant across the sweep.
    eff = max(3, min(reps, int(4000 / max(population, 1))))

    da = engine.density_analyzer
    raw = da.compute_raw_density(engine.agents)

    stages = {}
    stages["density | raw counts"] = measure(
        lambda: da.compute_raw_density(engine.agents), eff
    )
    stages["density | local density"] = measure(lambda: da.local_density(raw), eff)
    stages["density | gaussian smooth"] = measure(
        lambda: da.compute_smoothed_density(raw), eff
    )
    stages["density | flow vectors"] = measure(
        lambda: da.get_flow_vectors_for_viz(engine.agents), eff
    )
    stages["density | heatmap payload"] = measure(
        lambda: da.get_density_grid_for_heatmap(engine.agents), eff
    )
    stages["density | classify zones"] = measure(lambda: da.classify_zones(raw), eff)
    stages["density | detect hazards"] = measure(lambda: da.detect_hazards(raw), eff)

    stages["physics | 1 SFM substep"] = measure(
        lambda: engine.sfm.update_positions(engine.agents, engine.walls_cache), eff
    )
    stages["physics | all substeps/tick"] = measure(
        lambda: [
            engine.sfm.update_positions(engine.agents, engine.walls_cache)
            for _ in range(engine.physics_substeps)
        ],
        max(2, eff // 2),
    )

    # Rerouting only does work when a bottleneck sits near an agent's path, so one is
    # forced at the main chokepoint.
    engine.bottlenecks = [Point(x=15.0, y=14.0)]

    def reroute_cached():
        # tick_count NOT a multiple of the cooldown, so the existing field is reused.
        engine.tick_count += REROUTE_COOLDOWN_TICKS + 1
        engine._last_reroute.clear()
        engine._handle_rerouting(raw)

    stages["reroute | cached flow field"] = measure(reroute_cached, max(2, eff // 4))

    def reroute_rebuild():
        # Land exactly on a cooldown boundary so the Dijkstra rebuild is included.
        engine.tick_count += REROUTE_COOLDOWN_TICKS - (engine.tick_count % REROUTE_COOLDOWN_TICKS)
        engine._last_reroute.clear()
        engine._flow_cache = None
        engine._handle_rerouting(raw)

    stages["reroute | incl. field rebuild"] = measure(
        reroute_rebuild, max(2, eff // 8)
    )

    stages["forecast | 15 min horizon"] = measure(
        lambda: engine.prediction_engine.forecast(
            engine.agents, engine.grid, engine.walls_cache,
            engine.gates, engine.exits, sim_time_sec=engine.sim_time,
        ),
        max(2, eff // 10),
    )

    stages["serialise | get_state()"] = measure(lambda: engine.get_state(), eff)

    def spawn_batch():
        before = len(engine.agents)
        engine.spawn_agents(10)
        del engine.agents[before:]

    stages["spawn | 10 agents (A* each)"] = measure(spawn_batch, max(2, eff // 4))

    stages["TICK | update() end to end"] = measure(engine.update, max(3, eff // 2))

    return {
        "agents_requested": n_agents,
        "agents_populated": population,
        "initial_spawn_ms": round(spawn_ms, 1),
        "rss_mb": _rss_mb(),
        "stages": stages,
    }


def print_tables(results, seed):
    print(f"\n# Crowd engine benchmark - seed {seed}, Python {sys.version.split()[0]}")
    print(f"Tick budget at 1x speed: {TICK_DURATION_SEC * 1000:.0f} ms of real time per tick.\n")

    for r in results:
        rss = f", RSS {r['rss_mb']} MB" if r["rss_mb"] is not None else ""
        print(
            f"## {r['agents_populated']} agents{rss}  "
            f"(initial spawn of {r['agents_requested']}: {r['initial_spawn_ms']} ms)"
        )
        print("| stage | mean | p50 | p95 | max |")
        print("|---|---|---|---|---|")
        for name, m in r["stages"].items():
            print(
                f"| {name} | {m['mean_ms']:.2f} | {m['p50_ms']:.2f} "
                f"| {m['p95_ms']:.2f} | {m['max_ms']:.2f} |"
            )
        tick = r["stages"]["TICK | update() end to end"]["mean_ms"]
        headroom = TICK_DURATION_SEC * 1000.0 / tick if tick > 0 else float("inf")
        print(
            f"| **speed headroom** | {headroom:.2f}x realtime "
            f"({1000.0 / tick:.1f} ticks/s) | | | |"
        )
        print()


def main():
    ap = argparse.ArgumentParser(description="Per-stage latency benchmark.")
    ap.add_argument("--agents", type=int, nargs="+", default=[50, 200, 500, 1000])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else seed_from_env()

    results = []
    for n in args.agents:
        print(f"benchmarking {n} agents ...", flush=True)
        results.append(bench_one(n, seed, args.reps))

    print_tables(results, seed)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bench_baseline.json"
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "python": sys.version.split()[0],
                "tick_duration_sec": TICK_DURATION_SEC,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
