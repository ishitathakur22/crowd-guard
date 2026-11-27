"""
Consolidated verification checks for Crowd Guard.

Each check answers one question about the implementation that a test suite cannot
easily cover, either because it needs real models, real timing, or a measured
property rather than a pass/fail invariant. This replaced ~45 one-off scratch
scripts written while fixing individual defects.

Run all:
    python scripts/checks.py

Run one:
    python scripts/checks.py poisson
    python scripts/checks.py density
    python scripts/checks.py speed
    python scripts/checks.py sfm
    python scripts/checks.py ensemble
    python scripts/checks.py calibration
    python scripts/checks.py imports
"""

import argparse
import importlib.util
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def _header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ── imports and dependency surface ────────────────────────────────────


def check_imports():
    """Every backend module must import, and the optional deps must be reported honestly."""
    _header("imports")

    import ast

    for name in ("main.py", "simulation.py", "prediction.py", "physics.py",
                 "density.py", "pathfinding.py", "ingress.py", "autopilot.py",
                 "models.py", "rng.py"):
        src = (ROOT / "backend" / name).read_text(encoding="utf-8")
        ast.parse(src)
        print(f"  {name}: parses")

    import backend.main  # noqa: F401
    routes = [r.path for r in backend.main.app.routes if hasattr(r, "path")]
    print(f"  backend.main imports, {len(routes)} routes")
    assert any("calibration" in r for r in routes), "calibration endpoint missing"

    optional = {}
    for mod in ("llama_cpp", "cv2", "ultralytics", "torch", "lap"):
        optional[mod] = bool(importlib.util.find_spec(mod))
    print(f"  optional deps: {optional}")

    import backend.vision.models as vmodels
    assert not hasattr(vmodels, "DepthWorker"), "depth was removed; it must stay gone"
    print("  depth worker: absent (expected)")


# ── density: numerical identity and speed ─────────────────────────────


def check_density():
    """The separable convolution must match the original loop, and be much faster."""
    _header("density: equivalence and speed")

    from backend.density import DensityAnalyzer

    def reference_smooth(raw, kernel, kr):
        h, w = raw.shape
        ks = 2 * kr + 1
        padded = np.pad(raw, kr, mode="constant", constant_values=0)
        out = np.zeros_like(raw, dtype=np.float64)
        for y in range(h):
            for x in range(w):
                out[y][x] = np.sum(padded[y:y + ks, x:x + ks] * kernel)
        return out

    da = DensityAnalyzer(40, 30, kernel_radius=2)
    rng = np.random.default_rng(0)

    worst = 0.0
    for _ in range(5):
        raw = rng.random((30, 40)) * 3.0
        worst = max(worst, float(np.abs(reference_smooth(raw, da.kernel, da.kernel_radius)
                                        - da.compute_smoothed_density(raw)).max()))
    print(f"  max deviation from per-cell loop: {worst:.3e}")
    assert worst < 1e-12, "separable smoothing is not numerically equivalent"

    raw = rng.random((30, 40)) * 2.0
    t0 = time.perf_counter()
    for _ in range(50):
        reference_smooth(raw, da.kernel, da.kernel_radius)
    old = (time.perf_counter() - t0) / 50 * 1000
    t0 = time.perf_counter()
    for _ in range(50):
        da.compute_smoothed_density(raw)
    new = (time.perf_counter() - t0) / 50 * 1000
    print(f"  per-cell loop {old:.3f} ms -> separable {new:.3f} ms ({old / new:.1f}x)")
    assert new < old, "no speedup"


# ── crowd walking speed ───────────────────────────────────────────────


def check_speed():
    """
    Crowd speed must be plausible at low density and fall as density rises.

    An absolute band would be wrong: walking speed is a function of crowding, and a
    dense venue legitimately moves at about 0.4 m/s. What distinguishes correct
    behaviour from the old over-damping bug is the *shape* of the curve - real crowds
    slow monotonically with density, while the bug produced 0.42 m/s even in an empty
    corridor.
    """
    _header("crowd walking speed vs density")

    from backend.density import local_density
    from backend.simulation import SimulationEngine

    rows = []
    for target in (20, 120, 300):
        e = SimulationEngine(seed=1337)
        e.prediction_interval = 10**9
        e.auto_ingress = False          # hold the population steady
        e.spawn_agents(target)

        speeds, dens = [], []
        for tick in range(200):
            e.update()
            if tick >= 110 and e.agents:
                speeds.append(statistics.fmean(
                    (a.vel.x ** 2 + a.vel.y ** 2) ** 0.5 for a in e.agents))
                raw = e.density_analyzer.compute_raw_density(e.agents)
                ld = local_density(raw, 1)
                occupied = raw > 0
                dens.append(float(ld[occupied].mean()) if occupied.any() else 0.0)

        if speeds:
            rows.append((target, statistics.fmean(speeds), statistics.fmean(dens)))

    print("  spawn | mean speed | mean local density")
    for target, speed, density in rows:
        print(f"  {target:>5} | {speed:>10.3f} | {density:>18.3f}")

    sparse_speed = rows[0][1]
    print(f"\n  speed in a sparse venue: {sparse_speed:.3f} m/s")

    # The real discriminator: sparse must approach a real walking speed. The
    # over-damped bug never exceeded 0.46 m/s even when nearly empty.
    assert sparse_speed > 0.7, (
        f"only {sparse_speed:.3f} m/s when nearly empty - this is the over-damped "
        f"symptom, not crowding (the bug produced 0.42-0.46 m/s regardless of density)"
    )

    # And denser must be slower, monotonically.
    speeds = [s for _t, s, _d in rows]
    assert speeds[0] >= speeds[-1], (
        f"speed did not fall with density ({speeds[0]:.3f} -> {speeds[-1]:.3f}); "
        f"crowds should slow as they pack"
    )
    print(f"  monotonic slowing with density: {speeds[0]:.3f} -> {speeds[-1]:.3f}")


# ── SFM repulsion range ───────────────────────────────────────────────


def check_sfm():
    """
    The repulsion range must still let a sparse crowd walk freely.

    Measured at low occupancy, where density cannot mask the effect. At B=0.30 the
    braking term dominated even with nobody nearby and the crowd managed 0.45 m/s;
    at B=0.10 it should approach a real pedestrian's pace.
    """
    _header("SFM repulsion range (sparse crowd)")

    from backend.simulation import SimulationEngine

    results = {}
    for B in (0.30, 0.10):
        e = SimulationEngine(seed=1337)
        e.prediction_interval = 10**9
        e.auto_ingress = False
        e.sfm.B = B
        e.spawn_agents(40)

        speeds, exited = [], 0
        for tick in range(200):
            before = len(e.agents)
            e.update()
            exited += max(0, before - len(e.agents))
            if tick >= 100 and e.agents:
                speeds.append(statistics.fmean(
                    (a.vel.x ** 2 + a.vel.y ** 2) ** 0.5 for a in e.agents))

        results[B] = (statistics.fmean(speeds), exited)
        print(f"  B={B:.2f}: speed {results[B][0]:.3f} m/s, {exited} exited")

    assert results[0.10][0] > results[0.30][0], (
        "narrowing B should let the crowd move faster, not slower"
    )
    assert results[0.10][0] > 0.7, (
        f"B=0.10 gives only {results[0.10][0]:.3f} m/s in a sparse venue"
    )
    print(f"  B=0.10 is {results[0.10][0] / results[0.30][0]:.2f}x faster than B=0.30")


# ── Poisson arrivals ──────────────────────────────────────────────────


def check_poisson():
    """Arrivals must have variance equal to the mean, not zero."""
    _header("Poisson arrivals")

    from backend.ingress import ArrivalModel
    from backend.rng import Rng

    m = ArrivalModel(archetype="gate", attendance=100_000, scale=1.0,
                     start_hour=17.0, rng=Rng(1337))
    draws = [m.draw_arrivals(0.0, 0.5) for _ in range(20_000)]
    mean = statistics.fmean(draws)
    var = statistics.fmean([(d - mean) ** 2 for d in draws])
    ratio = var / mean

    print(f"  mean {mean:.3f}, variance {var:.3f}, ratio {ratio:.3f}")
    print(f"  range {min(draws)}..{max(draws)}")
    above = sum(1 for d in draws if d > 3 * mean)
    print(f"  ticks above 3x mean: {above} ({100 * above / len(draws):.2f}%)")
    assert 0.7 < ratio < 1.4, f"variance/mean {ratio:.3f} is not Poisson-like"

    # Determinism from the seed.
    a = ArrivalModel(attendance=50_000, start_hour=17.0, rng=Rng(7))
    b = ArrivalModel(attendance=50_000, start_hour=17.0, rng=Rng(7))
    assert [a.draw_arrivals(t * 0.5, 0.5) for t in range(200)] == \
           [b.draw_arrivals(t * 0.5, 0.5) for t in range(200)]
    print("  seeded draws reproducible: yes")


# ── exceedance ensemble ───────────────────────────────────────────────


def check_ensemble():
    """The ensemble must produce variance, and must be off by default."""
    _header("exceedance ensemble")

    from backend.models import CellType
    from backend.simulation import SimulationEngine

    e = SimulationEngine(seed=1337)
    print(f"  default ensemble_replicates: {e.prediction_engine.ensemble_replicates}")
    assert e.prediction_engine.ensemble_replicates == 0, (
        "the ensemble must stay off until it is calibrated"
    )

    e.prediction_interval = 10**9
    e.arrival_model.attendance = 90_000
    w = CellType.WALL.value
    for y in (13, 14, 16):
        e.grid[y][15] = w
    e.walls_cache = e._get_walls()
    e.sfm._wall_set_initialized = False
    e.prediction_engine._potential_cache.clear()
    e.spawn_agents(250)
    for _ in range(140):
        e.update()

    pe = e.prediction_engine
    pe.horizon_minutes = 6.0
    pe.ensemble_replicates = 4
    t0 = time.perf_counter()
    pe.forecast(e.agents, e.grid, e.walls_cache, e.gates, e.exits,
                sim_time_sec=e.sim_time)
    ms = (time.perf_counter() - t0) * 1000

    print(f"  4 replicates: {ms:.0f} ms, sigma max {pe.final_sigma.max():.3f}")
    print(f"  exceedance range {pe.exceedance.min():.2f}..{pe.exceedance.max():.2f}")
    assert pe.final_sigma.max() > 0, "replicates produced identical results"


# ── vision calibration ────────────────────────────────────────────────


def check_calibration():
    """A four-point calibration must persist, reload and validate."""
    _header("vision calibration")

    import tempfile

    from backend.vision.pipeline import VisionPipeline

    p = VisionPipeline(grid_width=40, grid_height=30)
    p.CALIBRATION_FILE = str(pathlib.Path(tempfile.mkdtemp()) / "cal.json")

    img = [[400, 900], [1500, 880], [1750, 300], [180, 320]]
    grid = [[0, 30], [40, 30], [40, 0], [0, 0]]

    assert p.save_calibration(img, grid), "valid calibration rejected"
    assert p.aggregator.is_proper_homography
    print(f"  saved and applied: {p.calibration_status()['proper_homography']}")

    p2 = VisionPipeline(grid_width=40, grid_height=30)
    p2.CALIBRATION_FILE = p.CALIBRATION_FILE
    assert p2.load_calibration(), "saved calibration did not reload"
    print("  reloaded by a fresh pipeline: yes")

    assert not p.save_calibration([[0, 0], [1, 1]], [[0, 0], [1, 1]]), \
        "a two-point calibration must be rejected"
    assert p.aggregator.is_proper_homography, "the good calibration must survive"
    print("  invalid calibration rejected, previous one intact")


CHECKS = {
    "imports": check_imports,
    "density": check_density,
    "speed": check_speed,
    "sfm": check_sfm,
    "poisson": check_poisson,
    "ensemble": check_ensemble,
    "calibration": check_calibration,
}


def main():
    ap = argparse.ArgumentParser(description="Verification checks.")
    ap.add_argument("checks", nargs="*", choices=sorted(CHECKS) + [[]],
                    help="which checks to run (default: all)")
    args = ap.parse_args()

    selected = args.checks or list(CHECKS)
    failures = 0
    for name in selected:
        try:
            CHECKS[name]()
            print(f"  -> PASS")
        except AssertionError as exc:
            failures += 1
            print(f"  -> FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  -> ERROR: {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 70}")
    print(f"{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
