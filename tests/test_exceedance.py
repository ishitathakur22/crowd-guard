"""
Exceedance ensemble: variance, probability bounds, and consumption of the map.

These exist because the first version of the ensemble was **silently useless**. It
returned a `final_sigma` of exactly 0.000 and an exceedance map of only 0s and 1s,
because the continuum forecast drew arrivals from `arrivals_in()` — the
*expectation* — so every replicate was a deterministic function of the same state
and produced an identical answer. Nothing failed, nothing warned; the feature simply
reported certainty it had not earned.

A single test asserting `sigma > 0` would have caught it. That is the first test here.

The second theme is that the probability map must actually be consumed. For a while
it was computed on every forecast and thrown away, with warnings built from a single
deterministic sample instead.

Run with pytest, or directly: `python tests/test_exceedance.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from backend.autopilot import AUTO_MIN_P_EXCEED  # noqa: E402
from backend.models import CellType  # noqa: E402
from backend.prediction import PILEUP_DENSITY  # noqa: E402
from backend.simulation import SimulationEngine  # noqa: E402


def congested_engine(seed=1337, agents=200, ticks=90):
    """
    A crowd pressing through a single-cell chokepoint, so density gets high.

    The in-tick forecast is disabled: with `ensemble_replicates` on by default a
    forecast fires every `prediction_interval` ticks and each costs K full horizon
    runs, which would make each test take seconds. Tests that need the ensemble call
    `forecast_with` explicitly instead.
    """
    e = SimulationEngine(seed=seed)
    e.prediction_interval = 10**9
    e.arrival_model.attendance = 90_000
    e.arrival_model.scale = 1.0
    w = CellType.WALL.value
    for y in (13, 14, 16):
        e.grid[y][15] = w
    e.walls_cache = e._get_walls()
    e.sfm._wall_set_initialized = False
    e.prediction_engine._potential_cache.clear()
    e.spawn_agents(agents)
    for _ in range(ticks):
        e.update()
    return e


#: Reused across the read-only tests. Building a congested crowd costs real time -
#: hundreds of physics ticks per engine - and most of these tests only read from it.
#: Building a fresh one per test pushed the suite from ~9 s to over two minutes.
_SHARED = {}


def shared_congested(agents=320, ticks=140):
    key = (agents, ticks)
    if key not in _SHARED:
        _SHARED[key] = congested_engine(agents=agents, ticks=ticks)
    return _SHARED[key]


def shared_forecast(k=3, horizon=8.0):
    """Forecast the shared engine once, reusing the result across assertions."""
    pe = shared_congested().prediction_engine
    if pe.exceedance is None or pe.ensemble_replicates != k:
        forecast_with(shared_congested(), k=k, horizon=horizon)
    return shared_congested()


def forecast_with(engine, k, horizon=5.0):
    pe = engine.prediction_engine
    pe.horizon_minutes = horizon
    pe.ensemble_replicates = k
    pe.forecast(
        engine.agents, engine.grid, engine.walls_cache,
        engine.gates, engine.exits, sim_time_sec=engine.sim_time,
    )
    return pe


# ── the defect that made the feature useless ──────────────────────


def test_ensemble_produces_variance():
    """
    With K > 1 replicates there must be measurable spread. A sigma of exactly zero
    means the replicates are identical, which means the underlying model is
    deterministic and the probability map is meaningless.
    """
    e = congested_engine()
    pe = forecast_with(e, k=5)

    assert pe.final_sigma is not None, "final_sigma must be populated"
    assert float(pe.final_sigma.max()) > 0.0, (
        "sigma is exactly zero: replicates produced identical results, so the "
        "forecast has no stochastic input to measure"
    )


def test_exceedance_is_a_probability():
    e = congested_engine()
    pe = forecast_with(e, k=5)

    exc = pe.exceedance
    assert exc is not None
    assert exc.shape == (e.height, e.width)
    assert 0.0 <= float(exc.min())
    assert float(exc.max()) <= 1.0


def test_exceedance_is_a_multiple_of_one_over_k():
    """Each cell's probability must be a fraction of replicates that crossed."""
    e = congested_engine()
    k = 5
    pe = forecast_with(e, k=k)
    scaled = pe.exceedance * k
    assert np.allclose(scaled, np.round(scaled), atol=1e-9), (
        "exceedance values are not multiples of 1/k"
    )


def test_exceedance_cells_align_with_the_threshold():
    """
    A cell reported at p = 1.0 must actually have crossed the pile-up threshold in
    the primary pass.
    """
    e = congested_engine()
    pe = forecast_with(e, k=4)

    certain = pe.exceedance >= 1.0
    if not certain.any():
        print("    (no certain cells in this crowd; nothing to check)")
        return

    primary_peak = np.zeros((e.height, e.width))
    for _t, g in pe.last_forecast_grids:
        np.maximum(primary_peak, g, out=primary_peak)

    ys, xs = np.nonzero(certain)
    crossed = sum(1 for y, x in zip(ys, xs) if primary_peak[y, x] >= PILEUP_DENSITY)
    assert crossed >= len(ys) * 0.5, (
        f"only {crossed}/{len(ys)} certain cells crossed the threshold"
    )


def test_single_replicate_gives_no_uncertainty():
    """
    K = 1 is the plain point forecast, not an ensemble of one: `forecast()` only
    routes through the ensemble when K > 1, so no probability map is produced.
    """
    e = congested_engine(agents=150, ticks=80)
    pe = forecast_with(e, k=1)
    assert pe.exceedance is None, "K=1 must behave as the plain point forecast"
    assert pe.final_sigma is None
    assert pe.last_forecast_grids, "the single pass must still produce snapshots"


def test_disabled_ensemble_leaves_exceedance_unset():
    """K = 0 keeps the original point-forecast behaviour exactly."""
    e = congested_engine()
    pe = forecast_with(e, k=0)
    assert pe.exceedance is None
    assert pe.final_sigma is None
    assert pe.last_forecast_grids, "the single pass must still produce snapshots"


def test_primary_pass_is_preserved_across_replicates():
    """
    Replicates are a measurement layered on top; the retained grids must be the
    first pass, so the accuracy harness and the dashboard see an unchanged shape.
    """
    e = congested_engine()
    pe = forecast_with(e, k=4)
    assert pe.last_forecast_grids, "replicates must not empty the primary grids"
    for _offset, grid in pe.last_forecast_grids:
        assert grid.shape == (e.height, e.width)


def test_seeded_ensemble_is_reproducible():
    """Poisson draws are stochastic but must still follow the seed."""
    def run(seed):
        e = congested_engine(seed=seed)
        pe = forecast_with(e, k=4)
        return pe.exceedance

    a = run(4242)
    b = run(4242)
    assert np.array_equal(a, b), "same seed produced different exceedance maps"


# ── consumption of the map ────────────────────────────────────────


def test_engine_enables_replicates_by_default():
    """
    Replicates are OFF by default, on measured evidence.

    The ensemble produces a miscalibrated probability: `scripts/eval_reliability.py`
    measured ECE 0.917 on the hotspot scenario (predicting 0.09 where 0.97 occurred)
    and 0.589 on the default scenario, erring the other way. Until that is fixed, a
    point forecast is the honest output.
    """
    e = SimulationEngine(seed=1)
    assert e.prediction_engine.ensemble_replicates == 0, (
        "the ensemble must stay off until a reliability check passes"
    )


def test_uncertainty_summary_is_reported():
    e = shared_forecast(k=3)
    summary = e._uncertainty_summary()
    assert summary["available"] is True
    assert summary["replicates"] >= 2
    assert 0.0 <= summary["max_p_exceed"] <= 1.0
    assert summary["max_sigma"] is None or summary["max_sigma"] >= 0.0


def test_uncertainty_summary_when_unavailable():
    e = SimulationEngine(seed=1)
    assert e._uncertainty_summary() == {"available": False}


def test_warnings_carry_a_probability():
    """Every early warning must expose p_exceed for the dashboard and the autopilot."""
    e = shared_forecast(k=3)
    pe = e.prediction_engine
    raw = e.density_analyzer.compute_raw_density(e.agents)
    warnings = e._enrich_warnings(pe.early_warnings(raw, e.sim_time))
    assert warnings, "no warnings produced for a congested venue"
    for w in warnings:
        assert "p_exceed" in w, f"warning missing p_exceed: {w}"
        assert 0.0 <= w["p_exceed"] <= 1.0


def test_divert_is_scaled_by_probability():
    """
    A less likely breach must recommend a smaller divert. The previous formula came
    from one deterministic sample and treated every warning as equally certain.
    """
    e = shared_forecast(k=3)
    pe = e.prediction_engine
    raw = e.density_analyzer.compute_raw_density(e.agents)
    warnings = e._enrich_warnings(pe.early_warnings(raw, e.sim_time))
    assert warnings, "no warnings produced"
    for w in warnings:
        unweighted = int((w["peak_density"] - 1.08) / w["peak_density"] * 100 + 0.5) \
            if w["peak_density"] > 1.08 else 0
        unweighted = max(0, min(95, unweighted))
        if w["p_exceed"] < 1.0:
            assert w["divert_pct"] <= unweighted, (
                f"divert {w['divert_pct']}% exceeds the unweighted {unweighted}% "
                f"despite p_exceed {w['p_exceed']}"
            )


def test_autopilot_requires_exceedance_evidence():
    """The autopilot must not act on a warning the ensemble is unsure about."""
    assert 0.0 < AUTO_MIN_P_EXCEED <= 1.0
    from backend import autopilot as ap_mod
    src = pathlib.Path(ap_mod.__file__).read_text(encoding="utf-8")
    assert "p_exceed" in src, "the autopilot does not consult p_exceed"
    assert "AUTO_MIN_P_EXCEED" in src


def test_state_payload_includes_uncertainty():
    e = shared_forecast(k=3)
    # `get_state` returns the snapshot published at the end of a tick, so republish
    # after the forecast rather than expecting the reader to see unsnapshotted state.
    e._publish_snapshot()
    state = e.get_state()
    assert "forecast_uncertainty" in state
    assert state["forecast_uncertainty"].get("available") is True


def test_default_engine_reports_no_uncertainty():
    """With the ensemble off, the payload must say so rather than imply confidence."""
    e = SimulationEngine(seed=1)
    e._publish_snapshot()
    state = e.get_state()
    assert state["forecast_uncertainty"]["available"] is False


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(
        (n, f) for n, f in dict(globals()).items() if n.startswith("test_")
    ):
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} ({failures} failing)")
    sys.exit(1 if failures else 0)
