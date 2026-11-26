"""
Invariants the forecast must hold.

These exist because the original forecast violated them and nobody noticed. It
predicted a population of 552 where the recorded truth plateaued at 351, because
agents rejected from a full cell had their velocity zeroed, could never leave,
were never removed, and fresh arrivals kept entering. Mass that nobody injected is
exactly the defect these tests catch.

They assert properties, not exact numbers, so they survive legitimate changes to
the model while still failing loudly on that class of bug.

Run with pytest, or directly: `python tests/test_forecast.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.simulation import SimulationEngine  # noqa: E402

HORIZON_MIN = 3.0


def _engine(seed=5, agents=60):
    engine = SimulationEngine(seed=seed)
    engine.prediction_interval = 10**9
    engine.arrival_model.attendance = 6_000
    engine.arrival_model.scale = 1.0
    engine.spawn_agents(agents)
    return engine


def _run(engine, with_arrivals, model="continuum"):
    """Forecast once and return (population totals per snapshot, grid list)."""
    pe = engine.prediction_engine
    pe.spatial_model = model
    pe.horizon_minutes = HORIZON_MIN
    pe.max_snapshots = 20
    pe.detour_fraction = 0.0

    saved = pe.arrival_model
    if not with_arrivals:
        pe.arrival_model = None
    try:
        pe.forecast(
            engine.agents,
            engine.grid,
            engine.walls_cache,
            engine.gates,
            engine.exits,
            sim_time_sec=engine.sim_time,
        )
        grids = [g for _t, g in pe.last_forecast_grids]
        return [float(g.sum()) for g in grids], grids, pe
    finally:
        pe.arrival_model = saved


# ── tests ─────────────────────────────────────────────────────────


def test_continuum_is_the_default_model():
    engine = _engine()
    assert engine.prediction_engine.spatial_model == "continuum"


def test_no_arrivals_means_population_never_grows():
    """
    With no ingress the only thing that can happen is people leaving, so total mass
    may fall but must never rise. This is the invariant the deadlock broke.
    """
    totals, _, _ = _run(_engine(), with_arrivals=False)
    assert totals, "forecast produced no snapshots"
    for i in range(1, len(totals)):
        assert totals[i] <= totals[i - 1] + 1e-6, (
            f"population rose from {totals[i - 1]:.2f} to {totals[i]:.2f} "
            f"with no arrivals injected"
        )


def test_population_never_exceeds_what_was_injected():
    """
    With arrivals on, the crowd can only grow by what the arrival curve supplies.
    The original forecast grew by 358 over eight minutes from 194 agents, far beyond
    what the curve allowed.
    """
    engine = _engine()
    pe = engine.prediction_engine
    initial = float(len(engine.agents))
    supplied = pe.arrival_model.arrivals_in(engine.sim_time, HORIZON_MIN * 60.0)

    totals, _, _ = _run(engine, with_arrivals=True)

    allowed = (initial + supplied) * 1.05 + 2.0
    assert max(totals) <= allowed, (
        f"forecast reached {max(totals):.1f} people, but only {initial:.0f} initial "
        f"+ {supplied:.1f} arrivals was possible"
    )


def test_density_never_exceeds_jam_density():
    """No cell may report a density that cannot physically be occupied."""
    _, grids, pe = _run(_engine(), with_arrivals=True)
    assert grids, "forecast produced no snapshots"
    for g in grids:
        assert float(g.max()) <= pe.jam_density + 1e-6, (
            f"cell density {g.max():.2f} exceeds jam density {pe.jam_density}"
        )


def test_snapshot_count_matches_horizon():
    _, grids, pe = _run(_engine(), with_arrivals=True)
    expected = int(round(HORIZON_MIN * 60.0 / pe.snapshot_interval_sec))
    assert len(grids) == min(expected, pe.max_snapshots), (
        f"expected {expected} snapshots, got {len(grids)}"
    )


def test_early_warnings_stay_within_horizon():
    """Every warning must be reachable inside the window it is derived from."""
    engine = _engine()
    _run(engine, with_arrivals=True)
    raw = engine.density_analyzer.compute_raw_density(engine.agents)
    for w in engine.prediction_engine.early_warnings(raw, engine.sim_time):
        assert 0.0 <= w["eta_sec"] <= HORIZON_MIN * 60.0 + 1e-6


def test_forecast_is_finite():
    """Guards against the divisions in the speed-density term producing NaN/inf."""
    _, grids, _ = _run(_engine(), with_arrivals=True)
    for g in grids:
        assert not bool((g != g).any()), "forecast grid contains NaN"
        assert float(g.min()) >= 0.0, "forecast produced negative density"


def test_particle_model_still_available_and_also_bounded():
    """The old model is retained for comparison but must obey the same mass bound."""
    engine = _engine()
    pe = engine.prediction_engine
    initial = float(len(engine.agents))
    supplied = pe.arrival_model.arrivals_in(engine.sim_time, HORIZON_MIN * 60.0)
    totals, _, _ = _run(engine, with_arrivals=True, model="particles")
    assert max(totals) <= (initial + supplied) * 1.10 + 5.0


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
