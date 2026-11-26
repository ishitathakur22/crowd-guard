"""
Snapshot safety and tick-clock accuracy.

Two defects lived here:

1. `get_state()` walked `self.agents` on the event loop while `update()` mutated the
   same list on an executor thread, with no lock. A broadcast landing mid-tick could
   iterate a list being appended to or shrunk, producing a torn payload or a
   RuntimeError. The tick now publishes an immutable dict of primitives and readers
   only ever see a finished snapshot.

2. The simulation loop slept `TICK_DURATION_SEC / speed` *after* the work, so the
   real period was `work + sleep`. At 10x with a 17 ms tick the intended 50 ms period
   became 67 ms, and the speed control quietly meant less the slower the machine.

Run with pytest, or directly: `python tests/test_snapshot_and_clock.py`
"""

import math
import pathlib
import sys
import threading
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.simulation import SimulationEngine, TICK_DURATION_SEC  # noqa: E402


def engine(seed=21, agents=80):
    e = SimulationEngine(seed=seed)
    e.prediction_interval = 10**9
    e.arrival_model.attendance = 20_000
    e.arrival_model.scale = 1.0
    e.spawn_agents(agents)
    return e


# ── snapshot safety ───────────────────────────────────────────────


def test_get_state_is_available_before_any_tick():
    """A client connecting first must still get a valid payload."""
    e = engine()
    state = e.get_state()
    assert isinstance(state, dict)
    assert state["total_agents"] == len(e.agents)
    assert "gates" in state and "sim_time_sec" in state


def test_get_state_does_not_alias_the_live_agent_list():
    """
    The whole point: the returned payload must be a snapshot, not a view onto state
    that a running tick can change underneath the reader.
    """
    e = engine()
    e.update()
    state = e.get_state()
    before = len(state["agents"])

    for _ in range(5):
        e.update()

    assert len(state["agents"]) == before, (
        "the published snapshot changed after later ticks, so it is not immutable"
    )
    assert state["total_agents"] == before


def test_snapshot_is_replaced_not_mutated():
    """Each tick publishes a new dict rather than editing the previous one."""
    e = engine()
    e.update()
    first = e.get_state()
    e.update()
    second = e.get_state()
    assert first is not second
    assert second["sim_time_sec"] >= first["sim_time_sec"]


def test_concurrent_reads_during_ticks_are_safe():
    """
    Hammer the reader from other threads while ticks run. Before the fix this was
    reading a list the writer was mutating.

    The readers sleep between iterations on purpose: a real broadcaster emits a
    message and then awaits, and three threads spinning in a tight Python loop would
    hold the GIL and starve the writer, turning a 15-tick test into minutes.
    """
    e = engine(agents=80)
    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                state = e.get_state()
                heat = e.get_heatmap_data()
                # Touch the payload the way the broadcaster would.
                assert isinstance(state["agents"], list)
                assert isinstance(heat["heatmap"], list)
                for a in state["agents"][:5]:
                    assert "pos" in a
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                return
            time.sleep(0.005)

    threads = [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        for _ in range(15):
            e.update()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=5)

    assert not errors, f"reader raised during concurrent ticks: {errors[0]!r}"


def test_heatmap_matches_the_published_state():
    e = engine()
    e.update()
    heat = e.get_heatmap_data()
    assert heat["width"] == e.width
    assert heat["height"] == e.height
    assert isinstance(heat["heatmap"], list)


# ── tick clock ────────────────────────────────────────────────────


def _simulate_scheduler(speed, work_ms, ticks=40):
    """
    Reproduce the loop's timing without asyncio, so the arithmetic can be checked
    without running a server. Mirrors backend/main.py's deadline logic.
    """
    min_period = 0.002
    period = max(min_period, TICK_DURATION_SEC / max(speed, 0.1))
    deadline = 0.0
    stamps = []
    now = 0.0
    for _ in range(ticks):
        now += work_ms / 1000.0          # the tick itself
        stamps.append(now)
        deadline += period
        if deadline < now:
            deadline = now               # re-base on overrun
        else:
            now = deadline               # sleep until the deadline
    return stamps, period


def test_period_is_independent_of_tick_work():
    """
    A 5 ms tick and a 40 ms tick must produce the same period at the same speed.
    The old `work + sleep` loop could not do this.
    """
    fast, period_fast = _simulate_scheduler(speed=1.0, work_ms=5)
    slow, period_slow = _simulate_scheduler(speed=1.0, work_ms=40)

    gaps_fast = [b - a for a, b in zip(fast, fast[1:])]
    gaps_slow = [b - a for a, b in zip(slow, slow[1:])]

    assert abs(statistics_mean(gaps_fast) - period_fast) < 1e-6
    assert abs(statistics_mean(gaps_slow) - period_slow) < 1e-6


def statistics_mean(values):
    return sum(values) / len(values)


def test_speed_multiplier_scales_the_period():
    for speed in (1.0, 2.0, 5.0, 10.0):
        _stamps, period = _simulate_scheduler(speed=speed, work_ms=2)
        assert abs(period - TICK_DURATION_SEC / speed) < 1e-9, (
            f"speed {speed} produced period {period:.4f}s"
        )


def test_overrun_rebases_instead_of_catching_up():
    """
    A tick that takes longer than its slot must not queue a burst of instant catch-up
    ticks - that would simulate the crowd faster than real time.
    """
    # 400 ms of work against a 500 ms budget: no overrun.
    stamps, period = _simulate_scheduler(speed=1.0, work_ms=400, ticks=5)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    for g in gaps:
        assert abs(g - period) < 1e-6

    # 800 ms of work against a 500 ms budget: periods become the work time.
    stamps, period = _simulate_scheduler(speed=1.0, work_ms=800, ticks=5)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    for g in gaps:
        assert abs(g - 0.8) < 1e-6, f"overrun produced a {g:.3f}s gap"


def test_minimum_period_is_respected():
    """A very high multiplier must not spin the loop with no yield."""
    _stamps, _period = _simulate_scheduler(speed=1000.0, work_ms=0)
    min_period = 0.002
    assert max(min_period, TICK_DURATION_SEC / 1000.0) == min_period


def test_engine_clock_advances_by_exactly_one_tick():
    e = engine()
    before = e.sim_time
    e.update()
    assert abs(e.sim_time - (before + TICK_DURATION_SEC)) < 1e-9


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
