"""
Gate admission, queue accounting and diverts.

The gate layer is where the simulation's people budget is decided: arrivals queue
outside, each gate admits up to its target rate per tick, and an operator can divert
a share of one gate's arrivals to another. The invariant that matters is that
nobody is invented or lost - everyone who arrives is either admitted, still queued,
or moved by a divert. A leak here would quietly corrupt both the crowd and the
forecast's ingress estimate.

Run with pytest, or directly: `python tests/test_gates.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    _ROOT_PARENT = str(pathlib.Path(_ROOT).parent)
    sys.path.insert(0, _ROOT)

from backend.models import GateAction  # noqa: E402
from backend.simulation import TICK_DURATION_SEC, SimulationEngine  # noqa: E402


def engine(seed=11):
    e = SimulationEngine(seed=seed)
    e.prediction_interval = 10**9
    e.auto_ingress = False        # drive arrivals explicitly in these tests
    e.arrival_model.attendance = 0
    return e


def total_queued(e):
    return sum(g.queue_length for g in e.gate_objects)


# ── basic admission ───────────────────────────────────────────────


def test_queues_start_empty():
    e = engine()
    assert len(e.gate_objects) > 0
    assert total_queued(e) == 0
    assert e.agents == []


def test_arrivals_enter_queues():
    e = engine()
    e._admit_arrivals(20)
    # Some may already have been admitted within the same call; every arrival is
    # either in a queue or in the venue.
    assert total_queued(e) + len(e.agents) == 20


def test_nobody_is_lost_across_many_ticks():
    """The core accounting invariant: arrivals = queued + admitted."""
    e = engine()
    arrivals = 0
    for _ in range(40):
        e._admit_arrivals(5)
        arrivals += 5
    assert total_queued(e) + len(e.agents) == arrivals, (
        f"{total_queued(e)} queued + {len(e.agents)} admitted != {arrivals} arrivals"
    )


def test_admission_respects_the_target_rate():
    """
    One gate open at 3.5 people/s over a 0.5 s tick admits at most about 1.75
    people, so a large queue must drain slowly rather than all at once.
    """
    e = engine()
    gate = e.gate_objects[0]
    for g in e.gate_objects[1:]:
        e.actuate_gate(g.gate_id, GateAction.CLOSE.value, 0.0)
    e.actuate_gate(gate.gate_id, GateAction.OPEN_FULL.value, 3.5)

    gate.queue_length = 100
    before = len(e.agents)
    e._admit_arrivals(0)

    admitted = len(e.agents) - before
    budget = gate.target_rate_per_sec * TICK_DURATION_SEC + 1.0   # + starting credit
    assert 0 < admitted <= budget + 1, (
        f"admitted {admitted} in one tick against a budget of {budget:.2f}"
    )
    assert gate.queue_length > 0, "a 100-person queue should not clear in one tick"


def test_closed_gate_admits_nobody():
    e = engine()
    for g in e.gate_objects:
        e.actuate_gate(g.gate_id, GateAction.CLOSE.value, 0.0)

    e._admit_arrivals(30)
    assert len(e.agents) == 0, "a closed gate must not admit anyone"
    # With every gate shut the venue is simply not accepting anyone; they wait
    # outside rather than being queued behind one particular door.
    assert total_queued(e) == 30


def test_manual_close_marks_the_gate_as_manual():
    e = engine()
    gate = e.gate_objects[0]
    e.actuate_gate(gate.gate_id, GateAction.CLOSE.value, 0.0)
    assert str(gate.status).startswith("MANUAL_")
    assert gate.action == GateAction.CLOSE


def test_reopening_returns_control_to_auto():
    e = engine()
    gate = e.gate_objects[0]
    e.actuate_gate(gate.gate_id, GateAction.CLOSE.value, 0.0)
    e.actuate_gate(gate.gate_id, GateAction.OPEN_FULL.value, 3.5)
    assert gate.status == "OPEN"


def test_wait_time_is_queue_over_rate():
    e = engine()
    gate = e.gate_objects[0]
    e.actuate_gate(gate.gate_id, GateAction.OPEN_FULL.value, 2.0)
    for g in e.gate_objects[1:]:
        e.actuate_gate(g.gate_id, GateAction.CLOSE.value, 0.0)

    gate.queue_length = 10
    e._admit_arrivals(0)
    expected = gate.queue_length / gate.target_rate_per_sec
    assert abs(gate.wait_time_sec - round(expected, 1)) < 0.2


def test_empty_queue_reports_no_wait():
    e = engine()
    e._admit_arrivals(0)
    for g in e.gate_objects:
        assert g.wait_time_sec == 0.0


# ── diverts ───────────────────────────────────────────────────────


def test_divert_requires_distinct_known_gates():
    e = engine()
    a, b = e.gate_objects[0].gate_id, e.gate_objects[1].gate_id
    assert e.divert_gate(a, b, 50) is True
    assert e.divert_gate(a, a, 50) is False, "diverting a gate to itself is a no-op"
    assert e.divert_gate(a, "gate_nonexistent", 50) is False
    assert e.divert_gate("gate_nonexistent", b, 50) is False


def test_divert_clamps_the_percentage():
    e = engine()
    a, b = e.gate_objects[0].gate_id, e.gate_objects[1].gate_id
    assert e.divert_gate(a, b, 500) is True
    assert 0 < e._diverts[a]["pct"] <= 100
    assert e.divert_gate(a, b, -20) is True
    assert e._diverts[a]["pct"] >= 1


def test_divert_records_an_active_entry():
    e = engine()
    a, b = e.gate_objects[0].gate_id, e.gate_objects[1].gate_id
    e.divert_gate(a, b, 40)
    assert a in e._diverts
    assert e._diverts[a]["target"] == b
    assert e._diverts[a]["until"] > e.sim_time


def test_no_arrivals_queue_at_a_closed_gate():
    """
    Arrivals must never be queued behind a door that never opens.

    This is the defect the earlier version of this file was written around: arrivals
    were assigned to a gate uniformly at random regardless of its status, queued
    there, and - because a closed gate admits at rate 0 - never moved again. Those
    people were silently removed from the venue and inflated both queue_length and
    wait_time_sec without bound.
    """
    e = engine()
    closed = e.gate_objects[1].gate_id
    for g in e.gate_objects:
        if g.gate_id != closed:
            e.actuate_gate(g.gate_id, GateAction.CLOSE.value, 0.0)
    e.actuate_gate(closed, GateAction.OPEN_FULL.value, 0.0)   # open, zero rate

    e._admit_arrivals(100)

    for g in e.gate_objects:
        if g.action == GateAction.CLOSE:
            assert g.queue_length == 0, (
                f"{g.gate_id} is closed but holds {g.queue_length} people"
            )
    assert total_queued(e) == 100, "every arrival should be held at the open gate"


def test_queue_at_a_gate_closed_mid_run_moves_to_an_open_gate():
    """People already queued when a gate shuts must walk to one that is open."""
    e = engine()
    a, b = e.gate_objects[0].gate_id, e.gate_objects[1].gate_id
    e.actuate_gate(a, GateAction.OPEN_FULL.value, 0.0)    # open, zero rate
    for g in e.gate_objects[1:]:
        e.actuate_gate(g.gate_id, GateAction.CLOSE.value, 0.0)

    e._admit_arrivals(50)
    gate_a = next(g for g in e.gate_objects if g.gate_id == a)
    assert gate_a.queue_length == 50

    # Shut A and open B. The 50 waiting outside A belong at B now.
    e.actuate_gate(a, GateAction.CLOSE.value, 0.0)
    e.actuate_gate(b, GateAction.OPEN_FULL.value, 0.0)
    e._admit_arrivals(0)

    gate_b = next(g for g in e.gate_objects if g.gate_id == b)
    assert gate_a.queue_length == 0, "a closed gate must not keep hold of its queue"
    assert gate_b.queue_length == 50, (
        f"the 50 displaced people ended up as {gate_b.queue_length}"
    )


def test_full_divert_redirects_the_diverting_gates_share():
    """
    With two gates open and a 100% divert from A to B, B should hold the arrivals
    that A would otherwise have taken, so B's queue grows relative to no divert.
    """
    a, b = "gate_0", "gate_1"

    def run(divert):
        e = engine()
        for g in e.gate_objects:
            e.actuate_gate(g.gate_id, GateAction.CLOSE.value, 0.0)
        # Both gates open at zero rate, so nobody is admitted and queues are exact.
        e.actuate_gate(a, GateAction.OPEN_FULL.value, 0.0)
        e.actuate_gate(b, GateAction.OPEN_FULL.value, 0.0)
        if divert:
            e.divert_gate(a, b, 100)
        e._admit_arrivals(400)
        gate_a = next(g for g in e.gate_objects if g.gate_id == a)
        gate_b = next(g for g in e.gate_objects if g.gate_id == b)
        return gate_a.queue_length, gate_b.queue_length, total_queued(e)

    a_without, b_without, total_without = run(divert=False)
    a_with, b_with, total_with = run(divert=True)

    assert total_without == 400 and total_with == 400, "arrivals must be conserved"
    assert b_with > b_without, (
        f"diverting A->B left B with {b_with}, no better than {b_without}"
    )
    assert a_with < a_without, (
        f"the diverting gate still took {a_with} against {a_without} without a divert"
    )


def test_expired_divert_is_dropped():
    e = engine()
    a, b = e.gate_objects[0].gate_id, e.gate_objects[1].gate_id
    e.divert_gate(a, b, 50)
    e.sim_time += 10_000.0      # well past DIVERT_DURATION_SEC
    e._admit_arrivals(0)
    assert a not in e._diverts, "an expired divert must not linger"


def test_divert_logs_to_the_activity_log():
    e = engine()
    a, b = e.gate_objects[0].gate_id, e.gate_objects[1].gate_id
    e.divert_gate(a, b, 30)
    entries = e.autopilot.recent(20)
    assert any(x.get("kind") == "divert" for x in entries), (
        "a divert must leave an audit entry"
    )


# ── state payload ─────────────────────────────────────────────────


def test_state_reports_gates_and_diverts():
    e = engine()
    a, b = e.gate_objects[0].gate_id, e.gate_objects[1].gate_id
    e._admit_arrivals(10)
    e.divert_gate(a, b, 25)

    state = e.get_state()
    assert "gates" in state
    assert len(state["gates"]) == len(e.gate_objects)
    assert len(state["active_diverts"]) == 1
    assert state["active_diverts"][0]["origin"] == a


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
