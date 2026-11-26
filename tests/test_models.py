"""
Model-level invariants.

`Agent.vel` once defaulted to a plain `Point(...)` instance, which pydantic shares
across every agent rather than copying. Mutating one agent's velocity therefore
mutated every agent's, and the effect was silent: positions and velocities looked
plausible while the crowd was quietly wrong. It surfaced only when the physics was
vectorised and began writing velocity in place instead of rebinding the attribute.

These tests assert that no default is a shared mutable object, because that class
of bug is invisible at the call site and expensive to diagnose.

Run with pytest, or directly: `python tests/test_models.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.models import Agent, GateState, Point  # noqa: E402


def _agent(i=0):
    return Agent(id=i, pos=Point(x=1.0, y=2.0), goal=Point(x=5.0, y=6.0))


def test_velocity_default_is_not_shared_between_agents():
    a, b = _agent(0), _agent(1)
    assert a.vel is not b.vel, "agents share one Point object for vel"

    a.vel.x = 99.0
    assert b.vel.x == 0.0, f"mutating one agent's velocity changed another's to {b.vel.x}"


def test_position_is_never_shared():
    a, b = _agent(0), _agent(1)
    assert a.pos is not b.pos
    a.pos.x = 7.0
    assert b.pos.x == 1.0


def test_path_list_is_not_shared():
    a, b = _agent(0), _agent(1)
    assert a.path is not b.path
    a.path.append(Point(x=3.0, y=3.0))
    assert b.path == [], "agents share one list object for path"


def test_gate_state_defaults_are_not_shared():
    g1 = GateState(gate_id="a", position=Point(x=0.0, y=2.0))
    g2 = GateState(gate_id="b", position=Point(x=0.0, y=8.0))
    g1.position.y = 5.0
    assert g2.position.y == 8.0


def test_many_agents_have_independent_velocity():
    """The failure mode at scale: spawning a crowd must not alias one Point widely."""
    agents = [_agent(i) for i in range(50)]
    for i, a in enumerate(agents):
        a.vel.x = float(i)
    assert [a.vel.x for a in agents] == [float(i) for i in range(50)]


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
