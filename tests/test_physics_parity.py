"""
Parity tests for the Social Force Model.

A vectorised rewrite was once committed and then reverted because it broke
bit-exact reproducibility: two identically-seeded engines diverged, with identical
`calculate_forces` output but different positions after integration. That is not a
numerical-tolerance problem, it is a hidden dependency on something outside the
inputs, and it is invisible unless tested for. (The root cause turned out to be a
shared mutable default on `Agent.vel` — see `tests/test_models.py`.)

These tests pin down the three properties that matter:

1. `calculate_forces` is a pure function of state — calling it twice gives exactly
   the same array.
2. Two engines built with the same seed stay bit-identical through integration.
3. The fast implementation agrees with the scalar reference on a fixed crowd.

Run with pytest, or directly: `python tests/test_physics_parity.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from backend.models import Agent, Point  # noqa: E402
from backend.rng import Rng  # noqa: E402
from backend.simulation import SimulationEngine  # noqa: E402
from backend.physics import SocialForceModel  # noqa: E402


def fixed_crowd(n=40, seed=99):
    """A fixed crowd with paths, goals and properties pinned exactly."""
    rng = Rng(seed)
    walls = [(15, y) for y in range(6, 13)]
    agents = []
    for i in range(n):
        x = 2.0 + rng.uniform(0.0, 8.0)
        y = 2.0 + rng.uniform(0.0, 26.0)
        agents.append(
            Agent(
                id=i,
                pos=Point(x=x, y=y),
                vel=Point(x=rng.uniform(-0.5, 0.5), y=rng.uniform(-0.5, 0.5)),
                goal=Point(x=39.0, y=5.0 + (i % 4) * 7.0),
                path=[Point(x=x + 1.0, y=y), Point(x=20.0, y=y)],
                desired_speed=1.1 + (i % 3) * 0.1,
                radius=0.3,
                mass=80.0,
            )
        )
    return agents, walls


def state_of(agents):
    return np.array([[a.pos.x, a.pos.y, a.vel.x, a.vel.y] for a in agents])


def test_forces_are_idempotent():
    """The same state must produce the same forces every time it is evaluated."""
    agents, walls = fixed_crowd()
    model = SocialForceModel(rng=Rng(1))

    f1 = model.calculate_forces(agents, walls)
    f2 = model.calculate_forces(agents, walls)
    f3 = model.calculate_forces(agents, walls)

    assert np.array_equal(f1, f2), (
        f"calculate_forces is not a pure function of state; "
        f"max delta {np.abs(f1 - f2).max()}"
    )
    assert np.array_equal(f2, f3)


def test_forces_identical_across_two_models():
    agents_a, walls = fixed_crowd()
    agents_b, _ = fixed_crowd()
    fa = SocialForceModel(rng=Rng(1)).calculate_forces(agents_a, walls)
    fb = SocialForceModel(rng=Rng(1)).calculate_forces(agents_b, walls)
    assert np.array_equal(fa, fb), f"max delta {np.abs(fa - fb).max()}"


def test_update_positions_is_idempotent_from_same_state():
    """Integrating identical states through identical models must agree exactly."""
    agents_a, walls = fixed_crowd()
    agents_b, _ = fixed_crowd()
    np.testing.assert_array_equal(state_of(agents_a), state_of(agents_b))

    SocialForceModel(rng=Rng(2)).update_positions(agents_a, walls)
    SocialForceModel(rng=Rng(2)).update_positions(agents_b, walls)

    after_a, after_b = state_of(agents_a), state_of(agents_b)
    delta = np.abs(after_a - after_b)
    assert np.array_equal(after_a, after_b), (
        f"update_positions diverged from identical state; max delta {delta.max()} "
        f"at agent {int(np.argmax(delta.max(axis=1)))}"
    )


def test_same_seed_engines_stay_identical():
    """End-to-end: the property the whole measurement suite depends on."""
    def run(seed):
        e = SimulationEngine(seed=seed)
        e.prediction_interval = 10**9
        e.arrival_model.attendance = 8000
        e.arrival_model.scale = 1.0
        e.spawn_agents(60)
        for _ in range(25):
            e.update()
        return state_of(e.agents), len(e.agents)

    sa, na = run(4242)
    sb, nb = run(4242)
    assert na == nb
    assert np.array_equal(sa, sb), f"same-seed engines diverged; max delta {np.abs(sa - sb).max()}"


def test_reference_implementation_parity():
    """
    Agreement with the scalar reference.

    The fast model drops interactions beyond 3 m, where repulsion is ~3e-8 N
    against desire forces of order 200 N, and it sums with np.add.at rather than
    sequentially — both move the last bits of a float sum. Parity is therefore
    asserted relative to the largest force, not exactly.
    """
    from backend.physics_scalar import SocialForceModel as ReferenceModel

    agents_a, walls = fixed_crowd()
    agents_b, _ = fixed_crowd()

    fast = SocialForceModel(rng=Rng(3)).calculate_forces(agents_a, walls)
    slow = ReferenceModel(rng=Rng(3)).calculate_forces(agents_b, walls)

    scale = max(float(np.abs(slow).max()), 1e-12)
    rel = float(np.abs(fast - slow).max()) / scale
    assert rel < 1e-5, (
        f"vectorised forces differ from the reference by {rel:.3e} relative; "
        f"more than truncation and summation order can explain"
    )


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
