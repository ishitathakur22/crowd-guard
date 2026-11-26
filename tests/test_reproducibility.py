"""
Reproducibility guarantees for the simulation.

These tests are the licence for every measurement: if two runs of the same seed
are not identical, a benchmark number cannot be compared across a change and a
forecast-accuracy score means nothing, because a "regression" might just be a
different random draw.

Exact equality is asserted deliberately, not a tolerance. Every draw comes from a
seeded generator and moves through deterministic float64 arithmetic, so states
must agree bit for bit. A tolerance here would hide exactly the kind of
nondeterminism this file exists to catch — an unseeded global sneaking back in, or
some code path depending on dict or set iteration order.

Run with pytest, or directly: `python tests/test_reproducibility.py`
"""

import hashlib
import os
import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.rng import Rng, seed_from_env  # noqa: E402
from backend.simulation import SimulationEngine  # noqa: E402


def fingerprint(engine) -> str:
    """A hash of every live agent's state, in list order."""
    h = hashlib.sha256()
    for a in engine.agents:
        h.update(
            f"{a.id}|{a.pos.x!r}|{a.pos.y!r}|{a.vel.x!r}|{a.vel.y!r}"
            f"|{a.goal.x!r}|{a.goal.y!r}|{a.status}".encode("utf-8")
        )
    return h.hexdigest()


def run(seed: int, ticks: int = 60, prefill: int = 80) -> SimulationEngine:
    """
    Build a seeded engine and advance it a fixed number of ticks.

    The internal forecast is switched off: it only reads state and cannot affect
    the crowd, but it costs real time and this test is about determinism, not
    throughput. The arrival rate is held down for the same reason — the shipped
    defaults draw several people a second, which turns a fast unit test into a
    slow one without testing anything extra.
    """
    engine = SimulationEngine(seed=seed)
    engine.prediction_interval = 10**9
    engine.arrival_model.attendance = 8_000
    engine.arrival_model.scale = 1.0
    engine.spawn_agents(prefill)
    for _ in range(ticks):
        engine.update()
    return engine


# ── tests ─────────────────────────────────────────────────────────


def test_same_seed_is_bit_identical():
    a = run(4242)
    b = run(4242)
    assert a.seed == 4242
    assert len(a.agents) == len(b.agents), "same seed produced a different population"
    assert fingerprint(a) == fingerprint(b)


def test_different_seed_diverges():
    a = run(1)
    b = run(2)
    assert fingerprint(a) != fingerprint(b), "different seeds produced identical runs"


def test_seed_survives_round_trip_through_env(monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("CROWD_SEED", "99")
        assert seed_from_env() == 99
        monkeypatch.setenv("CROWD_SEED", "not-a-number")
        assert seed_from_env() == 1337
        monkeypatch.delenv("CROWD_SEED")
        assert seed_from_env() == 1337
    else:
        old = os.environ.get("CROWD_SEED")
        try:
            os.environ["CROWD_SEED"] = "99"
            assert seed_from_env() == 99
            os.environ["CROWD_SEED"] = "  "
            assert seed_from_env() == 1337
            del os.environ["CROWD_SEED"]
            assert seed_from_env() == 1337
        finally:
            if old is None:
                os.environ.pop("CROWD_SEED", None)
            else:
                os.environ["CROWD_SEED"] = old


def test_substreams_are_stable_and_independent():
    parent = Rng(7)
    assert parent.derive("sfm").seed == Rng(7).derive("sfm").seed
    assert parent.derive("sfm").seed != parent.derive("predict").seed
    assert parent.derive("ingress").seed != Rng(8).derive("ingress").seed


def test_substream_is_unchanged_when_another_subsystem_draws_more():
    """
    The point of deriving substreams rather than sharing one generator: extra draws
    in physics must not shift the arrival sequence, or every benchmark would change
    after an unrelated edit.
    """
    a = Rng(11)
    b = Rng(11)
    ingest_a = a.derive("ingress")
    ingest_b = b.derive("ingress")
    for _ in range(50):
        b.derive("sfm").uniform(0.0, 1.0)
    assert [ingest_a.uniform(0, 1) for _ in range(5)] == [
        ingest_b.uniform(0, 1) for _ in range(5)
    ]


if __name__ == "__main__":
    failures = 0
    runner = None
    for name, fn in sorted(
        (n, f) for n, f in dict(globals()).items() if n.startswith("test_")
    ):
        try:
            fn(runner)
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'OK' if failures == 0 else 'FAILED'} ({failures} failing)")
    sys.exit(1 if failures else 0)
