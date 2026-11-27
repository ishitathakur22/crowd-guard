"""
Arrival model: curve normalisation, rate integration and arrival conservation.

The arrival model decides how many people enter the venue and, just as
importantly, how many the forecaster believes are still coming. Two properties
matter and are tested here:

  * the 24-hour curve is a probability distribution - it must sum to 1, so that
    `attendance` means what it says;
  * `draw_arrivals` must not lose or invent people over time. It carries a
    fractional remainder between ticks, and any error there compounds silently
    over thousands of ticks.

Known limitation, deliberately not asserted away: `draw_arrivals` is a
deterministic accumulator with zero variance, not a Poisson process. Real
pedestrian arrivals are approximately Poisson, so arrival surges - the thing that
actually forms bottlenecks - cannot happen in the model. Fixing that is tracked
separately; these tests pin the behaviour that exists.

Run with pytest, or directly: `python tests/test_ingress.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.ingress import ArrivalModel, _FALLBACK_CURVE  # noqa: E402
from backend.rng import Rng  # noqa: E402

DAY = 24 * 3600


def model(**kw):
    kw.setdefault("rng", Rng(1))
    return ArrivalModel(**kw)


# ── curve loading ─────────────────────────────────────────────────


def test_curve_is_a_normalised_distribution():
    m = model()
    assert len(m.curve) == 24
    assert abs(sum(m.curve) - 1.0) < 1e-4, f"curve sums to {sum(m.curve)}"
    assert all(c >= 0.0 for c in m.curve)


def test_unknown_archetype_falls_back_to_a_flat_curve():
    m = model(archetype="no-such-archetype")
    assert m.curve == _FALLBACK_CURVE
    assert "fallback" in m.meta.get("source", "")


def test_missing_profile_file_falls_back(tmp_path):
    m = ArrivalModel(profile_path=str(tmp_path / "absent.json"), rng=Rng(1))
    assert m.curve == _FALLBACK_CURVE


# ── clock ─────────────────────────────────────────────────────────


def test_hour_at_starts_at_start_hour_and_wraps():
    m = model(start_hour=17.0)
    assert abs(m.hour_at(0.0) - 17.0) < 1e-9
    assert abs(m.hour_at(3600.0) - 18.0) < 1e-9
    # 10 hours on from 17:00 is 03:00 the next day, not 27:00.
    assert abs(m.hour_at(10 * 3600.0) - 3.0) < 1e-9


# ── rate ──────────────────────────────────────────────────────────


def test_rate_is_never_negative():
    m = model()
    for t in range(0, DAY, 900):
        assert m.rate_at(float(t)) >= 0.0


def test_rate_integrates_to_the_attendance():
    """
    Integrating the arrival rate across a full simulated day must recover
    `attendance * scale`, because the curve sums to 1.
    """
    attendance, scale = 30_000, 1.0
    m = model(attendance=attendance, scale=scale, start_hour=0.0)

    step = 60.0
    total = sum(m.rate_at(t * step) * step for t in range(int(DAY / step)))

    expected = attendance * scale
    assert abs(total - expected) / expected < 0.01, (
        f"integrated {total:.0f} arrivals over a day, expected {expected:.0f}"
    )


def test_scale_multiplies_the_total():
    base = model(attendance=10_000, scale=1.0, start_hour=0.0)
    doubled = model(attendance=10_000, scale=2.0, start_hour=0.0)
    for t in (0.0, 6 * 3600.0, 12 * 3600.0):
        assert abs(doubled.rate_at(t) - 2.0 * base.rate_at(t)) < 1e-9


def test_arrivals_in_is_rate_times_duration():
    m = model()
    for t in (0.0, 3600.0, 43200.0):
        for d in (0.5, 60.0):
            assert abs(m.arrivals_in(t, d) - m.rate_at(t) * d) < 1e-9


def test_arrivals_in_grows_with_duration():
    m = model()
    assert m.arrivals_in(0.0, 120.0) > m.arrivals_in(0.0, 60.0)


# ── whole-person draws ────────────────────────────────────────────


def test_draw_arrivals_returns_whole_people():
    m = model()
    for t in range(0, 3600, 30):
        got = m.draw_arrivals(float(t), 0.5)
        assert isinstance(got, int)
        assert got >= 0


def test_draw_arrivals_is_poisson_not_deterministic():
    """
    The same configuration must produce *different* sequences on successive draws.

    This is the whole point of the change. The previous implementation carried a
    fractional remainder and emitted floor(carry), so it produced an identical
    sequence every time and had exactly zero variance - which makes arrival surges
    structurally impossible, and surges are what create bottlenecks.
    """
    m = model(attendance=100_000, scale=1.0, start_hour=17.0)
    # At 17:00 this is about two people per tick, so the rate is unambiguous.
    assert m.arrivals_in(0.0, 0.5) > 1.5, (
        f"expected ~2/tick at this attendance, got {m.arrivals_in(0.0, 0.5):.3f}"
    )

    draws = [m.draw_arrivals(0.0, 0.5) for _ in range(200)]
    assert len(set(draws)) > 1, (
        f"200 draws at a >1/tick rate all returned {draws[0]}; that is not Poisson"
    )
    mean = sum(draws) / len(draws)
    var = sum((d - mean) ** 2 for d in draws) / (len(draws) - 1)
    assert var > 0, "a counting process with zero variance cannot generate a surge"


def test_poisson_matches_the_expected_rate():
    """Poisson must preserve the expectation, so attendance is unchanged."""
    m = model(attendance=30_000, scale=1.0, start_hour=12.0)
    expected_per_tick = m.arrivals_in(0.0, 0.5)

    n = 4000
    total = sum(m.draw_arrivals(0.0, 0.5) for _ in range(n))
    observed = total / n

    # 4 sigma of the mean: sqrt(n * lambda) / n = sqrt(lambda / n)
    sigma = (expected_per_tick / n) ** 0.5
    assert abs(observed - expected_per_tick) < 4 * sigma + 1e-9, (
        f"drew {observed:.3f}/tick against an expected {expected_per_tick:.3f}"
    )


def test_draw_arrivals_conserves_people_over_time():
    """
    Summed over many ticks the draws must match the float integral within sampling
    error. The old test asserted near-exact equality; with a stochastic process the
    correct assertion is a statistical band, not a tighter one.
    """
    m = model(attendance=20_000, scale=1.0, start_hour=0.0)
    duration = 0.5
    ticks = 2400

    drawn = sum(m.draw_arrivals(t * duration, duration) for t in range(ticks))
    expected = sum(m.arrivals_in(t * duration, duration) for t in range(ticks))

    # Poisson over the whole window: sd = sqrt(total expected arrivals).
    sigma = max(1.0, expected ** 0.5)
    assert abs(drawn - expected) < 4 * sigma, (
        f"drew {drawn} against an expected {expected:.2f} (4 sigma = {4 * sigma:.1f})"
    )


def test_low_rates_still_admit_eventually():
    """
    A rate far below one person per tick must still produce people over time - a
    Poisson process handles this naturally, where a truncating implementation needed
    a fractional carry to avoid admitting nobody forever.

    At 03:00 the arrival curve is genuinely near zero (0.00035 of the daily total),
    so the attendance is set high enough that the *expectation* over the window is
    about twelve people: 0.003 per tick, four hundred times below one, yet non-zero
    overall. A lower attendance draws zero, correctly.
    """
    m = model(attendance=60_000, scale=1.0, start_hour=3.0)
    expected = sum(m.arrivals_in(t * 0.5, 0.5) for t in range(4000))
    assert expected > 5.0, f"test setup too weak: only {expected:.2f} expected"

    total = sum(m.draw_arrivals(t * 0.5, 0.5) for t in range(4000))
    assert total > 0, "a slow trickle should still admit somebody over 2000 seconds"


def test_draws_are_deterministic_for_a_given_seed():
    """Poisson sampling is stochastic but still reproducible from the seed."""
    a = model(rng=Rng(7))
    b = model(rng=Rng(7))
    seq_a = [a.draw_arrivals(t * 0.5, 0.5) for t in range(500)]
    seq_b = [b.draw_arrivals(t * 0.5, 0.5) for t in range(500)]
    assert seq_a == seq_b


def test_different_seeds_give_different_sequences():
    a = model(rng=Rng(7))
    b = model(rng=Rng(8))
    seq_a = [a.draw_arrivals(t * 0.5, 0.5) for t in range(500)]
    seq_b = [b.draw_arrivals(t * 0.5, 0.5) for t in range(500)]
    assert seq_a != seq_b


def test_zero_duration_draws_nothing():
    m = model()
    assert m.draw_arrivals(0.0, 0.0) == 0


# ── describe ──────────────────────────────────────────────────────


def test_describe_reports_the_configuration():
    m = model(attendance=12_345, start_hour=9.5, scale=1.5)
    d = m.describe()
    assert d["attendance"] == 12_345
    assert d["start_hour"] == 9.5
    assert d["scale"] == 1.5
    assert "source" in d


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(
        (n, f) for n, f in dict(globals()).items() if n.startswith("test_")
    ):
        try:
            if fn.__code__.co_argcount:
                continue   # needs pytest fixtures (tmp_path)
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
