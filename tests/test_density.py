"""
Density analysis: Fruin classification, the local-density measure, and units.

Two bugs lived here and each is now pinned by a test:

1. `_density_rise_rate` returned a per-history-sample slope and was consumed as a
   per-second rate, inflating every reported time-to-choke by about 2.5x. A crowd
   reported as further from a crush than it is, is the exact failure this dashboard
   exists to prevent.
2. `classify_zones` and `detect_hazards` used raw per-cell counts while the
   forecast's warnings used a neighbourhood maximum, so the same cell showed two
   different "densities" and a threshold applied to one meant nothing when compared
   against the other.

Run with pytest, or directly: `python tests/test_density.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from backend.density import (  # noqa: E402
    DensityAnalyzer, TICK_DURATION_SEC, classify_fruin, local_density,
)
from backend.models import FruinLoS  # noqa: E402


# ── Fruin boundaries ──────────────────────────────────────────────


def test_fruin_boundaries():
    cases = [
        (0.0, FruinLoS.A),
        (0.30, FruinLoS.A),
        (0.31, FruinLoS.B),
        (0.42, FruinLoS.B),
        (0.43, FruinLoS.C),
        (1.07, FruinLoS.C),
        (1.08, FruinLoS.D),
        (1.53, FruinLoS.D),
        (1.54, FruinLoS.E),
        (2.16, FruinLoS.E),
        (2.17, FruinLoS.F),
        (3.49, FruinLoS.F),
        (3.50, FruinLoS.CRITICAL),
        (9.0, FruinLoS.CRITICAL),
    ]
    for density, expected in cases:
        got = classify_fruin(density)
        assert got == expected, f"density {density} -> {got}, expected {expected}"


# ── the shared local-density measure ──────────────────────────────


def test_local_density_of_a_single_agent_is_not_los_c():
    """
    One person alone must not read as restricted movement.

    A raw per-cell count makes a lone pedestrian 1.0 p/m2, which is Fruin LoS C. A
    3x3 mean makes it 1/9 = 0.11 p/m2, which is LoS A - correct.
    """
    field = np.zeros((10, 10))
    field[5][5] = 1.0
    got = local_density(field, radius=1)
    assert abs(float(got[5][5]) - 1.0 / 9.0) < 1e-12
    assert classify_fruin(float(got[5][5])) == FruinLoS.A


def test_local_density_averages_a_small_cluster():
    """Three people in adjacent cells is 3/9 p/m2, not a peak of 1.0."""
    field = np.zeros((10, 10))
    field[5][5] = field[5][6] = field[6][5] = 1.0
    got = local_density(field, radius=1)
    assert abs(float(got[5][5]) - 3.0 / 9.0) < 1e-12


def test_local_density_is_zero_away_from_the_crowd():
    field = np.zeros((10, 10))
    field[0][0] = 5.0
    got = local_density(field, radius=1)
    assert float(got[9][9]) == 0.0


def test_local_density_preserves_total_mass():
    """Smoothing redistributes density; it must not create or destroy people."""
    field = np.zeros((12, 12))
    field[6][6] = 9.0
    got = local_density(field, radius=1)
    assert abs(float(got.sum()) - 9.0) < 1e-9


def test_radius_zero_is_a_no_op():
    field = np.arange(9.0).reshape(3, 3)
    assert np.array_equal(local_density(field, radius=0), field)


# ── zones and hazards agree with each other ───────────────────────


def test_classify_zones_uses_the_shared_measure():
    analyzer = DensityAnalyzer(10, 10, kernel_radius=1)
    field = np.zeros((10, 10))
    field[5][5] = 9.0

    zones = analyzer.classify_zones(field)
    assert len(zones) == 1
    zone = zones[0]
    expected = local_density(field, 1)[5][5]

    assert abs(zone.density - float(expected)) < 0.005
    # agent_count still reports the raw headcount, because that is what it means.
    assert zone.agent_count == 9
    assert zone.los_level == classify_fruin(float(expected))


def test_hazard_density_matches_zone_density_for_the_same_cell():
    """
    The bug this pins: a hazard and a zone label disagreeing about one cell.
    """
    analyzer = DensityAnalyzer(10, 10, kernel_radius=1)
    field = np.zeros((10, 10))
    field[5][5] = 12.0   # 12/9 = 1.33 p/m2 -> LoS D, above the hazard floor

    zones = {z.zone_id: z for z in analyzer.classify_zones(field)}
    hazards = {h.zone_id: h for h in analyzer.detect_hazards(field)}

    assert "cell_5_5" in hazards, "a 1.33 p/m2 cell should be flagged at the LoS D floor"
    assert abs(zones["cell_5_5"].density - hazards["cell_5_5"].current_density) < 0.02
    assert zones["cell_5_5"].los_level == hazards["cell_5_5"].fruin_los


def test_no_hazard_below_the_los_d_floor():
    analyzer = DensityAnalyzer(10, 10, kernel_radius=1)
    field = np.zeros((10, 10))
    field[5][5] = 9.0   # 9/9 = 1.0 p/m2, below 1.08
    assert analyzer.detect_hazards(field) == []


# ── the time-to-choke unit bug ────────────────────────────────────


def test_density_rise_rate_is_per_second_not_per_sample():
    """
    History rises by 1.0 p/m2 over 4 samples, i.e. over 4 * 0.5 s = 2.0 s.
    The rate must therefore be 0.5 per second. The old code divided by the sample
    count and returned 0.2, inflating every time-to-choke by 2.5x.
    """
    analyzer = DensityAnalyzer(4, 4, kernel_radius=1)
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        snapshot = np.zeros((4, 4))
        snapshot[2][2] = value
        analyzer.density_history.append(snapshot)

    rate = analyzer._density_rise_rate(2, 2)
    span = 4 * TICK_DURATION_SEC
    expected = 1.0 / span

    assert abs(rate - expected) < 1e-9, (
        f"rise rate {rate:.4f}/s, expected {expected:.4f}/s "
        f"(per-sample slope would be {1.0 / 5:.4f})"
    )
    assert abs(rate - 1.0 / 5) > 1e-3, "this is the per-sample slope, i.e. the old bug"


def test_rise_rate_is_zero_without_enough_history():
    analyzer = DensityAnalyzer(4, 4)
    assert analyzer._density_rise_rate(1, 1) == 0.0


def test_time_to_choke_is_in_seconds():
    """
    A cell at 3.2 p/m2 rising 0.5 p/m2 per second is 0.6 s from the 3.5 threshold.

    History is stored pre-smoothed (that is what `update_history` does), so the raw
    field fed to `detect_hazards` is scaled to smooth to the same value: a single
    occupied cell contributes one ninth of its count.
    """
    analyzer = DensityAnalyzer(6, 6, kernel_radius=1)
    for value in (2.2, 2.45, 2.7, 2.95, 3.2):
        snapshot = np.zeros((6, 6))
        snapshot[3][3] = value
        analyzer.density_history.append(snapshot)

    raw = np.zeros((6, 6))
    raw[3][3] = 3.2 * 9.0   # smooths to 3.2 p/m2 at the centre cell

    hazards = analyzer.detect_hazards(raw)
    assert hazards, "a 3.2 p/m2 cell must be reported"
    hz = hazards[0]
    assert abs(hz.current_density - 3.2) < 0.02

    rate = analyzer._density_rise_rate(3, 3)
    assert abs(rate - 0.5) < 1e-9, f"rise rate {rate}, expected 0.5/s"

    expected = (3.5 - hz.current_density) / rate
    assert abs(hz.time_to_choke_sec - round(expected, 1)) < 0.2, (
        f"time-to-choke {hz.time_to_choke_sec}s, expected about {expected:.1f}s"
    )


# ── hazard severity ordering ──────────────────────────────────────


def test_severity_increases_with_density():
    analyzer = DensityAnalyzer(6, 6, kernel_radius=1)

    def severity_for(cell_value):
        field = np.zeros((6, 6))
        field[3][3] = cell_value
        hazards = analyzer.detect_hazards(field)
        return hazards[0].severity if hazards else None

    # 12/9=1.33 D, 22/9=2.44 F, 40/9=4.44 CRITICAL
    ranks = {"INFO": 0, "WARNING": 1, "CRITICAL": 2, "EMERGENCY": 3}
    d = severity_for(12.0)
    f = severity_for(22.0)
    crit = severity_for(40.0)

    assert d is not None and f is not None and crit is not None
    assert ranks[d.value] < ranks[f.value] < ranks[crit.value]


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
