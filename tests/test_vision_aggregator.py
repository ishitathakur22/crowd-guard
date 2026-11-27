"""
Vision aggregator: de-duplication, calibration and out-of-plan handling.

Each test pins a defect that was present in the original implementation:

1. **Every person was counted two or three times.** Three detectors all saw
   pedestrians (`crowd` and `flow` with classes=[0], `anomaly` with classes=None
   covering all 80 COCO classes) and nothing merged their output. One pedestrian
   became two or three grid agents, inflating the density and the count fed to the
   forecaster.
2. **The homography was a disguised scale.** Feeding the image corners mapped to the
   grid corners yields an anisotropic stretch with no perspective correction, so a
   camera looking down at the floor mapped distant people to the same
   metres-per-pixel as nearby ones.
3. **Out-of-plan detections were clamped onto the border**, building a false density
   ridge along the grid edges - exactly where the gate zones and warnings are.

Run with pytest, or directly: `python tests/test_vision_aggregator.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from backend.vision.aggregator import VisionAggregator, dedupe, iou  # noqa: E402


def det(bbox, cls=0, conf=0.9, task="crowd", center=None, track_id=None):
    d = {
        "bbox": bbox,
        "class": cls,
        "conf": conf,
        "task": task,
        "center": center if center is not None else [(bbox[0] + bbox[2]) / 2, bbox[3]],
    }
    if track_id is not None:
        d["track_id"] = track_id
    return d


# ── IoU and de-duplication ────────────────────────────────────────


def test_iou_of_identical_boxes_is_one():
    assert abs(iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9


def test_iou_of_disjoint_boxes_is_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_is_symmetric():
    a, b = [0, 0, 10, 20], [5, 5, 15, 25]
    assert abs(iou(a, b) - iou(b, a)) < 1e-12


def test_duplicate_person_from_two_models_is_merged():
    """The core defect: two detectors reporting the same pedestrian."""
    crowd = det([100, 100, 150, 300], task="crowd", conf=0.95)
    flow = det([102, 101, 151, 299], task="flow", conf=0.80)
    merged = dedupe([crowd, flow])
    assert len(merged) == 1, f"{len(merged)} agents for one pedestrian"
    assert merged[0]["conf"] == 0.95, "the best evidence should survive"


def test_duplicate_from_all_three_detectors_is_merged():
    dets = [
        det([100, 100, 150, 300], task="crowd", conf=0.95),
        det([101, 101, 151, 301], task="flow", conf=0.90),
        det([100, 100, 150, 300], task="anomaly", conf=0.85),
    ]
    assert len(dedupe(dets)) == 1


def test_separate_people_are_kept():
    a = det([0, 0, 50, 200], conf=0.9)
    b = det([400, 0, 450, 200], conf=0.8)
    assert len(dedupe([a, b])) == 2


def test_different_classes_are_not_merged():
    """A car overlapping a pedestrian is not a duplicate of them."""
    person = det([0, 0, 100, 200], cls=0)
    car = det([0, 0, 100, 200], cls=2, task="traffic")
    assert len(dedupe([person, car])) == 2


def test_dedupe_is_order_independent():
    a = det([0, 0, 100, 200], conf=0.7)
    b = det([2, 2, 101, 201], conf=0.9)
    assert dedupe([a, b])[0]["conf"] == dedupe([b, a])[0]["conf"] == 0.9


def test_empty_input():
    assert dedupe([]) == []


# ── calibration ───────────────────────────────────────────────────


def test_corner_fallback_is_labelled_as_an_assumption():
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    assert agg.is_calibrated
    assert agg.is_proper_homography is False, (
        "an assumed corner stretch must not be reported as real calibration"
    )


def test_corner_fallback_scales_linearly():
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    gx, gy = agg.map_to_grid(960, 540)
    assert abs(gx - 20.0) < 1e-6 and abs(gy - 15.0) < 1e-6


def test_proper_homography_maps_measured_floor_points():
    """
    Four real ground points must land on their known grid coordinates, which is what
    an assumed corner stretch cannot do for a camera at an angle.
    """
    agg = VisionAggregator(40, 30)
    img = [[400, 900], [1500, 880], [1750, 300], [180, 320]]
    grid = [[0, 30], [40, 30], [40, 0], [0, 0]]

    assert agg.calibrate(img, grid) is True
    assert agg.is_proper_homography is True

    mapped = agg.map_points_to_grid(img)
    for (mx, my), (expected_x, expected_y) in zip(mapped, grid):
        assert abs(mx - expected_x) < 1.0, f"x {mx:.2f} != {expected_x}"
        assert abs(my - expected_y) < 1.0, f"y {my:.2f} != {expected_y}"


def test_calibration_rejects_too_few_points():
    agg = VisionAggregator(40, 30)
    assert agg.calibrate([[0, 0], [1, 1]], [[0, 0], [1, 1]]) is False
    assert agg.is_proper_homography is False


def test_depth_argument_is_accepted_but_does_not_move_the_point():
    """
    MiDaS emits relative inverse depth with no metric scale, so it cannot correct a
    ground-plane position. The parameter is kept for interface compatibility.
    """
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    plain = agg.map_to_grid(640, 720)
    with_depth = agg.map_to_grid(640, 720, depth_value=3.7)
    assert plain == with_depth


# ── aggregation ───────────────────────────────────────────────────


def test_off_plan_detections_are_dropped_not_clamped():
    """
    Clamping squashed every out-of-venue detection onto the border, which built a
    false density ridge along the grid edges.
    """
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)

    inside = det([0, 0, 100, 200], center=[960, 540])
    outside = det([0, 0, 100, 200], center=[5000, 5000])

    agents = agg.aggregate(1, [inside, outside])
    assert len(agents) == 1, "the out-of-plan detection should have been dropped"
    assert 0 <= agents[0]["x"] < 40
    assert 0 <= agents[0]["y"] < 30


def test_aggregate_dedupes_before_mapping():
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    a = det([0, 0, 100, 200], center=[500, 500], conf=0.9)
    b = det([1, 1, 101, 201], center=[500, 500], conf=0.8)
    agents = agg.aggregate(1, [a, b])
    assert len(agents) == 1


def test_tracked_ids_are_preserved():
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    a = det([0, 0, 100, 200], center=[500, 500], track_id="crowd_7")
    agents = agg.aggregate(1, [a])
    assert agents[0]["id"] == "crowd_7"
    assert agents[0]["tracked"] is True


def test_untracked_ids_are_marked_as_such():
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    a = det([0, 0, 100, 200], center=[500, 500])
    agents = agg.aggregate(1, [a])
    assert agents[0]["tracked"] is False


def test_aggregate_handles_empty_input():
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    assert agg.aggregate(1, []) == []


def test_mapped_positions_are_within_the_grid():
    agg = VisionAggregator(40, 30)
    agg.calibrate_from_corners(1920, 1080)
    # Distinct boxes, or de-duplication would (correctly) merge them into one.
    dets = [
        det([0, 0, 10, 10], center=[100, 100]),
        det([1000, 500, 1010, 510], center=[1800, 1000]),
        det([2000, 900, 2010, 910], center=[960, 540]),
    ]
    agents = agg.aggregate(1, dets)
    assert len(agents) == 3
    for a in agents:
        assert 0.0 <= a["x"] <= 40.0
        assert 0.0 <= a["y"] <= 30.0


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
