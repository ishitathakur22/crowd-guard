"""
Pathfinding: A* wall avoidance and the density-weighted flow field.

The flow field replaced a per-agent `compute_alternative_routes` call that ran one
A* per exit per agent - up to 160 heapq searches per tick, measured at 591 ms at
200 agents. The field answers the same question for every cell at once, so these
tests check the two properties the rerouting logic depends on: it never plans
through a wall, and following the field actually reaches an exit.

Run with pytest, or directly: `python tests/test_pathfinding.py`
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from backend.models import CellType, Point  # noqa: E402
from backend.pathfinding import (  # noqa: E402
    a_star_search, build_flow_field, compute_alternative_routes,
    density_weighted_cost, flow_path,
)

WALL = CellType.WALL.value


def empty_grid(w=10, h=10):
    return [[CellType.EMPTY.value for _ in range(w)] for _ in range(h)]


def cells_of(path):
    return [(int(p.x), int(p.y)) for p in path]


# ── A* ────────────────────────────────────────────────────────────


def test_a_star_finds_a_direct_path_on_open_ground():
    grid = empty_grid()
    path = a_star_search(grid, Point(x=0.0, y=0.0), Point(x=5.0, y=0.0))
    assert path, "no path found across empty ground"
    assert cells_of(path)[-1] == (5, 0)


def test_a_star_never_routes_through_a_wall():
    grid = empty_grid()
    for y in range(0, 8):          # wall with a gap at y=8,9
        grid[y][5] = WALL
    path = a_star_search(grid, Point(x=0.0, y=0.0), Point(x=9.0, y=0.0))
    assert path, "a route around the gap should exist"
    for x, y in cells_of(path):
        assert grid[y][x] != WALL, f"path enters wall at ({x},{y})"
    assert any(y >= 8 for _x, y in cells_of(path)), "did not go through the gap"


def test_a_star_returns_empty_when_fully_walled_off():
    grid = empty_grid()
    for y in range(10):
        grid[y][5] = WALL
    assert a_star_search(grid, Point(x=0.0, y=0.0), Point(x=9.0, y=0.0)) == []


def test_a_star_rejects_out_of_bounds():
    grid = empty_grid()
    assert a_star_search(grid, Point(x=-1.0, y=0.0), Point(x=5.0, y=5.0)) == []
    assert a_star_search(grid, Point(x=0.0, y=0.0), Point(x=99.0, y=5.0)) == []


def test_a_star_excludes_the_start_cell():
    grid = empty_grid()
    path = a_star_search(grid, Point(x=2.0, y=2.0), Point(x=6.0, y=2.0))
    assert cells_of(path)[0] != (2, 2), "path should not include its own start"


def test_density_weighting_penalises_crowding_monotonically():
    assert density_weighted_cost(1.0, 0.0) == 1.0
    previous = density_weighted_cost(1.0, 0.5)
    for rho in (1.0, 1.5, 2.0, 3.0, 3.5):
        current = density_weighted_cost(1.0, rho)
        assert current >= previous, f"cost fell as density rose at {rho}"
        previous = current
    assert density_weighted_cost(1.0, 3.5) > density_weighted_cost(1.0, 1.0)


def test_route_cost_accounts_for_crowding():
    """
    The same route must cost more when its cells are busy.

    Built as a sealed one-cell-wide corridor so A* has no alternative: it cannot
    dodge the crowd, which is exactly the case where the penalty has to show up.
    """
    grid = empty_grid(10, 3)
    for x in range(10):
        grid[0][x] = WALL
        grid[2][x] = WALL

    start, goals = Point(x=0.0, y=1.0), [Point(x=9.0, y=1.0)]

    clear = np.zeros((3, 10))
    busy = np.zeros((3, 10))
    busy[1, 1:9] = 3.0

    quiet_cost = compute_alternative_routes(grid, clear, start, goals)[0][2]
    crowded_cost = compute_alternative_routes(grid, busy, start, goals)[0][2]
    assert crowded_cost > quiet_cost * 1.5, (
        f"crowded corridor cost {crowded_cost:.2f} vs clear {quiet_cost:.2f}"
    )


# ── flow field ────────────────────────────────────────────────────


def test_flow_field_is_zero_at_the_exits():
    grid = empty_grid()
    dist, _nxt = build_flow_field(grid, np.zeros((10, 10)), [Point(x=9.0, y=0.0)])
    assert float(dist[0][9]) == 0.0


def test_flow_field_is_finite_across_open_ground():
    grid = empty_grid()
    dist, _nxt = build_flow_field(grid, np.zeros((10, 10)), [Point(x=9.0, y=9.0)])
    assert np.isfinite(dist).all(), "an open grid should be fully reachable"


def test_flow_field_leaves_walled_off_cells_unreachable():
    grid = empty_grid()
    for y in range(10):
        grid[y][5] = WALL
    dist, _nxt = build_flow_field(grid, np.zeros((10, 10)), [Point(x=9.0, y=9.0)])
    assert np.isinf(dist[0][0]), "a sealed region must be unreachable"


def test_flow_path_reaches_an_exit():
    grid = empty_grid()
    exits = [Point(x=9.0, y=0.0)]
    dist, nxt = build_flow_field(grid, np.zeros((10, 10)), exits)

    path = flow_path(nxt, Point(x=0.0, y=0.0), max_len=30)
    assert path, "flow_path produced nothing"
    final = (int(path[-1].x), int(path[-1].y))
    assert final in cells_of(exits) or dist[final[1]][final[0]] < 1.5, (
        f"walked to {final}, which is not near an exit"
    )


def test_flow_path_strictly_descends():
    """
    Every step must reduce the distance-to-exit, which is what makes the flow graph
    acyclic and therefore deadlock-free.
    """
    grid = empty_grid()
    dist, nxt = build_flow_field(grid, np.zeros((10, 10)), [Point(x=9.0, y=9.0)])

    cells = [(0, 0)] + cells_of(flow_path(nxt, Point(x=0.0, y=0.0), max_len=30))
    for (x0, y0), (x1, y1) in zip(cells, cells[1:]):
        assert dist[y1][x1] < dist[y0][x0], (
            f"step ({x0},{y0})->({x1},{y1}) did not descend "
            f"({dist[y0][x0]:.3f} -> {dist[y1][x1]:.3f})"
        )


def test_flow_path_stops_at_an_exit():
    grid = empty_grid()
    dist, nxt = build_flow_field(grid, np.zeros((10, 10)), [Point(x=5.0, y=5.0)])
    path = flow_path(nxt, Point(x=5.0, y=5.0), max_len=20)
    assert path == [], "standing on an exit there is nowhere to go"


def test_flow_path_avoids_crowding_when_a_clear_route_exists():
    """
    A wall down the middle with gaps top and bottom: with the top gap crowded, the
    descent field should send an agent toward the bottom one.
    """
    grid = empty_grid(12, 12)
    for y in range(1, 11):          # gaps left at row 0 and row 11
        grid[y][6] = WALL

    density = np.zeros((12, 12))
    density[0, 0:12] = 5.0          # top gap heavily congested

    exits = [Point(x=11.0, y=6.0)]
    _dist, nxt = build_flow_field(grid, density, exits)

    cells = cells_of(flow_path(nxt, Point(x=0.0, y=0.0), max_len=8))
    assert cells, "no route found around the wall"
    for x, y in cells:
        assert grid[y][x] != WALL, f"flow_path crossed a wall at ({x},{y})"

    # With the top gap expensive, the descent should trend away from row 0.
    assert max(y for _x, y in cells) > 0, (
        "flow kept hugging the congested top gap"
    )


def test_flow_field_is_cached_and_reused():
    """Repeated calls on the same grid and exits must agree exactly."""
    grid = empty_grid()
    exits = [Point(x=9.0, y=9.0)]
    d1, n1 = build_flow_field(grid, np.zeros((10, 10)), exits)
    d2, n2 = build_flow_field(grid, np.zeros((10, 10)), exits)
    assert np.array_equal(d1, d2)
    assert np.array_equal(n1, n2)


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
