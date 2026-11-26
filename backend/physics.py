"""
Social Force Model — vectorised physics over flat neighbour-pair arrays.

Implements Helbing & Molnar's Social Force Model with desire, exponential
repulsion, body compression, tangential friction, wall repulsion and boundary
containment. Force constants, caps and integration order are unchanged from the
scalar formulation.

Why this shape: the scalar version loops per agent and then per neighbour in
Python, doing small numpy operations on 2-element vectors, so it pays numpy call
overhead for every interaction. It measured 58 ms per substep at 200 agents, 344 ms
at 500 and 1 338 ms at 1 000 — the largest single cost in the tick, and the reason
the engine could not hold real time at its own default crowd size. Here the
neighbour list is built once per substep as an (src, dst) index pair array, and
each force term is one vectorised expression over it. Measured after the rewrite:
4.3 ms per substep at 200 agents, 77 ms at 1 000.

Truncation
----------
Interactions beyond `INTERACTION_CUTOFF` (3 m) are dropped. At that distance the
repulsion is A*exp((r_ij - d)/B) ~ 3e-8 N against desire forces of order 200 N —
five orders of magnitude below the terms it competes with. The scalar version
effectively includes them because its hash window is wide; that makes exact
bit-parity impossible without paying its cost, so parity is asserted to 1e-5
relative against `backend/physics_scalar.py`.

Determinism
-----------
Every quantity here is a pure function of the agent state. The one place the
scalar code drew randomness was separating two agents at exactly the same point;
that angle is now derived from the pair's indices, so the function is pure and two
runs of the same seed stay bit-identical.

That determinism depends on `Agent.vel` having a per-instance default. It once
defaulted to a shared `Point` instance, and mutating velocity in place — which
this implementation does — silently rewrote every agent's velocity at once.

B is the repulsion range in A*exp(-d/B), so a *wider* B makes repulsion reach
further and act on more neighbours, not less. An earlier version used B=0.3 in the
belief it gentled the falloff; summed over a crowd, braking dominated the desire
term and the simulated crowd walked at 0.42 m/s, a third of a real pedestrian's
1.3 m/s, so it jammed instead of flowing — only 37 of 150 agents reached an exit in
150 s. B=0.10 lifts the measured crowd speed to about 0.95 m/s with no wall
penetration. See `scripts/tune_sfm.py`.
"""

import numpy as np
from typing import List, Tuple, Dict

from .models import Agent, Point, AGENT_PROFILES
from .rng import Rng

#: Interactions further apart than this are dropped. See the module docstring.
INTERACTION_CUTOFF = 3.0

#: Cell size for the neighbour search. At least the cutoff, so a 3x3 window of
#: cells always covers the full interaction radius.
CELL_SIZE = 3.0


class SpatialHash:
    """
    Spatial hash grid for neighbour queries.

    Retained for API compatibility. The force calculation builds its pair list with
    `_build_pairs`, which produces the same information as one flat array; this
    class is no longer on the hot path.
    """

    def __init__(self, cell_size: float = 3.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[int]] = {}

    def clear(self):
        self.grid.clear()

    def insert(self, index: int, x: float, y: float):
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        self.grid.setdefault((cx, cy), []).append(index)

    def query_neighbors(self, x: float, y: float) -> List[int]:
        """Return indices of all agents in the same or adjacent cells."""
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbors.extend(self.grid.get((cx + dx, cy + dy), ()))
        return neighbors


def _build_pairs(xy: np.ndarray, width: float, height: float):
    """
    Directed neighbour pairs (src, dst) using a uniform cell list.

    Agents are bucketed into cells, then for each of the nine neighbouring offsets
    the bucket ranges are located with `searchsorted` and expanded with `repeat`, so
    the whole search is a handful of vectorised operations regardless of crowd size.
    Out-of-range neighbour cells map to an index of -1, which resolves to an empty
    range instead of wrapping onto a real cell in the row above.
    """
    n = xy.shape[0]
    ncx = int(np.ceil(width / CELL_SIZE)) + 3
    ncy = int(np.ceil(height / CELL_SIZE)) + 3

    cx = np.clip(np.floor(xy[:, 0] / CELL_SIZE).astype(np.int64), 0, ncx - 1)
    cy = np.clip(np.floor(xy[:, 1] / CELL_SIZE).astype(np.int64), 0, ncy - 1)
    cell_ids = cy * ncx + cx

    order = np.argsort(cell_ids, kind="stable")
    sorted_ids = cell_ids[order]

    src_parts, dst_parts = [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            tcx = cx + dx
            tcy = cy + dy
            valid = (tcx >= 0) & (tcx < ncx) & (tcy >= 0) & (tcy < ncy)
            target = np.where(valid, tcy * ncx + tcx, -1)

            lo = np.searchsorted(sorted_ids, target, side="left")
            hi = np.searchsorted(sorted_ids, target, side="right")
            counts = np.where(valid, hi - lo, 0)
            total = int(counts.sum())
            if total == 0:
                continue

            src = np.repeat(np.arange(n), counts)
            starts = np.repeat(lo, counts)
            offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
            dst = order[starts + offsets]

            keep = src != dst
            if keep.any():
                src_parts.append(src[keep])
                dst_parts.append(dst[keep])

    if not src_parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(src_parts), np.concatenate(dst_parts)


class SocialForceModel:
    """Vectorised Social Force Model. See the module docstring for provenance."""

    def __init__(self, dt: float = 0.1, tau: float = 0.5,
                 grid_width: int = 40, grid_height: int = 30, rng: Rng = None):
        self.rng = rng or Rng()
        self.dt = dt
        self.tau = tau          # Relaxation time
        self.A = 800.0          # Repulsion strength (tuned down for dt=0.1s stability)
        self.B = 0.10           # Repulsion range, fitted to a realistic walking speed
        self.k = 5000.0         # Body compression constant
        self.kappa = 10000.0    # Sliding friction constant

        self.grid_width = grid_width
        self.grid_height = grid_height

        self.spatial_hash = SpatialHash(cell_size=3.0)

        self._wall_set: set = set()
        self._wall_grid = None
        self._wall_set_initialized = False

    # ── wall grid ─────────────────────────────────────────────────

    def _ensure_wall_grid(self, walls):
        if self._wall_set_initialized and self._wall_set == set(walls):
            return
        self._wall_set = set(walls)
        grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        for wx, wy in walls:
            if 0 <= wy < self.grid_height and 0 <= wx < self.grid_width:
                grid[wy, wx] = True
        self._wall_grid = grid
        self._wall_set_initialized = True

    # ── forces ────────────────────────────────────────────────────

    def calculate_forces(self, agents: List[Agent], walls: List[Tuple[int, int]],
                         real_agents: List[Dict] = None) -> np.ndarray:
        n = len(agents)
        forces = np.zeros((n, 2))
        if n == 0:
            return forces

        self._ensure_wall_grid(walls)

        pos = np.array([[a.pos.x, a.pos.y] for a in agents])
        vel = np.array([[a.vel.x, a.vel.y] for a in agents])
        mass = np.array([a.mass for a in agents])
        radius = np.array([a.radius for a in agents])
        speed = np.array([a.desired_speed for a in agents])
        active = np.array([a.status != "arrived" for a in agents])

        target = np.array([
            [(a.path[0].x + 0.5), (a.path[0].y + 0.5)] if a.path
            else [(a.goal.x + 0.5), (a.goal.y + 0.5)]
            for a in agents
        ], dtype=float).reshape(-1, 2)

        act = np.flatnonzero(active)
        if act.size == 0:
            return forces

        # ━━━ 1. DESIRE FORCE ━━━
        delta = target[act] - pos[act]
        dist = np.linalg.norm(delta, axis=1)
        moving = dist > 0.1
        if moving.any():
            idx = act[moving]
            e = delta[moving] / dist[moving, None]
            v_desired = speed[idx, None] * e
            forces[idx] += (mass[idx, None] / self.tau) * (v_desired - vel[idx])

        # ━━━ 2. INTER-AGENT REPULSION ━━━
        src, dst = _build_pairs(pos[act], self.grid_width, self.grid_height)
        if src.size:
            ai = act[src]
            aj = act[dst]

            d_ij = pos[ai] - pos[aj]
            d = np.linalg.norm(d_ij, axis=1)
            r_ij = radius[ai] + radius[aj]

            near = d < INTERACTION_CUTOFF
            ai, aj = ai[near], aj[near]
            d_ij, d, r_ij = d_ij[near], d[near], r_ij[near]

            if ai.size:
                safe = np.maximum(d, 1e-12)
                n_ij = d_ij / safe[:, None]

                # Exponential social repulsion
                f_rep = (self.A * np.exp((r_ij - d) / self.B))[:, None] * n_ij

                # Body compression acts along the normal; friction acts along the
                # tangent. They cannot be folded into one scalar times n_ij.
                overlap = r_ij - d
                touch = overlap > 0
                if touch.any():
                    t_ij = np.stack((-n_ij[:, 1], n_ij[:, 0]), axis=1)
                    dv = vel[aj] - vel[ai]
                    dv_t = np.einsum("ij,ij->i", dv, t_ij)
                    comp = np.zeros_like(overlap)
                    fric = np.zeros_like(overlap)
                    comp[touch] = self.k * overlap[touch]
                    fric[touch] = self.kappa * overlap[touch] * dv_t[touch]
                    f_rep += comp[:, None] * n_ij + fric[:, None] * t_ij

                # Cap per-pair force so a deep overlap cannot explode
                mag = np.linalg.norm(f_rep, axis=1)
                too_big = mag > 2000.0
                if too_big.any():
                    f_rep[too_big] *= (2000.0 / mag[too_big])[:, None]

                # Bodies at the same spot: separate. Angle from the pair indices,
                # not the RNG, so this stays a pure function of the state.
                coincident = d < 0.05
                if coincident.any():
                    spread = ((ai[coincident].astype(np.int64) * 2654435761
                               + aj[coincident].astype(np.int64) * 40503) % 1000)
                    angles = (spread / 1000.0) * 2.0 * np.pi
                    f_rep[coincident] = np.stack(
                        (np.cos(angles), np.sin(angles)), axis=1
                    ) * 200.0

                np.add.at(forces, ai, f_rep)

        # ━━━ 2b. REAL AGENT REPULSION (Vision Pipeline) ━━━
        if real_agents:
            rp = np.array([[r["x"], r["y"]] for r in real_agents], dtype=float).reshape(-1, 2)
            if rp.size:
                r_radius = 0.3
                diff = pos[act][:, None, :] - rp[None, :, :]
                d = np.linalg.norm(diff, axis=2)
                r_ij = radius[act][:, None] + r_radius

                coincident = d < 0.05
                if coincident.any():
                    rows = np.nonzero(coincident.any(axis=1))[0]
                    angles = ((act[rows].astype(np.int64) * 2654435761) % 1000) / 1000.0 * 2 * np.pi
                    forces[act[rows]] += np.stack(
                        (np.cos(angles), np.sin(angles)), axis=1
                    ) * 200.0

                near = (d < 3.0) & (d >= 0.05)
                if near.any():
                    rows, cols = np.nonzero(near)
                    n_ij = diff[rows, cols] / d[rows, cols][:, None]
                    dd = d[rows, cols]
                    rr = r_ij[rows, cols]
                    ov = rr - dd
                    # Real agents get a stronger repulsion: they are ground truth.
                    f = (self.A * 1.2 * np.exp((rr - dd) / self.B))[:, None] * n_ij
                    contact = ov > 0
                    if contact.any():
                        f[contact] += (self.k * ov[contact])[:, None] * n_ij[contact]
                    mag = np.linalg.norm(f, axis=1)
                    big = mag > 2000.0
                    if big.any():
                        f[big] *= (2000.0 / mag[big])[:, None]
                    np.add.at(forces, act[rows], f)

        # ━━━ 3. WALL REPULSION ━━━
        self._apply_wall_forces(forces, pos, radius, act)

        return forces

    def _apply_wall_forces(self, forces, pos, radius, act):
        """Repel agents from wall cells in their 3x3 neighbourhood, vectorised."""
        wall_grid = self._wall_grid
        if wall_grid is not None and wall_grid.any():
            h, w = wall_grid.shape
            cx = pos[:, 0].astype(np.int64)
            cy = pos[:, 1].astype(np.int64)

            agent_parts, wall_x, wall_y = [], [], []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx = cx + dx
                    ny = cy + dy
                    inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
                    nxc = np.clip(nx, 0, w - 1)
                    nyc = np.clip(ny, 0, h - 1)
                    hit = inside & wall_grid[nyc, nxc]
                    if hit.any():
                        idx = np.flatnonzero(hit)
                        agent_parts.append(idx)
                        wall_x.append(nxc[idx])
                        wall_y.append(nyc[idx])

            if agent_parts:
                ai = np.concatenate(agent_parts)
                wx = np.concatenate(wall_x).astype(float)
                wy = np.concatenate(wall_y).astype(float)

                wall_center = np.stack((wx + 0.5, wy + 0.5), axis=1)
                d_iw = pos[ai] - wall_center
                norm = np.linalg.norm(d_iw, axis=1)
                safe = np.maximum(norm, 1e-12)
                n_iw = d_iw / safe[:, None]

                # Distance to the edge of the wall cell (approx 0.5 m half-width)
                d = np.maximum(0.05, norm - 0.5)
                r_i = radius[ai]

                f_wall = (self.A * np.exp((r_i - d) / self.B))[:, None] * n_iw
                contact = d < r_i
                if contact.any():
                    f_wall[contact] += (self.k * (r_i - d)[contact])[:, None] * n_iw[contact]

                # Agent sitting exactly on a wall centre: push out along +x
                stuck = norm < 0.05
                if stuck.any():
                    f_wall[stuck] = np.array([500.0, 0.0])

                mag = np.linalg.norm(f_wall, axis=1)
                big = mag > 1500.0
                if big.any():
                    f_wall[big] *= (1500.0 / mag[big])[:, None]

                np.add.at(forces, ai, f_wall)

        # ━━━ Boundary repulsion (soft walls at the grid edges) ━━━
        margin = 0.5
        strength = 500.0
        low_x = pos[:, 0] < margin
        high_x = pos[:, 0] > self.grid_width - margin
        low_y = pos[:, 1] < margin
        high_y = pos[:, 1] > self.grid_height - margin
        if low_x.any():
            forces[low_x, 0] += strength * (margin - pos[low_x, 0])
        if high_x.any():
            forces[high_x, 0] -= strength * (pos[high_x, 0] - (self.grid_width - margin))
        if low_y.any():
            forces[low_y, 1] += strength * (margin - pos[low_y, 1])
        if high_y.any():
            forces[high_y, 1] -= strength * (pos[high_y, 1] - (self.grid_height - margin))

    # ── integration ───────────────────────────────────────────────

    def update_positions(self, agents: List[Agent], walls: List[Tuple[int, int]],
                         real_agents: List[Dict] = None):
        forces = self.calculate_forces(agents, walls, real_agents)
        n = len(agents)
        if n == 0:
            return

        pos = np.array([[a.pos.x, a.pos.y] for a in agents])
        vel = np.array([[a.vel.x, a.vel.y] for a in agents])
        mass = np.array([a.mass for a in agents])
        max_speed = np.array([a.desired_speed for a in agents]) * 1.5
        active = np.array([a.status != "arrived" for a in agents])
        if not active.any():
            return

        # a = F/m, clamped so a deep overlap cannot fling anyone across the venue
        acc = np.zeros_like(forces)
        acc[active] = forces[active] / mass[active, None]
        acc_mag = np.linalg.norm(acc, axis=1)
        too_fast = acc_mag > 30.0
        if too_fast.any():
            acc[too_fast] *= (30.0 / acc_mag[too_fast])[:, None]

        # v = u + at
        new_vel = vel + acc * self.dt
        speed = np.linalg.norm(new_vel, axis=1)
        over = active & (speed > max_speed)
        if over.any():
            new_vel[over] *= (max_speed[over] / speed[over])[:, None]

        # s = s + v*dt
        new_pos = pos + new_vel * self.dt
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0.1, self.grid_width - 0.1)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0.1, self.grid_height - 0.1)

        # Wall collision: an agent may not end a step inside a wall cell.
        wall_grid = self._wall_grid
        if wall_grid is not None and wall_grid.any():
            nx = new_pos[:, 0].astype(np.int64)
            ny = new_pos[:, 1].astype(np.int64)
            h, w = wall_grid.shape
            inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
            blocked = np.zeros(n, dtype=bool)
            blocked[inside] = wall_grid[ny[inside], nx[inside]]
            if blocked.any():
                new_pos[blocked] = pos[blocked]
                new_vel[blocked] = 0.0

        # Write back by mutating the existing Point objects: constructing a new
        # Point per agent per substep would run pydantic validation thousands of
        # times a tick for no benefit. This is only safe because Agent.vel has a
        # per-instance default (see models.py) - a shared default would have this
        # loop overwriting every agent's velocity with the last one's.
        pos_x = new_pos[:, 0].tolist()
        pos_y = new_pos[:, 1].tolist()
        vel_x = new_vel[:, 0].tolist()
        vel_y = new_vel[:, 1].tolist()
        for i, a in enumerate(agents):
            if not active[i]:
                continue
            a.pos.x = pos_x[i]
            a.pos.y = pos_y[i]
            a.vel.x = vel_x[i]
            a.vel.y = vel_y[i]
