"""
Reference Social Force Model — the scalar implementation, kept for parity testing.

This is the knowingly-slow, obviously-correct statement of the intended physics:
per-agent Python loops over a spatial hash. `tests/test_physics_parity.py` compares
the vectorised `backend/physics.py` against it.

Do not optimise this file. Its only job is to be an independent reference.

B is the repulsion range in A*exp(-d/B): a *wider* B makes repulsion reach
further and act on more neighbours, not less. See `backend/physics.py`.
"""

import numpy as np
from typing import List, Tuple, Dict

from .models import Agent, Point, AGENT_PROFILES
from .rng import Rng


class SpatialHash:
    """Spatial hash grid for efficient neighbor queries."""

    def __init__(self, cell_size: float = 3.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[int]] = {}

    def clear(self):
        self.grid.clear()

    def insert(self, index: int, x: float, y: float):
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        key = (cx, cy)
        if key not in self.grid:
            self.grid[key] = []
        self.grid[key].append(index)

    def query_neighbors(self, x: float, y: float) -> List[int]:
        """Return indices of all agents in the same or adjacent cells."""
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (cx + dx, cy + dy)
                if key in self.grid:
                    neighbors.extend(self.grid[key])
        return neighbors


class SocialForceModel:
    """Scalar reference implementation of the Social Force Model."""

    def __init__(self, dt: float = 0.1, tau: float = 0.5,
                 grid_width: int = 40, grid_height: int = 30, rng: Rng = None):
        self.rng = rng or Rng()
        self.dt = dt
        self.tau = tau
        self.A = 800.0
        self.B = 0.10
        self.k = 5000.0
        self.kappa = 10000.0

        self.grid_width = grid_width
        self.grid_height = grid_height

        self.spatial_hash = SpatialHash(cell_size=3.0)

        self._wall_set: set = set()
        self._wall_set_initialized = False

    def calculate_forces(self, agents: List[Agent], walls: List[Tuple[int, int]],
                         real_agents: List[Dict] = None) -> np.ndarray:
        if not agents:
            return np.zeros((0, 2))

        n = len(agents)
        positions = np.array([[a.pos.x, a.pos.y] for a in agents])
        velocities = np.array([[a.vel.x, a.vel.y] for a in agents])

        forces = np.zeros((n, 2))

        # ━━━ 1. DESIRE FORCE ━━━
        for i, a in enumerate(agents):
            if a.status == "arrived":
                continue

            speed = a.desired_speed

            if a.path:
                target = np.array([a.path[0].x + 0.5, a.path[0].y + 0.5])
            else:
                target = np.array([a.goal.x + 0.5, a.goal.y + 0.5])

            direction = target - positions[i]
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                e_i = direction / dist
                v_desired = speed * e_i
                f_desire = (a.mass / self.tau) * (v_desired - velocities[i])
                forces[i] += f_desire

        # ━━━ 2. INTER-AGENT REPULSION ━━━
        self.spatial_hash.clear()
        for i, a in enumerate(agents):
            if a.status != "arrived":
                self.spatial_hash.insert(i, a.pos.x, a.pos.y)

        for i in range(n):
            if agents[i].status == "arrived":
                continue

            neighbors = self.spatial_hash.query_neighbors(positions[i][0], positions[i][1])

            for j in neighbors:
                if i == j or agents[j].status == "arrived":
                    continue

                r_ij = agents[i].radius + agents[j].radius
                d_ij = positions[i] - positions[j]
                d = np.linalg.norm(d_ij)

                if d < 0.05:
                    # Separation angle derived from the pair indices, not the RNG,
                    # so the function stays pure.
                    angle = ((i * 2654435761 + j * 40503) % 1000) / 1000.0 * 2 * np.pi
                    forces[i] += np.array([np.cos(angle), np.sin(angle)]) * 200.0
                    continue

                n_ij = d_ij / d

                f_repulsion = self.A * np.exp((r_ij - d) / self.B) * n_ij

                if d < r_ij:
                    overlap = r_ij - d
                    f_compression = self.k * overlap * n_ij

                    t_ij = np.array([-n_ij[1], n_ij[0]])
                    dv = velocities[j] - velocities[i]
                    f_friction = self.kappa * overlap * np.dot(dv, t_ij) * t_ij

                    f_repulsion += f_compression + f_friction

                f_mag = np.linalg.norm(f_repulsion)
                if f_mag > 2000.0:
                    f_repulsion = (f_repulsion / f_mag) * 2000.0

                forces[i] += f_repulsion

        # ━━━ 2b. REAL AGENT REPULSION ━━━
        if real_agents:
            for i in range(n):
                if agents[i].status == "arrived":
                    continue

                for r_agent in real_agents:
                    r_pos = np.array([r_agent["x"], r_agent["y"]])
                    r_radius = 0.3
                    r_ij = agents[i].radius + r_radius
                    d_ij = positions[i] - r_pos
                    d = np.linalg.norm(d_ij)

                    if d < 0.05:
                        angle = ((i * 2654435761) % 1000) / 1000.0 * 2 * np.pi
                        forces[i] += np.array([np.cos(angle), np.sin(angle)]) * 200.0
                        continue

                    if d < 3.0:
                        n_ij = d_ij / d
                        f_repulsion = (self.A * 1.2) * np.exp((r_ij - d) / self.B) * n_ij

                        if d < r_ij:
                            overlap = r_ij - d
                            f_repulsion += self.k * overlap * n_ij

                        f_mag = np.linalg.norm(f_repulsion)
                        if f_mag > 2000.0:
                            f_repulsion = (f_repulsion / f_mag) * 2000.0

                        forces[i] += f_repulsion

        # ━━━ 3. WALL REPULSION ━━━
        if not self._wall_set_initialized:
            self._wall_set = set(walls)
            self._wall_set_initialized = True

        for i in range(n):
            if agents[i].status == "arrived":
                continue

            p = positions[i]
            r_i = agents[i].radius

            cx, cy = int(p[0]), int(p[1])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    wx, wy = cx + dx, cy + dy
                    if (wx, wy) in self._wall_set:
                        wall_center = np.array([wx + 0.5, wy + 0.5])
                        d_iw = p - wall_center
                        d_iw_norm = np.linalg.norm(d_iw)

                        if d_iw_norm < 0.05:
                            forces[i] += np.array([1.0, 0.0]) * 500.0
                            continue

                        d = max(0.05, d_iw_norm - 0.5)
                        n_iw = d_iw / d_iw_norm

                        f_wall = self.A * np.exp((r_i - d) / self.B) * n_iw
                        if d < r_i:
                            f_wall += self.k * (r_i - d) * n_iw

                        f_w_mag = np.linalg.norm(f_wall)
                        if f_w_mag > 1500.0:
                            f_wall = (f_wall / f_w_mag) * 1500.0

                        forces[i] += f_wall

            margin = 0.5
            boundary_strength = 500.0
            if p[0] < margin:
                forces[i][0] += boundary_strength * (margin - p[0])
            if p[0] > self.grid_width - margin:
                forces[i][0] -= boundary_strength * (p[0] - (self.grid_width - margin))
            if p[1] < margin:
                forces[i][1] += boundary_strength * (margin - p[1])
            if p[1] > self.grid_height - margin:
                forces[i][1] -= boundary_strength * (p[1] - (self.grid_height - margin))

        return forces

    def update_positions(self, agents: List[Agent], walls: List[Tuple[int, int]],
                         real_agents: List[Dict] = None):
        forces = self.calculate_forces(agents, walls, real_agents)

        for i, a in enumerate(agents):
            if a.status == "arrived":
                continue

            acceleration = forces[i] / a.mass

            acc_mag = np.linalg.norm(acceleration)
            if acc_mag > 30.0:
                acceleration = (acceleration / acc_mag) * 30.0

            new_vel = np.array([a.vel.x, a.vel.y]) + acceleration * self.dt

            max_speed = a.desired_speed * 1.5
            speed = np.linalg.norm(new_vel)
            if speed > max_speed:
                new_vel = (new_vel / speed) * max_speed

            a.vel = Point(x=float(new_vel[0]), y=float(new_vel[1]))

            new_pos = np.array([a.pos.x, a.pos.y]) + new_vel * self.dt

            new_pos[0] = max(0.1, min(float(self.grid_width) - 0.1, new_pos[0]))
            new_pos[1] = max(0.1, min(float(self.grid_height) - 0.1, new_pos[1]))

            nx, ny = int(new_pos[0]), int(new_pos[1])
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                if (nx, ny) in self._wall_set:
                    new_pos = np.array([a.pos.x, a.pos.y])

            a.pos = Point(x=float(new_pos[0]), y=float(new_pos[1]))
