"""
Prediction Engine — 15-30 minute crowd forecast via cloned forward simulation.

Clones the current simulation state (agent positions, velocities, goals) and
runs it forward in accelerated time to identify future bottlenecks before they
form.

Two things make the forecast trustworthy rather than decorative:

  1. The fast-forward keeps crowd repulsion. Without it, cloned agents stack on
     the same waypoint, per-cell density saturates, and every horizon step
     reports the same number — a constant, not a prediction.

     Repulsion here is a density-gradient (pressure) term rather than the
     pairwise exponential the live physics uses. Agents are pushed down the
     local density gradient, which is the continuum-crowds formulation. It is
     O(agents + cells) instead of O(agents²), so a 15-minute horizon over a few
     hundred people stays in the tens of milliseconds. Fine-grained contact
     mechanics are left to the live Social Force Model; the forecast only needs
     the aggregate density field to be right.

  2. New arrivals are injected during the forward run using a measured arrival
     curve (see `ingress.ArrivalModel`). Real bottlenecks form because people
     keep arriving; a forecast that only drains the current crowd can only ever
     predict the congestion getting better.

The step loop is vectorised over agents so the whole horizon stays well under a
second and can run inside a thread executor without stalling the event loop.
"""

import heapq
import numpy as np
from typing import List, Optional

from .models import (
    Agent, Point, PredictionSnapshot, HazardZone, ZoneDensity,
    RerouteRecommendation, FruinLoS, AlertSeverity
)
from .density import DensityAnalyzer, classify_fruin, local_density
from .pathfinding import a_star_search
from .rng import Rng


# Density (people/m²) at which a zone counts as a "pile-up": Fruin LoS F begins here
# (severe congestion, body contact). The early-warning list flags any zone the
# forecast expects to reach this within the horizon.
PILEUP_DENSITY = 2.17

# 8-connected neighbourhood, used for the exit potential and its descent field.
_NEIGHBOURS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _local_max(field: np.ndarray, r: int = 1) -> np.ndarray:
    """
    Local density, as the one shared measure defined in `density.local_density`.

    This used to take the *maximum* of each cell's neighbourhood while the live
    analyzer reported a per-cell count, so a warning's "density" and the LoS label
    printed beside it were two different quantities. Both now go through the same
    function, so a threshold means the same thing on either side.

    The neighbourhood look is still what makes the comparison robust: forecast agents
    drift a cell or two from where the live ones stand, so a smoothed window agrees
    better than a single cell would.
    """
    return local_density(field, r)


class PredictionEngine:
    """
    Predictive forecast engine for crowd simulation.

    Takes a snapshot of the current simulation state and runs an accelerated
    clone forward to project density distributions 15-30 minutes into the future.
    """

    def __init__(self, width: int, height: int, rng: Rng = None):
        self.width = width
        self.height = height
        self.rng = rng or Rng()
        self.density_analyzer = DensityAnalyzer(width, height)

        # Forecast configuration
        self.horizon_minutes = 15
        self.snapshot_interval_sec = 60   # Output a snapshot every 60 sim seconds
        self.sim_dt = 0.5                 # Coarser than live physics; still sub-metre steps
        self.max_snapshots = 15

        # Crowd repulsion, as a density-gradient pressure term.
        self.pressure_strength = 1.5      # velocity added per unit density gradient
        self.max_pressure = 1.2           # m/s cap, so pressure cannot outrun walking

        # Hard occupancy limit, people per 1 m² cell. Fruin's LoS F begins at
        # 3.5 and jam density for standing crowds is around 5; above that bodies
        # simply do not fit. Without this bound the forecast happily stacks
        # dozens of people on one waypoint and reports impossible densities.
        self.cell_capacity = 5.0

        # Which forward model `forecast()` uses:
        #
        #  "continuum" - density advected along an exit potential with
        #                flux-conserving updates. O(cells) per step, and mass moves
        #                strictly downhill, so flow cannot cycle and nobody can be
        #                permanently trapped.
        #  "particles" - the original per-agent clone. Kept because it is the model
        #                the continuum version replaced, and because it is still
        #                useful to compare against.
        #
        # Evidence for the default: on recorded trajectories the particle clone
        # predicted a population of 552 where the truth plateaued at 351. Agents
        # rejected from a full cell had their velocity zeroed, could never leave,
        # were never removed, and fresh arrivals kept entering. It lost to
        # persistence at every horizon while costing ~1 s per forecast.
        self.spatial_model = "continuum"

        # Hard occupancy limit, people per 1 m2 cell. Used as the jam density in
        # the speed-density relation and as the cap on any single cell.
        self.jam_density = 5.0
        # Weidmann's published pedestrian coefficients.
        self.weidmann_gamma = 1.913
        self.min_speed_factor = 0.06
        # Congestion sensitivity: multiplier on the density fed into the
        # speed-density relation. Fitted against recorded runs rather than picked.
        self.congestion_scale = 1.0
        # Effective turnover, a multiplier on how fast density advances each step.
        self.turnover_scale = 1.0
        # Share of arrivals detouring via a concession. Off by default: measured
        # and found to cost peak-density accuracy, so the machinery is retained but
        # disabled.
        self.detour_fraction = 0.0
        self.dwell_sec = 90.0
        self._potential_cache = {}
        # Per-snapshot density grids from the most recent forecast, paired with the
        # offset each represents. Kept for the accuracy harness, which compares the
        # forecast cell by cell against recorded ground truth.
        self.last_forecast_grids = []

        # Uncertainty is reported instead of being hidden. Each replicate re-runs the
        # horizon with a fresh arrival draw, because arrivals are now Poisson and the
        # run-to-run spread is the dominant unknown. A single forecast is one sample;
        # what an operator actually needs is "how likely is this to go wrong", so
        # `exceedance` gives the per-cell probability of crossing the pile-up
        # threshold within the horizon.
        #
        # 0 disables the replicates (`forecast()` then behaves exactly as before);
        # the live engine sets this from its own configuration.
        self.ensemble_replicates = 0
        #: Per-cell probability in [0, 1] of reaching PILEUP_DENSITY within the horizon.
        self.exceedance = None
        #: Per-cell std-dev of density at the final snapshot, in people/m2.
        self.final_sigma = None

        # Ingress. Set by the simulation engine; None disables arrival injection.
        self.arrival_model = None
        self.gate_positions: List[Point] = []

        # Per-cell forecast summaries filled by forecast(): earliest time (sec after the
        # forecast was made) each cell's neighbourhood reaches PILEUP_DENSITY, and the
        # highest neighbourhood density seen anywhere in the horizon.
        self.forecast_eta: Optional[np.ndarray] = None
        self.forecast_peak: Optional[np.ndarray] = None
        self.forecast_made_at: float = 0.0

        # Legacy ingress estimate, kept for the vision pipeline path.
        self.recent_spawn_counts: List[int] = []
        self.estimated_ingress_rate = 0.5  # agents/sec default

    # ─────────────────────────────────────────────
    # INGRESS
    # ─────────────────────────────────────────────

    def set_arrival_model(self, model, gates: List[Point]):
        """Attach a measured arrival curve used to inject future arrivals."""
        self.arrival_model = model
        self.gate_positions = list(gates)

    def update_ingress_estimate(self, agents_spawned: int):
        """Track recent spawning to estimate future ingress rate."""
        self.recent_spawn_counts.append(agents_spawned)
        if len(self.recent_spawn_counts) > 20:
            self.recent_spawn_counts.pop(0)
        if self.recent_spawn_counts:
            self.estimated_ingress_rate = (
                sum(self.recent_spawn_counts) / max(len(self.recent_spawn_counts), 1)
            )

    # ─────────────────────────────────────────────
    # FORECAST
    # ─────────────────────────────────────────────

    def forecast(
        self,
        agents: List[Agent],
        grid: List[List[str]],
        walls: List,
        gates: List[Point],
        exits: List[Point],
        sim_time_sec: float = 0.0,
    ) -> List[PredictionSnapshot]:
        """Run the configured forward model. See `spatial_model`."""
        if self.ensemble_replicates and self.ensemble_replicates > 1:
            return self._forecast_ensemble(agents, grid, walls, gates, exits, sim_time_sec)
        if self.spatial_model == "particles":
            return self._forecast_particles(agents, grid, walls, gates, exits, sim_time_sec)
        return self._forecast_continuum(agents, grid, walls, gates, exits, sim_time_sec)

    # ─────────────────────────────────────────────
    # CONTINUUM FORECAST
    # ─────────────────────────────────────────────

    def _build_potential(self, free, targets, label="exit"):
        """
        Distance from every free cell to its nearest target, plus a descent field.

        Dijkstra rather than BFS because diagonal steps cost sqrt(2). Cached on the
        grid and target set, which do not change during a forecast.

        The descent field is what makes this model safe: every move strictly
        decreases the potential, so the flow graph is acyclic. No cycle means no
        deadlock, and nobody can be trapped the way the particle clone trapped them.
        """
        key = (
            free.shape,
            hash(free.tobytes()),
            tuple((float(t.x), float(t.y)) for t in targets),
        )
        cached = self._potential_cache.get(label)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]

        h, w = free.shape
        pot = np.full((h, w), np.inf)
        heap = []
        for t in targets:
            x, y = int(t.x), int(t.y)
            if 0 <= y < h and 0 <= x < w and free[y, x]:
                pot[y, x] = 0.0
                heapq.heappush(heap, (0.0, y, x))

        while heap:
            d, y, x = heapq.heappop(heap)
            if d > pot[y, x]:
                continue
            for dy, dx in _NEIGHBOURS8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and free[ny, nx]:
                    nd = d + (1.41421356 if (dy and dx) else 1.0)
                    if nd < pot[ny, nx]:
                        pot[ny, nx] = nd
                        heapq.heappush(heap, (nd, ny, nx))

        best = np.full((h, w), np.inf)
        downhill = np.full((h, w), -1, dtype=np.int32)
        ys = np.arange(h)[:, None]
        xs = np.arange(w)[None, :]
        for dy, dx in _NEIGHBOURS8:
            ty, tx = ys + dy, xs + dx
            inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
            src_y = np.clip(ty, 0, h - 1)
            src_x = np.clip(tx, 0, w - 1)
            nb = np.where(inside, pot[src_y, src_x], np.inf)
            flat = np.where(inside, ty * w + tx, -1).astype(np.int32)
            better = np.isfinite(nb) & (nb < best)
            best = np.where(better, nb, best)
            downhill = np.where(better, flat, downhill).astype(np.int32)

        downhill[~np.isfinite(pot)] = -1
        downhill[pot == 0.0] = -1

        self._potential_cache[label] = (key, pot, downhill)
        return pot, downhill

    def _concession_cells(self, grid):
        """Cells marked as concessions, taken from the grid itself."""
        cells = []
        for y, row in enumerate(grid[: self.height]):
            for x, cell in enumerate(row[: self.width]):
                if cell == "concession":
                    cells.append(Point(x=float(x), y=float(y)))
        return cells

    def _advect(self, mass, carry, flat_downhill, out_frac, n_cells, headroom):
        """
        Transport mass downhill in whole cells, carrying travel progress.

        Moving a *fraction* of a cell's contents each step is a blend, not transport:
        it smears a crowd across the venue and loses where people actually are.
        Each cell accumulates travel progress and hands over its whole content once
        a full cell has been covered, so a crowd stays concentrated as it moves.

        Mass is conserved: exactly what leaves a source arrives at its neighbour, all
        scaled by one factor where a receiver cannot take the whole flow. A receiver's
        capacity is its headroom plus whatever it vacates this step, so the front of
        a queue never blocks its own inflow - the defect that made the particle clone
        accumulate people forever.
        """
        progress = carry + out_frac
        whole = np.floor(progress)
        carry = progress - whole

        moving = (whole >= 1.0) & (flat_downhill >= 0) & (mass > 0.0)
        if not moving.any():
            return mass, carry

        src = np.flatnonzero(moving)
        tgt = flat_downhill[src]
        amt = mass[src]
        out_total = np.bincount(src, weights=amt, minlength=n_cells)
        in_total = np.bincount(tgt, weights=amt, minlength=n_cells)
        accept = headroom + out_total
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(
                in_total > 0,
                np.minimum(1.0, accept / np.maximum(in_total, 1e-12)),
                1.0,
            )
        amt = amt * scale[tgt]
        carry[src] = np.where(scale[tgt] < 1.0, 1.0, carry[src])

        return (
            mass
            - np.bincount(src, weights=amt, minlength=n_cells)
            + np.bincount(tgt, weights=amt, minlength=n_cells)
        ), carry

    def _cell_speed(self, rho_blurred):
        """Weidmann walking speed (m/s) for a density field."""
        r = np.maximum(rho_blurred * self.congestion_scale, 1e-3)
        inv = 1.0 / r - 1.0 / self.jam_density
        factor = 1.0 - np.exp(-self.weidmann_gamma * inv)
        return np.clip(factor, self.min_speed_factor, 1.0) * 1.3

    def _snapshot_from_density(self, raw_density, time_offset):
        """Build a PredictionSnapshot from a density grid."""
        hazards = self.density_analyzer.detect_hazards(raw_density)
        zones = self.density_analyzer.classify_zones(raw_density)
        max_d = float(raw_density.max()) if raw_density.size else 0.0
        critical_count = sum(
            1 for h in hazards
            if h.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)
        )
        return PredictionSnapshot(
            timestamp_offset_sec=round(time_offset, 1),
            hazard_zones=hazards[:20],
            zone_densities=zones[:50],
            max_density=round(max_d, 2),
            critical_zone_count=critical_count,
        )

    def _forecast_continuum(self, agents, grid, walls, gates, exits,
                            sim_time_sec: float = 0.0):
        """
        Advect a density field to the exits. Mass-conserving, O(cells) per step.

        State is people per 1 m2 cell - the same quantity the live analyzer reports,
        so the two are directly comparable.
        """
        live = [a for a in agents if a.status != "arrived"]
        if not live:
            self.reset_forecast()
            return []

        h, w = self.height, self.width
        free = ~self._build_wall_mask(grid)
        _pot, downhill_exit = self._build_potential(free, exits, "exit")
        concession_cells = self._concession_cells(grid)
        if concession_cells and self.detour_fraction > 0:
            _pc, downhill_conc = self._build_potential(
                free, concession_cells, "concession"
            )
        else:
            downhill_conc = None

        n_cells = h * w
        flat_exit = downhill_exit.ravel()
        flat_conc = downhill_conc.ravel() if downhill_conc is not None else None

        rho = np.zeros(n_cells, dtype=np.float64)
        rho_conc = np.zeros(n_cells, dtype=np.float64)
        holding = np.zeros(n_cells, dtype=np.float64)

        cx = np.clip(np.array([a.pos.x for a in live]).astype(np.int32), 0, w - 1)
        cy = np.clip(np.array([a.pos.y for a in live]).astype(np.int32), 0, h - 1)
        np.add.at(rho, cy * w + cx, 1.0)

        exit_cells = [int(e.y) * w + int(e.x) for e in exits
                      if 0 <= int(e.y) < h and 0 <= int(e.x) < w]
        exit_mask = np.zeros(n_cells, dtype=bool)
        exit_mask[exit_cells] = True
        gate_cells = [(int(g.y), int(g.x)) for g in gates
                      if 0 <= int(g.y) < h and 0 <= int(g.x) < w]
        conc_mask = np.zeros(n_cells, dtype=bool)
        if concession_cells and flat_conc is not None:
            conc_mask[[int(c.y) * w + int(c.x) for c in concession_cells]] = True

        total_steps = int((self.horizon_minutes * 60) / self.sim_dt)
        snapshot_every = max(1, int(self.snapshot_interval_sec / self.sim_dt))

        snapshots = []
        grids = []
        backlog = 0.0
        release_rate = min(1.0, self.sim_dt / max(self.dwell_sec, self.sim_dt))
        carry_exit = np.zeros(n_cells, dtype=np.float64)
        carry_conc = np.zeros(n_cells, dtype=np.float64)

        for step in range(total_steps):
            present = rho + rho_conc + holding

            # Inject arrivals at the gates, splitting off detours.
            if self.arrival_model is not None and gate_cells:
                elapsed = (step + 1) * self.sim_dt
                # Drawn, not expected. Using `arrivals_in` here made the forecast a
                # deterministic function of the current state, so every ensemble
                # replicate returned an identical answer and the exceedance map
                # collapsed to 0 or 1 with a sigma of exactly zero. Drawing from the
                # same Poisson process the live engine uses is both more correct and
                # what gives `ensemble_replicates` something to measure.
                expected = self.arrival_model.arrivals_in(
                    sim_time_sec + elapsed, self.sim_dt
                )
                backlog += float(self.rng.gen.poisson(expected)) if expected > 0 else 0.0
                wanting = int(backlog)
                admitted = 0
                for k in range(wanting):
                    gy, gx = gate_cells[k % len(gate_cells)]
                    gi = gy * w + gx
                    if present[gi] >= self.jam_density:
                        continue   # gate jammed; this arrival waits outside
                    if flat_conc is not None and self.rng.random() < self.detour_fraction:
                        rho_conc[gi] += 1.0
                    else:
                        rho[gi] += 1.0
                    present[gi] += 1.0
                    admitted += 1
                backlog -= admitted

            headroom = np.maximum(self.jam_density - present, 0.0)
            out_frac = np.clip(
                self._cell_speed(_blur3(present.reshape(h, w))).ravel()
                * self.sim_dt
                * self.turnover_scale,
                0.0,
                1.0,
            )
            rho, carry_exit = self._advect(
                rho, carry_exit, flat_exit, out_frac, n_cells, headroom
            )
            if flat_conc is not None:
                rho_conc, carry_conc = self._advect(
                    rho_conc, carry_conc, flat_conc, out_frac, n_cells, headroom
                )
                arrived = rho_conc[conc_mask]
                if arrived.any():
                    holding[conc_mask] += arrived
                    rho_conc[conc_mask] = 0.0

            if holding.any():
                released = holding * release_rate
                holding -= released
                rho += released

            rho[exit_mask] = 0.0
            np.clip(rho, 0.0, self.jam_density, out=rho)
            np.clip(rho_conc, 0.0, self.jam_density, out=rho_conc)

            if (step + 1) % snapshot_every == 0:
                time_offset = (step + 1) * self.sim_dt
                field = (rho + rho_conc + holding).reshape(h, w).copy()
                snapshots.append(self._snapshot_from_density(field, time_offset))
                grids.append((time_offset, field))
                if len(snapshots) >= self.max_snapshots:
                    break

        self._build_eta_maps(grids, sim_time_sec)
        self.last_forecast_grids = grids
        return snapshots

    def _run_once(self, agents, grid, walls, gates, exits, sim_time_sec):
        """One pass of the configured forward model."""
        if self.spatial_model == "particles":
            return self._forecast_particles(agents, grid, walls, gates, exits, sim_time_sec)
        return self._forecast_continuum(agents, grid, walls, gates, exits, sim_time_sec)

    def _forecast_ensemble(self, agents, grid, walls, gates, exits, sim_time_sec):
        """
        Run the horizon K times and report the spread as a probability map.

        A single forecast is one sample of a stochastic process, and quoting it as a
        point prediction hides exactly the information an operator needs. What matters
        is not "crush density at 17:43" but "62% chance of crush density here within
        8 minutes" - a number that can be acted on, and one the autopilot can weigh
        rather than treat every forecast as equally certain.

        Replicates differ only in the arrival draw, which is a genuine source of
        uncertainty now that ingress is Poisson. They do not capture error in the
        model's form, so this is narrower than a full predictive distribution: it
        answers "how much does the arrival randomness move the answer", not "is the
        model right".

        The first pass supplies the returned snapshots, so callers and the accuracy
        harness see the same shape of result as before.
        """
        k = max(1, int(self.ensemble_replicates))
        h, w = self.height, self.width

        final_grids = []
        exceed_hits = np.zeros((h, w), dtype=np.float64)
        primary = None

        for i in range(k):
            snapshots = self._run_once(agents, grid, walls, gates, exits, sim_time_sec)
            if i == 0:
                primary = snapshots
                primary_grids = list(self.last_forecast_grids or [])

            grids = list(self.last_forecast_grids or [])
            if not grids:
                continue

            peak = np.zeros((h, w), dtype=np.float64)
            for _t, g in grids:
                np.maximum(peak, g, out=peak)
            exceed_hits += (peak >= PILEUP_DENSITY)
            final_grids.append(grids[-1][1])

        runs = len(final_grids)
        if runs == 0:
            self.exceedance = None
            self.final_sigma = None
            return primary or []

        self.exceedance = exceed_hits / float(runs)
        if runs > 1:
            self.final_sigma = np.std(np.stack(final_grids, axis=0), axis=0)
        else:
            self.final_sigma = np.zeros((h, w), dtype=np.float64)

        # Restore the primary pass: the replicates are a measurement, not the result.
        self.last_forecast_grids = primary_grids if primary is not None else []
        return primary or []

    def _forecast_particles(
        self,
        agents: List[Agent],
        grid: List[List[str]],
        walls: List,
        gates: List[Point],
        exits: List[Point],
        sim_time_sec: float = 0.0,
    ) -> List[PredictionSnapshot]:
        """
        Run a cloned simulation forward and return prediction snapshots.

        Each snapshot contains forecast zone densities, predicted hazard zones,
        the venue-wide maximum density and a count of critical zones.
        """
        live = [a for a in agents if a.status != "arrived"]
        if not live:
            self.reset_forecast()
            return []

        wall_mask = self._build_wall_mask(grid)
        state = _ForecastState.from_agents(live, self.width, self.height)

        # One reusable route per gate, so injected arrivals follow corridors
        # instead of walking through walls. Computed once, not per agent.
        spawn_routes = self._build_spawn_routes(grid, gates, exits)

        total_steps = int((self.horizon_minutes * 60) / self.sim_dt)
        snapshot_every = max(1, int(self.snapshot_interval_sec / self.sim_dt))

        snapshots: List[PredictionSnapshot] = []
        grids = []   # (time_offset_sec, density grid) per snapshot, for the early-warning maps
        # People who have arrived but could not get through a gate yet. They
        # queue outside rather than materialising on top of the crowd already
        # standing in the gate cell.
        backlog = 0.0

        for step in range(total_steps):
            self._step(state, wall_mask)

            if self.arrival_model is not None and spawn_routes:
                elapsed = (step + 1) * self.sim_dt
                expected = self.arrival_model.arrivals_in(
                    sim_time_sec + elapsed, self.sim_dt
                )
                backlog += float(self.rng.gen.poisson(expected)) if expected > 0 else 0.0
                wanting_in = int(backlog)
                if wanting_in:
                    admitted = state.inject(
                        wanting_in, spawn_routes, self.cell_capacity
                    )
                    backlog -= admitted

            state.drop_arrived()

            if (step + 1) % snapshot_every == 0:
                time_offset = (step + 1) * self.sim_dt
                snapshots.append(self._capture_snapshot(state, time_offset))
                grids.append((time_offset, state.density_grid()))
                if len(snapshots) >= self.max_snapshots:
                    break

        self._build_eta_maps(grids, sim_time_sec)
        # Recorded so the accuracy harness can score either model through the
        # same path; the continuum model does the same.
        self.last_forecast_grids = grids
        return snapshots

    # ─────────────────────────────────────────────
    # EARLY WARNINGS (15-minute look-ahead per zone)
    # ─────────────────────────────────────────────

    def reset_forecast(self):
        self.forecast_eta = None
        self.forecast_peak = None
        self.last_forecast_grids = []
        self.exceedance = None
        self.final_sigma = None


    def _build_eta_maps(self, grids, sim_time_sec: float):
        h, w = self.height, self.width
        eta = np.full((h, w), np.inf)
        peak = np.zeros((h, w))
        for t, g in grids:
            lm = _local_max(g)
            peak = np.maximum(peak, lm)
            newly = (lm >= PILEUP_DENSITY) & np.isinf(eta)
            eta[newly] = t
        self.forecast_eta, self.forecast_peak = eta, peak
        self.forecast_made_at = sim_time_sec

    def early_warnings(self, raw_density: np.ndarray, sim_time_sec: float,
                       limit: int = 8, cluster_radius: float = 4.0) -> list:
        """
        Zones expected to pile up (>= PILEUP_DENSITY) within the forecast horizon, soonest
        first. Zones already there report eta_sec = 0. Nearby cells are merged so one
        crowd shows as one warning instead of dozens.
        """
        live_lm = _local_max(raw_density)
        if self.forecast_eta is None:
            eta = np.full(raw_density.shape, np.inf)
            peak = live_lm.copy()
        else:
            age = max(0.0, sim_time_sec - self.forecast_made_at)
            eta = np.maximum(self.forecast_eta - age, 0.0)   # inf stays inf
            peak = np.maximum(self.forecast_peak, live_lm)
        eta = np.where(live_lm >= PILEUP_DENSITY, 0.0, eta)

        ys, xs = np.where(np.isfinite(eta))
        if len(ys) == 0:
            return []
        order = sorted(range(len(ys)), key=lambda i: (eta[ys[i], xs[i]], -peak[ys[i], xs[i]]))

        picked = []
        for i in order:
            y, x = int(ys[i]), int(xs[i])
            if any(((x - px) ** 2 + (y - py) ** 2) ** 0.5 <= cluster_radius for px, py in
                   ((c["cell_x"], c["cell_y"]) for c in picked)):
                continue
            picked.append({
                "cell_x": x,
                "cell_y": y,
                "eta_sec": round(float(eta[y, x]), 0),
                # Bodies do not fit above ~5 people/m² (see cell_capacity); anything higher is
                # a forecast artefact, so never report it as a real density.
                "peak_density": round(min(float(peak[y, x]), self.cell_capacity), 2),
                "current_density": round(float(raw_density[y, x]), 2),
                "local_density": round(float(live_lm[y, x]), 2),   # busiest cell within 1 cell, right now
            })
            if len(picked) >= limit:
                break
        return picked

    def _build_wall_mask(self, grid: List[List[str]]) -> np.ndarray:
        """Boolean (height, width) mask, True where a cell is a wall."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        for y in range(min(self.height, len(grid))):
            row = grid[y]
            for x in range(min(self.width, len(row))):
                if row[x] == "wall":
                    mask[y][x] = True
        return mask

    def _build_spawn_routes(self, grid, gates: List[Point], exits: List[Point]):
        """Precompute one gate -> exit route per gate for injected arrivals."""
        routes = []
        if not gates or not exits:
            return routes
        for i, gate in enumerate(gates):
            target = exits[i % len(exits)]
            start = Point(x=float(int(gate.x)), y=float(int(gate.y)))
            goal = Point(x=float(int(target.x)), y=float(int(target.y)))
            path = a_star_search(grid, start, goal)
            if path:
                routes.append((gate, target, path))
        return routes

    # ─────────────────────────────────────────────
    # PHYSICS STEP (vectorised)
    # ─────────────────────────────────────────────

    def _step(self, s: "_ForecastState", wall_mask: np.ndarray):
        """
        Advance every agent one timestep.

        Desire force pulls each agent toward its next waypoint; pairwise
        repulsion pushes overlapping agents apart so density stays physical.
        """
        n = s.count
        if n == 0:
            return

        pos = s.pos[:n]
        vel = s.vel[:n]

        target = s.current_targets()
        occupancy = s.density_grid()

        # ── Desire force: steer toward the next waypoint ──
        delta = target - pos
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        safe = np.maximum(dist, 1e-6)
        desired = (delta / safe) * s.desired_speed[:n, None]
        new_vel = vel + (desired - vel) * 0.3   # relaxation toward desired velocity

        # ── Repulsion: push down the density gradient, away from crowding ──
        if n > 1:
            new_vel = new_vel + self._density_pressure(occupancy, pos)

        # ── Speed cap ──
        speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
        cap = s.desired_speed[:n, None] * 1.3
        too_fast = (speed > cap).ravel()
        if np.any(too_fast):
            new_vel[too_fast] = (
                new_vel[too_fast] / np.maximum(speed[too_fast], 1e-6) * cap[too_fast]
            )

        # ── Integrate, clamped to the venue bounds ──
        new_pos = pos + new_vel * self.sim_dt
        np.clip(new_pos[:, 0], 0, self.width - 1, out=new_pos[:, 0])
        np.clip(new_pos[:, 1], 0, self.height - 1, out=new_pos[:, 1])

        # ── Wall collision: reject moves that land inside a wall cell ──
        cx = new_pos[:, 0].astype(np.int32)
        cy = new_pos[:, 1].astype(np.int32)
        blocked = wall_mask[cy, cx]

        # ── Wall sliding: an agent whose diagonal move hits a wall corner would otherwise
        #    be frozen there forever (velocity zeroed, same target next step). Let it keep
        #    whichever axis of the move is still free, so it slides around the corner the
        #    way the live simulation's agents do.
        if np.any(blocked):
            src_cx = pos[:, 0].astype(np.int32)
            src_cy = pos[:, 1].astype(np.int32)
            x_only_free = blocked & ~wall_mask[src_cy, cx]     # move in x, stay in y
            y_only_free = blocked & ~x_only_free & ~wall_mask[cy, src_cx]
            if np.any(x_only_free):
                new_pos[x_only_free, 1] = pos[x_only_free, 1]
                new_vel[x_only_free, 1] = 0.0
            if np.any(y_only_free):
                new_pos[y_only_free, 0] = pos[y_only_free, 0]
                new_vel[y_only_free, 0] = 0.0
            slid = x_only_free | y_only_free
            cx = new_pos[:, 0].astype(np.int32)
            cy = new_pos[:, 1].astype(np.int32)
            blocked = wall_mask[cy, cx]

        # ── Capacity: a full cell cannot accept anyone new. Agents already
        #    inside one may still leave, so only cell-crossing moves are tested.
        #
        #    Every agent moves in the same step, so testing each one against the
        #    pre-move occupancy would let a crowd pour into one cell together,
        #    all of them seeing it as empty. Instead the intended arrivals are
        #    counted per cell, and where they exceed the remaining headroom the
        #    surplus is turned away — chosen at random, since within a step
        #    there is no meaningful ordering between them.
        src_x = pos[:, 0].astype(np.int32)
        src_y = pos[:, 1].astype(np.int32)
        crossing = (cx != src_x) | (cy != src_y)

        dest = (cy * self.width + cx).astype(np.int64)
        incoming = np.bincount(dest[crossing], minlength=self.width * self.height)
        headroom = np.maximum(self.cell_capacity - occupancy.ravel(), 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            admit_fraction = np.where(incoming > 0, headroom / incoming, 1.0)
        turn_away_p = np.clip(1.0 - admit_fraction, 0.0, 1.0)

        over_capacity = crossing & (np.random.random(n) < turn_away_p[dest])

        rejected = blocked | over_capacity
        if np.any(rejected):
            new_pos[rejected] = pos[rejected]
            new_vel[rejected] = 0.0

        s.pos[:n] = new_pos
        s.vel[:n] = new_vel

        s.advance_paths()

    def _density_pressure(self, occupancy: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """
        Velocity contribution pushing each agent away from local crowding.

        The occupancy grid is blurred so the gradient is smooth rather than
        stepping between whole-person counts, then each agent samples the
        negative gradient at its own cell. Cost is independent of how many
        agent pairs exist, which is what keeps the horizon affordable.
        """
        field = _blur3(occupancy)
        grad_y, grad_x = np.gradient(field)

        cx = np.clip(pos[:, 0].astype(np.int32), 0, self.width - 1)
        cy = np.clip(pos[:, 1].astype(np.int32), 0, self.height - 1)

        push = np.empty_like(pos)
        push[:, 0] = -grad_x[cy, cx]
        push[:, 1] = -grad_y[cy, cx]
        push *= self.pressure_strength

        # Cap so a dense cell cannot fling agents across the venue
        mag = np.linalg.norm(push, axis=1, keepdims=True)
        over = (mag > self.max_pressure).ravel()
        if np.any(over):
            push[over] = push[over] / np.maximum(mag[over], 1e-6) * self.max_pressure
        return push

    # ─────────────────────────────────────────────
    # SNAPSHOT
    # ─────────────────────────────────────────────

    def _capture_snapshot(self, s: "_ForecastState", time_offset: float) -> PredictionSnapshot:
        """Capture density state at a forecast timestep."""
        raw_density = s.density_grid()
        hazards = self.density_analyzer.detect_hazards(raw_density)
        zones = self.density_analyzer.classify_zones(raw_density)

        max_d = float(raw_density.max()) if raw_density.size > 0 else 0.0
        critical_count = sum(
            1 for h in hazards
            if h.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)
        )

        return PredictionSnapshot(
            timestamp_offset_sec=round(time_offset, 1),
            hazard_zones=hazards[:20],   # Cap to prevent massive payloads
            zone_densities=zones[:50],
            max_density=round(max_d, 2),
            critical_zone_count=critical_count,
        )


def _blur3(field: np.ndarray) -> np.ndarray:
    """
    Cheap separable 3-tap blur, used to smooth the occupancy grid before
    differentiating it. Without this the gradient steps between whole-person
    counts and agents jitter instead of flowing.
    """
    padded = np.pad(field, 1, mode="edge")
    horizontal = padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]
    return (horizontal[:-2] + horizontal[1:-1] + horizontal[2:]) / 9.0


class _ForecastState:
    """
    Struct-of-arrays agent state for the forecast run.

    Everything the step loop touches — positions, velocities, waypoint queues —
    lives in numpy arrays so no per-agent Python loop runs inside the horizon.
    Waypoint queues are padded to a fixed width rather than kept ragged, which
    is what allows the target lookup and arrival test to be vectorised.

    Arrays are over-allocated and tracked by `count`, so injected arrivals do
    not reallocate on every step.
    """

    GROWTH = 512
    MAX_PATH = 96   # A* paths on a 40x30 grid stay well under this

    def __init__(self, capacity: int, width: int, height: int):
        self.width = width
        self.height = height
        self.count = 0

        self.pos = np.zeros((capacity, 2), dtype=np.float64)
        self.vel = np.zeros((capacity, 2), dtype=np.float64)
        self.goal = np.zeros((capacity, 2), dtype=np.float64)
        self.desired_speed = np.full(capacity, 1.3, dtype=np.float64)
        self.radius = np.full(capacity, 0.3, dtype=np.float64)

        self.path_xy = np.zeros((capacity, self.MAX_PATH, 2), dtype=np.float32)
        self.path_len = np.zeros(capacity, dtype=np.int32)
        self.path_idx = np.zeros(capacity, dtype=np.int32)

    @classmethod
    def from_agents(cls, agents: List[Agent], width: int, height: int) -> "_ForecastState":
        s = cls(len(agents) + cls.GROWTH, width, height)
        for a in agents:
            s._append(
                pos=(a.pos.x, a.pos.y),
                vel=(a.vel.x, a.vel.y),
                goal=(a.goal.x, a.goal.y),
                desired_speed=a.desired_speed,
                radius=a.radius,
                path=[(p.x, p.y) for p in a.path],
            )
        return s

    def _ensure_capacity(self, extra: int):
        if self.count + extra <= self.pos.shape[0]:
            return
        new_cap = self.count + extra + self.GROWTH
        n = self.count
        for name in ("pos", "vel", "goal"):
            arr = getattr(self, name)
            grown = np.zeros((new_cap, 2), dtype=arr.dtype)
            grown[:n] = arr[:n]
            setattr(self, name, grown)
        for name, fill in (("desired_speed", 1.3), ("radius", 0.3)):
            arr = getattr(self, name)
            grown = np.full(new_cap, fill, dtype=arr.dtype)
            grown[:n] = arr[:n]
            setattr(self, name, grown)
        grown_xy = np.zeros((new_cap, self.MAX_PATH, 2), dtype=np.float32)
        grown_xy[:n] = self.path_xy[:n]
        self.path_xy = grown_xy
        for name in ("path_len", "path_idx"):
            arr = getattr(self, name)
            grown = np.zeros(new_cap, dtype=np.int32)
            grown[:n] = arr[:n]
            setattr(self, name, grown)

    def _append(self, pos, vel, goal, desired_speed, radius, path):
        self._ensure_capacity(1)
        i = self.count
        self.pos[i] = pos
        self.vel[i] = vel
        self.goal[i] = goal
        self.desired_speed[i] = desired_speed
        self.radius[i] = radius

        length = min(len(path), self.MAX_PATH)
        if length:
            self.path_xy[i, :length] = path[:length]
        self.path_len[i] = length
        self.path_idx[i] = 0
        self.count += 1

    def inject(self, n: int, spawn_routes, cell_capacity: float) -> int:
        """
        Admit up to `n` new arrivals across the gate routes.

        A gate cell that is already at capacity cannot take anyone: those people
        are still queueing outside. Returns how many actually got in, so the
        caller can carry the rest forward instead of losing them.
        """
        self._ensure_capacity(n)
        occupancy = self.density_grid()
        admitted = 0

        for k in range(n):
            gate, target, path = spawn_routes[k % len(spawn_routes)]
            gx = int(np.clip(gate.x, 0, self.width - 1))
            gy = int(np.clip(gate.y, 0, self.height - 1))
            if occupancy[gy, gx] >= cell_capacity:
                continue   # gate is jammed; this arrival waits outside

            occupancy[gy, gx] += 1.0
            self._append(
                pos=(gate.x + np.random.uniform(0.1, 0.9),
                     gate.y + np.random.uniform(-0.4, 0.4)),
                vel=(0.0, 0.0),
                goal=(target.x, target.y),
                desired_speed=float(np.random.uniform(1.0, 1.4)),
                radius=0.3,
                path=[(p.x, p.y) for p in path],
            )
            admitted += 1

        return admitted

    def _waypoints(self):
        """(has_waypoint, waypoint_centres) for every live agent."""
        n = self.count
        idx = self.path_idx[:n]
        has = idx < self.path_len[:n]
        safe = np.where(has, idx, 0)
        wp = self.path_xy[np.arange(n), safe].astype(np.float64) + 0.5
        return has, wp

    def current_targets(self) -> np.ndarray:
        """Next waypoint per agent, falling back to the final goal."""
        n = self.count
        has, wp = self._waypoints()
        return np.where(has[:, None], wp, self.goal[:n] + 0.5)

    def advance_paths(self):
        """Advance past waypoints that have been reached."""
        n = self.count
        if n == 0:
            return
        has, wp = self._waypoints()
        delta = self.pos[:n] - wp
        reached = has & (np.einsum("ij,ij->i", delta, delta) < 0.5)
        self.path_idx[:n] += reached

    def drop_arrived(self):
        """Remove agents that reached their goal, compacting the arrays."""
        n = self.count
        if n == 0:
            return
        delta = self.pos[:n] - (self.goal[:n] + 0.5)
        keep = np.einsum("ij,ij->i", delta, delta) >= 1.5
        if np.all(keep):
            return

        kept = np.flatnonzero(keep)
        m = kept.size
        for name in ("pos", "vel", "goal", "desired_speed", "radius",
                     "path_xy", "path_len", "path_idx"):
            arr = getattr(self, name)
            arr[:m] = arr[kept]
        self.count = m

    def density_grid(self) -> np.ndarray:
        """People-per-cell grid. Cells are 1 m², so this is people/m²."""
        grid = np.zeros((self.height, self.width), dtype=np.float64)
        n = self.count
        if n == 0:
            return grid
        cx = np.clip(self.pos[:n, 0].astype(np.int32), 0, self.width - 1)
        cy = np.clip(self.pos[:n, 1].astype(np.int32), 0, self.height - 1)
        np.add.at(grid, (cy, cx), 1.0)
        return grid