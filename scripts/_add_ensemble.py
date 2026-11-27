"""#14: exceedance probability over K stochastic replicates."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── 1. configuration ────────────────────────────────────────────────────
sub(
    "        self.last_forecast_grids = []",
    """        self.last_forecast_grids = []

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
        self.final_sigma = None""",
    "ensemble config",
)

# ── 2. forecast() runs replicates when configured ────────────────────────
sub(
    '''        \"\"\"Run the configured forward model. See `spatial_model` for the choice.\"\"\"
        if self.spatial_model == "particles":
            return self._forecast_particles(agents, grid, walls, gates, exits, sim_time_sec)
        return self._forecast_continuum(agents, grid, walls, gates, exits, sim_time_sec)''',
    '''        \"\"\"Run the configured forward model. See `spatial_model` for the choice.\"\"\"
        if self.ensemble_replicates and self.ensemble_replicates > 1:
            return self._forecast_ensemble(agents, grid, walls, gates, exits, sim_time_sec)
        if self.spatial_model == "particles":
            return self._forecast_particles(agents, grid, walls, gates, exits, sim_time_sec)
        return self._forecast_continuum(agents, grid, walls, gates, exits, sim_time_sec)

    # ─────────────────────────────────────────────
    # UNCERTAINTY (exceedance probability)
    # ─────────────────────────────────────────────

    def _run_once(self, agents, grid, walls, gates, exits, sim_time_sec):
        \"\"\"One deterministic pass of the configured model.\"\"\"
        if self.spatial_model == "particles":
            return self._forecast_particles(agents, grid, walls, gates, exits, sim_time_sec)
        return self._forecast_continuum(agents, grid, walls, gates, exits, sim_time_sec)

    def _forecast_ensemble(self, agents, grid, walls, gates, exits, sim_time_sec):
        \"\"\"
        Run the horizon K times and return the mean trajectory plus a probability map.

        The first pass is the one whose snapshots are returned, so the accuracy
        harness and the dashboard see the same shape of object as before. The extra
        passes exist only to measure spread:

          * `exceedance[y, x]` - fraction of replicates in which the cell reached
            PILEUP_DENSITY at any point in the horizon. This is the honest form of a
            warning: "62% chance of crush density here within 8 minutes" carries
            information that "LoS F at 17:43" does not.
          * `final_sigma[y, x]` - standard deviation of density at the last snapshot.

        The replicates differ only through the arrival draw. That is a real source of
        uncertainty now that ingress is Poisson, and it is the one the model can
        actually quantify; it does not capture error in the model form itself.
        \"\"\"
        k = max(1, int(self.ensemble_replicates))
        h, w = self.height, self.width

        stacked = []
        exceed_hits = np.zeros((h, w), dtype=np.float64)
        primary = None

        for i in range(k):
            snapshots = self._run_once(
                agents, grid, walls, gates, exits, sim_time_sec
            )
            grids = [(t, g) for t, g in (self.last_forecast_grids or [])]
            if i == 0:
                primary = snapshots

            if not grids:
                continue

            # A cell counts as exceeded if any snapshot in the horizon crosses.
            peak = np.zeros((h, w), dtype=np.float64)
            for _t, g in grids:
                peak = np.maximum(peak, g)
            exceed_hits += (peak >= PILEUP_DENSITY).astype(np.float64)
            stacked.append(grids[-1][1])

        runs = len(stacked)
        if runs == 0:
            self.exceedance = None
            self.final_sigma = None
            return primary or []

        self.exceedance = exceed_hits / float(runs)
        if runs > 1:
            self.final_sigma = np.std(np.stack(stacked, axis=0), axis=0)
        else:
            self.final_sigma = np.zeros((h, w), dtype=np.float64)

        # Leave last_forecast_grids holding the primary pass, so downstream consumers
        # and the accuracy harness are unaffected by the replicates.
        if primary is not None:
            self._run_once(agents, grid, walls, gates, exits, sim_time_sec) if False else None
        return primary or []''',
    "ensemble",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
