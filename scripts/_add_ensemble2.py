"""Insert the ensemble dispatcher and clear exceedance in reset_forecast."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label, count=1):
    global text
    n = text.count(old)
    report.append((label, n))
    if n:
        text = text.replace(old, new, count)


sub(
    "    def reset_forecast(self):\n        self.forecast_eta = None\n        self.forecast_peak = None\n        self.last_forecast_grids = []",
    "    def reset_forecast(self):\n        self.forecast_eta = None\n        self.forecast_peak = None\n        self.last_forecast_grids = []\n        self.exceedance = None\n        self.final_sigma = None",
    "reset exceedance",
)

# Dispatcher + ensemble, inserted before the particle model.
sub(
    '    def _forecast_particles(',
    '''    def _run_once(self, agents, grid, walls, gates, exits, sim_time_sec):
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

    def _forecast_particles(''',
    "ensemble dispatcher",
)

# Route forecast() through the ensemble when configured.
sub(
    '        """Run the configured forward model. See `spatial_model` for the choice."""\n        if self.spatial_model == "particles":',
    '        """Run the configured forward model. See `spatial_model` for the choice."""\n'
    "        if self.ensemble_replicates and self.ensemble_replicates > 1:\n"
    "            return self._forecast_ensemble(agents, grid, walls, gates, exits, sim_time_sec)\n"
    '        if self.spatial_model == "particles":',
    "forecast dispatch",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
