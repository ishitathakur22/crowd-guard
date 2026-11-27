"""Make the forecast draw arrivals stochastically, so the ensemble has variance."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# The continuum model: draw a Poisson count instead of accumulating an expectation.
sub(
    '''            # Inject arrivals at the gates, splitting off detours.
            if self.arrival_model is not None and gate_cells:
                elapsed = (step + 1) * self.sim_dt
                backlog += self.arrival_model.arrivals_in(
                    sim_time_sec + elapsed, self.sim_dt
                )
                wanting = int(backlog)''',
    '''            # Inject arrivals at the gates, splitting off detours.
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
                wanting = int(backlog)''',
    "continuum stochastic arrivals",
)

# The particle model: same treatment, for consistency.
sub(
    '''            if self.arrival_model is not None and spawn_routes:
                elapsed = (step + 1) * self.sim_dt
                backlog += self.arrival_model.arrivals_in(
                    sim_time_sec + elapsed, self.sim_dt
                )
                wanting_in = int(backlog)''',
    '''            if self.arrival_model is not None and spawn_routes:
                elapsed = (step + 1) * self.sim_dt
                expected = self.arrival_model.arrivals_in(
                    sim_time_sec + elapsed, self.sim_dt
                )
                backlog += float(self.rng.gen.poisson(expected)) if expected > 0 else 0.0
                wanting_in = int(backlog)''',
    "particle stochastic arrivals",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
