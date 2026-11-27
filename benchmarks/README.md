# Measurement baselines

Generated artefacts live here and are gitignored. This file records what the
harness measures, how to regenerate it, and what the numbers were at each stage of
the optimisation work — including the changes that made things worse and were
reverted.

Everything is seeded (`CROWD_SEED`, default `1337`), so a row is reproducible on any
machine, and a change in a number means a change in the code rather than a different
draw of the dice.

## Regenerating

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy pydantic pytest

:: correctness
.venv\Scripts\python.exe -m pytest tests -q

:: latency, per stage, across crowd sizes
.venv\Scripts\python.exe scripts\bench.py --agents 50 200 500 1000 --reps 10

:: ground truth, then accuracy against it
.venv\Scripts\python.exe scripts\record_trajectory.py --cuts 60 120 180 --horizon-minutes 15
.venv\Scripts\python.exe scripts\eval_forecast.py
.venv\Scripts\python.exe scripts\eval_forecast.py --model particles --label part

:: diagnosis, when numbers look wrong
.venv\Scripts\python.exe scripts\measure_speed.py
.venv\Scripts\python.exe scripts\tune_sfm.py
.venv\Scripts\python.exe scripts\fit_forecast.py --param congestion
```

A tick represents 0.5 simulated seconds, so holding 1x speed means finishing a tick
within **500 ms** of real time. "Headroom" is that budget divided by the measured
tick.

Run the recorder in the background and never alongside `bench.py`, or the latency
numbers will include contention.

---

## 1. Latency

Seed 1337, Python 3.12.10, Windows.

### Before optimisation

| agents | SFM/tick | reroute | forecast 15 min | `get_state()` | **full tick** | headroom |
|---|---|---|---|---|---|---|
| 50 | 44 ms | 143 ms | ~1956 ms | 1.1 ms | **65 ms** | 7.7x |
| 200 | 592 ms | 716 ms | 1133 ms | 3.2 ms | **609 ms** | 0.82x |
| 500 | 1803 ms | 455 ms | 1180 ms | 12.0 ms | **2051 ms** | 0.24x |
| 1000 | 7807 ms | 502 ms | 1376 ms | 22.9 ms | **6633 ms** | 0.08x |

### After optimisation

| agents | SFM/tick | reroute (cached) | forecast 15 min | `get_state()` | **full tick** | headroom |
|---|---|---|---|---|---|---|
| 50 | 3.28 ms | 0.25 ms | 178 ms | 0.42 ms | **8.3 ms** | 60.4x |
| 200 | 11.51 ms | 1.15 ms | 181 ms | 1.79 ms | **17.0 ms** | 29.4x |
| 500 | 56.32 ms | 2.35 ms | 178 ms | 5.42 ms | **57.0 ms** | 8.8x |
| 1000 | 226.21 ms | 3.89 ms | 173 ms | 19.21 ms | **199.6 ms** | 2.5x |

The engine now holds real time at 1 000 agents. Before, it fell behind its own clock
above roughly 200 — which is below the 220-agent prefill the dashboard does on first
connect.

What each change bought, at 200 agents:

| stage | before | after | factor |
|---|---|---|---|
| Physics, all substeps/tick | 313 ms | **11.5 ms** | ~27x |
| Rerouting | 591 ms | **1.15 ms** | ~500x |
| Forecast, 15 min horizon | ~1133 ms | **181 ms** | ~6x |

The flow-field rebuild costs 6.84 ms at 200 agents and happens at most once per
`REROUTE_COOLDOWN_TICKS`; between rebuilds every agent reads the same field, which
is why the cached path is 1.15 ms.

### Remaining hot spots

- **Physics still scales quadratically in crowd density.** 200 → 1000 agents takes
  a substep from 2.4 ms to 48.9 ms, ~20x for a 5x crowd. The cell list bounds the
  neighbour search to a fixed window, but the number of agents inside that window
  rises with density.
- **The forecast is ~175 ms regardless of crowd size**, because it is dominated by
  the fixed 1 800 timesteps of the horizon, not by the number of people.
- `get_state()` grows linearly (0.42 → 19.2 ms), since it serialises every agent.

---

## 2. A physics bug found while validating the forecast

The forecast kept draining the venue several times faster than the recorded truth,
and no re-parameterisation closed the gap without making the density field worse.
That pointed at the truth rather than the forecast, so `scripts/measure_speed.py`
measured the crowd directly:

```
mean walking speed     0.423 m/s
real pedestrian        1.300 m/s       -> 0.33x
```

**The simulated crowd walked at one third of real pedestrian speed.**
`backend/physics.py` used `B = 0.3` in `A*exp(-d/B)` and described the wide value as
giving "gentler falloff". That is backwards: a wider `B` makes repulsion reach
further and act on more neighbours, so braking dominated the goal-seeking term and
the crowd jammed instead of flowing.

`scripts/tune_sfm.py` swept it at fixed crowd size:

| B (m) | mean crowd speed | agents that exited | in-wall samples |
|---|---|---|---|
| 0.30 | 0.460 m/s | 37 | 0 |
| 0.20 | 0.663 m/s | 101 | 0 |
| 0.15 | 0.840 m/s | 156 | 0 |
| **0.10** | **0.951 m/s** | **157** | **0** |
| 0.08 | 0.995 m/s | 168 | 0 |

`B = 0.10` sits in a plausible band for a busy venue (0.9–1.1 m/s) and keeps the
widest avoidance range of the options that do. Measured crowd speed is now
**0.995 m/s**.

This matters beyond tidiness: a twin whose crowd moves at 0.42 m/s mis-estimates
crossing and evacuation times by roughly 3x.

---

## 3. Forecast accuracy

Truth is the seeded live simulation, recorded per tick. The forecast is rewound to
each cut, run forward, and compared cell by cell. `persistence` means "density now,
unchanged" — the baseline any forecast must beat.

Breach threshold is `PILEUP_DENSITY = 2.17` p/m^2 (Fruin LoS E).

Three metrics, because they disagree and each says something different:

- **MAE** — mean absolute error over all cells. Easy to game: predicting nothing
  scores well, because most cells are near-empty.
- **busy MAE** — error restricted to cells that are genuinely occupied. This is where
  a warning is either right or wrong.
- **corr** — Pearson correlation of the two fields. Scale-free, so it cannot be
  improved by predicting a flat zero.

### Before: particle clone, over-damped physics

| scenario | horizon | MAE | persist MAE | peak error | false-alarm cells | real breaches |
|---|---|---|---|---|---|---|
| default | 15 min | 0.156 | **0.154** | 2.67 | 34-44 | 0 |
| surge | 8 min | 0.429 | **0.279** | 3.0-5.0 | 159-182 | 0 |

The forecast lost to persistence in both scenarios, over-predicted congestion by up
to 5 p/m^2, and flagged 13-15% of the venue as crush density when the truth never
exceeded ~1.5. It also cost ~1 s per horizon.

### After: continuum advection, corrected physics

Recorded after arrival stochasticity was introduced (see section 4), so these are
the current numbers.

| model | MAE | persist MAE | busy MAE | persist busy | corr | persist corr | cost |
|---|---|---|---|---|---|---|---|
| **continuum** | **0.063** | 0.091 | 1.028 | **0.762** | 0.048 | **0.183** | **199 ms** |
| particles | 0.104 | 0.091 | 0.941 | **0.762** | 0.064 | **0.183** | 626 ms |

The continuum model **beats persistence on MAE** and the particle clone still loses.
Population bias tells the story: particles gain 22-57 people the arrival curve never
supplied (they accumulate), while continuum runs 15-38 short.

### What is still not good enough

**busy-cell MAE and correlation both favour persistence.** On cells that actually
contain people, copying the present is still the better answer, and correlation
0.048 against 0.183 means the forecast knows roughly how much crowd to expect but
not well enough *where*. For a safety warning, that is the question that matters.

This has been attempted five ways and every one measured worse; see section 5.

### Metrics that remain unexercised

Breach precision/recall/F1 and **bottleneck lead time are not measurable**: the truth
never reaches LoS E in either scenario, so there is nothing to detect. Lead time —
"how many seconds of warning did we get" — is the number that should matter most and
it is currently undefined.

Fixing that needs a scenario that genuinely reaches LoS E: a tighter venue, a forced
gate closure at high attendance, or a much higher attendance figure. Note the last
one is now feasible — 1 000 agents costs 200 ms per tick rather than 6.6 s.

---

## 4. Correctness fixes, with evidence

| Defect | Symptom | Fix |
|---|---|---|
| Over-damped physics (`B=0.3`) | crowd walked 0.42 m/s, jammed, only 37/150 reached an exit | `B=0.10`, fitted to a physical speed target |
| Forecast deadlock | predicted population 552 where truth plateaued at 351; agents rejected from full cells had velocity zeroed and could never leave | continuum advection, acyclic by construction |
| Shared `Agent.vel` default | pydantic shared one `Point` across every agent; mutating velocity in place rewrote the whole crowd | `Field(default_factory=...)` |
| Time-to-choke units | per-tick slope consumed as per-second, inflating every report ~2.5x | divide by elapsed simulated seconds |
| Split density measure | zones/hazards used per-cell counts, warnings used a neighbourhood max; same cell showed two numbers | one `density.local_density` everywhere |
| Closed-gate dead queue | arrivals queued behind gates that never open; those people vanished | assign only to open gates; displaced queues walk to one |
| `get_state` race | event loop iterated the agent list while the executor mutated it, unlocked | tick publishes an immutable snapshot under a lock |
| Dishonest tick clock | slept *after* the work, so 10x was really 7.5x | monotonic deadline scheduling |
| Deterministic arrivals | accumulator with exactly zero variance, so arrival surges were structurally impossible | seeded Poisson process |

### Poisson arrivals: measured effect

`draw_arrivals` used to carry a fractional remainder and emit `floor(carry)`, which
gave an identical sequence on every run and **zero variance**. Pedestrian arrivals
are a counting process with variance equal to the mean, so the old code understated
every queueing statistic and made a surge impossible to generate - and therefore
impossible to test whether the forecast would have predicted one.

Measured over 20 000 ticks at 100 000 attendance:

| statistic | value |
|---|---|
| mean arrivals/tick | 1.959 |
| variance | 1.971 |
| **variance / mean** | **1.006** (Poisson expects 1.0; the old code gave 0.0) |
| observed range | 0-10 per tick |
| ticks above 3x the mean | 1.53% |

The expectation is unchanged, so attendance totals and the arrival curve still mean
what they say; only dispersion was added. Draws stay seeded, so runs remain
reproducible.

---

## 5. Experiments that failed, and why

Recorded so a later session does not repeat them.

- **Ablating the capacity model** to test a gridlock hypothesis: wrong. Removing it
  made results *worse* (MAE 0.421 → 0.434, peak error 6.38). Removing arrival
  injection instead cut MAE to 0.262 and false alarms from 635 to 1, which identified
  accumulation as the real fault.
- **Weidmann speed-density alone** (no advection): structurally right, removed the
  deadlock, but alone made results worse (MAE 0.463, peak 9.5) because inflow still
  exceeded drain.
- **Concession detours with a dwell buffer**: cost peak accuracy badly (0.40 → 3.00)
  by parking mass on two concession cells. Off by default; machinery retained.
- **Whole-cell transport**: aimed at fixing smearing, made it worse (corr 0.074).
- **Splitting flow across equivalent routes**: corr 0.035 → 0.023. Reverted.
- **Parameter fits** for `turnover_scale` and `congestion_scale`: both preferred 1.0.
  Fitted, not guessed, and reported by `scripts/fit_forecast.py`.

The common thread: the forecast over-drains relative to the live simulation, and
every attempt to slow it uniformly concentrates the crowd into fewer cells and costs
peak accuracy. Closing the location gap needs flow that spreads across parallel
routes the way a real crowd does. That is unsolved.

---

## 6. Limitations, stated plainly

- **Truth is the simulation, not the world.** These scores measure whether the cheap
  forecast reproduces the Social Force Model. They do **not** establish that the model
  matches a real crowd. Doing that needs recorded footage with annotated ground-plane
  positions — which depends on the vision pipeline.
- **The physics fix is calibrated to a physical target** (plausible walking speed),
  not fitted to real trajectories. Better grounded than the previous constants, still
  not validated.
- **Two synthetic scenarios**, from Melbourne footpath sensors. No real venue.
- **Arrivals are now stochastic (Poisson).** More realistic, but a recorded
  trajectory is now one sample rather than the mean: two seeds give different crowd
  sizes at the same simulated time. Accuracy numbers are conditional on
  `CROWD_SEED=1337` and are only comparable between runs sharing it.
