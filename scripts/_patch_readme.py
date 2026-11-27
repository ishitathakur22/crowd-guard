"""Extend README.md with the engineering work, preserving existing content and structure."""

import pathlib

path = pathlib.Path("README.md")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new, 1)


# ── 1. Table of contents ────────────────────────────────────────────────
sub(
    "- [Usage](#usage)",
    "- [Usage](#usage)\n- [Testing & Benchmarks](#testing--benchmarks)",
    "toc",
)

# ── 2. Simulation features ──────────────────────────────────────────────
sub(
    "- **Density Heatmap & Analytics:** Fruin Level-of-Service classification and dynamic directional updates.",
    "- **Density Heatmap & Analytics:** Fruin Level-of-Service classification and dynamic directional updates.\n"
    "- **Vectorised Physics:** Cell-list neighbour search over flat NumPy pair arrays. A 1,000-agent crowd now simulates at 2.5× real time; the scalar implementation fell behind its own clock above ~200 agents.\n"
    "- **Seeded Reproducibility:** One seed (`CROWD_SEED`) drives arrivals, spawning, physics and forecasting through derived substreams, so any run can be reproduced exactly and a benchmark change always reflects the code rather than the dice.",
    "sim features",
)

# ── 3. Forecasting features ─────────────────────────────────────────────
sub(
    "- **15-Minute Predictive Forecasting:** Automated bottleneck detection with density-weighted A* rerouting.",
    "- **15-Minute Predictive Forecasting:** Mass-conserving density advection along an exit-potential field, so flow cannot cycle and nobody can be trapped.\n"
    "- **Flow-Field Rerouting:** One multi-source Dijkstra per venue replaces up to four A\\* searches *per agent*. Rerouting went from ~590 ms to ~1.2 ms per tick at 200 agents.\n"
    "- **Stochastic Arrivals:** Gate ingress follows a seeded Poisson process (variance/mean = 1.006), so arrival surges — the thing that actually forms bottlenecks — genuinely occur.",
    "forecast features",
)

# ── 4. Tech stack ──────────────────────────────────────────────────────
sub(
    "| **AI Models** | YOLOv8 (Nano/Small/Medium), MiDaS v2.1, SmolVLM2-2.2B-Instruct (GGUF) |",
    "| **AI Models** | YOLOv8 (Nano/Small/Medium), MiDaS v2.1, SmolVLM2-2.2B-Instruct (GGUF) |\n"
    "| **Vision Tracking** | ByteTrack (via ultralytics `model.track`) |\n"
    "| **Testing** | pytest — 117 tests across 10 files |",
    "tech stack",
)

# ── 5. New Testing & Benchmarks section, before Deploy ──────────────────
sub(
    "## Deploy on Render",
    """## Testing & Benchmarks

The engine is measured rather than assumed. Every optimisation in this repository is
justified by a number that can be regenerated, and every regression is recorded.

### Test suite

```cmd
.venv\\Scripts\\python.exe -m pytest tests -q
```

**117 tests across 10 files**, covering the properties that are expensive to
rediscover by hand:

| File | Guards |
| :--- | :--- |
| `test_reproducibility.py` | Same seed → bit-identical state; different seeds diverge |
| `test_physics_parity.py` | Vectorised SFM is a pure function and matches a scalar reference to 1e-5 |
| `test_models.py` | No shared mutable default (a shared `Point` once rewrote every agent's velocity) |
| `test_forecast.py` | Mass conservation, jam-density cap, snapshot counts, finiteness |
| `test_density.py` | Fruin boundaries, the unified local-density measure, rise-rate units |
| `test_pathfinding.py` | A\\* wall avoidance, flow field strictly descends |
| `test_ingress.py` | Poisson dispersion, rate integration, curve normalisation |
| `test_gates.py` | Admission budget, queue accounting, diverts, no dead queues |
| `test_snapshot_and_clock.py` | Concurrent reads are safe; tick period is independent of tick work |
| `test_vision_aggregator.py` | Detection de-duplication, homography, out-of-plan handling |

### Latency

```cmd
.venv\\Scripts\\python.exe scripts\\bench.py --agents 50 200 500 1000
```

A tick represents 0.5 simulated seconds, so holding 1× speed means finishing a tick
within **500 ms**.

| agents | tick (before) | tick (now) | headroom |
| :--- | :--- | :--- | :--- |
| 50 | 65 ms | **8.3 ms** | 60× |
| 200 | 609 ms | **17.0 ms** | 29× |
| 500 | 2,051 ms | **57.0 ms** | 8.8× |
| 1,000 | 6,633 ms | **200 ms** | 2.5× |

### Forecast accuracy

```cmd
.venv\\Scripts\\python.exe scripts\\record_trajectory.py --cuts 60 120 180 --horizon-minutes 15
.venv\\Scripts\\python.exe scripts\\eval_forecast.py
```

The recorder captures what the simulation actually did; the evaluator rewinds to a cut
point, forecasts forward, and scores every snapshot against that recorded truth —
always against a **persistence baseline** ("density now, unchanged"), because a
forecast that cannot beat standing still is decoration.

| metric | forecast | persistence |
| :--- | :--- | :--- |
| MAE | **0.063** | 0.091 |
| busy-cell MAE | 1.028 | **0.762** |
| field correlation | 0.048 | **0.183** |

The forecast wins on aggregate error but **loses on where the crowd actually is**.
For a safety warning that second question is the one that matters, and it is not yet
solved. `benchmarks/README.md` records the full history, including five approaches
that measured worse and were reverted.

### Diagnostics

| Script | Purpose |
| :--- | :--- |
| `scripts/measure_speed.py` | Mean crowd walking speed vs a real pedestrian |
| `scripts/tune_sfm.py` | Sweep the SFM repulsion range against a physical speed target |
| `scripts/fit_forecast.py` | Fit forecast parameters against recorded runs |
| `scripts/diagnose_forecast.py` | Ablate forecast components to find what is wrong |

---

## Deploy on Render""",
    "testing section",
)

# ── 6. Project structure ────────────────────────────────────────────────
sub(
    """crowd-guard/
├── backend/          # FastAPI app, simulation engine, physics, prediction, autopilot
├── frontend/         # Dashboard UI (HTML/CSS/JS, Canvas rendering)
├── scripts/          # One-off data-prep scripts (e.g. arrival profile builder)
└── weights/          # Downloaded AI model weights (populated by download_models.py)""",
    """crowd-guard/
├── backend/          # FastAPI app, simulation engine, physics, prediction, autopilot
│   ├── rng.py        # Seeded randomness with per-subsystem substreams
│   ├── physics.py    # Vectorised Social Force Model
│   ├── physics_scalar.py  # Scalar reference used by the parity tests
│   └── vision/       # YOLO + depth + VLM workers, aggregator, model manager
├── frontend/         # Dashboard UI (HTML/CSS/JS, Canvas rendering)
├── scripts/          # Benchmarks, ground-truth recorder, forecast evaluator, diagnostics
├── tests/            # 117 pytest tests
├── benchmarks/       # Written baseline record (generated artefacts are ignored)
└── weights/          # Downloaded AI model weights (populated by download_models.py)""",
    "project structure",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
