# Crowd Guard

Real-time crowd simulation, AI-powered vision pipeline, bottleneck forecasting, and automated rerouting engine. A predictive digital twin for venue crowd management.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Architecture & Simulation Pipeline](#architecture--simulation-pipeline)
- [Features & Engine Highlights](#features--engine-highlights)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
  - [Running the Server](#running-the-server)
  - [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Testing, Benchmarks & Forecast Evaluation](#testing-benchmarks--forecast-evaluation)
- [Deploy on Render](#deploy-on-render)
- [Project Structure](#project-structure)

---

## Live Demo

**[https://crowd-guard-mavt.onrender.com](https://crowd-guard-mavt.onrender.com)**

> **Note:**
> - Hosted on Render's free tier, so the first visit after 15 minutes of inactivity takes about a minute to wake up. Please wait for the page to load.
> - The live build runs the simulation, 15-minute forecasting, Recommended Actions (with Divert), and Autopilot. The AI vision pipeline (YOLO / MiDaS / SmolVLM2) is disabled there (`ENABLE_VISION=0`) because it needs a camera and GPU—run the project locally to use it.

---

## Architecture & Simulation Pipeline

Each engine tick (`TICK_DURATION_SEC = 0.5` s) is driven by a monotonic wall-clock deadline loop in `backend/main.py` and executed on a worker thread, publishing an immutable state and heatmap snapshot for WebSocket clients.

```mermaid
flowchart LR
    V["AI Vision Pipeline (Local) - YOLOv8 + MiDaS + SmolVLM2"] -->|"real detections"| S["SimulationEngine (Thread-Safe Snapshots)"]
    I["Ingress ArrivalModel (Melbourne Sensor Curves)"] -->|"gate arrivals"| S
    S -->|"substeps (dt=0.1s)"| P["Vectorised SFM Physics (3m Neighbour Cutoff)"]
    P --> D["Unified DensityAnalyzer (local_density + Fruin LoS)"]
    D -->|"W_e = L_e * [1 + α(ρ/ρ_c)^β]"| R["Pathfinding & Flow Fields (A* + Cached Descent)"]
    D -->|"15-min horizon"| F["Continuum PredictionEngine (Weidmann Scaling)"]
    F -->|"early warnings"| A["Autopilot & Gate Diverts (Guard-Railed JSONL Log)"]
    S -->|"WebSocket JSON"| UI["Dashboard UI (Canvas + Forecast + Controls)"]
    A --> S
```

---

## Features & Engine Highlights

### Vision & AI (Local)
- **AI Vision Pipeline:** End-to-end video analysis using YOLOv8 (Crowd, Traffic, Anomalies, Flow Seg) and MiDaS (Depth Estimation).
- **Vision Language Model (VLM):** Integrated SmolVLM2 (2.2B) for high-level scene understanding and safety hazard descriptions.
- **Hardware Acceleration:** Native PyTorch CUDA 12.8 support and GPU-compiled `llama-cpp-python` for fast local inference.

### Real-Time Physics, Density & Routing
- **Vectorised Social Force Model (`backend/physics.py`):** Evaluates Helbing & Molnar forces over flat `(src, dst)` neighbour-pair arrays within `INTERACTION_CUTOFF = 3.0` m (calibrated with `B = 0.10`; 4.3 ms/substep at 200 agents and 77 ms at 1,000 agents, down from 58 ms / 1,338 ms in the scalar loop) and verified to `1e-5` relative tolerance against the scalar reference (`backend/physics_scalar.py`).
- **Unified Local Density & Fruin LoS (`backend/density.py`):** Zone classification, hazard detection, and forecast early warnings all share one `local_density(field, radius=1)` spatial mean (people/m²), preventing mismatched readings across dashboard panels, with per-second rise rates and Fruin Level-of-Service classification (LoS A–F + Critical).
- **Density-Weighted A\* & Cached Flow Fields (`backend/pathfinding.py`):** Routes agents around congestion using $W_e = L_e \cdot [1 + \alpha (\rho_e / \rho_{\text{critical}})^\beta]$ ($\alpha = 2.0$, $\beta = 4.0$, $\rho_{\text{critical}} = 3.5$ p/m²) plus cached exit potential flow fields, bounded per tick (`REROUTE_COOLDOWN_TICKS = 10`, `MAX_REROUTES_PER_TICK = 40`).

### Forecasting, Reproducibility & Automation
- **15-Minute Continuum Forecasting (`backend/prediction.py`):** Uses a flux-conserving `continuum` advection model along an 8-connected exit potential with Weidmann speed-density scaling (`jam_density = 5.0` p/m², `PILEUP_DENSITY = 2.17` p/m²) and future arrivals injected from measured pedestrian curves (`backend/ingress.py`, City of Melbourne CC BY 4.0 profiles for `gate`, `concourse`, and `food`). The legacy `particles` clone model remains selectable for benchmark comparisons.
- **Autopilot & Gate Diverts (`backend/autopilot.py`):** Automatically diverts gate arrivals if an operator does not respond within a configurable real-time window, backed by strict guard rails—acts only on imminent pile-ups (`<= 300s` ETA) that are already busy (`>= 1.08` p/m², LoS D+), caps diverts at `70%` (`DIVERT_DURATION_SEC = 120s`), never stacks diverts or closes/throttles gates, resets to OFF on restart, and persists an activity log (`backend/data/autopilot_log.jsonl`, up to 200 entries).
- **Deterministic Seeded Engine (`backend/rng.py`):** Derives independent SHA-256 substreams (`sfm`, `predict`, `ingress`) from a single seed (`DEFAULT_SEED = 1337`, overridable via `CROWD_SEED`), making simulation runs and forecast accuracy checks bit-identical.
- **Thread-Safe Snapshots & Monotonic Tick Clock (`backend/main.py`, `backend/simulation.py`):** Schedules ticks against an absolute `time.monotonic()` deadline (re-basing on overrun instead of firing catch-up bursts, bounded by `MIN_TICK_PERIOD_SEC = 0.002` s) and swaps immutable state/heatmap snapshots under `_state_lock` so WebSocket broadcasts never race with the simulation thread.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, Uvicorn, WebSockets, NumPy, Pydantic |
| **AI & Vision** | PyTorch (CUDA 12.8 / CPU / MPS), `torchvision`, `ultralytics` (YOLOv8), `timm` (MiDaS v2.1), `filterpy`, `opencv-python`, `llama-cpp-python` (SmolVLM2-2.2B-Instruct GGUF), `huggingface-hub` |
| **Frontend** | Vanilla HTML/CSS/JS with Canvas API |
| **Testing & Benchmarks** | `pytest`, recorded `.npz` ground-truth trajectories, per-stage latency profiler |

---

## Getting Started

### Prerequisites

- **Windows OS** (recommended for CUDA GPU vision stack)
- **Python 3.10+** (3.12 used in `setup.bat`)
- **NVIDIA GPU** (Recommended: 8GB+ VRAM, e.g., RTX 4060)
- **CUDA Toolkit** and **C++ Build Tools** (Required for GPU acceleration of the VLM)

> **macOS / Linux:** Use `./setup.sh` and `./start.sh` instead of the `.bat` scripts below. There is no CUDA support on these platforms: PyTorch runs on CPU (or MPS on Apple Silicon), and the VLM's `llama-cpp-python` build is optional (Metal-accelerated on macOS, CPU-only otherwise). See the comments inside [setup.sh](file:///f:/CrowdFlowOptimiser/crowd-guard/setup.sh) for the exact commands.

### Installation & Setup

Automated batch scripts are provided for Windows to handle virtual environments, dependencies, and downloading all AI model weights (~1.2GB).

1. **Clone the repository:**
   ```cmd
   git clone https://github.com/ishitathakur22/crowd-guard.git
   cd crowd-guard
   ```

2. **Run the automated setup script** (creates `.venv`, installs PyTorch with CUDA 12.8, installs `llama-cpp-python` for CUDA, installs Python dependencies, and downloads all models to `weights/`):
   ```cmd
   .\setup.bat
   ```

### Running the Server

Once setup is complete, launch the backend and frontend simultaneously:

| Mode | Command | Host / Port | Notes |
| :--- | :--- | :--- | :--- |
| **GPU (Default)** | `.\start.bat` | `0.0.0.0:8001` | Uses CUDA acceleration when available (`--reload`) |
| **CPU Only** | `.\start-cpu.bat` | `127.0.0.1:8001` | Runs all models on CPU and opens the dashboard automatically |
| **macOS / Linux** | `./start.sh` | `127.0.0.1:8001` | Simulation + CPU/MPS (`--reload`) |

Open the dashboard in your browser at:
```text
http://localhost:8001/static/index.html
```
*(or `http://localhost:8001/` — on Windows, avoid opening `http://0.0.0.0:8001/` printed by Uvicorn when running `start.bat`, as browsers treat `0.0.0.0` as `ERR_ADDRESS_INVALID`.)*

### Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_VISION` | `1` | Set to `0`, `false`, or `no` to disable the YOLO/MiDaS/SmolVLM2 vision stack (simulation-only mode). |
| `CROWD_SEED` | `1337` | Integer RNG seed controlling all derived simulation, physics, and ingress substreams. |
| `PORT` | `8001` | HTTP/WebSocket port when running `python -m backend.main` (binds `0.0.0.0` if `PORT` is set, else `127.0.0.1`). |

---

## Usage

### 1. AI Vision Pipeline (Local Only)
- Click **Add Files** or **Add Folder** to select local video files (`.mp4`, `.avi`, etc.).
- Click **Scan** to process the videos through the AI pipeline.
- View live **FPS**, **Latency**, **VRAM usage**, and **active models** on the dashboard.
- Review periodic text-based scene intelligence reports generated by the VLM.

### 2. Simulation & Digital Twin
- Click **Spawn** to manually add synthetic agents to the simulation.
- Use the **speed controls** (`1×`, `2×`, `5×`, `10×`) to adjust simulation time.
- Toggle visualization layers (**Heatmap**, **Flow**, **Paths**, **Agents**, **Labels**).
- Configure **Expected crowd size**, **Doors open**, and **Surge intensity** in **Event Setup**—or pick a **Day type** preset (*Weekday commute* / *Match day* / *Sell-out surge*)—to control how heavily the venue fills up.
- Monitor density metrics, alerts, and 15-minute predictions in the sidebars.

### 3. Recommended Actions & Autopilot
- **Recommended Actions:** Lists zones the 15-minute forecast expects to hit critical density (soonest first), each with a one-click **Divert** button that reroutes a share of a congested gate's new arrivals to a less-loaded gate.
- **Autopilot:** Turn on **Autopilot** to apply that same divert automatically if nobody responds within a configurable window (`Now` / `15s` / `30s` / `60s` / `90s` / `2m`). It only acts on pile-ups that are both imminent and already busy right now, never closes or throttles a gate, and writes all actions to the **Activity Log** for review.

---

## Testing, Benchmarks & Forecast Evaluation

The repository includes an automated test suite and standalone scripts to profile engine latency and score forecast accuracy against recorded ground-truth trajectories.

### Running Tests
```cmd
.\.venv\Scripts\python.exe -m pytest tests -q
```
Covers unified density & Fruin thresholds (`test_density.py`), continuum forecast bounds (`test_forecast.py`), gate queues & diverts (`test_gates.py`), sensor ingress curves (`test_ingress.py`), model instance isolation (`test_models.py`), density-weighted A\* & flow fields (`test_pathfinding.py`), vectorised vs. scalar SFM parity (`test_physics_parity.py`), seeded determinism (`test_reproducibility.py`), and thread-safe snapshots / monotonic tick timing (`test_snapshot_and_clock.py`).

### Per-Stage Latency Benchmark
Profiles `density`, `physics`, `reroute`, `forecast`, `serialise`, `spawn`, and end-to-end `update()` across 50, 200, 500, and 1,000 agents (writing `benchmarks/bench_baseline.json`):
```cmd
.\.venv\Scripts\python.exe scripts/bench.py
.\.venv\Scripts\python.exe scripts/bench.py --agents 200 500 --reps 30
```

### Trajectory Recording & Forecast Scoring
Records a deterministic live trajectory (`default` or `surge`) to `benchmarks/trajectory_<scenario>.npz` and evaluates `continuum` vs. `particles` against a persistence baseline (`MAE`, `busy MAE`, `corr`, `peak err`, `lead time`):
```cmd
.\.venv\Scripts\python.exe scripts/record_trajectory.py --scenario default
.\.venv\Scripts\python.exe scripts/eval_forecast.py --scenario default --model continuum
```

---

## Deploy on Render

Live deployment: **[https://crowd-guard-mavt.onrender.com](https://crowd-guard-mavt.onrender.com)**

This repository includes a `render.yaml` blueprint configured for Render's free tier (Python web service, persistent WebSockets, 512 MB RAM).

### Deployment Steps

1. Fork or push this repository to your GitHub account.
2. Sign in to [Render](https://render.com) and navigate to **Blueprints → New Blueprint Instance**.
3. Connect your GitHub repository. Render will detect `render.yaml` and configure the service automatically:

   | Setting | Value |
   | :--- | :--- |
   | **Environment** | Python |
   | **Plan** | Free |
   | **Build Command** | `pip install -r requirements-deploy.txt` |
   | **Start Command** | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
   | **Health Check Path** | `/health` |
   | **Environment Variables** | `ENABLE_VISION=0` |

4. Click **Apply**. Once deployed, open the assigned `https://<service-name>.onrender.com` URL to view the live simulation.

> **Note on Free Tier:**
> - Free web services on Render automatically spin down (sleep) after 15 minutes of inactivity. Incoming requests wake the instance up, which takes about a minute on a cold start.
> - The deployed build runs with `ENABLE_VISION=0` to stay within the 512 MB RAM allowance. The simulation, digital twin physics, 15-minute predictive forecasting, Recommended Actions, gate divert routing, and Autopilot are fully functional, while the local AI vision stack (YOLO/MiDaS/SmolVLM) is disabled.

---

## Project Structure

```text
crowd-guard/
├── backend/
│   ├── autopilot.py          # Guard-railed auto-divert engine & JSONL activity log
│   ├── data/                 # Sensor arrival profiles & runtime autopilot log
│   ├── density.py            # Unified local_density() & Fruin LoS classification
│   ├── ingress.py            # Measured pedestrian arrival curves (Melbourne CC BY 4.0)
│   ├── main.py               # FastAPI app, WebSocket hub & monotonic tick loop
│   ├── models.py             # Pydantic schemas, agent profiles & gate state models
│   ├── pathfinding.py        # Density-weighted A* & cached exit flow fields
│   ├── physics.py            # Vectorised Social Force Model over neighbour pairs
│   ├── physics_scalar.py     # Scalar SFM reference implementation for parity tests
│   ├── prediction.py         # 15-min continuum advection & particle forecast models
│   ├── rng.py                # SHA-256 derived deterministic RNG substreams
│   ├── simulation.py         # Core simulation engine & thread-safe snapshot publisher
│   └── vision/               # Local YOLOv8, MiDaS & SmolVLM2 vision pipeline
├── frontend/                 # Dashboard UI (HTML/CSS/JS, Canvas rendering)
├── benchmarks/               # Saved latency baselines, trajectories & forecast scores
├── scripts/
│   ├── bench.py              # Per-stage latency profiler across crowd sizes
│   ├── build_arrival_profiles.py  # Arrival curve builder from sensor data
│   ├── eval_forecast.py      # Forecast accuracy scorer vs. persistence baseline
│   └── record_trajectory.py  # Deterministic ground-truth trajectory recorder
├── tests/                    # Pytest suite (physics parity, determinism, concurrency, etc.)
├── weights/                  # Downloaded AI model weights (populated by download_models.py)
├── download_models.py        # Downloads YOLOv8, MiDaS, and SmolVLM2 weights
├── render.yaml               # Render free-tier deployment blueprint
├── requirements.txt          # Full local dependencies (including vision/GPU stack)
├── requirements-deploy.txt   # Lightweight deployment dependencies (simulation-only)
├── setup.bat / setup.sh      # Automated environment & model setup scripts
└── start.bat / start-cpu.bat / start.sh  # Server launchers (GPU / CPU / macOS-Linux)
```
