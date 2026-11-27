from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import time
import os
import json
import logging

logger = logging.getLogger("CFO")
from .simulation import SimulationEngine, TICK_DURATION_SEC

# Smallest gap between ticks, so a high speed multiplier cannot spin the loop
# with no opportunity to serve websockets.
MIN_TICK_PERIOD_SEC = 0.002

ENABLE_VISION_ENV = os.environ.get("ENABLE_VISION", "1").strip().lower()
ENABLE_VISION = ENABLE_VISION_ENV not in ("0", "false", "no")

app = FastAPI(
    title="Crowd Guard",
    description="Real-time crowd simulation, bottleneck forecasting, and automated rerouting engine.",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure frontend directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="static")

sim_engine = SimulationEngine()

# Initialize Vision Pipeline conditionally
vision_pipeline = None
latest_vision_update = {}

if ENABLE_VISION:
    try:
        from .vision.pipeline import VisionPipeline
        vision_pipeline = VisionPipeline(grid_width=sim_engine.width, grid_height=sim_engine.height)
        vision_pipeline.on_agents_detected = sim_engine.ingest_real_data

        def _handle_vision_update(data: dict):
            global latest_vision_update
            latest_vision_update = data

        vision_pipeline.on_vision_update = _handle_vision_update
        logger.info("Vision pipeline initialized")
    except Exception as _vision_err:
        logger.warning("Vision pipeline unavailable, running simulation-only: %s", _vision_err)
        vision_pipeline = None
else:
    logger.info("Vision pipeline disabled via ENABLE_VISION=%s", ENABLE_VISION_ENV)

DISABLED_VISION_MSG = {"status": "disabled", "message": "Vision pipeline disabled in this deployment"}

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/api/config")
async def get_config():
    return {"vision_enabled": bool(ENABLE_VISION and vision_pipeline is not None)}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMULATION LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())

async def simulation_loop():
    """
    Drive the simulation on a monotonic deadline.

    The old loop slept `TICK_DURATION_SEC / speed` *after* doing the tick, so the
    real period was `tick_work + sleep`. At 1x with a 17 ms tick that ran ~3% fast;
    at 10x with the same work the intended 50 ms period became 67 ms, so "10x" was
    really 7.5x - and the slower the machine, the less the speed control meant. The
    measured headroom reported by scripts/bench.py is the honest figure; this keeps
    the simulation clock agreeing with it.

    Scheduling against an absolute deadline makes the period independent of how long
    the tick took. When a tick overruns its slot the deadline is re-based rather than
    firing a burst of catch-up ticks, which would simulate a crowd faster than real
    time and distort every forecast made from it.
    """
    sim_engine.running = True
    loop = asyncio.get_event_loop()
    next_deadline = time.monotonic()

    while sim_engine.running:
        speed = sim_engine.sim_speed          # read once — it can change mid-tick
        if speed <= 0:
            await asyncio.sleep(0.5)
            next_deadline = time.monotonic()  # resuming must not replay the pause
            continue

        period = max(MIN_TICK_PERIOD_SEC, TICK_DURATION_SEC / max(speed, 0.1))
        try:
            await loop.run_in_executor(None, sim_engine.update)
        except Exception:
            logger.exception("Simulation tick failed")
            await asyncio.sleep(0.5)
            next_deadline = time.monotonic()
            continue

        next_deadline += period
        now = time.monotonic()
        if next_deadline < now:
            # The tick overran its slot. Re-base instead of catching up.
            next_deadline = now
        else:
            await asyncio.sleep(next_deadline - now)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEBSOCKET CONNECTION MANAGER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

        # Seed the venue with an initial crowd so it isn't empty on first
        # paint — the arrival curve alone takes a minute or two to build
        # this up naturally.
        if not sim_engine.agents:
            sim_engine.spawn_agents(220)

        # Send initial grid config + venue metadata
        await websocket.send_json({
            "type": "grid_config",
            "width": sim_engine.width,
            "height": sim_engine.height,
            "grid": sim_engine.grid,
            "gates": [
                {
                    "gate_id": g.gate_id,
                    "x": g.position.x,
                    "y": g.position.y,
                    "status": g.status,
                    "target_rate": g.target_rate_per_sec,
                }
                for g in sim_engine.gate_objects
            ],
            "exits": [{"x": e.x, "y": e.y} for e in sim_engine.exits],
            "concessions": [{"x": c.x, "y": c.y} for c in sim_engine.concessions],
            "signs": [
                {
                    "sign_id": s.sign_id,
                    "x": s.position.x,
                    "y": s.position.y,
                    "message": s.message,
                    "direction": s.direction,
                    "active": s.active,
                }
                for s in sim_engine.signs
            ],
        })

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
                action = command.get("action")
                
                if action == "spawn":
                    count = command.get("count", 10)
                    sim_engine.spawn_agents(count)
                    
                elif action == "clear":
                    sim_engine.agents.clear()
                    sim_engine.suggested_routes.clear()
                    
                elif action == "gate_control":
                    gate_id = command.get("gate_id")
                    gate_action = command.get("gate_action", "OPEN_FULL")
                    rate = command.get("rate", 3.5)
                    if gate_id:
                        sim_engine.actuate_gate(gate_id, gate_action, rate)

                elif action == "divert_gate":
                    sim_engine.divert_gate(
                        command.get("origin_gate_id"),
                        command.get("target_gate_id"),
                        command.get("pct", 0),
                    )
                        
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def broadcast_state():
    while True:
        if manager.active_connections:
            state = sim_engine.get_state()
            heatmap_data = sim_engine.get_heatmap_data()
            
            await manager.broadcast({
                "type": "state_update",
                "state": state,
                "heatmap": heatmap_data,
            })
        speed = sim_engine.sim_speed
        sleep_time = max(0.1, 0.5 / max(speed, 0.1)) if speed > 0 else 0.5
        await asyncio.sleep(sleep_time)

async def broadcast_vision_state():
    """Broadcast vision pipeline metrics to all connected clients."""
    if not ENABLE_VISION or vision_pipeline is None:
        return
    while True:
        if manager.active_connections and latest_vision_update:
            await manager.broadcast({
                "type": "vision_update",
                "data": latest_vision_update,
            })
        await asyncio.sleep(0.5)

@app.on_event("startup")
async def start_broadcaster():
    asyncio.create_task(broadcast_state())
    if ENABLE_VISION and vision_pipeline is not None:
        asyncio.create_task(broadcast_vision_state())

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# REST API ENDPOINTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VisionBoundingBox(BaseModel):
    track_id: int
    x: float
    y: float
    confidence: float

class VisionIngestionPayload(BaseModel):
    camera_id: str
    timestamp: float
    detections: List[VisionBoundingBox]

@app.post("/ingest/vision")
async def ingest_vision_data(payload: VisionIngestionPayload):
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    return {"status": "success", "processed": len(payload.detections)}

@app.post("/api/vision/start")
async def start_vision_pipeline():
    """Start the real-time AI vision pipeline (webcam)."""
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    if not vision_pipeline.running:
        vision_pipeline.set_source(0)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, vision_pipeline.initialize)
        if not result["loaded"]:
            return {"status": "error", "detail": "No vision models could be loaded", **result}
        asyncio.create_task(vision_pipeline.start())
        return {"status": "started", "mode": "live", **result}
    return {"status": "already_running", "mode": "live"}

@app.post("/api/vision/stop")
async def stop_vision_pipeline():
    """Stop the real-time AI vision pipeline."""
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    vision_pipeline.stop()
    return {"status": "stopped"}

class VideoScanRequest(BaseModel):
    paths: List[str]  # List of file paths and/or folder paths

@app.post("/api/vision/scan-video")
async def scan_video(request: VideoScanRequest):
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    if vision_pipeline.running:
        vision_pipeline.stop()
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, vision_pipeline.initialize)
    asyncio.create_task(vision_pipeline.scan_videos(request.paths))
    return {"status": "scanning", "paths": request.paths}

class CalibrationRequest(BaseModel):
    reference_points_img: List[List[float]]
    reference_points_grid: List[List[float]]


@app.get("/api/vision/calibration")
async def get_calibration():
    """Report whether the vision pipeline has a real ground-plane calibration."""
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    return vision_pipeline.calibration_status()


@app.post("/api/vision/calibration")
async def set_calibration(request: CalibrationRequest):
    """
    Supply four point pairs measured on the floor, in image and venue coordinates.

    Validated before it is applied: if the homography cannot be solved the previous
    calibration is left untouched rather than silently degraded.
    """
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    if len(request.reference_points_img) != len(request.reference_points_grid):
        return {"status": "error", "detail": "point lists must be the same length"}
    if len(request.reference_points_img) < 4:
        return {"status": "error", "detail": "four point pairs are required"}

    ok = vision_pipeline.save_calibration(
        request.reference_points_img, request.reference_points_grid
    )
    if not ok:
        return {"status": "error", "detail": "homography could not be solved from those points"}
    return {"status": "calibrated", **vision_pipeline.calibration_status()}


@app.get("/api/vision/status")
async def vision_status():
    """Return current vision pipeline status."""
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    try:
        from .vision.model_manager import model_manager
        mm_status = model_manager.get_status()
    except Exception:
        mm_status = {"gpu_available": False, "models": {}}
    return {
        "running": vision_pipeline.running,
        "mode": vision_pipeline.mode,
        "fps": vision_pipeline.fps,
        "frame": vision_pipeline.frame_count,
        "total_frames": vision_pipeline.total_frames,
        "vlm_analysis": vision_pipeline.last_vlm_analysis,
        "model_manager": mm_status,
    }

@app.get("/api/vision/pick-files")
async def pick_files():
    """Open a native file picker dialog and return selected video file paths."""
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    import threading
    result = {"paths": []}

    def _pick():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            files = filedialog.askopenfilenames(
                title="Select Video Files",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm"),
                    ("All files", "*.*"),
                ],
            )
            result["paths"] = list(files)
            root.destroy()
        except Exception as e:
            logger.warning("File picker error: %s", e)

    t = threading.Thread(target=_pick)
    t.start()
    t.join(timeout=120)
    return result

@app.get("/api/vision/pick-folder")
async def pick_folder():
    """Open a native folder picker dialog and return the selected folder path."""
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    import threading
    result = {"path": ""}

    def _pick():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            folder = filedialog.askdirectory(title="Select Video Folder")
            result["path"] = folder
            root.destroy()
        except Exception as e:
            logger.warning("Folder picker error: %s", e)

    t = threading.Thread(target=_pick)
    t.start()
    t.join(timeout=120)
    return result

class SpawnRequest(BaseModel):
    count: int = 10

@app.post("/simulation/spawn")
async def spawn_agents_endpoint(request: SpawnRequest):
    spawned = sim_engine.spawn_agents(request.count)
    return {"status": "success", "spawned": spawned, "total_agents": len(sim_engine.agents)}

@app.post("/simulation/clear")
async def clear_agents():
    sim_engine.agents.clear()
    sim_engine.suggested_routes.clear()
    return {"status": "success", "total_agents": 0}

@app.get("/simulation/state")
async def get_state_endpoint():
    return sim_engine.get_state()

@app.get("/simulation/heatmap")
async def get_heatmap_endpoint():
    return sim_engine.get_heatmap_data()

@app.get("/simulation/predictions")
async def get_predictions_endpoint():
    return {"predictions": sim_engine.predictions}

class GateControlRequest(BaseModel):
    gate_id: str
    action: str = "OPEN_FULL"
    rate: float = 3.5

@app.post("/gates/control")
async def control_gate(request: GateControlRequest):
    sim_engine.actuate_gate(request.gate_id, request.action, request.rate)
    return {"status": "success", "gate_id": request.gate_id, "action": request.action}

class ResolveAlertRequest(BaseModel):
    alert_id: str

@app.post("/alerts/resolve")
async def resolve_alert(request: ResolveAlertRequest):
    for alert in sim_engine.alerts:
        if alert.alert_id == request.alert_id:
            alert.resolved = True
            break
    return {"status": "success", "alert_id": request.alert_id}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVENT SETUP / INGRESS CURVES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IngressConfigRequest(BaseModel):
    archetype: Optional[str] = None
    attendance: Optional[int] = None
    start_hour: Optional[float] = None
    scale: Optional[float] = None
    weekend: Optional[bool] = None
    enabled: Optional[bool] = None

@app.get("/api/ingress")
async def get_ingress():
    """Current arrival model configuration + live admission rate."""
    model = sim_engine.arrival_model
    hour = model.hour_at(sim_engine.sim_time)
    return {
        "enabled": sim_engine.auto_ingress,
        "config": model.describe(),
        "clock_hour": round(hour, 2),
        "current_rate_per_sec": round(model.rate_at(sim_engine.sim_time), 3),
        "current_rate_per_hour": round(model.rate_at(sim_engine.sim_time) * 3600, 1),
        "curve": model.curve,
    }

@app.post("/api/ingress")
async def set_ingress(request: IngressConfigRequest):
    """Reconfigure the arrival model. Rebuilds it when the profile changes."""
    from .ingress import ArrivalModel

    if request.enabled is not None:
        sim_engine.auto_ingress = request.enabled

    model = sim_engine.arrival_model
    needs_rebuild = any(
        v is not None for v in (request.archetype, request.weekend)
    )

    if needs_rebuild:
        model = ArrivalModel(
            archetype=request.archetype or model.archetype,
            attendance=request.attendance or model.attendance,
            start_hour=model.start_hour if request.start_hour is None else request.start_hour,
            weekend=model.weekend if request.weekend is None else request.weekend,
            scale=model.scale if request.scale is None else request.scale,
        )
        sim_engine.arrival_model = model
        sim_engine.prediction_engine.set_arrival_model(model, sim_engine.gates)
    else:
        if request.attendance is not None:
            model.attendance = request.attendance
        if request.start_hour is not None:
            model.start_hour = request.start_hour
        if request.scale is not None:
            model.scale = request.scale

    return {"status": "success", "enabled": sim_engine.auto_ingress, **model.describe()}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTOPILOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AutopilotConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    wait_sec: Optional[int] = None

@app.get("/api/autopilot")
async def get_autopilot(limit: int = 100):
    """Autopilot status plus its activity log (newest first)."""
    ap = sim_engine.autopilot
    return {**ap.status(), "log": ap.recent(max(1, min(limit, 200)))}

@app.post("/api/autopilot")
async def set_autopilot(request: AutopilotConfigRequest):
    """Turn Autopilot on/off and set how long it waits for an operator before acting."""
    ap = sim_engine.autopilot
    ap.set_config(enabled=request.enabled, wait_sec=request.wait_sec, sim_time=sim_engine.sim_time)
    return {**ap.status(), "log": ap.recent(100)}

@app.post("/api/autopilot/ack")
async def ack_autopilot():
    """Mark everything in the activity log as read."""
    ap = sim_engine.autopilot
    ap.acknowledge()
    return {**ap.status(), "log": ap.recent(100)}

class SpeedControlRequest(BaseModel):
    speed: float

@app.post("/simulation/speed")
async def set_speed(request: SpeedControlRequest):
    """Set simulation speed multiplier."""
    sim_engine.set_sim_speed(request.speed)
    return {"status": "success", "speed": sim_engine.sim_speed}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
