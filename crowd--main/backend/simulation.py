"""
Simulation Engine — Core crowd simulation with integrated density analysis,
prediction forecasting, dynamic rerouting, and gate actuation.
"""

import os
import random
import time
import numpy as np
from typing import List, Dict, Tuple
from .models import (
    Point, Agent, CellType, SimulationState, FruinLoS,
    ZoneDensity, HazardZone, AlertEvent, GateState, GateAction,
    SignageState, PredictionSnapshot, RerouteRecommendation,
    AGENT_PROFILES, AlertSeverity
)
from .pathfinding import a_star_search, compute_alternative_routes
from .physics import SocialForceModel
from .density import DensityAnalyzer
from .prediction import PredictionEngine
from .ingress import ArrivalModel
from .autopilot import Autopilot


# Simulated seconds advanced by one engine tick. The physics integrator runs at
# a finer dt, so a tick executes several substeps to cover this span — keeping
# the displayed clock honest about how much movement actually happened.
TICK_DURATION_SEC = 0.5

# Rerouting runs an A* per exit per agent, which dominates the tick cost once a
# crowd is dense. Each agent is re-routed at most once per cooldown window, and
# at most MAX_REROUTES_PER_TICK agents are processed per tick, so a crowd of
# several hundred stays simulable in real time.
REROUTE_COOLDOWN_TICKS = 10      # 10 ticks = 5 simulated seconds
MAX_REROUTES_PER_TICK = 40

# Operator gate diverts ("send N% of Gate A's new arrivals to Gate B") last this long
# in simulated seconds unless re-issued.
DIVERT_DURATION_SEC = 120.0


class SimulationEngine:
    def __init__(self, width: int = 40, height: int = 30):
        self.width = width
        self.height = height
        self.grid = self._generate_default_grid()
        self.agents: List[Agent] = []
        self.real_agents: List[Dict] = []  # Added for Vision Pipeline
        self.bottlenecks: List[Point] = []
        self.suggested_routes: Dict[int, List[Point]] = {}
        self.agent_id_counter = 0
        
        self.running = False
        self.sim_time = 0.0        # Simulation clock (seconds)
        self.sim_speed = 1.0       # Speed multiplier
        self.tick_count = 0
        
        # Physics engine. One tick covers TICK_DURATION_SEC of simulated time,
        # which takes several integrator substeps at dt=0.1s.
        self.sfm = SocialForceModel(dt=0.1)
        self.physics_substeps = max(1, int(round(TICK_DURATION_SEC / self.sfm.dt)))
        self.walls_cache = self._get_walls()
        
        # Density analyzer
        self.density_analyzer = DensityAnalyzer(width, height)
        self.zone_densities: List[ZoneDensity] = []
        self.hazard_zones: List[HazardZone] = []
        self.density_heatmap: List[List[float]] = []
        self.flow_vectors: List[Dict] = []
        
        # Prediction engine
        self.prediction_engine = PredictionEngine(width, height)
        self.predictions: List[PredictionSnapshot] = []
        self.prediction_interval = 10  # Run forecast every N ticks
        
        # Alert system
        self.alerts: List[AlertEvent] = []
        self.alert_history: List[AlertEvent] = []
        self.max_alerts = 50
        
        # Gates
        self.gate_objects: List[GateState] = []
        # Spread evenly down the left edge (grid height=30) instead of
        # clustering at y=10-16, so each gate has its own zone to serve.
        self.gates: List[Point] = [
            Point(x=0.0, y=2.0),
            Point(x=0.0, y=8.0),
            Point(x=0.0, y=14.0),
            Point(x=0.0, y=20.0),
            Point(x=0.0, y=26.0),
        ]
        # Same idea for exits on the right edge — was y=10,11 stacked
        # together plus y=20,25. Now evenly spaced across the height.
        self.exits: List[Point] = [
            Point(x=39.0, y=5.0),
            Point(x=39.0, y=13.0),
            Point(x=39.0, y=21.0),
            Point(x=39.0, y=27.0),
        ]
        self.concessions: List[Point] = [
            Point(x=20.0, y=5.0), Point(x=20.0, y=25.0)
        ]
        
        # Digital signage
        self.signs: List[SignageState] = [
            SignageState(sign_id="sign_corridor_a", position=Point(x=10.0, y=14.0), message="→ Main Concourse", direction="RIGHT"),
            SignageState(sign_id="sign_fork_b", position=Point(x=22.0, y=10.0), message="→ Exit A", direction="RIGHT"),
            SignageState(sign_id="sign_fork_c", position=Point(x=22.0, y=20.0), message="↓ Exit B", direction="DOWN"),
        ]
        
        # Initialize gates on grid
        self._setup_venue_elements()

        # Arrival model — real hourly ingress curve measured from pedestrian
        # sensors, used both to drive live spawning and to tell the forecaster
        # how many people are still on their way in.
        self.arrival_model = ArrivalModel(
            archetype="gate",
            attendance=55_000,
            start_hour=17.0,   # matches the default slider values in index.html
            scale=2.0,
        )
        self.auto_ingress = True
        self.prediction_engine.set_arrival_model(self.arrival_model, self.gates)

        # Fractional admit capacity carried between ticks, per gate, so rates
        # below one person per tick (e.g. a throttled 0.5/s gate) still admit.
        self._gate_credit: Dict[str, float] = {g.gate_id: 0.0 for g in self.gate_objects}
        self._last_reroute: Dict[int, int] = {}   # agent id -> tick of last reroute
        self._diverts: Dict[str, dict] = {}       # origin gate id -> {target, pct, until}
        self.forecast_warnings: List[dict] = []   # early pile-up warnings (see PredictionEngine.early_warnings)
        self._landmarks = None                    # cached (x, y, label) for named zones
        self.autopilot = Autopilot(
            log_path=os.path.join(os.path.dirname(__file__), "data", "autopilot_log.jsonl")
        )

        # Active reroute tracking
        self.active_reroutes: List[RerouteRecommendation] = []
        
    def _generate_default_grid(self) -> List[List[str]]:
        grid = [[CellType.EMPTY.value for _ in range(self.width)] for _ in range(self.height)]
        
        # Wall layout: creates a venue with corridors, chokepoints, and multiple paths
        # Central wall with gap (main chokepoint)
        for y in range(6, 24):
            if y not in (13, 14, 15, 16):  # Gap in the middle
                grid[y][15] = CellType.WALL.value
        
        # Upper barrier (forces traffic through specific corridors)
        for y in range(0, 8):
            grid[y][25] = CellType.WALL.value
            
        # Lower barrier
        for y in range(22, 30):
            grid[y][25] = CellType.WALL.value
            
        # Small obstacles creating secondary chokepoints
        for x in range(30, 34):
            grid[14][x] = CellType.WALL.value
            grid[15][x] = CellType.WALL.value
        
        # Corridor walls (upper passage)
        for x in range(8, 14):
            grid[5][x] = CellType.WALL.value
            grid[8][x] = CellType.WALL.value
        
        return grid
        
    def _get_walls(self) -> List[Tuple[int, int]]:
        walls = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == CellType.WALL.value:
                    walls.append((x, y))
        return walls
    
    def _setup_venue_elements(self):
        """Place gates, exits, concessions on the grid."""
        for i, g in enumerate(self.gates):
            gy, gx = int(g.y), int(g.x)
            if 0 <= gy < self.height and 0 <= gx < self.width:
                self.grid[gy][gx] = CellType.GATE.value
            self.gate_objects.append(GateState(
                gate_id=f"gate_{i}",
                position=g,
                throughput_per_sec=3.5,
                target_rate_per_sec=3.5,
            ))
            
        for e in self.exits:
            ey, ex = int(e.y), int(e.x)
            if 0 <= ey < self.height and 0 <= ex < self.width:
                self.grid[ey][ex] = CellType.EXIT.value
                
        for c in self.concessions:
            cy, cx = int(c.y), int(c.x)
            if 0 <= cy < self.height and 0 <= cx < self.width:
                self.grid[cy][cx] = CellType.CONCESSION.value

    def spawn_agents(self, count: int, gate_idx: int = None, record_estimate: bool = True):
        """
        Spawn agents at gate positions with demographic variance.

        gate_idx pins every agent to one gate (used by the queue admit step).
        When None, a random gate is chosen per agent (manual spawn button).
        """
        profiles = list(AGENT_PROFILES.keys())
        profile_weights = [0.6, 0.15, 0.1, 0.15]  # adult, elderly, child, luggage
        
        spawned = 0
        for _ in range(count):
            # Check gate throttling
            g_idx = gate_idx if gate_idx is not None else random.randint(0, len(self.gates) - 1)
            gate_state = self.gate_objects[g_idx]
            
            if gate_state.action == GateAction.CLOSE:
                continue  # Gate is closed
            
            start = self.gates[g_idx]
            
            # Jitter to prevent stacking
            jitter_x = random.uniform(0.1, 0.9)
            jitter_y = random.uniform(-0.4, 0.4)
            actual_start = Point(x=start.x + jitter_x, y=start.y + jitter_y)
            
            # Choose destination
            if random.random() > 0.6:
                goal = random.choice(self.concessions)
            else:
                goal = random.choice(self.exits)
            
            # Choose demographic profile
            profile_name = random.choices(profiles, weights=profile_weights, k=1)[0]
            profile = AGENT_PROFILES[profile_name]
                
            agent = Agent(
                id=self.agent_id_counter,
                pos=actual_start,
                goal=goal,
                path=[],
                desired_speed=profile.desired_speed + random.uniform(-0.15, 0.15),
                radius=profile.radius,
                mass=profile.mass,
                profile=profile_name,
            )
            self.agent_id_counter += 1
            
            # Initial path calculation
            start_int = Point(x=float(int(actual_start.x)), y=float(int(actual_start.y)))
            goal_int = Point(x=float(int(goal.x)), y=float(int(goal.y)))
            
            path = a_star_search(self.grid, start_int, goal_int)
            if path:
                agent.path = path[1:] if len(path) > 1 else path
            
            self.agents.append(agent)
            gate_state.agents_passed += 1
            spawned += 1

        # Record actual admissions (may be below `count` when gates are closed)
        # so the fallback ingress estimate reflects reality.
        if record_estimate:
            self.prediction_engine.update_ingress_estimate(spawned)
        return spawned

    def _admit_arrivals(self, arrivals: int):
        """
        Queue new arrivals at gates, then admit as many as each gate's current
        target rate allows this tick. Everyone else keeps waiting outside, so
        throttling a gate builds a real queue instead of making people vanish.
        """
        n_gates = len(self.gate_objects)

        # Drop expired diverts, then route each new arrival. An arrival headed for a
        # diverted gate is sent to the target gate with probability pct/100 (never to a
        # closed gate), so the split is real queue movement, not a display trick.
        for k in [k for k, d in self._diverts.items() if d["until"] <= self.sim_time]:
            d = self._diverts.pop(k)
            self.autopilot.record(
                "divert_end", "system",
                f"Divert ended: {self._gate_name(k)} -> {self._gate_name(d['target'])} ({int(d['pct'])}%) "
                f"after {int(DIVERT_DURATION_SEC)}s",
                self.sim_time,
                {"origin": k, "target": d["target"], "pct": d["pct"], "started_by": d.get("source", "operator")},
            )
        by_id = {g.gate_id: g for g in self.gate_objects}
        for _ in range(arrivals):
            gate = self.gate_objects[random.randrange(n_gates)]
            d = self._diverts.get(gate.gate_id)
            if d and random.random() < d["pct"] / 100.0:
                tgt = by_id.get(d["target"])
                if tgt is not None and tgt.action != GateAction.CLOSE:
                    tgt.queue_length += 1
                    continue
            gate.queue_length += 1

        total_admitted = 0
        for idx, gate in enumerate(self.gate_objects):
            if gate.action == GateAction.CLOSE:
                rate = 0.0
            else:
                rate = max(0.0, gate.target_rate_per_sec)

            credit = self._gate_credit[gate.gate_id] + rate * TICK_DURATION_SEC
            admit = min(int(credit), gate.queue_length)
            credit -= admit
            # Don't bank unused capacity while the queue is empty.
            if gate.queue_length - admit <= 0:
                credit = min(credit, 1.0)
            self._gate_credit[gate.gate_id] = credit

            if admit > 0:
                gate.queue_length -= self.spawn_agents(admit, gate_idx=idx, record_estimate=False)
                total_admitted += admit

            gate.wait_time_sec = round(gate.queue_length / rate, 1) if rate > 0 else 0.0

        self.prediction_engine.update_ingress_estimate(total_admitted)


    def ingest_real_data(self, real_agents_data: List[Dict]):
        """
        Phase 1: Purely Real Update.
        Accepts data from the Vision Pipeline and updates the real_agents list.
        Data format: [{"id": str, "x": float, "y": float, "type": str, "class_id": int, "conf": float, "depth": float}]
        """
        self.real_agents = real_agents_data
        
        # Note for Phase 2 (Hybrid Merge):
        # Real agents now influence synthetic agents via the Social Force Model.

        self.prediction_engine.update_ingress_estimate(len(real_agents_data))
            
    def update(self):
        """Main simulation tick."""
        self.tick_count += 1
        # The clock advances by the world time this tick actually simulates.
        # sim_speed controls how often ticks run in wall-clock time, not how
        # much simulated time each one represents.
        self.sim_time += TICK_DURATION_SEC

        # 0. Ingress — admit arrivals from the measured hourly curve
        arrivals = 0
        if self.auto_ingress:
            arrivals = self.arrival_model.draw_arrivals(self.sim_time, TICK_DURATION_SEC)
        # Always run the admit step: queued people keep entering even if new
        # arrivals are paused.
        self._admit_arrivals(arrivals)

        # 1. Update positions using Social Force Model
        for _ in range(self.physics_substeps):
            self.sfm.update_positions(self.agents, self.walls_cache, self.real_agents)
        
        # 2. Check path progress and goals
        for agent in self.agents:
            if agent.status == "arrived":
                continue
                
            # Reached next path node?
            if agent.path:
                next_node = agent.path[0]
                dist_sq = (agent.pos.x - (next_node.x + 0.5))**2 + (agent.pos.y - (next_node.y + 0.5))**2
                if dist_sq < 0.5:
                    agent.path.pop(0)
                    
            # Reached ultimate goal?
            dist_to_goal = (agent.pos.x - (agent.goal.x + 0.5))**2 + (agent.pos.y - (agent.goal.y + 0.5))**2
            if dist_to_goal < 1.0:
                gx, gy = int(agent.goal.x), int(agent.goal.y)
                if 0 <= gy < self.height and 0 <= gx < self.width:
                    if self.grid[gy][gx] == CellType.CONCESSION.value:
                        # Visit concession then head to exit
                        agent.goal = random.choice(self.exits)
                        start_int = Point(x=float(int(agent.pos.x)), y=float(int(agent.pos.y)))
                        goal_int = Point(x=float(int(agent.goal.x)), y=float(int(agent.goal.y)))
                        agent.path = a_star_search(self.grid, start_int, goal_int)
                        agent.rerouted = False
                    else:
                        agent.status = "arrived"
                else:
                    agent.status = "arrived"
                    
        # Remove arrived agents
        self.agents = [a for a in self.agents if a.status != "arrived"]
        
        # 3. Density analysis
        raw_density = self.density_analyzer.compute_raw_density(self.agents)
        self.density_heatmap = self.density_analyzer.get_density_grid_for_heatmap(self.agents)
        self.flow_vectors = self.density_analyzer.get_flow_vectors_for_viz(self.agents)
        self.zone_densities = self.density_analyzer.classify_zones(raw_density)
        self.density_analyzer.update_history(raw_density)
        
        # 4. Detect bottlenecks & hazards
        self.hazard_zones = self.density_analyzer.detect_hazards(raw_density)
        self.bottlenecks = [
            Point(x=float(hz.cell_x), y=float(hz.cell_y)) 
            for hz in self.hazard_zones 
            if hz.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)
        ]
        
        # 5. Generate alerts
        new_alerts = self.density_analyzer.generate_alerts(self.hazard_zones)
        self.alerts = new_alerts[:self.max_alerts]
        self.alert_history.extend(new_alerts)
        if len(self.alert_history) > 200:
            self.alert_history = self.alert_history[-200:]
        
        # 6. Dynamic rerouting
        self._handle_rerouting(raw_density)
        
        # 7. Auto gate actuation
        self._auto_gate_control(raw_density)
        
        # 8. Update digital signage
        self._update_signage()
        
        # 9. Run prediction forecast periodically
        if self.tick_count % self.prediction_interval == 0 and len(self.agents) > 5:
            self.predictions = self.prediction_engine.forecast(
                self.agents, self.grid, self.walls_cache,
                self.gates, self.exits,
                sim_time_sec=self.sim_time,
            )
        elif len(self.agents) <= 5:
            self.prediction_engine.reset_forecast()

        # 10. Early warnings: zones the forecast expects to pile up in the next 15 minutes
        self.forecast_warnings = self._enrich_warnings(
            self.prediction_engine.early_warnings(raw_density, self.sim_time)
        )

        # 11. Autopilot: apply recommendations nobody has acted on within the response window
        self.autopilot.evaluate(self)
    
    def _handle_rerouting(self, raw_density: np.ndarray):
        """Reroute agents whose paths intersect critical zones."""
        if not self.bottlenecks:
            self.suggested_routes.clear()
            self.active_reroutes.clear()
            for agent in self.agents:
                agent.color = "#3b82f6"
                agent.rerouted = False
            return
        
        bottleneck_set = set()
        for p in self.bottlenecks:
            bottleneck_set.add((int(p.x), int(p.y)))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    bottleneck_set.add((int(p.x) + dx, int(p.y) + dy))
        
        reroute_count = 0
        self.active_reroutes.clear()
        if self.tick_count % 50 == 0:
            live = {a.id for a in self.agents}
            self._last_reroute = {i: t for i, t in self._last_reroute.items() if i in live}
        
        for agent in self.agents:
            if agent.status == "arrived":
                continue
            
            # Check if path goes through bottleneck
            path_through_bottleneck = False
            for p in agent.path[:15]:
                if (int(p.x), int(p.y)) in bottleneck_set:
                    path_through_bottleneck = True
                    break
            
            if path_through_bottleneck:
                # Skip agents re-routed recently, and cap the work done per tick
                if self.tick_count - self._last_reroute.get(agent.id, -10**9) < REROUTE_COOLDOWN_TICKS:
                    continue
                if reroute_count >= MAX_REROUTES_PER_TICK:
                    continue
                start_int = Point(x=float(int(agent.pos.x)), y=float(int(agent.pos.y)))
                
                # Use density-weighted pathfinding to find best alternative
                alt_routes = compute_alternative_routes(
                    self.grid, raw_density, start_int, self.exits
                )
                
                if alt_routes:
                    best_goal, best_path, best_cost = alt_routes[0]
                    if best_path:
                        agent.path = best_path[1:] if len(best_path) > 1 else best_path
                        agent.goal = best_goal
                        self.suggested_routes[agent.id] = best_path
                        agent.color = "#f59e0b"  # Amber for rerouted
                        agent.rerouted = True
                        reroute_count += 1
                        self._last_reroute[agent.id] = self.tick_count
                        
                        # Record active reroute
                        if len(self.active_reroutes) < 10:
                            self.active_reroutes.append(RerouteRecommendation(
                                reroute_id=f"reroute_{agent.id}",
                                origin_zone=f"cell_{int(agent.pos.x)}_{int(agent.pos.y)}",
                                congested_zone=f"bottleneck_zone",
                                target_zone=f"cell_{int(best_goal.x)}_{int(best_goal.y)}",
                                path_polyline=best_path[:10],
                            ))
            elif not agent.rerouted:
                agent.color = "#3b82f6"
    
    def _auto_gate_control(self, raw_density: np.ndarray):
        """Automatically throttle gates that feed into congested downstream zones."""
        for gate_state in self.gate_objects:
            # Operator override wins: don't let auto-control undo a manual
            # Throttle/Close on the next tick. Manual "Open" hands control back.
            if str(gate_state.status).startswith("MANUAL_"):
                continue
            gx, gy = int(gate_state.position.x), int(gate_state.position.y)
            
            # Check density in downstream area (5-cell radius ahead of gate)
            max_downstream_density = 0.0
            for dx in range(1, 8):
                for dy in range(-2, 3):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        max_downstream_density = max(max_downstream_density, raw_density[ny][nx])
            
            # Auto-throttle logic
            if max_downstream_density > 3.5:
                gate_state.action = GateAction.THROTTLE_FLOW
                gate_state.target_rate_per_sec = 0.5
                gate_state.status = "THROTTLED"
            elif max_downstream_density > 2.0:
                gate_state.action = GateAction.THROTTLE_FLOW
                gate_state.target_rate_per_sec = 2.0
                gate_state.status = "RESTRICTED"
            else:
                gate_state.action = GateAction.OPEN_FULL
                gate_state.target_rate_per_sec = 3.5
                gate_state.status = "OPEN"
    
    def _update_signage(self):
        """Update digital signage based on current congestion state."""
        has_bottleneck = len(self.bottlenecks) > 0
        
        for sign in self.signs:
            if has_bottleneck:
                # Check if any bottleneck is near this sign's default direction
                sign.active = True
                # Simple logic: redirect traffic away from bottlenecks
                closest_bottleneck = None
                min_dist = float('inf')
                for bp in self.bottlenecks:
                    d = (sign.position.x - bp.x)**2 + (sign.position.y - bp.y)**2
                    if d < min_dist:
                        min_dist = d
                        closest_bottleneck = bp
                
                if closest_bottleneck and min_dist < 100:
                    # Redirect: if bottleneck is to the right, suggest going up/down
                    if closest_bottleneck.x > sign.position.x:
                        if closest_bottleneck.y > sign.position.y:
                            sign.direction = "UP"
                            sign.message = "↑ Divert: Use Upper Corridor"
                        else:
                            sign.direction = "DOWN"
                            sign.message = "↓ Divert: Use Lower Corridor"
                    else:
                        sign.direction = "RIGHT"
                        sign.message = "→ Continue: Path Clear"
            else:
                sign.active = False
                sign.message = "→ Normal Flow"
                
    def set_sim_speed(self, speed: float):
        """Set simulation speed multiplier."""
        self.sim_speed = max(0.0, min(10.0, speed))
    
    def actuate_gate(self, gate_id: str, action: str, rate: float = 3.5):
        """Manually control a gate."""
        for gate in self.gate_objects:
            if gate.gate_id == gate_id:
                gate.action = GateAction(action)
                gate.target_rate_per_sec = rate
                # Open = give the gate back to auto-control; anything else is a manual hold
                gate.status = "OPEN" if action == "OPEN_FULL" else "MANUAL_" + action
                break
                
    # ─────────────────────────────────────────────
    # Recommendations, zone names and diverts
    # ─────────────────────────────────────────────

    @staticmethod
    def _gate_name(gate_id: str) -> str:
        return str(gate_id).replace("gate_", "Gate ")

    def zone_label(self, cell_x: int, cell_y: int) -> str:
        """Same naming the dashboard uses: nearest concession cell within 6 cells is 'Food Court N'."""
        if self._landmarks is None:
            self._landmarks, n = [], 0
            for y, row in enumerate(self.grid):
                for x, cell in enumerate(row):
                    if cell == "concession":
                        n += 1
                        self._landmarks.append((x, y, f"Food Court {n}"))
        best, best_d = None, float("inf")
        for x, y, label in self._landmarks:
            d = ((x - cell_x) ** 2 + (y - cell_y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = label, d
        return best if best is not None and best_d <= 6 else f"Zone ({cell_x}, {cell_y})"

    def _enrich_warnings(self, warnings: List[dict]) -> List[dict]:
        """
        Attach the recommended divert to each early warning: the gate feeding the zone
        (nearest by position), the least-loaded open gate to send arrivals to instead,
        and how much inflow has to drop for the forecast peak to fall under the LoS-C
        ceiling. The dashboard button and Autopilot both use these exact values.
        """
        def open_gate(g):
            status = str(g.status)
            return g.action != GateAction.CLOSE and "THROTTL" not in status and "RESTRICT" not in status

        for w in warnings:
            origin = min(
                self.gate_objects,
                key=lambda g: (g.position.x - w["cell_x"]) ** 2 + (g.position.y - w["cell_y"]) ** 2,
                default=None,
            )
            candidates = [g for g in self.gate_objects if open_gate(g) and (origin is None or g.gate_id != origin.gate_id)]
            target = min(candidates, key=lambda g: (g.queue_length, -g.target_rate_per_sec), default=None)

            peak = w["peak_density"]
            pct = 0
            if peak > 1.08:
                pct = int((peak - 1.08) / peak * 100 + 0.5)
                pct = max(5, min(95, pct))

            w["origin_gate"] = origin.gate_id if origin else None
            w["target_gate"] = target.gate_id if target else None
            w["divert_pct"] = pct
        return warnings

    def divert_gate(self, origin_id: str, target_id: str, pct, source: str = "operator",
                    context: dict = None) -> bool:
        """Send pct% of new arrivals bound for origin_id to target_id for DIVERT_DURATION_SEC."""
        try:
            pct = max(1.0, min(100.0, float(pct)))
        except (TypeError, ValueError):
            return False
        ids = {g.gate_id for g in self.gate_objects}
        if origin_id not in ids or target_id not in ids or origin_id == target_id:
            return False

        self._diverts[origin_id] = {
            "target": target_id,
            "pct": pct,
            "until": self.sim_time + DIVERT_DURATION_SEC,
            "source": source,
        }
        self.autopilot.clear_pending(origin_id)

        by_id = {g.gate_id: g for g in self.gate_objects}
        o, t = by_id[origin_id], by_id[target_id]
        head = f"diverted {int(pct)}% of {self._gate_name(origin_id)} arrivals to {self._gate_name(target_id)}"
        if source == "autopilot" and context:
            eta = context.get("eta_sec", 0)
            when = "now" if eta <= 0 else (f"in {int(round(eta / 60))} min" if eta >= 60 else "in <1 min")
            message = (f"Autopilot {head}. {context.get('zone')} forecast LoS "
                       f"{'CRITICAL' if context.get('peak_density', 0) >= 3.5 else 'F'} "
                       f"({context.get('peak_density', 0):.1f} p/m²) {when}; "
                       f"no operator response for {context.get('waited_sec', 0)}s.")
        else:
            message = f"Operator {head}."
        self.autopilot.record(
            "divert", source, message, self.sim_time,
            {
                "origin": origin_id, "target": target_id, "pct": int(pct),
                "origin_queue": o.queue_length, "target_queue": t.queue_length,
                **(context or {}),
            },
        )
        return True

    def get_state(self) -> dict:
        total_rerouted = sum(1 for a in self.agents if a.rerouted)
        densities = [zd.density for zd in self.zone_densities] if self.zone_densities else [0.0]
        
        return SimulationState(
            agents=self.agents,
            real_agents=self.real_agents,
            bottlenecks=self.bottlenecks,
            suggested_routes=self.suggested_routes,
            zone_densities=self.zone_densities[:100],  # Cap payload
            hazard_zones=self.hazard_zones[:30],
            alerts=self.alerts[:20],
            gates=self.gate_objects,
            signs=self.signs,
            predictions=self.predictions[:10],
            total_agents=len(self.agents),
            total_rerouted=total_rerouted,
            avg_density=round(sum(densities) / max(len(densities), 1), 2),
            max_density=round(max(densities), 2) if densities else 0.0,
            sim_time_sec=round(self.sim_time, 1),
            sim_speed=self.sim_speed,
            clock_hour=round(self.arrival_model.hour_at(self.sim_time), 2),
            forecast_warnings=self.forecast_warnings,
            autopilot=self.autopilot.status(),
            active_diverts=[
                {"origin": o, "target": d["target"], "pct": d["pct"],
                 "remaining_sec": round(max(0.0, d["until"] - self.sim_time), 1)}
                for o, d in self._diverts.items() if d["until"] > self.sim_time
            ],
        ).model_dump()
    
    def get_heatmap_data(self) -> dict:
        """Return heatmap + flow vector data for frontend rendering."""
        return {
            "heatmap": self.density_heatmap,
            "flow_vectors": self.flow_vectors,
            "width": self.width,
            "height": self.height,
        }