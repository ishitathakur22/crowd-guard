"""
Density Analyzer — Grid-based spatial density estimation with Fruin LoS classification.

Computes per-cell density using Gaussian kernel smoothing, classifies zones by
Fruin's Level-of-Service thresholds, tracks density trends over time, and
calculates flow rates for corridor segments.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from .models import (
    Agent, Point, FruinLoS, ZoneDensity, HazardZone, 
    AlertSeverity, AlertEvent
)
import time
import uuid


# Simulated seconds advanced per density-history sample. Must match
# `simulation.TICK_DURATION_SEC`: history is appended once per tick, and
# `_density_rise_rate` converts a per-sample change into a per-second rate
# using this value. Duplicated rather than imported to avoid a circular import.
TICK_DURATION_SEC = 0.5

# Fruin LoS thresholds (people/m²)
FRUIN_THRESHOLDS = [
    (0.31, FruinLoS.A),
    (0.43, FruinLoS.B),
    (1.08, FruinLoS.C),
    (1.54, FruinLoS.D),
    (2.17, FruinLoS.E),
    (3.50, FruinLoS.F),
    (float('inf'), FruinLoS.CRITICAL),
]

# Map LoS to alert severity
LOS_SEVERITY = {
    FruinLoS.A: None,
    FruinLoS.B: None,
    FruinLoS.C: None,
    FruinLoS.D: AlertSeverity.INFO,
    FruinLoS.E: AlertSeverity.WARNING,
    FruinLoS.F: AlertSeverity.CRITICAL,
    FruinLoS.CRITICAL: AlertSeverity.EMERGENCY,
}


def local_density(field, radius: int = 1):
    """
    The single measure of local density, in people/m2.

    A cell is 1 m2, so the mean people-count over a (2r+1) x (2r+1) window is a
    density in people/m2. Smoothing is necessary because a raw per-cell count is a
    very noisy estimate of how crowded a place is:

      * One person standing alone in a cell reads as 1.0 p/m2, which is Fruin LoS
        C ("restricted movement"). That is plainly wrong - a lone pedestrian is
        LoS A.
      * Three people spread across two adjacent cells is genuine congestion, but a
        per-cell maximum reports 2.0 rather than the 1.5 the pair really averages.

    Everything that reports a density must go through this function. Previously
    `classify_zones` and `detect_hazards` used per-cell counts while the forecast's
    early warnings used a neighbourhood maximum, so the same corner of the venue
    showed two different numbers on the same dashboard - and a threshold applied to
    one was meaningless when compared against the other.
    """
    import numpy as np

    field = np.asarray(field, dtype=np.float64)
    r = int(radius)
    if r <= 0:
        return field
    h, w = field.shape
    padded = np.pad(field, r, mode="constant", constant_values=0.0)
    size = 2 * r + 1
    total = np.zeros_like(field)
    for dy in range(size):
        for dx in range(size):
            total += padded[dy:dy + h, dx:dx + w]
    return total / float(size * size)


def classify_fruin(density: float) -> FruinLoS:
    """Classify density value to Fruin Level-of-Service."""
    for threshold, los in FRUIN_THRESHOLDS:
        if density < threshold:
            return los
    return FruinLoS.CRITICAL


class DensityAnalyzer:
    """
    Spatial density analysis engine.
    
    Uses a grid-based approach where each cell represents 1m². Computes raw
    headcount density, applies Gaussian kernel smoothing for heatmap visualization,
    and classifies each cell by Fruin's Level-of-Service.
    """
    
    def __init__(self, width: int, height: int, kernel_radius: int = 2):
        self.width = width
        self.height = height
        self.kernel_radius = kernel_radius
        
        # Pre-compute Gaussian kernel for smoothing
        self.kernel = self._build_gaussian_kernel(kernel_radius, sigma=1.0)
        
        # Density history for trend analysis (rolling window)
        self.density_history: List[np.ndarray] = []
        self.history_max_len = 20  # Keep last 20 snapshots
        
        # Alert deduplication
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_cooldown: Dict[str, float] = {}
        
    def _build_gaussian_kernel(self, radius: int, sigma: float) -> np.ndarray:
        """Build a 2D Gaussian kernel for density smoothing."""
        size = 2 * radius + 1
        kernel = np.zeros((size, size))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                kernel[dy + radius][dx + radius] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel
    
    def compute_raw_density(self, agents: List[Agent]) -> np.ndarray:
        """Compute raw per-cell agent count density grid."""
        density = np.zeros((self.height, self.width), dtype=np.float64)
        
        for agent in agents:
            if agent.status == "arrived":
                continue
            cx = int(agent.pos.x)
            cy = int(agent.pos.y)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                density[cy][cx] += 1.0
                
        return density
    
    def local_density(self, raw_density: np.ndarray, radius: int = 1) -> np.ndarray:
        """Local density in people/m2 for every cell. See the module-level helper."""
        return local_density(raw_density, radius)

    def compute_smoothed_density(self, raw_density: np.ndarray) -> np.ndarray:
        """Apply Gaussian kernel smoothing to raw density for heatmap visualization."""
        h, w = raw_density.shape
        kr = self.kernel_radius
        ks = 2 * kr + 1
        
        # Pad the density grid
        padded = np.pad(raw_density, kr, mode='constant', constant_values=0)
        smoothed = np.zeros_like(raw_density)
        
        for y in range(h):
            for x in range(w):
                region = padded[y:y + ks, x:x + ks]
                smoothed[y][x] = np.sum(region * self.kernel)
                
        return smoothed
    
    def compute_flow_vectors(self, agents: List[Agent]) -> np.ndarray:
        """
        Compute per-cell average velocity vectors for flow visualization.
        Returns (height, width, 2) array with [vx, vy] per cell.
        """
        flow = np.zeros((self.height, self.width, 2), dtype=np.float64)
        counts = np.zeros((self.height, self.width), dtype=np.float64)
        
        for agent in agents:
            if agent.status == "arrived":
                continue
            cx = int(agent.pos.x)
            cy = int(agent.pos.y)
            if 0 <= cx < self.width and 0 <= cy < self.height:
                flow[cy][cx][0] += agent.vel.x
                flow[cy][cx][1] += agent.vel.y
                counts[cy][cx] += 1.0
        
        # Average velocities
        mask = counts > 0
        flow[mask, 0] /= counts[mask]
        flow[mask, 1] /= counts[mask]
        
        return flow
    
    def classify_zones(self, raw_density: np.ndarray) -> List[ZoneDensity]:
        """
        Classify every non-zero cell by Fruin LoS.

        `density` and `los_level` come from `local_density`; `agent_count` stays the
        raw per-cell headcount, because that is what it says it is. Only cells that
        actually hold someone are emitted, so the payload stays bounded.
        """
        zones = []
        smoothed = local_density(raw_density, self.kernel_radius)

        ys, xs = np.nonzero(raw_density > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            d = float(smoothed[y][x])
            zones.append(ZoneDensity(
                zone_id=f"cell_{x}_{y}",
                cell_x=x,
                cell_y=y,
                density=round(d, 2),
                agent_count=int(raw_density[y][x]),
                los_level=classify_fruin(d),
                trend=self._compute_trend(x, y),
            ))

        return zones
    
    def detect_hazards(
        self, 
        raw_density: np.ndarray, 
        predicted_density: Optional[np.ndarray] = None
    ) -> List[HazardZone]:
        """
        Detect hazard zones where density exceeds LoS D threshold.
        If predicted_density is provided, include future forecast data.
        """
        hazards = []

        # Same measure as `classify_zones`, so a hazard and a zone label never
        # disagree about how crowded the same cell is.
        current = local_density(raw_density, self.kernel_radius)
        if predicted_density is not None:
            predicted = local_density(predicted_density, self.kernel_radius)
        else:
            predicted = current

        ys, xs = np.nonzero(raw_density > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
                current_d = float(current[y][x])
                predicted_d = float(predicted[y][x])
                
                # Flag if current OR predicted density is concerning (LoS D+)
                max_d = max(current_d, predicted_d)
                if max_d >= 1.08:  # LoS D threshold
                    los = classify_fruin(max_d)
                    severity_map = {
                        FruinLoS.D: AlertSeverity.INFO,
                        FruinLoS.E: AlertSeverity.WARNING,
                        FruinLoS.F: AlertSeverity.CRITICAL,
                        FruinLoS.CRITICAL: AlertSeverity.EMERGENCY,
                    }
                    severity = severity_map.get(los, AlertSeverity.INFO)
                    
                    # Estimate time to choke: how fast is density rising?
                    trend_rate = self._density_rise_rate(x, y)
                    if trend_rate > 0 and max_d < 3.5:
                        ttc = (3.5 - current_d) / trend_rate  # seconds to critical
                    else:
                        ttc = 0.0
                    
                    hazards.append(HazardZone(
                        zone_id=f"cell_{x}_{y}",
                        cell_x=x,
                        cell_y=y,
                        current_density=round(current_d, 2),
                        predicted_density=round(predicted_d, 2),
                        severity=severity,
                        fruin_los=los,
                        time_to_choke_sec=round(ttc, 1),
                    ))
                    
        return hazards
    
    def generate_alerts(self, hazards: List[HazardZone]) -> List[AlertEvent]:
        """Generate de-duplicated alerts from hazard zones."""
        now = time.time()
        new_alerts = []
        
        for hz in hazards:
            if hz.severity in (AlertSeverity.INFO,):
                continue  # Don't alert on INFO-level
                
            # Cooldown: don't re-alert same zone within 10s
            if hz.zone_id in self.alert_cooldown:
                if now - self.alert_cooldown[hz.zone_id] < 10.0:
                    continue
            
            severity_messages = {
                AlertSeverity.WARNING: f"⚠ Density rising at ({hz.cell_x},{hz.cell_y}): {hz.current_density:.1f} p/m² (LoS {hz.fruin_los.value})",
                AlertSeverity.CRITICAL: f"🔴 CRITICAL congestion at ({hz.cell_x},{hz.cell_y}): {hz.current_density:.1f} p/m² — rerouting active",
                AlertSeverity.EMERGENCY: f"🚨 EMERGENCY: Crush risk at ({hz.cell_x},{hz.cell_y})! Density {hz.current_density:.1f} p/m² exceeds safety limit",
            }
            
            alert = AlertEvent(
                alert_id=str(uuid.uuid4())[:8],
                timestamp=now,
                severity=hz.severity,
                zone_id=hz.zone_id,
                message=severity_messages.get(hz.severity, f"Density alert at {hz.zone_id}"),
                density=hz.current_density,
                los_level=hz.fruin_los,
            )
            new_alerts.append(alert)
            self.alert_cooldown[hz.zone_id] = now
            self.active_alerts[hz.zone_id] = alert
            
        return new_alerts
    
    def update_history(self, raw_density: np.ndarray):
        """
        Store a density snapshot for trend analysis.

        The snapshot is smoothed with the same `local_density` measure as everything
        else, because `_compute_trend` and `_density_rise_rate` read this history and
        their results are compared against the densities those functions report.
        Storing raw counts here while reporting smoothed ones would have made the
        trend and the time-to-choke describe a different quantity from the LoS label
        printed beside them.
        """
        self.density_history.append(local_density(raw_density, self.kernel_radius))
        if len(self.density_history) > self.history_max_len:
            self.density_history.pop(0)
    
    def _compute_trend(self, x: int, y: int) -> str:
        """Determine if density is rising, falling, or stable at a cell."""
        if len(self.density_history) < 3:
            return "stable"
        
        recent = [h[y][x] for h in self.density_history[-5:]]
        if len(recent) < 2:
            return "stable"
        
        # Thresholds for a smoothed density. A per-cell count moves in whole
        # people, so 0.5 was a reasonable cut there; a 3x3 mean moves in ninths, so
        # the equivalent sensitivity is about a fifth.
        diff = recent[-1] - recent[0]
        if diff > 0.2:
            return "rising"
        elif diff < -0.2:
            return "falling"
        return "stable"
    
    def _density_rise_rate(self, x: int, y: int) -> float:
        """Density rise rate at a cell, in people/m2 per second."""
        if len(self.density_history) < 3:
            return 0.0
        
        recent = [h[y][x] for h in self.density_history[-5:]]
        if len(recent) < 2:
            return 0.0
        
        # Linear regression-ish: just slope over recent window
        # `detect_hazards` divides a density gap by this value to get a
        # time-to-choke in seconds, so the denominator has to be elapsed simulated
        # time - not the number of samples. Dividing by the sample count gave a
        # per-sample slope, which overstated the rise by len(recent)/2 and inflated
        # every reported time-to-choke by about 2.5x. A crowd reported as 2.5x
        # further from a crush than it is, is exactly the error this dashboard
        # exists to prevent.
        span_sec = (len(recent) - 1) * TICK_DURATION_SEC
        if span_sec <= 0:
            return 0.0
        return (recent[-1] - recent[0]) / span_sec
    
    def get_density_grid_for_heatmap(self, agents: List[Agent]) -> List[List[float]]:
        """
        Compute smoothed density and return as nested list for JSON serialization.
        Values are clamped to [0, 5] range for consistent color mapping.
        """
        raw = self.compute_raw_density(agents)
        smoothed = self.compute_smoothed_density(raw)
        # Clamp for visualization
        smoothed = np.clip(smoothed, 0, 5.0)
        return smoothed.tolist()
    
    def get_flow_vectors_for_viz(self, agents: List[Agent]) -> List[Dict]:
        """
        Return flow vectors as a list of {x, y, vx, vy, magnitude} dicts
        for cells with significant flow. Only returns cells with agents.
        """
        flow = self.compute_flow_vectors(agents)
        vectors = []
        
        for y in range(self.height):
            for x in range(self.width):
                vx, vy = flow[y][x]
                mag = np.sqrt(vx**2 + vy**2)
                if mag > 0.05:  # Skip negligible flow
                    vectors.append({
                        "x": x,
                        "y": y,
                        "vx": round(float(vx), 3),
                        "vy": round(float(vy), 3),
                        "mag": round(float(mag), 3),
                    })
                    
        return vectors
