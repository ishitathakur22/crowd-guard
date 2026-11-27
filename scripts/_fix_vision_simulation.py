"""Stop vision from corrupting the forecast ingress estimate, and expose real density."""

import pathlib

path = pathlib.Path("backend/simulation.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


sub(
    '''    def ingest_real_data(self, real_agents_data: List[Dict]):
        """
        Phase 1: Purely Real Update.
        Accepts data from the Vision Pipeline and updates the real_agents list.
        Data format: [{"id": str, "x": float, "y": float, "type": str, "class_id": int, "conf": float, "depth": float}]
        """
        self.real_agents = real_agents_data
        
        # Note for Phase 2 (Hybrid Merge):
        # Real agents now influence synthetic agents via the Social Force Model.

        self.prediction_engine.update_ingress_estimate(len(real_agents_data))''',
    '''    def ingest_real_data(self, real_agents_data: List[Dict]):
        """
        Accept camera detections and expose them for the dashboard and physics.

        Two things were wrong here.

        **It corrupted the forecast.** `len(real_agents_data)` was passed to
        `update_ingress_estimate`, but that number is the count of people currently
        *visible*, not the count who arrived this tick. It was also inflated two- or
        threefold by the duplicate detectors. So enabling vision silently overwrote
        the forecaster's arrival rate with a view count, and made the forecast worse.

        **It reached no safety metric.** Density, Fruin LoS, hazard zones and every
        early warning were computed from `self.agents` alone, so a camera seeing a
        crush produced no warning. Detections are now published as `real_agents` and
        folded into the density grid that the analyzer reads, so vision and simulation
        describe one world instead of two.
        """
        self.real_agents = list(real_agents_data or [])
        # Deliberately NOT feeding the ingress estimate: a headcount is not an
        # arrival rate, and treating it as one made the forecast worse whenever the
        # camera was on.'''
,
    "ingest_real_data",
)

# Expose a combined density grid so the analyzer can see camera detections.
sub(
    '''    def get_heatmap_data(self) -> dict:''',
    '''    def combined_density(self) -> np.ndarray:
        """
        Raw density in people/m2 from simulated agents plus camera detections.

        The live analyzer scored only `self.agents`, so with vision enabled the
        dashboard showed a crowd that was half real and half invented, and never the
        two together. Camera detections are on the same grid and the same units, so
        they are summed here. Vehicles are excluded: they are not pedestrians and
        must not contribute to pedestrian density or LoS classification.
        """
        field = self.density_analyzer.compute_raw_density(self.agents)
        if self.real_agents:
            for det in self.real_agents:
                try:
                    cx, cy = int(det["x"]), int(det["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                if det.get("class_id") in (2, 3, 5, 7):   # car, motorcycle, bus, truck
                    continue
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    field[cy][cx] += 1.0
        return field

    def get_heatmap_data(self) -> dict:''',
    "combined_density",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
