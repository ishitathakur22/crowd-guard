"""Unify the local-density measure across every safety surface in density.py."""

import pathlib

path = pathlib.Path("backend/density.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── 1. one shared local-density measure ─────────────────────────────────
sub(
    "def classify_fruin(density: float) -> FruinLoS:",
    '''def local_density(field, radius: int = 1):
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


def classify_fruin(density: float) -> FruinLoS:''',
    "local_density function",
)

# ── 2. instance method so analyzers can call it ──────────────────────────
sub(
    "    def compute_smoothed_density(self, raw_density: np.ndarray) -> np.ndarray:",
    '''    def local_density(self, raw_density: np.ndarray, radius: int = 1) -> np.ndarray:
        """Local density in people/m2 for every cell. See the module-level helper."""
        return local_density(raw_density, radius)

    def compute_smoothed_density(self, raw_density: np.ndarray) -> np.ndarray:''',
    "local_density method",
)

# ── 3. classify_zones uses the shared measure ────────────────────────────
sub(
    '''    def classify_zones(self, raw_density: np.ndarray) -> List[ZoneDensity]:
        """Classify every non-zero cell by Fruin LoS."""
        zones = []
        
        for y in range(self.height):
            for x in range(self.width):
                d = raw_density[y][x]
                if d > 0:
                    trend = self._compute_trend(x, y)
                    zones.append(ZoneDensity(
                        zone_id=f"cell_{x}_{y}",
                        cell_x=x,
                        cell_y=y,
                        density=round(d, 2),
                        agent_count=int(d),
                        los_level=classify_fruin(d),
                        trend=trend,
                    ))
                    
        return zones''',
    '''    def classify_zones(self, raw_density: np.ndarray) -> List[ZoneDensity]:
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

        return zones''',
    "classify_zones",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
