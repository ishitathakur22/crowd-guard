"""Point detect_hazards and the density history at the shared local-density measure."""

import pathlib

path = pathlib.Path("backend/density.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── detect_hazards: same measure as classify_zones ───────────────────────
sub(
    '''        hazards = []
        
        for y in range(self.height):
            for x in range(self.width):
                current_d = raw_density[y][x]
                predicted_d = predicted_density[y][x] if predicted_density is not None else current_d''',
    '''        hazards = []

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
                predicted_d = float(predicted[y][x])''',
    "detect_hazards measure",
)

# ── history stores the measure that trend and rise-rate are defined on ──
sub(
    '''    def update_history(self, raw_density: np.ndarray):
        """Store density snapshot for trend analysis."""
        self.density_history.append(raw_density.copy())''',
    '''    def update_history(self, raw_density: np.ndarray):
        """
        Store a density snapshot for trend analysis.

        The snapshot is smoothed with the same `local_density` measure as everything
        else, because `_compute_trend` and `_density_rise_rate` read this history and
        their results are compared against the densities those functions report.
        Storing raw counts here while reporting smoothed ones would have made the
        trend and the time-to-choke describe a different quantity from the LoS label
        printed beside them.
        """
        self.density_history.append(local_density(raw_density, self.kernel_radius))''',
    "update_history",
)

# ── trend threshold for smoothed values ────────────────────────────────
sub(
    '''        diff = recent[-1] - recent[0]
        if diff > 0.5:
            return "rising"
        elif diff < -0.5:
            return "falling"
        return "stable"''',
    '''        # Thresholds for a smoothed density. A per-cell count moves in whole
        # people, so 0.5 was a reasonable cut there; a 3x3 mean moves in ninths, so
        # the equivalent sensitivity is about a fifth.
        diff = recent[-1] - recent[0]
        if diff > 0.2:
            return "rising"
        elif diff < -0.2:
            return "falling"
        return "stable"''',
    "trend thresholds",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
