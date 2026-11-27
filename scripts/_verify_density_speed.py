"""Verify the vectorised smoothing is numerically identical to the original loop."""

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.density import DensityAnalyzer


def reference_smooth(raw, kernel, kr):
    """The original per-cell implementation, kept here as the reference."""
    h, w = raw.shape
    ks = 2 * kr + 1
    padded = np.pad(raw, kr, mode="constant", constant_values=0)
    out = np.zeros_like(raw, dtype=np.float64)
    for y in range(h):
        for x in range(w):
            out[y][x] = np.sum(padded[y:y + ks, x:x + ks] * kernel)
    return out


da = DensityAnalyzer(40, 30, kernel_radius=2)
print(f"kernel_radius={da.kernel_radius}, kernel shape {da.kernel.shape}")
print(f"kernel sum = {da.kernel.sum():.12f}")

rng = np.random.default_rng(0)
worst = 0.0
for trial in range(5):
    raw = rng.random((30, 40)) * 3.0
    ref = reference_smooth(raw, da.kernel, da.kernel_radius)
    got = da.compute_smoothed_density(raw)
    delta = float(np.abs(ref - got).max())
    worst = max(worst, delta)
    print(f"  trial {trial}: max abs delta {delta:.3e}")

print(f"\nworst delta across trials: {worst:.3e}")
print("identical within float rounding:", worst < 1e-12)

# Mass preservation.
raw = np.zeros((30, 40))
raw[15, 20] = 9.0
print(f"mass in {raw.sum():.1f}, mass out {da.compute_smoothed_density(raw).sum():.6f}")

# Timing, old vs new.
raw = rng.random((30, 40)) * 2.0
t0 = time.perf_counter()
for _ in range(50):
    reference_smooth(raw, da.kernel, da.kernel_radius)
old_ms = (time.perf_counter() - t0) / 50 * 1000

t0 = time.perf_counter()
for _ in range(50):
    da.compute_smoothed_density(raw)
new_ms = (time.perf_counter() - t0) / 50 * 1000

print(f"\nper-cell loop : {old_ms:.3f} ms")
print(f"separable     : {new_ms:.3f} ms")
print(f"speedup       : {old_ms / new_ms:.1f}x")

# Heatmap payload.
class A:
    def __init__(self, x, y):
        self.pos = type("P", (), {"x": x, "y": y})()
        self.vel = type("V", (), {"x": 0.0, "y": 0.0})()
        self.status = "moving"

agents = [A(rng.uniform(0, 39), rng.uniform(0, 29)) for _ in range(300)]
t0 = time.perf_counter()
for _ in range(20):
    grid = da.get_density_grid_for_heatmap(agents)
heat_ms = (time.perf_counter() - t0) / 20 * 1000
print(f"\nheatmap payload: {heat_ms:.2f} ms, peak {max(max(r) for r in grid):.2f}")
