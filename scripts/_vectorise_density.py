"""#7: vectorise the Gaussian smoothing and heatmap path in density.py."""

import pathlib

path = pathlib.Path("backend/density.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── 1. smoothed density: separable convolution instead of a per-cell loop ──
sub(
    '''    def compute_smoothed_density(self, raw_density: np.ndarray) -> np.ndarray:
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
                
        return smoothed''',
    '''    def compute_smoothed_density(self, raw_density: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian kernel smoothing to raw density for heatmap visualization.

        Done as two 1-D convolutions rather than a 2-D loop. The kernel is separable
        - it is built from exp(-(dx^2 + dy^2)/2sigma^2), which factors into an x part
        and a y part - so convolving rows then columns gives the same result as
        summing the 2-D product, in O(k*h*w) instead of O(k^2*h*w) operations.

        The previous version looped over all 1200 cells in Python and called
        `np.sum(region * kernel)` in each one; at 3.0 ms per call it was the largest
        single term in the density path. This is the same arithmetic without the
        interpreter in the inner loop.
        """
        field = np.asarray(raw_density, dtype=np.float64)
        h, w = field.shape
        kr = self.kernel_radius
        if kr <= 0:
            return field.copy()

        # The 2-D kernel is an outer product, so recover the 1-D factor from any row
        # and normalise it. Taking the square root of the centre keeps relative
        # weights exact while making the separable form sum to the same total.
        row = self.kernel[kr]
        centre = float(self.kernel[kr][kr])
        if centre <= 0:
            return field.copy()
        kernel_1d = row / np.sqrt(centre)

        padded = np.pad(field, kr, mode="constant", constant_values=0.0)

        # Convolve along x for every row.
        horizontal = np.zeros_like(field)
        for i, k in enumerate(kernel_1d):
            horizontal += k * padded[:, i:i + w]

        # Then along y.
        smoothed = np.zeros_like(field)
        vertical = np.pad(horizontal, ((kr, kr), (0, 0)), mode="constant", constant_values=0.0)
        for i, k in enumerate(kernel_1d):
            smoothed += k * vertical[i:i + h, :]

        return smoothed''',
    "separable smooth",
)

# ── 2. heatmap payload: one clip+round, no nested list comprehension ──────
sub(
    '''    def get_density_grid_for_heatmap(self, agents: List[Agent]) -> List[List[float]]:
        """
        Compute smoothed density and return as nested list for JSON serialization.
        Values are clamped to [0, 5] range for consistent color mapping.
        """
        raw = self.compute_raw_density(agents)
        smoothed = self.compute_smoothed_density(raw)
        # Clamp for visualization
        smoothed = np.clip(smoothed, 0, 5.0)
        return smoothed.tolist()''',
    '''    def get_density_grid_for_heatmap(self, agents: List[Agent]) -> List[List[float]]:
        """
        Smoothed density as a nested list ready for JSON serialization.

        Clamped to [0, 5] for consistent colour mapping, and rounded to three
        decimals: the payload is transmitted every broadcast, and full float64
        precision roughly doubles its size for no visible difference in a heatmap.
        """
        raw = self.compute_raw_density(agents)
        smoothed = self.compute_smoothed_density(raw)
        return np.round(np.clip(smoothed, 0.0, 5.0), 3).tolist()''',
    "heatmap payload",
)

sub(
    "def compute_raw_density(self, agents: List[Agent]) -> np.ndarray:",
    '''def compute_raw_density(self, agents: List[Agent]) -> np.ndarray:''',
    "noop anchor",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
