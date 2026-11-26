"""Switch prediction.py's early warnings onto the shared local-density measure."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── replace the bespoke neighbourhood-max helper ─────────────────────────
sub(
    '''def _local_max(field: np.ndarray, r: int = 1) -> np.ndarray:
    """Max of each cell's (2r+1)x(2r+1) neighbourhood. Forecast agents drift a cell or two
    from where the live ones stand, so matching on the neighbourhood is more robust than
    on the single cell."""
    h, w = field.shape
    padded = np.pad(field, r, mode="constant", constant_values=0)
    out = np.zeros_like(field)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            np.maximum(out, padded[dy:dy + h, dx:dx + w], out=out)
    return out''',
    '''def _local_max(field: np.ndarray, r: int = 1) -> np.ndarray:
    """
    Local density, as the one shared measure defined in `density.local_density`.

    This used to take the *maximum* of each cell's neighbourhood while the live
    analyzer reported a per-cell count, so a warning's "density" and the LoS label
    printed beside it were two different quantities. Both now go through the same
    function, so a threshold means the same thing on either side.

    The neighbourhood look is still what makes the comparison robust: forecast agents
    drift a cell or two from where the live ones stand, so a smoothed window agrees
    better than a single cell would.
    """
    return local_density(field, r)''',
    "_local_max delegation",
)

sub(
    "from .density import DensityAnalyzer, classify_fruin",
    "from .density import DensityAnalyzer, classify_fruin, local_density",
    "import local_density",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
