"""Re-apply the density time-to-choke unit fix (per-sample slope -> per-second rate)."""

import pathlib

path = pathlib.Path("backend/density.py")
text = path.read_text(encoding="utf-8")
before = text

report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


sub(
    "# Fruin LoS thresholds",
    "# Simulated seconds advanced per density-history sample. Must match\n"
    "# `simulation.TICK_DURATION_SEC`: history is appended once per tick, and\n"
    "# `_density_rise_rate` converts a per-sample change into a per-second rate\n"
    "# using this value. Duplicated rather than imported to avoid a circular import.\n"
    "TICK_DURATION_SEC = 0.5\n\n"
    "# Fruin LoS thresholds",
    "TICK_DURATION_SEC constant",
)

sub(
    "        return (recent[-1] - recent[0]) / len(recent)",
    "        # `detect_hazards` divides a density gap by this value to get a\n"
    "        # time-to-choke in seconds, so the denominator has to be elapsed simulated\n"
    "        # time - not the number of samples. Dividing by the sample count gave a\n"
    "        # per-sample slope, which overstated the rise by len(recent)/2 and inflated\n"
    "        # every reported time-to-choke by about 2.5x. A crowd reported as 2.5x\n"
    "        # further from a crush than it is, is exactly the error this dashboard\n"
    "        # exists to prevent.\n"
    "        span_sec = (len(recent) - 1) * TICK_DURATION_SEC\n"
    "        if span_sec <= 0:\n"
    "            return 0.0\n"
    "        return (recent[-1] - recent[0]) / span_sec",
    "per-second rate",
)

sub(
    '        """Compute density rise rate (people/m\u00b2/step) at a cell."""',
    '        """Density rise rate at a cell, in people/m2 per second."""',
    "docstring unit",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(before)} -> {len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
print("still has per-sample division:", "len(recent)" in text.split("_density_rise_rate")[-1][:900])
