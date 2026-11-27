"""Phase C: give exits a finite discharge capacity so surplus queues instead of vanishing."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    if n:
        text = text.replace(old, new, 1)


# ── configuration ───────────────────────────────────────────────────────
sub(
    "        self.turnover_scale = 1.0\n        self._potential_cache = {}",
    "        self.turnover_scale = 1.0\n\n"
    "        # Discharge capacity of one exit cell, people per second.\n"
    "        #\n"
    "        # The continuum model used to do `rho[exit_mask] = 0.0`, removing everyone\n"
    "        # who reached an exit no matter how many arrived at once. A door is not\n"
    "        # infinitely wide: the SFPE/Fruin specific flow through a 1 m opening is\n"
    "        # about 1.3 people per second. Without this limit the forecast drains a\n"
    "        # crowded venue faster than reality, which is the measured failure - at\n"
    "        # the hotspot scenario it loses 476-826 people the truth retains.\n"
    "        #\n"
    "        # Surplus now queues at the threshold rather than disappearing, which is\n"
    "        # what a real crowd does and what makes a bottleneck a bottleneck.\n"
    "        self.exit_capacity_per_sec = 1.3\n"
    "        self._potential_cache = {}",
    "config",
)

# ── discharge at capacity instead of instant removal ────────────────────
sub(
    "            rho[exit_mask] = 0.0\n            np.clip(rho, 0.0, self.jam_density, out=rho)",
    "            # Discharge at a finite rate. Anything beyond the exit's throughput\n"
    "            # stays and forms a queue at the door, which is the behaviour the\n"
    "            # instant-zero version could not represent.\n"
    "            if exit_cells:\n"
    "                per_step = self.exit_capacity_per_sec * self.sim_dt\n"
    "                idx = np.asarray(exit_cells)\n"
    "                rho[idx] = np.maximum(0.0, rho[idx] - per_step)\n"
    "            np.clip(rho, 0.0, self.jam_density, out=rho)",
    "discharge",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
