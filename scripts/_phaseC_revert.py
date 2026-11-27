"""Phase C stopping rule: revert finite exit capacity (measured, no improvement)."""

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


# Remove the attribute and its explanation.
start = text.find("        # Discharge capacity of one exit cell, people per second.")
if start != -1:
    end = text.find("        self.exit_capacity_per_sec = 1.3", start)
    if end != -1:
        end = text.find("\n", end) + 1
        text = text[:start] + text[end:]
        report.append(("attribute removed", 1))

# Restore instant discharge.
sub(
    "            # Discharge at a finite rate. Anything beyond the exit's throughput\n"
    "            # stays and forms a queue at the door, which is the behaviour the\n"
    "            # instant-zero version could not represent.\n"
    "            if exit_cells:\n"
    "                per_step = self.exit_capacity_per_sec * self.sim_dt\n"
    "                idx = np.asarray(exit_cells)\n"
    "                rho[idx] = np.maximum(0.0, rho[idx] - per_step)\n"
    "            np.clip(rho, 0.0, self.jam_density, out=rho)",
    "            # Exits absorb at once. A finite-throughput version was measured and\n"
    "            # reverted: see benchmarks/README.md section 3d. It changed hotspot\n"
    "            # busy-MAE from 1.526 to 1.525 - noise - so exit capacity is not the\n"
    "            # source of the over-drain and the simpler behaviour is kept.\n"
    "            rho[exit_mask] = 0.0\n"
    "            np.clip(rho, 0.0, self.jam_density, out=rho)",
    "instant discharge restored",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
print(f"  exit_capacity references remaining: {text.count('exit_capacity_per_sec')}")
