"""Make the particle forecast record its grids too, so the harness can score both."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")

old = (
    "        self._build_eta_maps(grids, sim_time_sec)\n"
    "        return snapshots\n\n"
    "    # ─────────────────────────────────────────────\n"
    "    # EARLY WARNINGS"
)
new = (
    "        self._build_eta_maps(grids, sim_time_sec)\n"
    "        # Recorded so the accuracy harness can score either model through the\n"
    "        # same path; the continuum model does the same.\n"
    "        self.last_forecast_grids = grids\n"
    "        return snapshots\n\n"
    "    # ─────────────────────────────────────────────\n"
    "    # EARLY WARNINGS"
)
n = text.count(old)
if n:
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
print(f"particle grids recorded: {n}")
