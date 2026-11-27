"""Add forecast_uncertainty to the SimulationState schema so it reaches the client."""

import pathlib

path = pathlib.Path("backend/models.py")
text = path.read_text(encoding="utf-8")

old = '    active_diverts: List[Dict] = []  # [{origin, target, pct, remaining_sec}] operator gate diverts in effect'
new = (
    '    active_diverts: List[Dict] = []  # [{origin, target, pct, remaining_sec}] operator gate diverts in effect\n'
    '    # Headline numbers from the exceedance ensemble: how confident the forecast is.\n'
    '    # {"available": bool, "replicates": int, "max_p_exceed": float,\n'
    '    #  "cells_above_50pct": int, "cells_above_80pct": int, "max_sigma": float}\n'
    '    forecast_uncertainty: Dict = {}'
)

n = text.count(old)
print(f"anchor occurrences: {n}")
if n:
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print("added forecast_uncertainty to SimulationState")
