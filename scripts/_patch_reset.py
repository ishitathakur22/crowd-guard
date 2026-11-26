"""Clear last_forecast_grids in reset_forecast, then verify the module imports."""

import pathlib

path = pathlib.Path("backend/prediction.py")
text = path.read_text(encoding="utf-8")

old = "    def reset_forecast(self):\n        self.forecast_eta = None\n        self.forecast_peak = None"
new = (
    "    def reset_forecast(self):\n"
    "        self.forecast_eta = None\n"
    "        self.forecast_peak = None\n"
    "        self.last_forecast_grids = []"
)
n = text.count(old)
if n:
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
print(f"reset_forecast cleared: {n}")
