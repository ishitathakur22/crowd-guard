"""Add the rng parameter to ArrivalModel and PredictionEngine."""

import pathlib

report = []


def patch(path, pairs):
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    for label, old, new in pairs:
        n = text.count(old)
        report.append((f"{path}: {label}", n))
        text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")


patch(
    "backend/ingress.py",
    [
        (
            "import ArrivalModel rng import",
            "from typing import List, Optional\n",
            "from typing import List, Optional\n\nfrom .rng import Rng\n",
        ),
        (
            "rng param",
            "        profile_path: Optional[str] = None,\n    ):",
            "        profile_path: Optional[str] = None,\n"
            "        rng: Optional[Rng] = None,\n    ):",
        ),
        (
            "rng init",
            "        self.curve, self.meta = self._load_curve(profile_path or _PROFILE_PATH)",
            "        # Arrival randomness lives on a derived substream so that later work\n"
            "        # (Poisson arrivals) does not shift the physics draws.\n"
            "        self.rng = rng or Rng()\n"
            "        self.curve, self.meta = self._load_curve(profile_path or _PROFILE_PATH)",
        ),
    ],
)

patch(
    "backend/prediction.py",
    [
        (
            "rng import",
            "from .pathfinding import a_star_search",
            "from .pathfinding import a_star_search\nfrom .rng import Rng",
        ),
        (
            "rng param",
            "    def __init__(self, width: int, height: int):",
            "    def __init__(self, width: int, height: int, rng: Rng = None):",
        ),
        (
            "rng init",
            "        self.density_analyzer = DensityAnalyzer(width, height)",
            "        self.rng = rng or Rng()\n"
            "        self.density_analyzer = DensityAnalyzer(width, height)",
        ),
    ],
)

bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"\nmisses: {bad}")
