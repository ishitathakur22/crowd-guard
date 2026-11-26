"""Re-apply the seeded-RNG wiring to backend/simulation.py (it was reverted)."""

import pathlib

path = pathlib.Path("backend/simulation.py")
text = path.read_text(encoding="utf-8")
original = text

checks = []


def sub(old, new, label):
    global text
    n = text.count(old)
    checks.append((label, n))
    text = text.replace(old, new)
    return n


# 1. imports
sub("import random\n", "", "drop 'import random'")
sub(
    "from typing import List, Dict, Tuple\n",
    "from typing import List, Dict, Tuple, Optional\n",
    "typing Optional",
)
sub(
    "from .autopilot import Autopilot\n",
    "from .autopilot import Autopilot\nfrom .rng import Rng, seed_from_env\n",
    "import Rng",
)

# 2. constructor signature + rng construction
sub(
    "    def __init__(self, width: int = 40, height: int = 30):",
    "    def __init__(self, width: int = 40, height: int = 30, seed: Optional[int] = None):",
    "__init__ seed param",
)
sub(
    "        self.height = height\n        self.grid = self._generate_default_grid()",
    "        self.height = height\n"
    "        # One seed decides the whole run: arrivals, spawn jitter, demographic\n"
    "        # draws, gate selection and the physics separation all come from\n"
    "        # substreams derived from it. Without this two runs are incomparable and\n"
    "        # no accuracy claim about the forecast can be checked against anything.\n"
    "        self.rng = Rng(seed if seed is not None else seed_from_env())\n"
    "        self.seed = self.rng.seed\n\n"
    "        self.grid = self._generate_default_grid()",
    "rng construction",
)

# 3. pass substreams into subsystems
sub(
    "self.sfm = SocialForceModel(dt=0.1)",
    'self.sfm = SocialForceModel(dt=0.1, rng=self.rng.derive("sfm"))',
    "sfm rng",
)
sub(
    "self.prediction_engine = PredictionEngine(width, height)",
    'self.prediction_engine = PredictionEngine(width, height, rng=self.rng.derive("predict"))',
    "prediction rng",
)
sub(
    "            scale=2.0,\n        )",
    "            scale=2.0,\n            rng=self.rng.derive(\"ingress\"),\n        )",
    "arrival rng",
)

# 4. call sites
call_sites = [
    ("random.randint(0, len(self.gates) - 1)", "self.rng.randint(0, len(self.gates) - 1)"),
    ("random.uniform(0.1, 0.9)", "self.rng.uniform(0.1, 0.9)"),
    ("random.uniform(-0.4, 0.4)", "self.rng.uniform(-0.4, 0.4)"),
    ("if random.random() > 0.6:", "if self.rng.random() > 0.6:"),
    ("random.choice(self.concessions)", "self.rng.choice(self.concessions)"),
    ("random.choice(self.exits)", "self.rng.choice(self.exits)"),
    ("random.choices(profiles", "self.rng.choices(profiles"),
    ("random.uniform(-0.15, 0.15)", "self.rng.uniform(-0.15, 0.15)"),
    ("random.randrange(n_gates)", "self.rng.randrange(n_gates)"),
    ('if d and random.random() < d["pct"] / 100.0:', 'if d and self.rng.random() < d["pct"] / 100.0:'),
]
for old, new in call_sites:
    sub(old, new, old.split("(")[0])

path.write_text(text, encoding="utf-8")

print(f"patched {path}  ({len(original)} -> {len(text)} bytes)\n")
bad = 0
for label, n in checks:
    flag = "ok " if n >= 1 else "MISS"
    if n < 1:
        bad += 1
    print(f"  {flag} {label}: {n}")
print(f"\nremaining 'random.' references: {text.count('random.')}")
print(f"unexpected misses: {bad}")
