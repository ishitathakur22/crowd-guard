"""
Deterministic randomness for the whole engine.

Before this module the simulation drew from the unseeded module-level `random`
and `np.random` globals — agent spawning, gate admission, forecast arrival
injection, and the physics separation jitter. Two runs of the same scenario
therefore produced different crowds, which made every accuracy claim about the
forecast unfalsifiable: there was no way to say whether a change made the
prediction better or just different.

One seed now decides the run. It is threaded explicitly rather than kept as a
module global so that two engines with different seeds can coexist in the same
process — which is exactly what the reproducibility test does.

Substreams
----------
Each subsystem gets its own generator derived from the parent seed:

    engine.rng.derive("sfm")       # physics separation
    engine.rng.derive("predict")   # forecast arrival placement
    engine.rng.derive("ingress")   # arrival process

Deriving rather than sharing matters because a shared generator makes draws
order-dependent across subsystems: adding one extra random call in the physics
term would shift every subsequent arrival and change the scenario under test. A
derived substream is stable under that kind of edit, so a benchmark stays
comparable after unrelated code changes.
"""

import hashlib
import os
import random
from typing import Sequence

import numpy as np

DEFAULT_SEED = 1337

#: Environment variable that overrides the default seed for the running server.
SEED_ENV = "CROWD_SEED"


def seed_from_env(default: int = DEFAULT_SEED) -> int:
    """Read ``CROWD_SEED`` from the environment, falling back to ``default``."""
    raw = os.environ.get(SEED_ENV)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


class Rng:
    """
    A pair of seeded generators: one for Python-level draws, one for numpy.

    ``random()`` is a method here, not an attribute: the underlying Python
    generator is ``self._py``. That keeps call sites reading like the stdlib they
    replace.
    """

    __slots__ = ("seed", "_py", "gen")

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = int(seed)
        self._py = random.Random(self.seed)
        # default_rng is the modern numpy API and is stable across versions, unlike
        # the legacy global RandomState whose algorithm numpy reserves the right to
        # change. A seed that means something different next year is not a seed.
        self.gen = np.random.default_rng(self.seed)

    # ── derivation ────────────────────────────────────────────────

    def derive(self, label: str) -> "Rng":
        """
        An independent generator for a named subsystem.

        The label is hashed with SHA-256 rather than Python's built-in ``hash``,
        because ``hash`` on str is salted per process — the derived seed would
        change on every restart and nothing would reproduce.
        """
        digest = hashlib.sha256(f"{self.seed}:{label}".encode("utf-8")).digest()
        return Rng(int.from_bytes(digest[:8], "big") % (2**63 - 1))

    # ── Python-level draws (mirror the stdlib signatures) ─────────

    def random(self) -> float:
        return self._py.random()

    def uniform(self, a: float, b: float) -> float:
        return self._py.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self._py.randint(a, b)

    def randrange(self, start: int, stop: int = None, step: int = 1) -> int:
        if stop is None:
            return self._py.randrange(start)
        return self._py.randrange(start, stop, step)

    def choice(self, seq: Sequence):
        return self._py.choice(seq)

    def choices(self, seq: Sequence, weights=None, k: int = 1):
        return self._py.choices(seq, weights=weights, k=k)

    # ── numpy-level draws ─────────────────────────────────────────

    def uniform_array(self, low: float, high: float, size):
        return self.gen.uniform(low, high, size)

    def random_array(self, size):
        return self.gen.random(size)
