"""Fix #10: publish an immutable snapshot from inside the tick, under a lock."""

import pathlib

path = pathlib.Path("backend/simulation.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── 1. imports ────────────────────────────────────────────────────────────
sub(
    "import os\nimport time\n",
    "import os\nimport threading\nimport time\n",
    "threading import",
)

# ── 2. snapshot buffers, created before anything that can be read ────────
sub(
    "        self.agents: List[Agent] = []",
    """        # Published snapshots. `update()` runs on an executor thread while the
        # websocket broadcaster reads state on the event loop, so the reader must
        # never touch `self.agents` - it would iterate a list that is being mutated
        # underneath it. Instead the tick builds a plain dict of primitives at its
        # end, under this lock, and readers only ever see a finished snapshot.
        self._state_lock = threading.Lock()
        self._state_snapshot: Optional[dict] = None
        self._heatmap_snapshot: Optional[dict] = None

        self.agents: List[Agent] = []""",
    "snapshot buffers",
)

# ── 3. publish at the end of the tick ────────────────────────────────────
sub(
    """        # 11. Autopilot: apply recommendations nobody has acted on within the response window
        self.autopilot.evaluate(self)""",
    """        # 11. Autopilot: apply recommendations nobody has acted on within the response window
        self.autopilot.evaluate(self)

        # 12. Publish an immutable snapshot for the event loop to broadcast. Built
        # here, on the thread that owns the data, so no reader can observe a partly
        # updated crowd.
        self._publish_snapshot()""",
    "publish call",
)

# ── 4. reader-side accessors and the publish helper ─────────────────────
sub(
    """    def get_state(self) -> dict:
        total_rerouted = sum(1 for a in self.agents if a.rerouted)""",
    '''    def _publish_snapshot(self):
        """Build the state and heatmap payloads once, then swap them in."""
        state = self._build_state()
        heatmap = {
            "heatmap": self.density_heatmap,
            "flow_vectors": self.flow_vectors,
            "width": self.width,
            "height": self.height,
        }
        with self._state_lock:
            self._state_snapshot = state
            self._heatmap_snapshot = heatmap

    def get_state(self) -> dict:
        """
        The most recently published state snapshot.

        Returns the dict built at the end of the last tick rather than walking the
        live agent list, because this is called from the event loop while `update()`
        runs on an executor thread. If no tick has completed yet, one is built here
        so a fresh client still gets a valid payload - it can only race with the very
        first tick, when the crowd is empty and there is nothing to corrupt.
        """
        with self._state_lock:
            snapshot = self._state_snapshot
        if snapshot is None:
            snapshot = self._build_state()
            with self._state_lock:
                self._state_snapshot = snapshot
        return snapshot

    def _build_state(self) -> dict:
        total_rerouted = sum(1 for a in self.agents if a.rerouted)''',
    "get_state wrapper",
)

# ── 5. heatmap reader ───────────────────────────────────────────────────
sub(
    '''    def get_heatmap_data(self) -> dict:
        """Return heatmap + flow vector data for frontend rendering."""
        return {
            "heatmap": self.density_heatmap,
            "flow_vectors": self.flow_vectors,
            "width": self.width,
            "height": self.height,
        }''',
    '''    def get_heatmap_data(self) -> dict:
        """The most recently published heatmap payload. See `get_state`."""
        with self._state_lock:
            snapshot = self._heatmap_snapshot
        if snapshot is None:
            snapshot = {
                "heatmap": self.density_heatmap,
                "flow_vectors": self.flow_vectors,
                "width": self.width,
                "height": self.height,
            }
            with self._state_lock:
                self._heatmap_snapshot = snapshot
        return snapshot''',
    "heatmap reader",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
