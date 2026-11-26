"""Fix the closed-gate dead queue in _admit_arrivals."""

import pathlib

path = pathlib.Path("backend/simulation.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── 1. redistribute existing queues away from closed gates ───────────────
sub(
    """        by_id = {g.gate_id: g for g in self.gate_objects}
        for _ in range(arrivals):
            gate = self.gate_objects[self.rng.randrange(n_gates)]
            d = self._diverts.get(gate.gate_id)
            if d and self.rng.random() < d["pct"] / 100.0:
                tgt = by_id.get(d["target"])
                if tgt is not None and tgt.action != GateAction.CLOSE:
                    tgt.queue_length += 1
                    continue
            gate.queue_length += 1""",
    """        by_id = {g.gate_id: g for g in self.gate_objects}

        # People already queued at a gate that has since closed walk to an open one.
        # Without this they wait forever: a closed gate admits at rate 0, so its
        # queue never drains and its reported wait_time_sec grows without bound.
        open_gates = [g for g in self.gate_objects if g.action != GateAction.CLOSE]
        if open_gates:
            stranded = 0
            for g in self.gate_objects:
                if g.action == GateAction.CLOSE and g.queue_length:
                    stranded += g.queue_length
                    g.queue_length = 0
                    g.wait_time_sec = 0.0
            for _ in range(stranded):
                # Weighted by each open gate's arrival share; defaults are equal.
                target = self.rng.choices(
                    open_gates, weights=[max(0.0, g.arrival_weight) for g in open_gates], k=1
                )[0]
                target.queue_length += 1

        for _ in range(arrivals):
            # New arrivals cannot enter through a closed gate, so they must be
            # assigned to one that is open. Choosing uniformly across all gates and
            # queuing regardless - which is what this used to do - put people behind
            # a door that never opens, and they were never seen again.
            gate = (
                self.rng.choices(
                    open_gates, weights=[max(0.0, g.arrival_weight) for g in open_gates], k=1
                )[0]
                if open_gates
                else self.gate_objects[0]
            )
            # A divert redirects a share of this gate's arrivals to its target.
            d = self._diverts.get(gate.gate_id)
            if d and self.rng.random() < d["pct"] / 100.0:
                tgt = by_id.get(d["target"])
                if tgt is not None and tgt.action != GateAction.CLOSE:
                    tgt.queue_length += 1
                    continue
            gate.queue_length += 1""",
    "arrival assignment",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
