"""#13: replace the deterministic arrival accumulator with a seeded Poisson process."""

import pathlib

path = pathlib.Path("backend/ingress.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


sub(
    "        # Fractional-arrival carry, so rates below 1/tick still spawn eventually.\n"
    "        self._carry = 0.0",
    """        # Expected arrivals accumulated but not yet drawn, so a rate below one
        # person per tick still produces people. Retained from the deterministic
        # implementation, where it converted a fractional expectation into whole
        # people; with Poisson sampling it is now redundant for that purpose but
        # still records the residual for diagnostics.
        self._carry = 0.0""",
    "carry comment",
)

sub(
    '''    def draw_arrivals(self, sim_time_sec: float, duration_sec: float) -> int:
        """
        Whole number of arrivals for one tick, carrying the fractional remainder.

        Without the carry, any rate below one person per tick would truncate to
        zero and the venue would never fill.
        """
        self._carry += self.arrivals_in(sim_time_sec, duration_sec)
        whole = int(self._carry)
        self._carry -= whole
        return whole''',
    '''    def draw_arrivals(self, sim_time_sec: float, duration_sec: float) -> int:
        """
        People arriving in a window, drawn from a Poisson process.

        Why Poisson rather than a deterministic accumulator: pedestrian arrivals are
        a counting process with variance equal to the mean. The previous
        implementation carried a fractional remainder and emitted `floor(carry)`, so
        given the same configuration it produced *exactly* the same sequence every
        time and had zero variance. That is not a small statistical nicety - it makes
        arrival surges structurally impossible, and surges are what actually create
        bottlenecks. A simulator that cannot generate a surge cannot be used to test
        whether the forecast would have predicted one. It also understated every
        queueing statistic, because real crowds bunch up and an evenly-spaced stream
        never does.

        Poisson keeps the same expectation - `E[N] = rate * duration` - so attendance
        totals and the arrival curve are unchanged; only the dispersion is added.
        Draws come from the seeded generator, so a run is still exactly reproducible.

        The integral uses the rate at the start of the window (left-endpoint
        rectangle). At the tick granularity used here (0.5 s against an hourly curve)
        the error is well under 0.1%.
        """
        expected = self.arrivals_in(sim_time_sec, duration_sec)
        if expected <= 0.0:
            return 0
        # Record the expectation for diagnostics; not used to produce the count.
        self._carry = expected
        return int(self.rng.gen.poisson(expected))''',
    "draw_arrivals",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
