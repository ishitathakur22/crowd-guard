"""Fix #11: schedule ticks on a monotonic deadline so sim_speed is honest."""

import pathlib

path = pathlib.Path("backend/main.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


sub(
    '''async def simulation_loop():
    sim_engine.running = True
    loop = asyncio.get_event_loop()
    while sim_engine.running:
        speed = sim_engine.sim_speed          # read once — it can change mid-tick
        if speed > 0:
            try:
                await loop.run_in_executor(None, sim_engine.update)
            except Exception:
                logger.exception("Simulation tick failed")
                await asyncio.sleep(0.5)
                continue
            await asyncio.sleep(max(0.02, TICK_DURATION_SEC / max(speed, 0.1)))
        else:
            await asyncio.sleep(0.5)''',
    '''async def simulation_loop():
    """
    Drive the simulation on a monotonic deadline.

    The old loop slept `TICK_DURATION_SEC / speed` *after* doing the tick, so the
    real period was `tick_work + sleep`. At 1x with a 17 ms tick that ran ~3% fast;
    at 10x with the same work the intended 50 ms period became 67 ms, so "10x" was
    really 7.5x - and the slower the machine, the less the speed control meant. The
    measured headroom reported by scripts/bench.py is the honest figure; this keeps
    the simulation clock agreeing with it.

    Scheduling against an absolute deadline makes the period independent of how long
    the tick took. When a tick overruns its slot the deadline is re-based rather than
    firing a burst of catch-up ticks, which would simulate a crowd faster than real
    time and distort every forecast made from it.
    """
    sim_engine.running = True
    loop = asyncio.get_event_loop()
    next_deadline = time.monotonic()

    while sim_engine.running:
        speed = sim_engine.sim_speed          # read once — it can change mid-tick
        if speed <= 0:
            await asyncio.sleep(0.5)
            next_deadline = time.monotonic()  # resuming must not replay the pause
            continue

        period = max(MIN_TICK_PERIOD_SEC, TICK_DURATION_SEC / max(speed, 0.1))
        try:
            await loop.run_in_executor(None, sim_engine.update)
        except Exception:
            logger.exception("Simulation tick failed")
            await asyncio.sleep(0.5)
            next_deadline = time.monotonic()
            continue

        next_deadline += period
        now = time.monotonic()
        if next_deadline < now:
            # The tick overran its slot. Re-base instead of catching up.
            next_deadline = now
        else:
            await asyncio.sleep(next_deadline - now)''',
    "scheduler",
)

# The old loop imported `time` only indirectly; make the dependency explicit.
if "import time" not in text.split("@app.get(\"/\")")[0]:
    sub(
        "import asyncio\n",
        "import asyncio\nimport time\n",
        "time import",
    )

sub(
    "ENABLE_VISION_ENV = os.environ.get",
    "# Smallest gap between ticks, so a high speed multiplier cannot spin the loop\n"
    "# with no opportunity to serve websockets.\n"
    "MIN_TICK_PERIOD_SEC = 0.002\n\n"
    "ENABLE_VISION_ENV = os.environ.get",
    "MIN_TICK_PERIOD_SEC",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
