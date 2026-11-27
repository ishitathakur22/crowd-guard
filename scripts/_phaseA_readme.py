"""Finish Phase A: update the README test table and add the exceedance section."""

import pathlib

path = pathlib.Path("README.md")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    if n:
        text = text.replace(old, new, 1)


sub(
    "**117 tests across 10 files**, covering core correctness, physics parity, and concurrency safety:",
    "**132 tests across 11 files**, covering core correctness, physics parity, concurrency safety, and forecast uncertainty:",
    "test count",
)

sub(
    "| `test_vision_aggregator.py` | Detection IoU de-duplication, 4-point homography, out-of-plan handling |",
    "| `test_vision_aggregator.py` | Detection IoU de-duplication, 4-point homography, out-of-plan handling |\n"
    "| `test_exceedance.py` | Ensemble variance (`sigma > 0`), probability bounds, divert scaling, autopilot gating |",
    "test table row",
)

sub(
    "> **Honest Assessment:** The continuum forecast beats persistence on overall MAE (and runs in `199 ms` vs. `626 ms` for the particle clone), but still trails persistence on busy-cell MAE and spatial correlation. See [benchmarks/README.md](file:///f:/CrowdFlowOptimiser/crowd-guard/benchmarks/README.md) for the full accuracy breakdown, reverted experiments, and scenario limitations.",
    "> **Honest Assessment:** The continuum forecast beats persistence on overall MAE (and runs in `199 ms` vs. `626 ms` for the particle clone), but still trails persistence on busy-cell MAE and spatial correlation. See [benchmarks/README.md](file:///f:/CrowdFlowOptimiser/crowd-guard/benchmarks/README.md) for the full accuracy breakdown, reverted experiments, and scenario limitations.\n"
    "\n"
    "#### Exceedance Probability & Bottleneck Lead Time\n"
    "\n"
    "Each forecast runs `ensemble_replicates = 4` passes and reports a per-cell **probability** of reaching the pile-up threshold within the horizon (`p_exceed`) rather than a single point estimate. Warnings carry it, the recommended divert is scaled by it, and the Autopilot refuses to act below `AUTO_MIN_P_EXCEED = 0.6` — a warning derived from one sample cannot distinguish a likely pile-up from a fluke, so autonomy now requires evidence.\n"
    "\n"
    "A tightened-chokepoint scenario (`scripts/record_hotspot.py`: 956 agents, peak 8.67 p/m², 96% of ticks above LoS E) made **bottleneck lead time** measurable for the first time:\n"
    "\n"
    "| Cut | Agents | Lead time | Matched | Missed |\n"
    "| :--- | :--- | :--- | :--- | :--- |\n"
    "| `60s` | 464 | **110.0 s** | 24 | 261 |\n"
    "| `120s` | 515 | 55.4 s | 26 | 273 |\n"
    "| `180s` | 590 | 55.6 s | 27 | 288 |\n"
    "\n"
    "Up to **110 seconds of warning** — but in that dense regime the forecast loses to persistence harder than anywhere else (MAE `0.653` vs `0.496`, population drained 476–826 short, ~8% recall). The over-drain is therefore **scale-dependent**, not merely spatial.\n"
    "\n"
    "The ensemble costs latency: at 200 agents headroom falls `29× → 13×`, at 500 agents `8.8× → 2.7×`. Set `ensemble_replicates = 0` to restore the point forecast exactly.",
    "exceedance section",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
