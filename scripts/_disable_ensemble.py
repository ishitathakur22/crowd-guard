"""Disable the exceedance ensemble by default, on the strength of the reliability check."""

import pathlib

report = []


def patch(path, pairs):
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    for label, old, new in pairs:
        n = text.count(old)
        report.append((f"{path}: {label}", n))
        if n:
            text = text.replace(old, new, 1)
    p.write_text(text, encoding="utf-8")


patch("backend/simulation.py", [
    (
        "disable by default",
        "        # Replicates per forecast, for the exceedance probability. Each one re-runs\n"
        "        # the whole horizon, so this multiplies the forecast cost; 4 keeps the\n"
        "        # 15-minute forecast inside a single tick budget at a few hundred agents\n"
        "        # (see scripts/bench.py). Set to 0 to disable and get the old point\n"
        "        # forecast back.\n"
        "        self.prediction_engine.ensemble_replicates = 4",
        "        # Replicates per forecast, for the exceedance probability. OFF by default.\n"
        "        #\n"
        "        # The mechanism works and costs real latency (4 replicates took the forecast\n"
        "        # from 181 ms to 1579 ms at 200 agents). What it does not do is produce a\n"
        "        # *calibrated* probability. Measured by scripts/eval_reliability.py against\n"
        "        # recorded truth:\n"
        "        #\n"
        "        #   hotspot scenario (base rate 0.97): ECE 0.917, mean predicted 0.09\n"
        "        #     against 0.97 observed - 1859 of 2078 cells were called 'unlikely'\n"
        "        #     and then breached every time.\n"
        "        #   default scenario (base rate 0.24): ECE 0.589, cells predicted at 1.0\n"
        "        #     breached 0% of the time.\n"
        "        #\n"
        "        # Erring in opposite directions across scenarios means `p_exceed` is not a\n"
        "        # probability but a rescaled replicate count, and a safety number that\n"
        "        # claims confidence it has not earned is worse than no number at all:\n"
        "        # an operator who trusts it decides worse than one given nothing.\n"
        "        #\n"
        "        # The ensemble stays available and tested. Enable it only after a reliability\n"
        "        # check passes (ECE <= 0.05) on the scenario of interest, and until then treat\n"
        "        # any p_exceed value as a ranking, never a probability.\n"
        "        self.prediction_engine.ensemble_replicates = 0",
    ),
])

patch("backend/autopilot.py", [
    (
        "document the rail",
        "AUTO_MIN_P_EXCEED = 0.6        # ...and only where the ensemble is at least this confident it happens.\n"
        "                               # A warning derived from one sample cannot distinguish a likely\n"
        "                               # pile-up from a fluke; the exceedance map can, so autonomy\n"
        "                               # requires evidence rather than a point estimate.",
        "AUTO_MIN_P_EXCEED = 0.6        # ...and only where the ensemble is at least this confident it happens.\n"
        "                               # Inert while the ensemble is disabled (p_exceed defaults to 1.0\n"
        "                               # for a point forecast, so this rail never fires). It exists for\n"
        "                               # the case where the ensemble is switched on after passing a\n"
        "                               # reliability check - the guard is already in place for that.",
    ),
])

bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
