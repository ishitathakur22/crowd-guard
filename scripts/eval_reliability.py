"""
Reliability check: is `p_exceed` an honest probability?

The dashboard now shows statements like "62% chance of exceeding LoS E here within
10 minutes". Nothing has ever verified that events predicted at 70% actually occur
about 70% of the time. A systematically over-confident probability is worse than no
probability at all: an operator who trusts it makes worse decisions than one given
nothing, because the number carries false authority.

Method
------
For each cut point in a recorded trajectory:

  1. forecast from that exact state, with the ensemble enabled;
  2. read the per-cell exceedance probability `p_exceed`;
  3. check the recorded truth: did that cell actually reach the threshold at any
     point inside the forecast horizon?

Then bin every (cell, cut) observation by predicted probability and compare with the
observed frequency. A well-calibrated map lies on the diagonal.

Metrics
-------
  Brier score   mean squared error of the probability forecast. Lower is better.
                0.25 is what always predicting 0.5 scores; 0.0 is perfect.
  ECE           expected calibration error - the bin-count-weighted gap between
                predicted and observed, in probability points. The headline number.
  resolution    how much the predictions vary. A model that always says 0.06 can be
                "calibrated" while being useless, so spread matters as much as bias.

Usage:
    python scripts/eval_reliability.py
    python scripts/eval_reliability.py --scenario hotspot --replicates 4
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from backend.prediction import PILEUP_DENSITY  # noqa: E402
from eval_forecast import agents_from_cut, build_engine  # noqa: E402


def observed_breach(density, cut_tick, horizon_steps, dt):
    """Did each cell reach the threshold at any point inside the horizon?"""
    peak = np.zeros(density.shape[1:], dtype=np.float64)
    for step in range(horizon_steps + 1):
        tick = cut_tick + step
        if tick >= density.shape[0]:
            break
        np.maximum(peak, density[tick], out=peak)
    return peak >= PILEUP_DENSITY


def reliability_report(predicted, observed, bins=10):
    """Bin predictions and compare with outcomes."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    n_total = len(predicted)

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (predicted >= lo) & (predicted < hi) if i < bins - 1 else (
            (predicted >= lo) & (predicted <= hi)
        )
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        mean_pred = float(predicted[mask].mean())
        obs = float(observed[mask].mean())
        ece += (n / n_total) * abs(mean_pred - obs)
        rows.append((lo, hi, n, mean_pred, obs))

    brier = float(np.mean((predicted - observed.astype(float)) ** 2))
    base_rate = float(observed.mean())

    return {
        "bins": rows,
        "brier": brier,
        "ece": ece,
        "base_rate": base_rate,
        "n": n_total,
        "resolution": float(np.std(predicted)),
    }


def main():
    ap = argparse.ArgumentParser(description="Check whether p_exceed is calibrated.")
    ap.add_argument("--scenario", default="hotspot")
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    npz_path = out_dir / f"trajectory_{args.scenario}.npz"
    meta_path = out_dir / f"trajectory_{args.scenario}.json"
    if not npz_path.exists():
        sys.exit(f"missing {npz_path.name}; run scripts/record_hotspot.py first")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)

    dt = meta["tick_duration_sec"]
    density = npz["density"].astype(np.float64)
    horizon_steps = int(round(meta["horizon_minutes"] * 60.0 / dt))

    print(
        f"\n# Reliability of p_exceed - scenario '{args.scenario}', "
        f"{args.replicates} replicates\n"
    )
    print(
        f"horizon {meta['horizon_minutes']:.0f} min "
        f"({horizon_steps} ticks), threshold {PILEUP_DENSITY} p/m2, "
        f"population {meta['population']['min']}-{meta['population']['max']}\n"
    )

    all_pred, all_obs = [], []

    for cut_sec in meta["cuts_sec"]:
        cut_tick = int(round(cut_sec / dt))
        engine = build_engine(meta)
        pe = engine.prediction_engine
        pe.ensemble_replicates = args.replicates

        pe.forecast(
            agents_from_cut(npz, cut_tick),
            engine.grid, engine.walls_cache, engine.gates, engine.exits,
            sim_time_sec=float(cut_sec),
        )

        if pe.exceedance is None:
            print(f"  cut {cut_sec:.0f}s: no exceedance produced; skipping")
            continue

        obs = observed_breach(density, cut_tick, horizon_steps, dt)

        # Restrict to cells the forecast even considered: an empty part of the venue
        # sits at p = 0 and would otherwise dominate every bin.
        relevant = (pe.exceedance > 0) | obs
        predicted = pe.exceedance[relevant].ravel()
        observed = obs[relevant].ravel()

        all_pred.append(predicted)
        all_obs.append(observed)

        print(
            f"  cut {cut_sec:.0f}s: {int(relevant.sum())} cells considered, "
            f"{int(observed.sum())} actually breached, "
            f"mean p {predicted.mean():.3f}, observed {observed.mean():.3f}"
        )

    if not all_pred:
        sys.exit("no exceedance data produced")

    predicted = np.concatenate(all_pred)
    observed = np.concatenate(all_obs)
    rep = reliability_report(predicted, observed)

    print(f"\n## Reliability diagram (all cuts, n={rep['n']})\n")
    print("| predicted range | cells | mean predicted | observed | gap |")
    print("|---|---|---|---|---|")
    for lo, hi, n, mp, obs in rep["bins"]:
        if n == 0:
            continue
        print(
            f"| {lo:.1f}-{hi:.1f} | {n} | {mp:.3f} | {obs:.3f} | {mp - obs:+.3f} |"
        )

    print(f"\n**Brier score:** {rep['brier']:.4f}  (always-0.5 scores 0.25)")
    print(f"**Expected calibration error:** {rep['ece']:.4f} "
          f"({rep['ece'] * 100:.1f} probability points)")
    print(f"**Base rate** (how often breach actually happened): {rep['base_rate']:.3f}")
    print(f"**Resolution** (sd of predictions): {rep['resolution']:.3f}")

    # A verdict, because a table alone does not tell an operator what to do.
    if rep["ece"] <= 0.05 and rep["brier"] < 0.20:
        verdict = (
            "CALIBRATED - predicted probabilities match observed frequencies. "
            "Safe to surface as a probability and to gate the autopilot on it."
        )
    elif rep["ece"] <= 0.15:
        verdict = (
            "MARGINALLY CALIBRATED - usable, but the probability should be read as "
            "a band rather than a precise number."
        )
    else:
        verdict = (
            "NOT CALIBRATED - the probability is misleading. Either disable the "
            "ensemble (ensemble_replicates = 0) or treat p_exceed as a ranking only, "
            "never as a probability."
        )
    print(f"\n**Verdict:** {verdict}")

    path = out_dir / f"reliability_{args.scenario}.json"
    path.write_text(json.dumps({
        "scenario": args.scenario,
        "replicates": args.replicates,
        "threshold": PILEUP_DENSITY,
        "brier": rep["brier"],
        "ece": rep["ece"],
        "base_rate": rep["base_rate"],
        "resolution": rep["resolution"],
        "n": rep["n"],
        "verdict": verdict,
        "bins": [
            {"lo": lo, "hi": hi, "n": n, "predicted": mp, "observed": obs}
            for lo, hi, n, mp, obs in rep["bins"]
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
