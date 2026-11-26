"""
Score the forecast against recorded ground truth.

This is the oracle. For every cut point in a recorded trajectory it rebuilds the
exact crowd state at that moment, runs the forecast, and compares each predicted
density grid against the grid the live simulation actually produced at the same
offset.

Two things are scored, and the second is the one that matters:

  * the forecast itself;
  * persistence - "density now, unchanged" - which is the baseline any forecast
    must beat to be worth anything. Without it, a low absolute error looks
    impressive purely because most cells are empty most of the time.

Metrics:

  MAE        mean absolute error over all cells. Easy to game: predicting nothing
             scores well, because most cells are near-empty.
  busy MAE   error restricted to cells that are genuinely occupied. This is where
             a warning is either right or wrong.
  corr       Pearson correlation of the fields. Scale-free, so it cannot be
             improved by predicting a flat zero.
  peak err   absolute difference in maximum density - whether the crush was
             called correctly.
  lead time  seconds of warning, measured only for cells that really did breach.

Usage:
    python scripts/eval_forecast.py
    python scripts/eval_forecast.py --scenario surge --model particles
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.models import Agent, Point  # noqa: E402
from backend.prediction import PILEUP_DENSITY  # noqa: E402
from backend.simulation import SimulationEngine  # noqa: E402


def build_engine(meta):
    """Rebuild an engine configured exactly as the recorder's was."""
    engine = SimulationEngine(seed=meta["seed"])
    engine.prediction_interval = 10**9

    arrival = meta["arrival_model"]
    engine.arrival_model.attendance = arrival["attendance"]
    engine.arrival_model.scale = arrival["scale"]
    engine.arrival_model.start_hour = arrival["start_hour"]
    engine.arrival_model.weekend = arrival["weekend"]

    pred = meta["prediction"]
    pe = engine.prediction_engine
    pe.horizon_minutes = meta["horizon_minutes"]
    pe.snapshot_interval_sec = pred["snapshot_interval_sec"]
    pe.sim_dt = pred["sim_dt"]
    pe.max_snapshots = pred["max_snapshots"]
    pe.jam_density = pred["jam_density"]
    return engine


def agents_from_cut(npz, tick):
    """Reconstruct the crowd at a cut point from the recorded struct-of-arrays."""
    pos = npz[f"cut_{tick}_pos"]
    vel = npz[f"cut_{tick}_vel"]
    goal = npz[f"cut_{tick}_goal"]
    speed = npz[f"cut_{tick}_speed"]
    radius = npz[f"cut_{tick}_radius"]
    path_xy = npz[f"cut_{tick}_path_xy"]
    path_len = npz[f"cut_{tick}_path_len"]

    agents = []
    for i in range(pos.shape[0]):
        length = int(path_len[i])
        path = [Point(x=float(path_xy[i, j, 0]), y=float(path_xy[i, j, 1]))
                for j in range(length)]
        agents.append(
            Agent(
                id=i,
                pos=Point(x=float(pos[i, 0]), y=float(pos[i, 1])),
                vel=Point(x=float(vel[i, 0]), y=float(vel[i, 1])),
                goal=Point(x=float(goal[i, 0]), y=float(goal[i, 1])),
                path=path,
                desired_speed=float(speed[i]),
                radius=float(radius[i]),
            )
        )
    return agents


def _div(a, b):
    return a / b if b else float("nan")


def compare(pred, truth, threshold):
    """Field and breach metrics for one predicted grid against one truth grid."""
    diff = pred - truth
    pb = pred >= threshold
    tb = truth >= threshold
    tp = int(np.count_nonzero(pb & tb))
    fp = int(np.count_nonzero(pb & ~tb))
    fn = int(np.count_nonzero(~pb & tb))

    precision = _div(tp, tp + fp) if (tp + fp) else (1.0 if not tb.any() else 0.0)
    recall = _div(tp, tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))

    busy = truth >= 0.5
    busy_mae = float(np.abs(diff[busy]).mean()) if busy.any() else float("nan")

    pf = pred.ravel()
    tf = truth.ravel()
    pf_c = pf - pf.mean()
    tf_c = tf - tf.mean()
    denom = float(np.sqrt((pf_c ** 2).sum() * (tf_c ** 2).sum()))
    corr = float((pf_c * tf_c).sum() / denom) if denom > 0 else float("nan")

    return {
        "mae": float(np.abs(diff).mean()),
        "busy_mae": busy_mae,
        "corr": corr,
        "pop_pred": float(pred.sum()),
        "pop_truth": float(truth.sum()),
        "peak_error": float(abs(float(pred.max()) - float(truth.max()))),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def first_breach(masks):
    """First offset (seconds) at which each cell crosses the threshold."""
    first = {}
    for offset, mask in masks:
        for y, x in zip(*np.nonzero(mask)):
            key = (int(y), int(x))
            if key not in first:
                first[key] = offset
    return first


def score_cut(engine, npz, meta, cut_sec):
    dt = meta["tick_duration_sec"]
    density = npz["density"]
    cut_tick = int(round(cut_sec / dt))
    cut_grid = density[cut_tick].astype(np.float64)

    agents = agents_from_cut(npz, cut_tick)

    t0 = time.perf_counter()
    engine.prediction_engine.forecast(
        agents, engine.grid, engine.walls_cache, engine.gates, engine.exits,
        sim_time_sec=cut_sec,
    )
    forecast_ms = (time.perf_counter() - t0) * 1000.0

    pred_grids = engine.prediction_engine.last_forecast_grids or []

    rows = []
    for offset, pgrid in pred_grids:
        truth_tick = cut_tick + int(round(offset / dt))
        if truth_tick >= density.shape[0]:
            continue
        truth = density[truth_tick].astype(np.float64)
        rows.append({
            "offset_sec": offset,
            "model": compare(pgrid.astype(np.float64), truth, PILEUP_DENSITY),
            "persistence": compare(cut_grid, truth, PILEUP_DENSITY),
        })

    model_breach = first_breach([(o, g >= PILEUP_DENSITY) for o, g in pred_grids])
    truth_breach = first_breach([
        (o, density[cut_tick + int(round(o / dt))].astype(np.float64) >= PILEUP_DENSITY)
        for o, _ in pred_grids
        if cut_tick + int(round(o / dt)) < density.shape[0]
    ])

    leads, missed = [], 0
    for cell, t_truth in truth_breach.items():
        t_pred = model_breach.get(cell)
        if t_pred is None:
            missed += 1
        else:
            leads.append(t_truth - t_pred)

    return {
        "cut_sec": cut_sec,
        "agents": len(agents),
        "forecast_ms": round(forecast_ms, 1),
        "rows": rows,
        "lead": {
            "mean_sec": round(statistics.fmean(leads), 1) if leads else None,
            "matched_cells": len(leads),
            "missed_cells": missed,
            "truth_breach_cells": len(truth_breach),
            "false_alarm_cells": sum(1 for c in model_breach if c not in truth_breach),
        },
    }


def _agg(values):
    vals = [v for v in values if v == v]
    return statistics.fmean(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Score the forecast against recorded truth.")
    ap.add_argument("--scenario", default="default")
    ap.add_argument("--out", default=str(ROOT / "benchmarks"))
    ap.add_argument("--model", choices=["continuum", "particles"], default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    npz_path = out_dir / f"trajectory_{args.scenario}.npz"
    meta_path = out_dir / f"trajectory_{args.scenario}.json"
    if not npz_path.exists():
        sys.exit(f"missing {npz_path.name}; run scripts/record_trajectory.py first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path)

    engine = build_engine(meta)
    if args.model:
        engine.prediction_engine.spatial_model = args.model

    cuts = [score_cut(engine, npz, meta, float(c)) for c in meta["cuts_sec"]]

    print(f"\n# Forecast accuracy - scenario '{meta['scenario']}', seed {meta['seed']}")
    print(f"model={engine.prediction_engine.spatial_model}, "
          f"horizon {meta['horizon_minutes']:.0f} min, "
          f"population {meta['population']['min']}-{meta['population']['max']}, "
          f"breach threshold {PILEUP_DENSITY}\n")

    print("| horizon | MAE | persist MAE | busy MAE | busy persist | corr | persist corr | peak err | pop err |")
    print("|---|---|---|---|---|---|---|---|---|")

    by_offset = {}
    for c in cuts:
        for r in c["rows"]:
            by_offset.setdefault(r["offset_sec"], []).append(r)

    for offset in sorted(by_offset):
        rs = by_offset[offset]
        pop_err = statistics.fmean(
            r["model"]["pop_pred"] - r["model"]["pop_truth"] for r in rs
        )
        print(
            f"| {offset / 60:.0f} min "
            f"| {_agg([r['model']['mae'] for r in rs]):.3f} "
            f"| {_agg([r['persistence']['mae'] for r in rs]):.3f} "
            f"| {_agg([r['model']['busy_mae'] for r in rs]):.3f} "
            f"| {_agg([r['persistence']['busy_mae'] for r in rs]):.3f} "
            f"| {_agg([r['model']['corr'] for r in rs]):.3f} "
            f"| {_agg([r['persistence']['corr'] for r in rs]):.3f} "
            f"| {_agg([r['model']['peak_error'] for r in rs]):.3f} "
            f"| {pop_err:+.0f} |"
        )

    print("\n| cut | agents | forecast ms | lead mean | matched | missed | false cells |")
    print("|---|---|---|---|---|---|---|")
    for c in cuts:
        lead = c["lead"]
        mean = "-" if lead["mean_sec"] is None else f"{lead['mean_sec']}s"
        print(f"| {c['cut_sec']:.0f}s | {c['agents']} | {c['forecast_ms']:.0f} "
              f"| {mean} | {lead['matched_cells']} | {lead['missed_cells']} "
              f"| {lead['false_alarm_cells']} |")

    mae = _agg([r["model"]["mae"] for c in cuts for r in c["rows"]])
    pmae = _agg([r["persistence"]["mae"] for c in cuts for r in c["rows"]])
    bmae = _agg([r["model"]["busy_mae"] for c in cuts for r in c["rows"]])
    pbmae = _agg([r["persistence"]["busy_mae"] for c in cuts for r in c["rows"]])
    corr = _agg([r["model"]["corr"] for c in cuts for r in c["rows"]])
    pcorr = _agg([r["persistence"]["corr"] for c in cuts for r in c["rows"]])

    print(f"\nOverall: MAE {mae:.3f} vs persistence {pmae:.3f}; "
          f"busy-cell MAE {bmae:.3f} vs {pbmae:.3f}; "
          f"correlation {corr:.3f} vs {pcorr:.3f}.")
    print(f"Verdict: the forecast {'beats' if mae < pmae else 'does NOT beat'} "
          f"persistence on MAE; mean cost "
          f"{_agg([c['forecast_ms'] for c in cuts]):.0f} ms per horizon.")

    path = out_dir / f"forecast{'_' + args.label if args.label else ''}.json"
    path.write_text(json.dumps({
        "scenario": meta["scenario"],
        "model": engine.prediction_engine.spatial_model,
        "threshold": PILEUP_DENSITY,
        "notes": ("Truth is the seeded live simulation, so this measures whether the "
                  "forecast reproduces the Social Force Model - not real-world validity."),
        "cuts": cuts,
    }, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
