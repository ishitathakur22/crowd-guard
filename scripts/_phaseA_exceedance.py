"""Phase A step 1: wire exceedance probability into warnings, autopilot and state."""

import pathlib

report = []


def patch(path, pairs):
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    for label, old, new in pairs:
        n = text.count(old)
        report.append((f"{path}: {label}", n))
        if n:
            text = text.replace(old, new)
    p.write_text(text, encoding="utf-8")


# ── simulation: attach p_exceed, scale the divert, run replicates ────────
patch("backend/simulation.py", [
    (
        "enrich warnings",
        '            w["origin_gate"] = origin.gate_id if origin else None\n'
        '            w["target_gate"] = target.gate_id if target else None\n'
        '            w["divert_pct"] = pct\n'
        '        return warnings',
        '''            # Exceedance probability: the share of forecast replicates in which this
            # cell reached the pile-up threshold within the horizon. The ensemble
            # computes it on every forecast, and until now nothing consumed it, so a
            # graded probability was being thrown away in favour of a single sample.
            p_exceed = 1.0
            pe = self.prediction_engine
            if pe.exceedance is not None:
                try:
                    p_exceed = float(pe.exceedance[int(w["cell_y"]), int(w["cell_x"])])
                except (IndexError, KeyError, TypeError, ValueError):
                    p_exceed = 1.0

            # Scale the recommended divert by how likely the breach actually is. A
            # 30%-likely pile-up should not attract the same intervention as a 90%
            # one; the previous formula came from a single deterministic sample and
            # treated every warning as equally certain.
            pct = int(round(pct * float(p_exceed)))
            pct = max(0, min(95, pct))

            w["p_exceed"] = round(p_exceed, 3)
            w["origin_gate"] = origin.gate_id if origin else None
            w["target_gate"] = target.gate_id if target else None
            w["divert_pct"] = pct
        return warnings''',
    ),
    (
        "ensemble config in init",
        "        self.prediction_interval = 10  # Run forecast every N ticks",
        "        self.prediction_interval = 10  # Run forecast every N ticks\n"
        "        # Replicates per forecast, for the exceedance probability. Each one re-runs\n"
        "        # the whole horizon, so this multiplies the forecast cost; 4 keeps the\n"
        "        # 15-minute forecast inside a single tick budget at a few hundred agents\n"
        "        # (see scripts/bench.py). Set to 0 to disable and get the old point\n"
        "        # forecast back.\n"
        "        self.prediction_engine.ensemble_replicates = 4",
    ),
    (
        "state payload",
        "            forecast_warnings=self.forecast_warnings,",
        "            forecast_warnings=self.forecast_warnings,\n"
        "            # Summary of the uncertainty map, so the dashboard can show how\n"
        "            # confident the forecast is rather than implying certainty.\n"
        "            forecast_uncertainty=self._uncertainty_summary(),",
    ),
    (
        "uncertainty summary",
        "    def get_heatmap_data(self) -> dict:",
        '''    def _uncertainty_summary(self) -> dict:
        """Headline numbers from the exceedance map, without shipping the full grid."""
        pe = self.prediction_engine
        if pe.exceedance is None:
            return {"available": False}
        exc = pe.exceedance
        sigma = pe.final_sigma
        return {
            "available": True,
            "replicates": int(pe.ensemble_replicates),
            "max_p_exceed": round(float(exc.max()), 3),
            "cells_above_50pct": int((exc >= 0.5).sum()),
            "cells_above_80pct": int((exc >= 0.8).sum()),
            "max_sigma": round(float(sigma.max()), 3) if sigma is not None else None,
        }

    def get_heatmap_data(self) -> dict:''',
    ),
])

# ── autopilot: require a minimum exceedance probability ─────────────────
patch("backend/autopilot.py", [
    (
        "constant",
        "AUTO_MIN_LIVE_DENSITY = 1.08   # ...and only where it is ALREADY busy now (LoS D, p/m²): the forecast",
        "AUTO_MIN_P_EXCEED = 0.6        # ...and only where the ensemble is at least this confident it happens.\n"
        "                               # A warning derived from one sample cannot distinguish a likely\n"
        "                               # pile-up from a fluke; the exceedance map can, so autonomy\n"
        "                               # requires evidence rather than a point estimate.\n"
        "AUTO_MIN_LIVE_DENSITY = 1.08   # ...and only where it is ALREADY busy now (LoS D, p/m²): the forecast",
    ),
    (
        "gate check",
        '                if w.get("local_density", 0.0) < AUTO_MIN_LIVE_DENSITY:\n'
        '                    continue      # forecast only: leave it as a recommendation for the operator',
        '                if w.get("local_density", 0.0) < AUTO_MIN_LIVE_DENSITY:\n'
        '                    continue      # forecast only: leave it as a recommendation for the operator\n'
        '                if w.get("p_exceed", 1.0) < AUTO_MIN_P_EXCEED:\n'
        '                    continue      # the ensemble is not confident enough to act on its own',
    ),
    (
        "status",
        '                "min_live_density": AUTO_MIN_LIVE_DENSITY,',
        '                "min_live_density": AUTO_MIN_LIVE_DENSITY,\n'
        '                "min_p_exceed": AUTO_MIN_P_EXCEED,',
    ),
])

bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
