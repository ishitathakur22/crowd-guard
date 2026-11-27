"""Add the fundamental diagram check to scripts/checks.py, after check_density."""

import pathlib

path = pathlib.Path("scripts/checks.py")
text = path.read_text(encoding="utf-8")

CHECK = '''def check_fundamental_diagram():
    """
    Does the simulated crowd slow with density the way real people do?

    The only check here that compares the simulation to measurements of humans rather
    than to itself. Weidmann (1993) is a closed-form fit to empirical pedestrian data
    and the reference used throughout the pedestrian-simulation literature.

    Measured result, seed 1337:

        density  1.26 p/m2  ->  simulation 1.326 m/s, Weidmann 0.922 m/s

    The crowd loses 0.012 m/s across the range where real pedestrians lose 0.409. It
    walks at a near-constant speed regardless of crowding, so it cannot reproduce
    queueing or crush behaviour. See benchmarks/README.md section 3e.
    """
    _header("fundamental diagram vs Weidmann (1993)")

    import math

    from backend.models import Agent, CellType, Point
    from backend.simulation import SimulationEngine

    v_free, gamma, rho_jam = 1.34, 1.913, 5.4

    def weidmann(rho):
        rho = max(rho, 1e-6)
        return v_free * (1.0 - math.exp(-gamma * (1.0 / rho - 1.0 / rho_jam)))

    def sample(n):
        e = SimulationEngine(seed=1337)
        e.auto_ingress = False
        e.prediction_interval = 10**9
        e.grid = [[CellType.EMPTY.value for _ in range(e.width)]
                  for _ in range(e.height)]
        e.walls_cache = []
        e.sfm._wall_set_initialized = False
        e.sfm._wall_set = set()

        rng = e.rng
        for _ in range(n):
            x = rng.uniform(1.0, 6.0)
            y = rng.uniform(1.0, e.height - 1.0)
            e.agents.append(Agent(
                id=e.agent_id_counter, pos=Point(x=x, y=y),
                goal=Point(x=e.width - 2.0, y=y), path=[],
                desired_speed=v_free, radius=0.3, mass=80.0,
            ))
            e.agent_id_counter += 1

        speeds, dens = [], []
        # Agents cross 40 m in ~57 ticks, so the sampling window has to sit inside
        # the crossing; a longer warmup leaves an empty venue and no measurement.
        for tick in range(60):
            e.update()
            if tick < 20 or not e.agents:
                continue
            speeds.append(statistics.fmean(
                (a.vel.x ** 2 + a.vel.y ** 2) ** 0.5 for a in e.agents))
            raw = e.density_analyzer.compute_raw_density(e.agents)
            ld = local_density(raw, 1)
            occ = raw > 0
            if occ.any():
                dens.append(float(ld[occ].mean()))
        if not speeds or not dens:
            return None
        return statistics.fmean(dens), statistics.fmean(speeds)

    rows = []
    for n in (60, 200, 450, 800):
        got = sample(n)
        if got:
            rows.append((n, got[0], got[1]))

    print("  spawn | density | simulated | Weidmann | error")
    for n, rho, v in rows:
        print(f"  {n:>5} | {rho:>7.3f} | {v:>9.3f} | {weidmann(rho):>8.3f} "
              f"| {v - weidmann(rho):+.3f}")

    if len(rows) < 3:
        raise AssertionError("not enough measurements to judge the curve")

    sim_drop = rows[0][2] - rows[-1][2]
    real_drop = weidmann(rows[0][1]) - weidmann(rows[-1][1])
    print(f"\\n  speed lost over the range: simulation {sim_drop:.3f} m/s, "
          f"real {real_drop:.3f} m/s")

    if real_drop <= 0.3:
        raise AssertionError("reference curve is not behaving as expected")
    if sim_drop < 0.25 * real_drop:
        raise AssertionError(
            f"the crowd walks at a near-constant speed ({sim_drop:.3f} m/s lost "
            f"against {real_drop:.3f} for real people), so it cannot reproduce "
            f"queueing or crush behaviour"
        )
    print("  -> density sensitivity present")


def check_speed():'''

n = text.count("def check_speed():")
print(f"check_speed anchors: {n}")
if n:
    text = text.replace("def check_speed():", CHECK, 1)

# Register in the CHECKS dict.
old_entry = '    "density": check_density,\n'
new_entry = '    "density": check_density,\n    "fundamental": check_fundamental_diagram,\n'
m = text.count(old_entry)
print(f"CHECKS entry anchors: {m}")
if m:
    text = text.replace(old_entry, new_entry, 1)

path.write_text(text, encoding="utf-8")
print(f"patched, {len(text)} bytes")
print(f"  fundamental registered: {'\"fundamental\"' in text}")
