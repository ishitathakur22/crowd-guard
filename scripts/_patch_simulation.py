"""One-off patch: switch rerouting from per-agent A* to a cached flow field."""

import pathlib

path = pathlib.Path("backend/simulation.py")
text = path.read_text(encoding="utf-8")
lines = text.split("\n")

# ── 1. imports ────────────────────────────────────────────────────────────
for i, ln in enumerate(lines):
    if ln.startswith("from .pathfinding import"):
        lines[i] = "from .pathfinding import (\n    a_star_search, compute_alternative_routes, build_flow_field, flow_path\n)"
        break

# ── 2. cache attribute ────────────────────────────────────────────────────
for i, ln in enumerate(lines):
    if "self._last_reroute: Dict[int, int] = {}" in ln:
        indent = ln[: len(ln) - len(ln.lstrip())]
        lines.insert(
            i + 1,
            indent
            + "# Flow field cache: (tick bucket, id(density array)) -> (dist, next_step).\n"
            + indent
            + "# Rerouting asks one question per cell, not per agent, so the field is\n"
            + indent
            + "# rebuilt a few times a second and shared by everyone. See build_flow_field.\n"
            + indent
            + "self._flow_cache = None",
        )
        break

# ── 3. compute the field once per rerouting pass ─────────────────────────
for i, ln in enumerate(lines):
    if "def _handle_rerouting(self, raw_density: np.ndarray):" in ln:
        indent = ln[: len(ln) - len(ln.lstrip())] + "    "
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        lines.insert(
            j + 1,
            indent
            + "# Rebuild at most every REROUTE_COOLDOWN_TICKS. Between rebuilds every\n"
            + indent
            + "# agent reuses the same field, which is what removes the per-agent search.\n"
            + indent
            + "if self._flow_cache is None or self.tick_count % REROUTE_COOLDOWN_TICKS == 0:\n"
            + indent
            + "    self._flow_cache = build_flow_field(self.grid, raw_density, self.exits)\n"
            + indent
            + "flow_next_step = self._flow_cache[1]",
        )
        break

# ── 4. replace the per-agent A* block ────────────────────────────────────
start = next(
    i for i, ln in enumerate(lines) if "alt_routes = compute_alternative_routes(" in ln
)
end = next(
    i
    for i, ln in enumerate(lines[start:], start)
    if "if best_path:" in ln
)
indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
lines[start : end + 1] = [
    indent + "# Read the precomputed field: one shared computation for the whole venue",
    indent + "# instead of up to four full A* searches per agent per tick.",
    indent + "best_path = flow_path(flow_next_step, start_int, max_len=20)",
    indent + "if best_path:",
]

text = "\n".join(lines)

# ── 5. drop the now-unused best_goal references ──────────────────────────
text = text.replace(
    "agent.path = best_path[1:] if len(best_path) > 1 else best_path",
    "agent.path = best_path",
)
text = text.replace("agent.goal = best_goal", "agent.goal = best_path[-1]")
text = text.replace(
    "target_zone=f\"cell_{int(best_goal.x)}_{int(best_goal.y)}\",",
    "target_zone=f\"cell_{int(agent.goal.x)}_{int(agent.goal.y)}\",",
)

path.write_text(text, encoding="utf-8")
print("patched", path)
for probe in ("build_flow_field", "_flow_cache", "flow_path(", "best_goal"):
    print(f"  {probe}: {text.count(probe)} occurrences")
