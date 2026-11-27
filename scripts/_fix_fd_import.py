import pathlib

p = pathlib.Path("scripts/checks.py")
t = p.read_text(encoding="utf-8")

old = ("    from backend.models import Agent, CellType, Point\n"
       "    from backend.simulation import SimulationEngine")
new = ("    from backend.density import local_density\n"
       "    from backend.models import Agent, CellType, Point\n"
       "    from backend.simulation import SimulationEngine")

# Only the occurrence inside check_fundamental_diagram (the one preceded by the
# "import math" line in that function).
idx = t.find("def check_fundamental_diagram")
if idx == -1:
    raise SystemExit("function not found")

before, after = t[:idx], t[idx:]
n = after.count(old)
print(f"occurrences after the function head: {n}")
if n:
    after = after.replace(old, new, 1)
    p.write_text(before + after, encoding="utf-8")
    print("import added")
else:
    print("anchor not found; showing context")
    print(after[:1200])
