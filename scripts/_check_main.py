import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the server module to prove the scheduler edit is syntactically valid and
# the app still constructs. FastAPI is not installed here, so failure to import
# fastapi is reported distinctly from a broken edit.
try:
    import fastapi  # noqa: F401
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

import ast

src = (ROOT / "backend" / "main.py").read_text(encoding="utf-8")
ast.parse(src)
print("main.py parses OK")

for token in ("next_deadline", "MIN_TICK_PERIOD_SEC", "import time"):
    print(f"  {token}: {src.count(token)}")

if HAVE_FASTAPI:
    import backend.main  # noqa: F401
    print("backend.main imported OK")
else:
    print("fastapi not installed; skipped importing the app")
