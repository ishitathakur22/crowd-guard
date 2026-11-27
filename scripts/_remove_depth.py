"""Step 3: remove depth estimation (option C) - it has no consumer."""

import pathlib
import re

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


# ── 1. delete the DepthWorker class outright ────────────────────────────
p = pathlib.Path("backend/vision/models.py")
text = p.read_text(encoding="utf-8")

start = text.find("# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n# DEPTH WORKER")
if start == -1:
    start = text.find("# DEPTH WORKER")
end = text.find("# VLM WORKER")
if start != -1 and end != -1 and end > start:
    removed = end - start
    text = text[:start] + text[end:]
    p.write_text(text, encoding="utf-8")
    report.append(("models.py: DepthWorker class", 1))
    print(f"removed {removed} chars of DepthWorker")
else:
    report.append(("models.py: DepthWorker class", 0))

# ── 2. drop depth from the pipeline ─────────────────────────────────────
patch("backend/vision/pipeline.py", [
    (
        "import",
        "from .models import YOLOWorker, DepthWorker, VLMWorker",
        "from .models import YOLOWorker, VLMWorker",
    ),
    (
        "worker",
        '        # ━━━ Depth Worker ━━━\n        self.depth_worker = DepthWorker(_w("midas_v21_small.pt"))\n',
        "        # Depth estimation was removed: MiDaS could not be loaded without a torch.hub\n"
        "        # trust prompt that blocks non-interactive startup, and nothing downstream\n"
        "        # ever consumed the result - `map_to_grid` explicitly ignored it. Keeping a\n"
        "        # worker whose output is discarded, behind a blocker, was dead weight.\n"
        "        # Restoring it would need a four-point ground-plane calibration first, at\n"
        "        # which point the depth map would still be relative rather than metric.\n",
    ),
    (
        "use_depth flag",
        "        #: Depth estimation is off by default; see the note where it is dispatched.\n        self.use_depth = False",
        "        #: Retained so callers that set it do not fail; has no effect without depth.\n        self.use_depth = False",
    ),
    (
        "load",
        '            if self.depth_worker.loaded or self.depth_worker.load():\n'
        '                loaded["depth"] = True\n'
        '            else:\n'
        '                failed["depth"] = "load returned false (MiDaS needs timm installed)"\n\n',
        "",
    ),
    (
        "dispatch",
        '            # Depth is opt-in. It used to run on every frame and cost a full-frame\n'
        '            # colour conversion, an H2D copy, a 256->1080 bicubic upsample on the GPU\n'
        '            # and an 8.3 MB transfer back - and the only consumer ignored the value.\n'
        '            depth_task = (\n'
        '                loop.run_in_executor(self.executor, self.depth_worker.predict, frame)\n'
        '                if self.use_depth else None\n'
        '            )\n',
        '            # No depth worker: see the note in __init__.\n            depth_task = None\n',
    ),
])

# ── 3. requirements: retire timm ────────────────────────────────────────
p = pathlib.Path("requirements.txt")
text = p.read_text(encoding="utf-8")
start = text.find("# --- Depth estimation: loadable in principle, BLOCKED in practice ------------")
end = text.find("# --- Declared but not used at runtime ----------------------------------------")
if start != -1 and end != -1 and end > start:
    text = (
        text[:start]
        + "# --- Depth estimation: REMOVED -------------------------------------------------\n"
        "# MiDaS is no longer used. It required `timm` for its torch.hub model definition\n"
        "# AND a nested `rwightman_gen-efficientnet-pytorch` hub repo that prompts\n"
        "# \"Do you trust this repository? (y/N)\" on first load - which raises EOFError in a\n"
        "# non-interactive service, so the model never loaded. Nothing consumed its output:\n"
        "# `VisionAggregator.map_to_grid` ignored the depth value, because MiDaS emits\n"
        "# relative inverse depth with no metric scale and so cannot correct a ground-plane\n"
        "# position anyway. The worker, its weights and the `timm` dependency are gone.\n\n"
        + text[end:]
    )
    p.write_text(text, encoding="utf-8")
    report.append(("requirements.txt: timm retired", 1))

bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
