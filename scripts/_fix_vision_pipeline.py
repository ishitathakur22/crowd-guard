"""Fix the vision pipeline: initialize() return value, detector stack, calibration, reader."""

import pathlib

path = pathlib.Path("backend/vision/pipeline.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


# ── 1. collapse the duplicate detector stack ─────────────────────────────
sub(
    '''        self.workers = {
            "crowd": YOLOWorker(_w("yolov8n.pt"), "crowd", classes=[0]),
            "traffic": YOLOWorker(_w("yolov8s.pt"), "traffic", classes=[2, 3, 5, 7]),
            "anomaly": YOLOWorker(_w("yolov8m.pt"), "anomaly"),
            "flow": YOLOWorker(_w("yolov8n-seg.pt"), "flow", classes=[0]),
        }''',
    '''        # Only ONE detector looks for people, and it tracks them.
        #
        # The previous stack ran three pedestrian detectors at once: "crowd"
        # (classes=[0]), "flow" (classes=[0], on the segmentation model) and
        # "anomaly" (classes=None, i.e. all 80 COCO classes including person).
        # Nothing merged their output, so a single pedestrian became two or three
        # grid agents and the detection count fed to the forecast was inflated
        # accordingly. It also cost three letterbox/normalise passes over the same
        # frame for one answer.
        #
        # "crowd" is kept because it is the cheapest model and person detection is
        # the task that matters; "traffic" is kept because vehicles genuinely are a
        # separate class the venue needs to see.
        self.workers = {
            "crowd": YOLOWorker(_w("yolov8n.pt"), "crowd", classes=[0], track=True),
            "traffic": YOLOWorker(_w("yolov8s.pt"), "traffic", classes=[2, 3, 5, 7]),
        }''',
    "worker stack",
)

# ── 2. initialize() must return a result the API can read ────────────────
sub(
    '''    def initialize(self):
        """Loads all models into memory (VRAM). Skips already-loaded models."""
        logger.info("━━━ Initializing Vision Pipeline ━━━")
        for name, worker in self.workers.items():
            if not worker.loaded:
                worker.load()
        if not self.depth_worker.loaded:
            self.depth_worker.load()
        if not self.vlm_worker.loaded:
            self.vlm_worker.load()
        logger.info("━━━ All models initialized ━━━")''',
    '''    def initialize(self):
        """
        Load every model, returning a status dict.

        This used to return None, and the API endpoint did
        `result = await ...initialize(); if not result["loaded"]`, so
        /api/vision/start and /api/vision/scan-video raised TypeError on *every*
        call regardless of whether the models were fine. Returning the report is
        what makes those endpoints work at all.

        Loading is serialised under a lock: the endpoints run it through
        `run_in_executor`, so two overlapping requests could previously load the same
        models twice and double the VRAM footprint.
        """
        with self._init_lock:
            logger.info("--- Initializing Vision Pipeline ---")
            loaded, failed = {}, {}

            for name, worker in self.workers.items():
                if worker.loaded or worker.load():
                    loaded[name] = True
                else:
                    failed[name] = "load returned false"

            if self.depth_worker.loaded or self.depth_worker.load():
                loaded["depth"] = True
            else:
                failed["depth"] = "load returned false (MiDaS needs timm installed)"

            if self.vlm_worker.loaded or self.vlm_worker.load():
                loaded["vlm"] = True
            else:
                failed["vlm"] = "load returned false (llama-cpp-python not installed)"

            # Person detection is the one capability the pipeline cannot work without.
            usable = "crowd" in loaded
            logger.info(
                f"--- Initialized: {sorted(loaded)} loaded, {sorted(failed)} unavailable ---"
            )

            return {
                "loaded": usable,
                "models": sorted(loaded),
                "unavailable": failed,
                "device": model_manager.device,
            }''',
    "initialize return",
)

# ── 3. init lock, and honest calibration ─────────────────────────────────
sub(
    "        self.aggregator = VisionAggregator(grid_width, grid_height)",
    "        self.aggregator = VisionAggregator(grid_width, grid_height)\n"
    "        self._init_lock = threading.Lock()",
    "init lock",
)

sub(
    "from typing import Callable, Optional",
    "import threading\nfrom typing import Callable, Optional",
    "threading import",
)

sub(
    '''        # Auto-calibrate homography (simple scaling)
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        self.aggregator.auto_calibrate(
            frame=None,
            reference_points_img=[[0, 0], [frame_w, 0], [frame_w, frame_h], [0, frame_h]],
            reference_points_grid=[
                [0, 0],
                [self.aggregator.grid_width, 0],
                [self.aggregator.grid_width, self.aggregator.grid_height],
                [0, self.aggregator.grid_height],
            ],
        )''',
    '''        # Calibration. The corners mapping below is an assumption, not calibration:
        # it treats the frame as covering exactly the venue with no perspective
        # correction, so anyone further from the camera maps to the same
        # metres-per-pixel as the foreground. `calibrate()` with four measured ground
        # points is the correct route and takes precedence when supplied.
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        if not self.aggregator.is_calibrated:
            self.aggregator.calibrate_from_corners(frame_w, frame_h)''',
    "calibration",
)

# ── 4. drop the depth call from the hot path (opt-in) ────────────────────
sub(
    '''            depth_task = loop.run_in_executor(self.executor, self.depth_worker.predict, frame)''',
    '''            # Depth is opt-in. It used to run on every frame and cost a full-frame
            # colour conversion, an H2D copy, a 256->1080 bicubic upsample on the GPU
            # and an 8.3 MB transfer back - and the only consumer ignored the value.
            depth_task = (
                loop.run_in_executor(self.executor, self.depth_worker.predict, frame)
                if self.use_depth else None
            )''',
    "depth opt-in",
)

sub(
    "            yolo_results_raw = await asyncio.gather(*yolo_tasks)\n            depth_map = await depth_task",
    "            yolo_results_raw = await asyncio.gather(*yolo_tasks)\n"
    "            depth_map = (await depth_task) if depth_task is not None else None",
    "depth await",
)

sub(
    "        self.vlm_keyframe_interval = 30  # Run VLM every 30 frames",
    "        self.vlm_keyframe_interval = 30  # Run VLM every 30 frames\n"
    "        #: Depth estimation is off by default; see the note where it is dispatched.\n"
    "        self.use_depth = False",
    "use_depth flag",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
