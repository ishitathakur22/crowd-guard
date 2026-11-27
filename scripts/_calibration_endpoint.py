"""Step 4: expose vision calibration as an API endpoint and a persisted JSON file."""

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


# ── pipeline: accept and persist a calibration ──────────────────────────
patch("backend/vision/pipeline.py", [
    (
        "calibration helpers",
        "    def set_source(self, source):",
        '''    #: Where an operator-supplied calibration is kept between restarts.
    CALIBRATION_FILE = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "vision_calibration.json"
    )

    def load_calibration(self) -> bool:
        """
        Apply a saved four-point calibration, if one exists.

        Stored as JSON so an operator can supply ground-truth points without editing
        code or restarting into a hard-coded homography.
        """
        try:
            with open(self.CALIBRATION_FILE) as f:
                data = json.load(f)
            img = data["reference_points_img"]
            grid = data["reference_points_grid"]
        except (OSError, KeyError, json.JSONDecodeError, TypeError):
            return False

        ok = self.aggregator.calibrate(img, grid)
        if ok:
            logger.info("Applied saved vision calibration from %s", self.CALIBRATION_FILE)
        return ok

    def save_calibration(self, reference_points_img, reference_points_grid) -> bool:
        """Validate, apply and persist a calibration. Returns False if unusable."""
        if not self.aggregator.calibrate(reference_points_img, reference_points_grid):
            return False
        try:
            os.makedirs(os.path.dirname(self.CALIBRATION_FILE), exist_ok=True)
            with open(self.CALIBRATION_FILE, "w") as f:
                json.dump(
                    {
                        "reference_points_img": reference_points_img,
                        "reference_points_grid": reference_points_grid,
                        "grid_width": self.aggregator.grid_width,
                        "grid_height": self.aggregator.grid_height,
                    },
                    f,
                    indent=2,
                )
        except OSError as exc:
            logger.warning("Calibration applied but could not be saved: %s", exc)
            return True
        return True

    def calibration_status(self) -> dict:
        return {
            "calibrated": self.aggregator.is_calibrated,
            "proper_homography": self.aggregator.is_proper_homography,
            "saved_file": os.path.exists(self.CALIBRATION_FILE),
            "grid_width": self.aggregator.grid_width,
            "grid_height": self.aggregator.grid_height,
            "note": (
                "A proper homography needs four point pairs measured on the floor. "
                "Without one the pipeline assumes the frame covers the whole venue "
                "with no perspective correction, which misplaces anyone further from "
                "the camera."
            ),
        }

    def set_source(self, source):''',
    ),
    (
        "json import",
        "import threading\nfrom typing import Callable, Optional",
        "import json\nimport threading\nfrom typing import Callable, Optional",
    ),
    (
        "apply saved calibration on start",
        "        if not self.aggregator.is_calibrated:\n            self.aggregator.calibrate_from_corners(frame_w, frame_h)",
        "        if not self.aggregator.is_calibrated and not self.load_calibration():\n"
        "            self.aggregator.calibrate_from_corners(frame_w, frame_h)",
    ),
])

# ── main.py: endpoints ──────────────────────────────────────────────────
patch("backend/main.py", [
    (
        "endpoints",
        '@app.get("/api/vision/status")',
        '''class CalibrationRequest(BaseModel):
    reference_points_img: List[List[float]]
    reference_points_grid: List[List[float]]


@app.get("/api/vision/calibration")
async def get_calibration():
    """Report whether the vision pipeline has a real ground-plane calibration."""
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    return vision_pipeline.calibration_status()


@app.post("/api/vision/calibration")
async def set_calibration(request: CalibrationRequest):
    """
    Supply four point pairs measured on the floor, in image and venue coordinates.

    Validated before it is applied: if the homography cannot be solved the previous
    calibration is left untouched rather than silently degraded.
    """
    if not ENABLE_VISION or vision_pipeline is None:
        return DISABLED_VISION_MSG
    if len(request.reference_points_img) != len(request.reference_points_grid):
        return {"status": "error", "detail": "point lists must be the same length"}
    if len(request.reference_points_img) < 4:
        return {"status": "error", "detail": "four point pairs are required"}

    ok = vision_pipeline.save_calibration(
        request.reference_points_img, request.reference_points_grid
    )
    if not ok:
        return {"status": "error", "detail": "homography could not be solved from those points"}
    return {"status": "calibrated", **vision_pipeline.calibration_status()}


@app.get("/api/vision/status")''',
    ),
])

bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
