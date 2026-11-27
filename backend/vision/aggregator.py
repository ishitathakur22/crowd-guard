"""
Vision aggregator — turns raw model output into grid-space agents.

Three defects lived in the previous version and are fixed here.

**Every person was counted two or three times.** The pipeline runs three detectors
that all see pedestrians: `crowd` (classes=[0]), `flow` (classes=[0], on the
segmentation model) and `anomaly` (which passed `classes=None`, so it detected all
80 COCO classes including person). Nothing merged their output, so one pedestrian
became two or three grid agents, and the count fed to the forecast's ingress
estimate was inflated accordingly. Detections are now de-duplicated per class with
greedy IoU suppression.

**Identity was positional.** IDs were `f"real_{frame_id}_{len(real_agents)}"`, so
they changed every frame and every reordering of the detection list. Nothing could
be tracked across time, which rules out velocity, flow and any temporal smoothing.
IDs now come from the tracker when one is supplied.

**The homography was a disguised scale.** `auto_calibrate` was fed the image corners
mapped to the grid corners, which yields exactly `x * (grid_w / frame_w)` — an
anisotropic stretch with no perspective correction at all. A surveillance view is
never parallel to the floor, so distance from the camera was mapped to the same
metres-per-pixel as the foreground. `calibrate` now takes four real ground points;
the corner stretch remains only as an explicit, clearly-labelled fallback.

Depth is retained but optional. It was previously computed every frame at 1080p
(8.3 MB of transfer per frame) and then *ignored* by `map_to_grid`, which took the
argument and never used it.
"""

import logging
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger("VisionAggregator")


def iou(a, b) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedupe(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Merge detections of the same class coming from different models.

    Keeps the highest-confidence box and drops any same-class box overlapping it
    beyond `iou_threshold`. Ordered by confidence so the survivor is the best
    evidence available rather than whichever model happened to run last.
    """
    ordered = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
    kept: List[Dict] = []
    for det in ordered:
        box = det.get("bbox")
        cls = det.get("class")
        if box is None:
            kept.append(det)
            continue
        duplicate = any(
            k.get("class") == cls
            and k.get("bbox") is not None
            and iou(box, k["bbox"]) >= iou_threshold
            for k in kept
        )
        if not duplicate:
            kept.append(det)
    return kept


class VisionAggregator:
    """
    Maps detections onto the simulation grid.

    `calibrate` is the entry point that matters: it needs four pixel positions whose
    real grid coordinates are known (typically floor markings), which is what makes a
    perspective view correctable.
    """

    def __init__(self, grid_width: float, grid_height: float):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.homography_matrix = None
        self.is_calibrated = False
        #: True when calibrated from measured ground points rather than assumed corners.
        self.is_proper_homography = False
        self._frame_size: Optional[Tuple[int, int]] = None

    # ── calibration ───────────────────────────────────────────────

    def calibrate(self, reference_points_img, reference_points_grid) -> bool:
        """
        Compute the ground-plane homography from four measured point pairs.

        `reference_points_img` are pixels on the floor, `reference_points_grid` their
        metres in the venue. Four non-collinear pairs determine the transform, which
        is what corrects a camera looking down at an angle.
        """
        if len(reference_points_img) < 4 or len(reference_points_grid) < 4:
            logger.warning("Calibration needs at least four point pairs.")
            return False

        pts_src = np.array(reference_points_img, dtype=np.float32)
        pts_dst = np.array(reference_points_grid, dtype=np.float32)
        matrix, _status = cv2.findHomography(pts_src, pts_dst)

        if matrix is None or not np.all(np.isfinite(matrix)):
            logger.warning("Homography could not be solved from those points.")
            return False

        self.homography_matrix = matrix
        self.is_calibrated = True
        self.is_proper_homography = True
        logger.info("Ground-plane homography calibrated from measured points.")
        return True

    def calibrate_from_corners(self, frame_width: int, frame_height: int) -> None:
        """
        Fallback: assume the frame covers exactly the venue, edge to edge.

        This is an anisotropic stretch, not perspective correction, and it is wrong
        for any real camera that sees the floor at an angle. It is kept so the
        pipeline degrades predictably when no calibration has been supplied, and it
        is labelled as an assumption rather than presented as calibration.
        """
        self._frame_size = (int(frame_width), int(frame_height))
        self.homography_matrix = None
        self.is_calibrated = True
        self.is_proper_homography = False
        logger.warning(
            "Vision calibrated by assumption: the frame is treated as covering the "
            "whole venue with no perspective correction. Supply four ground points "
            "via VisionAggregator.calibrate() for correct positions."
        )

    # ── mapping ───────────────────────────────────────────────────

    def map_to_grid(self, pixel_x: float, pixel_y: float, depth_value: float = None):
        """
        Convert one pixel to venue coordinates.

        `depth_value` is accepted for interface compatibility but deliberately unused:
        MiDaS emits relative inverse depth with per-frame normalisation and no metric
        scale, so it cannot correct a ground-plane position. Real range needs a
        calibrated homography or a stereo baseline.
        """
        if not self.is_calibrated:
            return self._scale_by_frame(pixel_x, pixel_y)

        if self.homography_matrix is None:
            return self._scale_by_frame(pixel_x, pixel_y)

        pts = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(pts, self.homography_matrix)
        gx, gy = float(dst[0][0][0]), float(dst[0][0][1])
        return gx, gy

    def _scale_by_frame(self, pixel_x: float, pixel_y: float):
        fw, fh = self._frame_size or (1920, 1080)
        return (pixel_x / fw) * self.grid_width, (pixel_y / fh) * self.grid_height

    def map_points_to_grid(self, points, depth_map=None) -> np.ndarray:
        """Vectorised conversion of many pixels; one transform call, not one per point."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if self.is_calibrated and self.homography_matrix is not None:
            dst = cv2.perspectiveTransform(pts, self.homography_matrix)
            return dst.reshape(-1, 2).astype(np.float64)
        fw, fh = self._frame_size or (1920, 1080)
        out = pts.reshape(-1, 2).astype(np.float64)
        out[:, 0] = (out[:, 0] / fw) * self.grid_width
        out[:, 1] = (out[:, 1] / fh) * self.grid_height
        return out

    # ── aggregation ───────────────────────────────────────────────

    def aggregate(self, frame_id: int, yolo_results: list, depth_map: np.ndarray = None,
                  vlm_analysis: dict = None, in_plan: bool = True):
        """
        Merge detections from every model into one list of grid agents.

        Detections whose mapped position falls outside the venue are dropped rather
        than clamped. Clamping used to squash every off-plan detection onto the border,
        which built a false density ridge along the edges — exactly where the gate
        zones and the early warnings are.
        """
        detections = dedupe(yolo_results)

        if not detections:
            return []

        # One vectorised transform for the whole batch.
        centres = np.array([d["center"] for d in detections], dtype=np.float64)
        mapped = self.map_points_to_grid(centres)

        agents = []
        for det, (gx, gy) in zip(detections, mapped):
            if in_plan and not (0.0 <= gx < self.grid_width and 0.0 <= gy < self.grid_height):
                continue
            agents.append({
                # Stable when the caller supplies tracked detections; falls back to a
                # positional id only when no tracker ran, which is stated in the docstring.
                "id": det.get("track_id", f"det_{frame_id}_{len(agents)}"),
                "x": round(float(gx), 2),
                "y": round(float(gy), 2),
                "type": det.get("task", "unknown"),
                "class_id": det.get("class"),
                "class_name": det.get("class_name", ""),
                "conf": det.get("conf", 0.0),
                "tracked": det.get("track_id") is not None,
            })

        return agents
