import cv2
import time
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Callable, Optional

from .models import YOLOWorker, DepthWorker, VLMWorker
from .aggregator import VisionAggregator
from .model_manager import model_manager

logger = logging.getLogger("VisionPipeline")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WEIGHTS DIRECTORY (relative to project root)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "weights")


def _w(filename: str) -> str:
    """Resolve a weight filename to its full path."""
    return os.path.join(WEIGHTS_DIR, filename)


class VisionPipeline:
    """
    Multi-model vision pipeline for real-time and recorded video analysis.

    Runs 4 YOLO models + 1 Depth model + 1 VLM concurrently per frame.
    The VLM only runs on keyframes (every N frames) to avoid bottlenecking.
    """

    def __init__(self, source=0, grid_width=100.0, grid_height=100.0):
        self.source = source
        self.cap = None
        self.running = False
        self.mode = "idle"  # idle | live | video | batch

        self.aggregator = VisionAggregator(grid_width, grid_height)
        self._init_lock = threading.Lock()

        # ━━━ YOLO Workers (using correct downloaded weights) ━━━
        # Only ONE detector looks for people, and it tracks them.
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
        }

        # ━━━ Depth Worker ━━━
        self.depth_worker = DepthWorker(_w("midas_v21_small.pt"))

        # ━━━ VLM Worker (SmolVLM2 2.2B Q4 on GPU) ━━━
        self.vlm_worker = VLMWorker(
            model_path=_w("SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"),
            mmproj_path=_w("mmproj-SmolVLM2-2.2B-Instruct-f16.gguf"),
            n_gpu_layers=-1,
        )
        self.vlm_keyframe_interval = 30  # Run VLM every 30 frames
        #: Depth estimation is off by default; see the note where it is dispatched.
        self.use_depth = False

        # Thread pool for concurrent inference
        self.executor = ThreadPoolExecutor(max_workers=7)

        # Callbacks
        self.on_agents_detected: Optional[Callable] = None
        self.on_vision_update: Optional[Callable] = None  # For dashboard metrics

        # Metrics
        self.fps = 0.0
        self.frame_count = 0
        self.total_frames = 0  # For video files
        self.last_vlm_analysis = None

        # Batch queue
        self._video_queue: list = []
        self._current_video_name = ""

    def initialize(self):
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
            }

    def set_source(self, source):
        """Change the video source (0=webcam, or file path string)."""
        self.source = source

    async def start(self):
        """Start processing the current video source."""
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return

        self.running = True
        is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        self.mode = "video" if is_file else "live"
        self._current_video_name = os.path.basename(self.source) if is_file else "Webcam"

        if is_file:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_frames = 0

        logger.info(f"Vision Pipeline started — mode={self.mode}, source={self.source}")

        # Calibration. The corners mapping below is an assumption, not calibration:
        # it treats the frame as covering exactly the venue with no perspective
        # correction, so anyone further from the camera maps to the same
        # metres-per-pixel as the foreground. `calibrate()` with four measured ground
        # points is the correct route and takes precedence when supplied.
        frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        if not self.aggregator.is_calibrated:
            self.aggregator.calibrate_from_corners(frame_w, frame_h)

        self.frame_count = 0
        fps_timer = time.time()
        fps_frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.mode == "video":
                    logger.info(f"Video '{self._current_video_name}' analysis complete.")
                    self.running = False
                    break
                else:
                    await asyncio.sleep(0.5)
                    continue

            self.frame_count += 1
            fps_frame_count += 1
            start_t = time.time()

            # ━━━ Dispatch YOLO + Depth concurrently ━━━
            loop = asyncio.get_running_loop()
            yolo_tasks = [
                loop.run_in_executor(self.executor, worker.predict, frame)
                for worker in self.workers.values()
            ]
            # Depth is opt-in. It used to run on every frame and cost a full-frame
            # colour conversion, an H2D copy, a 256->1080 bicubic upsample on the GPU
            # and an 8.3 MB transfer back - and the only consumer ignored the value.
            depth_task = (
                loop.run_in_executor(self.executor, self.depth_worker.predict, frame)
                if self.use_depth else None
            )

            # ━━━ VLM on keyframes only ━━━
            vlm_task = None
            if self.frame_count % self.vlm_keyframe_interval == 0:
                vlm_task = loop.run_in_executor(self.executor, self.vlm_worker.predict, frame)

            # Wait for YOLO + Depth
            yolo_results_raw = await asyncio.gather(*yolo_tasks)
            depth_map = (await depth_task) if depth_task is not None else None

            # Wait for VLM if it was dispatched
            vlm_analysis = None
            if vlm_task is not None:
                vlm_analysis = await vlm_task
                if vlm_analysis:
                    self.last_vlm_analysis = vlm_analysis

            # Flatten YOLO results
            all_detections = []
            for r in yolo_results_raw:
                all_detections.extend(r)

            # Aggregate
            real_agents = self.aggregator.aggregate(
                self.frame_count, all_detections, depth_map, vlm_analysis
            )

            # Inject into simulation
            if self.on_agents_detected:
                self.on_agents_detected(real_agents)

            # Calculate FPS
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self.fps = round(fps_frame_count / elapsed, 1)
                fps_frame_count = 0
                fps_timer = time.time()

            inference_ms = round((time.time() - start_t) * 1000, 1)

            # Build detection summary for dashboard
            det_summary = {}
            for det in all_detections:
                task = det["task"]
                det_summary[task] = det_summary.get(task, 0) + 1

            # Push vision update to frontend via callback
            if self.on_vision_update:
                self.on_vision_update({
                    "fps": self.fps,
                    "latency_ms": inference_ms,
                    "frame": self.frame_count,
                    "total_frames": self.total_frames,
                    "mode": self.mode,
                    "detections": det_summary,
                    "total_detections": len(all_detections),
                    "vlm_analysis": self.last_vlm_analysis,
                    "model_manager": model_manager.get_status(),
                })

            # Yield control
            await asyncio.sleep(0.005)

        # Cleanup
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mode = "idle"

    async def scan_videos(self, paths: list):
        """
        Batch scan multiple video files sequentially.
        `paths` can contain file paths and/or folder paths.
        """
        self._video_queue = []
        for p in paths:
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm')):
                        self._video_queue.append(os.path.join(p, f))
            elif os.path.isfile(p):
                self._video_queue.append(p)

        logger.info(f"Batch scan: {len(self._video_queue)} videos queued.")
        self.mode = "batch"

        for i, video_path in enumerate(self._video_queue):
            logger.info(f"Processing video {i + 1}/{len(self._video_queue)}: {video_path}")
            self.set_source(video_path)
            await self.start()
            # Small pause between videos
            await asyncio.sleep(0.1)

        self.mode = "idle"
        logger.info("Batch scan complete.")

    def stop(self):
        """Stop the pipeline."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.mode = "idle"
        logger.info("Vision Pipeline stopped.")
