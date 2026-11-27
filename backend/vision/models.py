import cv2
import os
import base64
import json
import time
import numpy as np
import torch
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .model_manager import model_manager

logger = logging.getLogger("VisionModels")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASE CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseVisionModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = model_manager.device  # Use manager's device
        self.model = None
        self.loaded = False

    def load(self):
        raise NotImplementedError

    def predict(self, frame: np.ndarray):
        raise NotImplementedError


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YOLO WORKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class YOLOWorker(BaseVisionModel):
    """Runs a specific YOLOv8 model for object detection/segmentation."""

    #: Detection confidence floor. Ultralytics' default is 0.25; pedestrians in a
    #: crowd are routinely detected below that at distance, and a missed person is
    #: worse for a safety system than a spurious one that deduplication can remove.
    DEFAULT_CONF = 0.20

    def __init__(self, model_path: str, task_name: str, classes: list = None,
                 device: str = 'cuda', conf: float = None, track: bool = False):
        super().__init__(model_path, device)
        self.task_name = task_name
        self.classes = classes
        self.conf = self.DEFAULT_CONF if conf is None else conf
        #: When True, `predict` uses ultralytics' tracker so ids are stable across
        #: frames rather than positional.
        self.track = track
        self._names = None

        # Register with the model manager
        model_manager.register(task_name, "yolo", model_path)

    def load(self):
        if YOLO is None:
            logger.warning("ultralytics not installed. YOLOWorker disabled.")
            return False

        def _loader():
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.loaded = True
            return True

        return model_manager.load_model(self.task_name, _loader)

    def predict(self, frame: np.ndarray):
        if not self.loaded or self.model is None:
            return []

        start_t = time.time()

        # `track` assigns persistent ids via ByteTrack, so the same pedestrian keeps
        # the same id between frames. Without it every id is positional and nothing
        # downstream can derive velocity or hold a track together.
        if self.track:
            results = self.model.track(
                frame, classes=self.classes, conf=self.conf,
                persist=True, verbose=False,
            )
        else:
            results = self.model(frame, classes=self.classes, conf=self.conf, verbose=False)

        if self._names is None:
            self._names = self.model.names or {}

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # One host transfer for the whole frame. The previous version called
            # `.cpu().numpy()` three times per box - xyxy, conf and cls - and each of
            # those forces a device synchronisation. At 40 people per frame that was
            # about 120 syncs per model per frame, which dwarfed the inference itself.
            data = boxes.data.cpu().numpy()
            has_ids = boxes.id is not None
            ids = boxes.id.cpu().numpy().astype(int) if has_ids else None

            for row_idx in range(data.shape[0]):
                x1, y1, x2, y2, conf, cls = data[row_idx][:6]
                cls = int(cls)
                det = {
                    "task": self.task_name,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    # Bottom edge, centre-x: the ground contact point, which is the
                    # correct anchor for a floor-plane mapping.
                    "center": [float((x1 + x2) / 2), float(y2)],
                    "conf": round(float(conf), 3),
                    "class": cls,
                    "class_name": self._names.get(cls, str(cls)),
                }
                if ids is not None:
                    det["track_id"] = f"{self.task_name}_{int(ids[row_idx])}"
                detections.append(det)

        latency = (time.time() - start_t) * 1000
        model_manager.record_inference(self.task_name, latency)

        return detections


# VLM WORKER (SmolVLM2 2.2B via llama-cpp-python)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VLM_SECURITY_PROMPT = (
    "You are a crowd safety AI analyst. Analyze this surveillance frame and respond ONLY with a JSON object. "
    "Do NOT include markdown formatting, code fences, or any text outside the JSON. "
    "Keys: "
    '"crowd_density" (one of: "low", "moderate", "high", "critical"), '
    '"estimated_people" (integer), '
    '"safety_hazards" (list of strings, empty if none), '
    '"unusual_behavior" (list of strings, empty if none), '
    '"stampede_risk" (one of: "none", "low", "moderate", "high"), '
    '"summary" (one sentence overall assessment).'
)


class VLMWorker:
    """
    Vision Language Model worker using SmolVLM2 2.2B Q4 via llama-cpp-python.
    Loaded and managed through the ModelManager for VRAM tracking.
    """

    def __init__(self, model_path: str = "weights/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
                 mmproj_path: str = "weights/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
                 n_gpu_layers: int = -1):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_gpu_layers = n_gpu_layers  # -1 = offload all layers to GPU
        self.model = None
        self.loaded = False
        self.last_analysis = None

        model_manager.register("vlm", "vlm", model_path)

    def load(self):
        if Llama is None:
            logger.warning("llama-cpp-python not installed. VLMWorker disabled.")
            return False
        if not os.path.exists(self.mmproj_path):
            logger.error(f"VLM mmproj file not found: {self.mmproj_path}")
            return False

        def _loader():
            self.model = Llama(
                model_path=self.model_path,
                chat_format="llama-3",
                n_ctx=2048,
                n_gpu_layers=self.n_gpu_layers,
                mmproj=self.mmproj_path,
                verbose=False,
            )
            self.loaded = True
            return True

        return model_manager.load_model("vlm", _loader)

    def predict(self, frame: np.ndarray):
        """Analyze a frame and return structured scene analysis dict."""
        if not self.loaded or self.model is None:
            return None

        start_t = time.time()

        try:
            # Encode frame to base64 JPEG for the VLM
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            data_uri = f"data:image/jpeg;base64,{img_b64}"

            response = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VLM_SECURITY_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.3,
            )

            raw_text = response["choices"][0]["message"]["content"].strip()

            # Try to parse as JSON
            try:
                analysis = json.loads(raw_text)
            except json.JSONDecodeError:
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                if start != -1 and end != -1:
                    analysis = json.loads(raw_text[start:end + 1])
                else:
                    analysis = {
                        "crowd_density": "unknown",
                        "estimated_people": 0,
                        "safety_hazards": [],
                        "unusual_behavior": [],
                        "stampede_risk": "unknown",
                        "summary": raw_text[:200],
                    }

            self.last_analysis = analysis

            latency = (time.time() - start_t) * 1000
            model_manager.record_inference("vlm", latency)

            return analysis

        except Exception as e:
            logger.error(f"VLM inference error: {e}")
            return None
