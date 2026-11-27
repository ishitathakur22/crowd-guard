"""Rewrite YOLOWorker for batched transfers + tracking, and make DepthWorker honest."""

import pathlib

path = pathlib.Path("backend/vision/models.py")
text = path.read_text(encoding="utf-8")
report = []


def sub(old, new, label):
    global text
    n = text.count(old)
    report.append((label, n))
    text = text.replace(old, new)


sub(
    '''    def __init__(self, model_path: str, task_name: str, classes: list = None, device: str = 'cuda'):
        super().__init__(model_path, device)
        self.task_name = task_name
        self.classes = classes''',
    '''    #: Detection confidence floor. Ultralytics' default is 0.25; pedestrians in a
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
        self._names = None''',
    "YOLOWorker init",
)

sub(
    '''        results = self.model(frame, classes=self.classes, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names.get(cls, str(cls)) if self.model.names else str(cls)
                detections.append({
                    "task": self.task_name,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float((x1 + x2) / 2), float(y2)],
                    "conf": round(conf, 3),
                    "class": cls,
                    "class_name": cls_name,
                })''',
    '''        # `track` assigns persistent ids via ByteTrack, so the same pedestrian keeps
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
                detections.append(det)''',
    "YOLOWorker predict",
)

sub(
    '''            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False, trust_repo=True)
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
            else:
                logger.warning(f"Local weights not found at {self.model_path}, using torch.hub fallback.")
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)

            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform''',
    '''            # MiDaS's torch.hub entry point pulls the model definition from GitHub,
            # and that definition imports `timm`. Without timm installed the loader
            # fails with ModuleNotFoundError before the local weights are ever read,
            # so `timm` is a hard requirement for depth even when the .pt file is
            # already on disk. That was not obvious from the code, hence this note.
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False, trust_repo=True)
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
            else:
                logger.warning(f"Local weights not found at {self.model_path}, using torch.hub fallback.")
                self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)

            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform''',
    "MiDaS timm note",
)

path.write_text(text, encoding="utf-8")

print(f"patched {path} ({len(text)} bytes)")
bad = 0
for label, n in report:
    if n < 1:
        bad += 1
    print(f"  {'ok ' if n >= 1 else 'MISS'} {label}: {n}")
print(f"misses: {bad}")
