import importlib.util as u

for name in ("llama_cpp", "timm", "cv2", "ultralytics", "torch", "filterpy", "websockets"):
    print(f"{name}: {bool(u.find_spec(name))}")
