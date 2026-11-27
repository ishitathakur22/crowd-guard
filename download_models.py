"""
Model Downloader — Downloads the AI model weights used by the vision pipeline.

Models:
  1. YOLOv8 Nano   — pedestrian detection, tracked with ByteTrack
  2. YOLOv8 Small  — vehicle detection
  3. SmolVLM2 2.2B — Vision Language Model (optional; needs llama-cpp-python)

Depth estimation (MiDaS) is no longer downloaded: it was removed because its
torch.hub definition requires a nested repo that prompts for trust on first load,
which cannot complete in a non-interactive service, and nothing consumed the
result. See `backend/vision/pipeline.py` for the note.

It is left to ultralytics to fetch the YOLO checkpoints it knows about. Pass
`--with-vlm` to also pull the (large) VLM weights.
"""

import argparse
import os

from ultralytics import YOLO

WEIGHTS_DIR = "weights"

#: The two detectors the pipeline actually instantiates. Keep in step with
#: `VisionPipeline.__init__` — a checkpoint listed here but not there is dead weight,
#: and one used there but missing here will not be downloaded.
YOLO_CHECKPOINTS = [
    ("yolov8n.pt", "YOLOv8 Nano (pedestrians, ByteTrack)"),
    ("yolov8s.pt", "YOLOv8 Small (vehicles)"),
]

VLM_REPO = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"
VLM_FILES = [
    "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
    "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
]


def download_yolo():
    for i, (filename, label) in enumerate(YOLO_CHECKPOINTS, start=1):
        print(f"{i}. {label} -> {filename}")
        YOLO(os.path.join(WEIGHTS_DIR, filename))
        print("   done\n")


def download_vlm():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  huggingface-hub not installed; skipping the VLM.")
        print("  Run: pip install huggingface-hub")
        return

    for filename in VLM_FILES:
        dest = os.path.join(WEIGHTS_DIR, filename)
        if os.path.exists(dest):
            print(f"  {filename}: already present")
            continue
        print(f"  {filename}: downloading from {VLM_REPO}...")
        hf_hub_download(repo_id=VLM_REPO, filename=filename, local_dir=WEIGHTS_DIR)
        print(f"  {filename}: done")


def main():
    ap = argparse.ArgumentParser(description="Download model weights.")
    ap.add_argument(
        "--with-vlm",
        action="store_true",
        help="also download the ~1.9 GB SmolVLM2 weights (only useful with llama-cpp-python)",
    )
    args = ap.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"--- Downloading model weights to '{WEIGHTS_DIR}/' ---\n")

    download_yolo()

    if args.with_vlm:
        print(f"{len(YOLO_CHECKPOINTS) + 1}. SmolVLM2 2.2B Q4_K_M (vision language model)")
        download_vlm()
        print()

    print("--- Done ---")
    if not args.with_vlm:
        print("VLM weights skipped. Re-run with --with-vlm if you have compiled")
        print("llama-cpp-python and want scene-description analysis.")


if __name__ == "__main__":
    main()
