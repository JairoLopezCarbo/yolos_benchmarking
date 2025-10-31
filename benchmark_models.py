# ================================
# File: benchmark_models.py
# ================================
"""
Benchmark trained .pt models by predicting all images in a folder.
Two modes:
  1) EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS: evaluate all .pt files in MODELS_DIR with common predict params.
  2) EVAL_SINGLE_MODEL_PARAM_SWEEP: evaluate ONE .pt across a grid of predict params (e.g., conf, iou);
     results saved under predictions/<model>_CONF-x_IOU-y.

Rules:
- For each image, run 10 predictions, but only save the *first* predicted image with masks overlaid.
- For every prediction (repeat), log timing row with columns: total_ms, preprocess_ms, inference_ms, postprocess_ms, image_id, repeat_idx.
- Per-model timing CSV lives in predictions/<subfolder>/timings.csv
- Saved image filenames are consistent across model folders (use original basename as ID).
- Also writes a helper script predictions/grade_predictions.py to manually score images 1–10 across models.
"""
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------
# === CONSTANTS ===
# --------------------
MODE = "EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS"  # "EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS" | "EVAL_SINGLE_MODEL_PARAM_SWEEP"

# Images to benchmark
IMAGES_DIR = Path("./benchmark_images")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
N_REPEATS_PER_IMAGE = 10

# Where your trained models live (for mode 1)
MODELS_DIR = Path("./trained_models")

# Single model for param sweep (mode 2)
SINGLE_MODEL_PATH = MODELS_DIR / "yolo11n-seg_epochs-50_imgsz-640.pt"

# Common predict params (mode 1)
COMMON_PREDICT = dict(
    imgsz=640,
    conf=0.25,
    iou=0.5,
    classes=None,        # e.g., [0]
    half=True,
    device="cpu",
    stream=False,
    verbose=False,
    retina_masks=True,
)

# Param grid for mode 2
PREDICT_PARAM_VARIANTS = [
    {"conf": 0.25, "iou": 0.50},
    {"conf": 0.50, "iou": 0.50},
    {"conf": 0.25, "iou": 0.75},
]

# Predictions root folder
PREDICTIONS_ROOT = Path("./predictions")
PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------
# === UTILS ===
# --------------------

def list_images(folder: Path) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    paths.sort()
    return paths


def rand_color() -> Tuple[int, int, int]:
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def overlay_masks(img_bgr: np.ndarray, result) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    masks = getattr(result, "masks", None)
    if masks is not None and masks.data is not None:
        m = masks.data.cpu().numpy().astype(np.float32)  # (N, Hm, Wm)
        overlay = np.zeros_like(out, dtype=np.uint8)
        colors = [rand_color() for _ in range(m.shape[0])]
        for i in range(m.shape[0]):
            mask_small = m[i]
            mask_resized = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask_resized > 0.5
            overlay[mask] = colors[i]
        out = cv2.addWeighted(overlay, 0.45, out, 1 - 0.45, 0)
    return out


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_header_if_new(csv_path: Path, header: List[str]):
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def predict_one_image(model: YOLO, img_path: Path, predict_kwargs: Dict, save_overlay_to: Path) -> List[Dict]:
    """Run N_REPEATS_PER_IMAGE predictions; save overlay for the first; return list of timing dicts per repeat."""
    timings = []
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    for rep in range(N_REPEATS_PER_IMAGE):
        t0 = time.perf_counter()
        res_list = model.predict(source=img, **predict_kwargs)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0

        # Ultralytics returns list of Results; take first
        r = res_list[0]
        speed = getattr(r, "speed", {})  # dict with preprocess, inference, postprocess in ms
        row = {
            "total_ms": round(total_ms, 3),
            "preprocess_ms": round(float(speed.get("preprocess", float("nan"))), 3),
            "inference_ms": round(float(speed.get("inference", float("nan"))), 3),
            "postprocess_ms": round(float(speed.get("postprocess", float("nan"))), 3),
        }
        timings.append(row)

        if rep == 0:
            drawn = overlay_masks(img, r)
            cv2.imwrite(str(save_overlay_to), drawn)

    return timings


def save_timings(csv_path: Path, image_id: str, timings: List[Dict]):
    write_header_if_new(csv_path, ["image_id", "repeat_idx", "total_ms", "preprocess_ms", "inference_ms", "postprocess_ms"])
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        for i, t in enumerate(timings):
            w.writerow([image_id, i, t["total_ms"], t["preprocess_ms"], t["inference_ms"], t["postprocess_ms"]])



# --------------------
# === MODES ===
# --------------------

def eval_multiple_models_default():
    images = list_images(IMAGES_DIR)
    if not images:
        raise FileNotFoundError(f"No images found under {IMAGES_DIR}")

    model_files = sorted([p for p in MODELS_DIR.iterdir() if p.suffix == ".pt"])
    if not model_files:
        raise FileNotFoundError(f"No .pt models found under {MODELS_DIR}")

    model_folder_names = []

    for mpath in model_files:
        model_name = mpath.stem  # subfolder name under predictions
        model_folder_names.append(model_name)
        out_dir = PREDICTIONS_ROOT / model_name
        ensure_dir(out_dir)
        timings_csv = out_dir / "timings.csv"

        print(f"\n>>> Evaluating model: {mpath}")
        model = YOLO(str(mpath))

        for img_path in images:
            image_id = img_path.stem
            save_to = out_dir / f"{image_id}{img_path.suffix.lower()}"
            timings = predict_one_image(model, img_path, COMMON_PREDICT, save_to)
            save_timings(timings_csv, image_id, timings)




def eval_single_model_param_sweep():
    images = list_images(IMAGES_DIR)
    if not images:
        raise FileNotFoundError(f"No images found under {IMAGES_DIR}")

    if not SINGLE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {SINGLE_MODEL_PATH}")

    model = YOLO(str(SINGLE_MODEL_PATH))
    base_name = SINGLE_MODEL_PATH.stem

    model_folder_names = []

    for variant in PREDICT_PARAM_VARIANTS:
        conf = variant.get("conf", COMMON_PREDICT["conf"])  # default
        iou = variant.get("iou", COMMON_PREDICT["iou"])    # default
        subname = f"{base_name}_CONF-{conf}_IOU-{iou}"
        model_folder_names.append(subname)
        out_dir = PREDICTIONS_ROOT / subname
        ensure_dir(out_dir)
        timings_csv = out_dir / "timings.csv"

        predict_kwargs = {**COMMON_PREDICT, **variant}

        print(f"\n>>> Evaluating {SINGLE_MODEL_PATH.name} with params: {variant}")
        for img_path in images:
            image_id = img_path.stem
            save_to = out_dir / f"{image_id}{img_path.suffix.lower()}"
            timings = predict_one_image(model, img_path, predict_kwargs, save_to)
            save_timings(timings_csv, image_id, timings)



if __name__ == "__main__":
    if MODE == "EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS":
        eval_multiple_models_default()
    elif MODE == "EVAL_SINGLE_MODEL_PARAM_SWEEP":
        eval_single_model_param_sweep()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
