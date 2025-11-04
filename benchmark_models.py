# ================================
# File: benchmark_models.py
# ================================
"""
Benchmark trained .pt models using a single CONFIG dict (readable, editable).

Modes:
    - EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS: evaluate all .pt in CONFIG['models']['dir'] with common predict params.
    - EVAL_SINGLE_MODEL_PARAM_SWEEP: evaluate one .pt over a grid of predict params (conf/iou), saving under
        predictions/<model>_CONF-x_IOU-y.

Rules:
- For each image, run N repeats (CONFIG['data']['repeats_per_image']), but only save the first overlay image.
- For every repeat, log timing row with: total_ms, preprocess_ms, inference_ms, postprocess_ms, image_id, repeat_idx.
- Per-model timing CSV: predictions/<subfolder>/benchmark.csv
- Saved image filenames: original basename preserved across model folders.
"""
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# =============================
# Centralized configuration
# =============================
CONFIG = {
    "mode": "EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS",  # or "EVAL_SINGLE_MODEL_PARAM_SWEEP"
    "data": {
        "images_dir": "benchmark_images",
        "image_exts": [".jpg", ".jpeg", ".png", ".bmp"],
        "repeats_per_image": 10,
    },
    "models": {
        "dir": "trained_models",  # where your .pt models live (mode 1)
        "single_model": "trained_models/yolo11n-seg_epochs-50_imgsz-640.pt",  # used in mode 2
    },
    "predict": {
        "common": {
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.5,
            "classes": None,   # e.g., [0]
            "half": True,
            "device": "cpu",
            "stream": False,
            "verbose": False,
            "retina_masks": True,
        },
        "variants": [  # param sweep for mode 2
            {"conf": 0.25, "iou": 0.50},
            {"conf": 0.50, "iou": 0.50},
            {"conf": 0.25, "iou": 0.75},
        ],
    },
    "predictions": {
        "root": "predictions",
    },
}

# --------------------
# === UTILS ===
# --------------------

def list_images(folder: Path, image_exts: set[str]) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in image_exts]
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


def predict_one_image(model: YOLO, img_path: Path, predict_kwargs: Dict, save_overlay_to: Path, repeats: int) -> List[Dict]:
    """Run `repeats` predictions; save overlay for the first; return list of timing dicts per repeat."""
    benchmark = []
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    for rep in range(repeats):
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
        benchmark.append(row)

        if rep == 0:
            drawn = overlay_masks(img, r)
            cv2.imwrite(str(save_overlay_to), drawn)

    return benchmark


def save_benchmark(csv_path: Path, image_id: str, benchmark: List[Dict]):
    write_header_if_new(csv_path, ["image_id", "repeat_idx", "total_ms", "preprocess_ms", "inference_ms", "postprocess_ms"])
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        for i, t in enumerate(benchmark):
            w.writerow([image_id, i, t["total_ms"], t["preprocess_ms"], t["inference_ms"], t["postprocess_ms"]])



# --------------------
# === MODES ===
# --------------------

def eval_multiple_models_default(cfg: dict):
    images_dir = Path(cfg["data"]["images_dir"]) 
    image_exts = set([e.lower() for e in cfg["data"]["image_exts"]])
    repeats = int(cfg["data"]["repeats_per_image"])
    models_dir = Path(cfg["models"]["dir"]) 
    pred_root = Path(cfg["predictions"]["root"]).resolve()
    pred_root.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir, image_exts)
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}")

    model_files = sorted([p for p in models_dir.iterdir() if p.suffix == ".pt"])
    if not model_files:
        raise FileNotFoundError(f"No .pt models found under {models_dir}")

    for mpath in model_files:
        model_name = mpath.stem  # subfolder name under predictions
        out_dir = pred_root / model_name
        ensure_dir(out_dir)
        benchmark_csv = out_dir / "benchmark.csv"

        print(f"\n>>> Evaluating model: {mpath}")
        model = YOLO(str(mpath))

        for img_path in images:
            image_id = img_path.stem
            save_to = out_dir / f"{image_id}{img_path.suffix.lower()}"
            benchmark = predict_one_image(model, img_path, cfg["predict"]["common"], save_to, repeats)
            save_benchmark(benchmark_csv, image_id, benchmark)




def eval_single_model_param_sweep(cfg: dict):
    images_dir = Path(cfg["data"]["images_dir"]) 
    image_exts = set([e.lower() for e in cfg["data"]["image_exts"]])
    repeats = int(cfg["data"]["repeats_per_image"])
    pred_root = Path(cfg["predictions"]["root"]).resolve()
    pred_root.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir, image_exts)
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}")

    single_model_path = Path(cfg["models"]["single_model"]).resolve()
    if not single_model_path.exists():
        raise FileNotFoundError(f"Model not found: {single_model_path}")

    model = YOLO(str(single_model_path))
    base_name = single_model_path.stem

    for variant in cfg["predict"]["variants"]:
        conf = variant.get("conf", cfg["predict"]["common"]["conf"])  # default
        iou = variant.get("iou", cfg["predict"]["common"]["iou"])    # default
        subname = f"{base_name}_CONF-{conf}_IOU-{iou}"
        out_dir = pred_root / subname
        ensure_dir(out_dir)
        benchmark_csv = out_dir / "benchmark.csv"

        predict_kwargs = {**cfg["predict"]["common"], **variant}

        print(f"\n>>> Evaluating {single_model_path.name} with params: {variant}")
        for img_path in images:
            image_id = img_path.stem
            save_to = out_dir / f"{image_id}{img_path.suffix.lower()}"
            benchmark = predict_one_image(model, img_path, predict_kwargs, save_to, repeats)
            save_benchmark(benchmark_csv, image_id, benchmark)



if __name__ == "__main__":
    mode = (CONFIG.get("mode") or "").upper()
    if mode == "EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS":
        eval_multiple_models_default(CONFIG)
    elif mode == "EVAL_SINGLE_MODEL_PARAM_SWEEP":
        eval_single_model_param_sweep(CONFIG)
    else:
        raise ValueError(f"Unknown CONFIG['mode']: {CONFIG.get('mode')} (expected 'EVAL_MULTIPLE_MODELS_DEFAULT_PARAMS' or 'EVAL_SINGLE_MODEL_PARAM_SWEEP')")
