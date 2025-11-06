# ================================
# File: benchmark_models.py
# ================================
"""
Benchmark trained .pt models using a single CONFIG dict (readable, editable).

Modes:
    - MULTI_MODELS: evaluate all .pt in CONFIG['models']['dir'] with common predict params.
    - SINGLE_MODEL_MULTI_CFG: evaluate one .pt over a grid of predict params (conf/iou), saving under
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
    "mode": "SINGLE_MODEL_MULTI_CFG",  # or "MULTI_MODELS"
    "data": {
        "images_dir": "benchmark_images",
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
            {"conf": 0.4, "iou": 0.25},
            {"conf": 0.4, "iou": 0.50},
            {"conf": 0.4, "iou": 0.75},
        ],
    },
    "predictions": {
        "root": "predictions",
    },
    "visual": {
        # mode: 'masks' to color segmentation masks if available, otherwise fallback to boxes;
        #       'boxes' to color bounding boxes/regions regardless of masks
        "mode": "masks",  # "masks" | "boxes"
        # color_mode: 'random' assigns a random color per object instance;
        #             'by_class' assigns a deterministic color per class id and shows legend (if enabled)
        "color_mode": "random",  # "random" | "by_class"
        # visual styles (unified)
        "alpha": 0.3,            # opacity for overlays (masks and boxes)
        "thickness": 5,           # outline thickness for both masks and boxes
        # legend settings (used only with color_mode='by_class')
        "show_legend": False,
        "legend_loc": "top-left",  # "top-left" | "top-right" | "bottom-left" | "bottom-right"
    },
}

# Default image extensions (used if CONFIG no longer provides image_exts)
DEFAULT_IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}

# --------------------
# === UTILS ===
# --------------------

def list_images(folder: Path, image_exts: set[str]) -> List[Path]:
    paths = [p for p in folder.iterdir() if p.suffix.lower() in image_exts]
    paths.sort()
    return paths


def rand_color() -> Tuple[int, int, int]:
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def class_color(cls_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR color for a class id (stable across runs)."""
    rnd = random.Random(int(cls_id) * 10007 + 42)
    return (rnd.randint(40, 215), rnd.randint(40, 215), rnd.randint(40, 215))


def _draw_legend(canvas: np.ndarray, entries: list[Tuple[str, Tuple[int,int,int]]], loc: str = "top-left") -> None:
    """Draw a simple legend: colored squares + class names on a semi-transparent box."""
    if not entries:
        return
    h, w = canvas.shape[:2]
    pad = 8
    swatch = 14
    line_h = max(18, swatch + 6)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    # compute legend box size
    text_widths = []
    for name, _ in entries:
        (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        text_widths.append(tw)
    box_w = pad*3 + swatch + (max(text_widths) if text_widths else 0)
    box_h = pad*2 + line_h * len(entries)
    # position
    if loc == "top-left":
        x0, y0 = pad, pad
    elif loc == "top-right":
        x0, y0 = max(0, w - box_w - pad), pad
    elif loc == "bottom-left":
        x0, y0 = pad, max(0, h - box_h - pad)
    else:  # bottom-right
        x0, y0 = max(0, w - box_w - pad), max(0, h - box_h - pad)
    x1, y1 = x0 + box_w, y0 + box_h
    # draw semi-transparent background
    roi = canvas[y0:y1, x0:x1].copy()
    bg = roi.copy()
    cv2.rectangle(bg, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(bg, 0.5, roi, 0.5, 0, dst=roi)
    canvas[y0:y1, x0:x1] = roi
    # draw entries
    cy = y0 + pad + line_h//2
    for name, color in entries:
        # swatch
        sx0 = x0 + pad
        sy0 = cy - swatch//2
        cv2.rectangle(canvas, (sx0, sy0), (sx0 + swatch, sy0 + swatch), color, -1)
        # text
        tx = sx0 + swatch + pad
        ty = cy + swatch//3
        cv2.putText(canvas, name, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cy += line_h


def overlay_visuals(img_bgr: np.ndarray, result, visual_cfg: Dict) -> np.ndarray:
    """Overlay masks or boxes with coloring rules and optional legend."""
    out = img_bgr.copy()
    h, w = out.shape[:2]
    mode = (visual_cfg.get("mode") or "masks").lower()
    color_mode = (visual_cfg.get("color_mode") or "by_class").lower()
    show_legend = bool(visual_cfg.get("show_legend", True)) and color_mode == "by_class"
    legend_loc = (visual_cfg.get("legend_loc") or "top-left").lower()

    # Extract detections
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    class_ids = []
    if boxes is not None and getattr(boxes, 'cls', None) is not None:
        try:
            class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        except Exception:
            try:
                class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
            except Exception:
                class_ids = []
    # Prepare colors
    instance_colors: list[Tuple[int,int,int]] = []
    if color_mode == "random":
        n = masks.data.shape[0] if (masks is not None and masks.data is not None) else (boxes.xyxy.shape[0] if boxes is not None and getattr(boxes, 'xyxy', None) is not None else 0)
        instance_colors = [rand_color() for _ in range(n)]
    else:  # by_class
        instance_colors = [class_color(cid if cid is not None else 0) for cid in class_ids]

    drew_any = False
    # Draw according to mode
    if mode == "masks" and masks is not None and masks.data is not None:
        m = masks.data.detach().cpu().numpy().astype(np.float32)  # (N, Hm, Wm)
        overlay = np.zeros_like(out, dtype=np.uint8)
        alpha = float(visual_cfg.get("alpha", 0.45))
        thick = int(visual_cfg.get("thickness", 2))
        outlines: list[Tuple[list[np.ndarray], Tuple[int,int,int]]] = []  # (contours, color)
        for i in range(m.shape[0]):
            mask_small = m[i]
            mask_resized = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask_resized > 0.5
            color = instance_colors[i] if i < len(instance_colors) else rand_color()
            overlay[mask] = color
            # draw contour for the mask using the same color and thickness
            try:
                mask_u8 = (mask.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # store for drawing on top (post-blend) to keep outlines fully opaque
                outlines.append((contours, color))
            except Exception:
                pass
        # First blend the filled mask overlays
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        # Then draw contours on top with full opacity for visibility
        for contours, color in outlines:
            try:
                cv2.drawContours(out, contours, -1, color, thick)
            except Exception:
                pass
        drew_any = m.shape[0] > 0
    else:
        # Boxes mode (or fallback if no masks)
        if boxes is not None and getattr(boxes, 'xyxy', None) is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            alpha = float(visual_cfg.get("alpha", 0.45))
            thick = int(visual_cfg.get("thickness", 2))
            overlay = out.copy()
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                color = instance_colors[i] if i < len(instance_colors) else rand_color()
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                # filled rectangle area goes to overlay (alpha blended)
                cv2.rectangle(overlay, pt1, pt2, color, -1)
            # blend filled areas first
            out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
            # draw borders on top with full opacity for visibility
            for i, (x1, y1, x2, y2) in enumerate(xyxy):
                color = instance_colors[i] if i < len(instance_colors) else rand_color()
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(out, pt1, pt2, color, thick)
            drew_any = xyxy.shape[0] > 0

    # Legend for class colors
    if show_legend and drew_any and class_ids:
        # Collect unique classes present in this result
        try:
            names_map = getattr(result, 'names', None)
        except Exception:
            names_map = None
        entries = []
        for cid in sorted(set(class_ids)):
            cname = str(cid)
            if isinstance(names_map, (list, tuple)) and cid < len(names_map):
                cname = str(names_map[cid])
            elif isinstance(names_map, dict):
                cname = str(names_map.get(cid, cname))
            entries.append((cname, class_color(cid)))
        _draw_legend(out, entries, loc=legend_loc)

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
            drawn = overlay_visuals(img, r, CONFIG["visual"])
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

def eval_multi_models(cfg: dict):
    images_dir = Path(cfg["data"]["images_dir"]) 
    # Use default set of image extensions (CONFIG no longer contains image_exts)
    image_exts = {e.lower() for e in DEFAULT_IMAGE_EXTS}
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




def single_model_multi_cfg(cfg: dict):
    images_dir = Path(cfg["data"]["images_dir"]) 
    # Use default set of image extensions (CONFIG no longer contains image_exts)
    image_exts = {e.lower() for e in DEFAULT_IMAGE_EXTS}
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
    if mode == "MULTI_MODELS":
        eval_multi_models(CONFIG)
    elif mode == "SINGLE_MODEL_MULTI_CFG":
        single_model_multi_cfg(CONFIG)
    else:
        raise ValueError(f"Unknown CONFIG['mode']: {CONFIG.get('mode')} (expected 'MULTI_MODELS' or 'SINGLE_MODEL_  MULTIPLE_CFG')")
