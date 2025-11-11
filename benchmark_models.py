"""
Simplified YOLO benchmark + evaluator (boxes OR masks).
- Two modes: MULTI_MODELS or SINGLE_MODEL_MULTI_CFG
- For each image, run N repeats; save ONE overlay (first repeat only)
- CSV per model/variant with: timing + TP/FP/FN + mean TP IoU (%) on first repeat

Assumptions (kept simple on purpose):
- Dataset root at CONFIG['images_dir'] with subfolders images_subdir and labels_subdir.
- Labels are YOLO format .txt files under the labels subfolder.
- Label filenames may be prefixed by a hash, but must contain the image stem (e.g. 0abc-frame_20.txt for frame_20.jpg).
- Detection labels:  cls cx cy w h (normalized)
- Segmentation labels: cls x1 y1 x2 y2 ... (normalized polygon)
- If eval_mode = 'auto', use masks when model returns masks; otherwise use boxes.

If something is missing or mismatched, just run and inspect—minimal guards by design.
"""

import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# =============================
# Config (minimal)
# =============================
CONFIG = {
    "mode": "MULTI_MODELS",  # "MULTI_MODELS" or "SINGLE_MODEL_MULTI_CFG"
    # Model configuration mirroring train_models.py structure
    "models_dir": "trained_models",
    "models": {
        # Trained checkpoints (filenames relative to models_dir)
        "multi_models": [
            "yolo11n-seg_epochs-50_imgsz-640.pt",
            "yolo11s-seg_epochs-50_imgsz-640.pt",
            "yolo11m-seg_epochs-50_imgsz-640.pt",
            "yolo11l-seg_epochs-50_imgsz-640.pt",
            "yolo11x-seg_epochs-50_imgsz-640.pt",
        ],
        # Single base model for SINGLE_MODEL_MULTI_CFG mode (relative filename)
        "single_model": "yolo11n-seg_epochs-50_imgsz-640.pt",
    },
    # Directory containing the trained model .pt files
    
    "predict_common": {
        "imgsz": 640,
        "conf": 0.4,
        "iou": 0.5,
        "classes": None,
        "half": False,
        "device": "cpu",
        "verbose": False,
        "retina_masks": True,
    },
    "variants": [
        {"conf": 0.4, "iou": 0.25},
        {"conf": 0.4, "iou": 0.50},
        {"conf": 0.4, "iou": 0.75},
    ],
    "eval": {
        "enabled": True,
        "iou_thr": 0.5,
        "match_by_class": False,   # set True to require class match
        "eval_mode": "masks",       # "boxes" | "masks" 
    },
    
    
    # Dataset root containing two subfolders (always assumed):
    #   <images_dir>/<images_subdir>
    #   <images_dir>/<labels_subdir>
    # Labels may have a hash prefix, so any .txt whose filename contains the
    # image stem will be considered its label file (e.g. 0abc123-frame_20.txt → frame_20.jpg).
    "images_dir": "benchmark_images/containers_test",
    "images_subdir": "images",
    "labels_subdir": "labels",
    "repeats": 5,
    
    "out_root": "predictions",
    
    "visual": {
        "alpha": 0.35,
        "thickness": 3,
    },

}

# =============================
# Small helpers
# =============================

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# =============================
# Labels (YOLO) → boxes/polygons
# =============================

def _parse_yolo_label_line(line: str, iw: int, ih: int) -> Tuple[int, np.ndarray, np.ndarray | None]:
    """Return (cls, xyxy, polygon) in pixels. polygon=None for detection labels.
    - Detection: cls cx cy w h (normalized)
    - Segmentation: cls x1 y1 x2 y2 ... (normalized), even length >= 6
    """
    parts = line.strip().split()
    cls_id = int(float(parts[0]))
    vals = [float(x) for x in parts[1:]]

    # Segmentation polygon (>= 3 points)
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs = vals[0::2]
        ys = vals[1::2]
        poly_px = np.stack([np.array(xs) * iw, np.array(ys) * ih], axis=1).astype(np.float32)
        x1, y1 = poly_px[:, 0].min(), poly_px[:, 1].min()
        x2, y2 = poly_px[:, 0].max(), poly_px[:, 1].max()
        return cls_id, np.array([x1, y1, x2, y2], dtype=np.float32), poly_px

    # Detection box
    cx, cy, w, h = vals[:4]
    bw, bh = w * iw, h * ih
    x1 = cx * iw - bw / 2.0
    y1 = cy * ih - bh / 2.0
    x2 = x1 + bw
    y2 = y1 + bh
    return cls_id, np.array([x1, y1, x2, y2], dtype=np.float32), None


def _labels_path_for(img_path: Path) -> Path:
    """Return path to label file for an image.
    Logic (always uses configured dataset root):
    1. images_dir/<images_subdir>/.../file.jpg → images_dir/<labels_subdir>/.../<hash>-file.txt
       We don't know the hash; search recursively under labels_subdir for any .txt whose stem contains the image stem.
    2. If multiple matches: pick the shortest filename (heuristic).
    3. If none found: fall back to expected plain path without hash (mirror relative path, same directories, stem.txt).
    """
    root = Path(CONFIG["images_dir"]).resolve()
    images_sub = str(CONFIG.get("images_subdir", "images"))
    labels_sub = str(CONFIG.get("labels_subdir", "labels"))
    images_base = (root / images_sub).resolve()
    labels_base = (root / labels_sub).resolve()
    img_resolved = img_path.resolve()
    try:
        rel = img_resolved.relative_to(images_base)
    except Exception:
        # Image not under images_base; fallback to simple heuristic
        return labels_base / f"{img_path.stem}.txt"
    stem = img_path.stem
    # Collect candidates whose stem contains the image stem
    candidates = []
    if labels_base.exists():
        for p in labels_base.rglob("*.txt"):
            if stem in p.stem:
                candidates.append(p)
    if candidates:
        # Choose shortest filename (heuristic: likely single hash + stem)
        candidates.sort(key=lambda p: (len(p.name), p.name))
        return candidates[0]
    # Fallback: mirror relative path (no hash)
    return labels_base / rel.with_suffix(".txt")


def load_gt(img_path: Path, img_shape: Tuple[int, int, int]) -> List[Dict]:
    h, w = img_shape[:2]
    label_path = _labels_path_for(img_path)
    if not label_path.exists():
        return []
    items = []
    with open(label_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            cls_id, xyxy, poly = _parse_yolo_label_line(line, w, h)
            items.append({"cls": int(cls_id), "xyxy": xyxy, "poly": poly})
    return items


# =============================
# Predictions → boxes/masks
# =============================

def pred_boxes(result) -> List[Dict]:
    if result.boxes is None or result.boxes.xyxy is None:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros((xyxy.shape[0],), int)
    return [{"cls": int(cls_ids[i]), "xyxy": xyxy[i]} for i in range(xyxy.shape[0])]


def pred_masks(result, H: int, W: int) -> List[Dict]:
    if result.masks is None or result.masks.data is None:
        return []
    m = result.masks.data.cpu().numpy().astype(np.float32)  # (N, Hm, Wm)
    cls_ids = result.boxes.cls.cpu().numpy().astype(int) if (result.boxes is not None and result.boxes.cls is not None) else np.zeros((m.shape[0],), int)
    out = []
    for i in range(m.shape[0]):
        mask = cv2.resize(m[i], (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
        out.append({"cls": int(cls_ids[i]), "mask": mask})
    return out


# =============================
# IoU (boxes or masks) + greedy matching
# =============================

def iou_boxes(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    ua = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    ub = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = ua + ub - inter
    return float(inter / union) if union > 0 else 0.0


def rect_mask(xyxy: np.ndarray, H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def poly_mask(poly: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    pts = poly.reshape(-1, 1, 2).astype(np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def iou_masks(ma: np.ndarray, mb: np.ndarray) -> float:
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union > 0 else 0.0


def greedy_match(gt_items: List[Dict], pr_items: List[Dict], iou_thr: float, match_by_class: bool, iou_fn) -> Tuple[List[Tuple[int,int,float]], List[int], List[int]]:
    if not gt_items or not pr_items:
        return [], list(range(len(gt_items))), list(range(len(pr_items)))
    pairs = []
    for gi, g in enumerate(gt_items):
        for pi, p in enumerate(pr_items):
            if match_by_class and g["cls"] != p["cls"]:
                continue
            i = iou_fn(g, p)
            if i >= iou_thr:
                pairs.append((i, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)
    matched_g, matched_p, matches = set(), set(), []
    for iou, gi, pi in pairs:
        if gi in matched_g or pi in matched_p:
            continue
        matched_g.add(gi); matched_p.add(pi)
        matches.append((gi, pi, iou))
    unmatched_g = [i for i in range(len(gt_items)) if i not in matched_g]
    unmatched_p = [i for i in range(len(pr_items)) if i not in matched_p]
    return matches, unmatched_g, unmatched_p


# =============================
# Drawing (simple, rectangles only for clarity)
# =============================

def draw_eval_overlay(img: np.ndarray,
                      gt_boxes: List[np.ndarray],
                      pr_boxes: List[np.ndarray],
                      matches: List[Tuple[int,int,float]],
                      unmatched_gt: List[int],
                      unmatched_pr: List[int],
                      thickness: int = 3) -> np.ndarray:
    out = img.copy()
    # TP: green on predicted boxes
    for gi, pi, _ in matches:
        x1, y1, x2, y2 = pr_boxes[pi].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 170, 0), thickness)
    # FP: red on predicted boxes not matched
    for pi in unmatched_pr:
        x1, y1, x2, y2 = pr_boxes[pi].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    # FN: yellow on GT boxes not matched
    for gi in unmatched_gt:
        x1, y1, x2, y2 = gt_boxes[gi].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), thickness)
    return out


def _blend_mask(base: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """Blend a single boolean mask onto base with given BGR color and alpha."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    overlay = np.zeros_like(base, dtype=base.dtype)
    overlay[mask] = color
    return cv2.addWeighted(base, 1.0, overlay, alpha, 0)


def draw_eval_overlay_masks(img: np.ndarray,
                            gt_masks: List[np.ndarray],
                            pr_masks: List[np.ndarray],
                            matches: List[Tuple[int,int,float]],
                            unmatched_gt: List[int],
                            unmatched_pr: List[int],
                            alpha: float = 0.35,
                            thickness: int = 3) -> np.ndarray:
    """Draw evaluation overlay for mask mode.
    - TP: blend predicted mask in green and draw contour.
    - FP: blend predicted mask in red and draw contour.
    - FN: blend GT mask in yellow and draw contour.
    """
    out = img.copy()

    # Helper to draw contours
    def draw_contours(dst: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]):
        if mask.dtype != np.uint8:
            m8 = mask.astype(np.uint8)
        else:
            m8 = mask
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(dst, cnts, -1, color, thickness)

    # Build quick lookups
    matched_pr_idx = {pi for _, pi, _ in matches}

    # Draw TP (green, darker) on predicted masks
    for gi, pi, _ in matches:
        pm = pr_masks[pi]
        out = _blend_mask(out, pm, (0, 170, 0), alpha)
        draw_contours(out, pm.astype(np.uint8), (0, 170, 0))

    # Draw FP (red) on predicted masks not matched
    for pi in unmatched_pr:
        pm = pr_masks[pi]
        out = _blend_mask(out, pm, (0, 0, 255), alpha)
        draw_contours(out, pm.astype(np.uint8), (0, 0, 255))

    # Draw FN (yellow) on GT masks not matched
    for gi in unmatched_gt:
        gm = gt_masks[gi]
        out = _blend_mask(out, gm, (0, 255, 255), alpha)
        draw_contours(out, gm.astype(np.uint8), (0, 255, 255))

    return out


def draw_counts_top_right(img: np.ndarray, tp: int, fp: int, fn: int,
                          font_scale: float = 0.8,
                          thickness: int = 2,
                          pad: int = 10) -> np.ndarray:
    """Draw TP/FP/FN counts at top-right of the image with a readable background.
    Colors: TP in green, FP in red, FN in yellow.
    """
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    seg_tp = f"TP: {tp}  "
    seg_fp = f"FP: {fp}  "
    seg_fn = f"FN: {fn}"

    (w_tp, h_tp), base_tp = cv2.getTextSize(seg_tp, font, font_scale, thickness)
    (w_fp, h_fp), base_fp = cv2.getTextSize(seg_fp, font, font_scale, thickness)
    (w_fn, h_fn), base_fn = cv2.getTextSize(seg_fn, font, font_scale, thickness)

    tw = w_tp + w_fp + w_fn
    th = max(h_tp, h_fp, h_fn)
    baseline = max(base_tp, base_fp, base_fn)

    x = out.shape[1] - tw - pad
    y = pad + th

    # Background rectangle
    x1, y1 = x - pad, y - th - pad
    x2, y2 = x + tw + pad, y + baseline + pad
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # Draw segments with colors
    # Colors in BGR
    col_tp = (0, 170, 0)
    col_fp = (0, 0, 255)
    col_fn = (0, 255, 255)

    x_cur = x
    cv2.putText(out, seg_tp, (x_cur, y), font, font_scale, col_tp, thickness, cv2.LINE_AA)
    x_cur += w_tp
    cv2.putText(out, seg_fp, (x_cur, y), font, font_scale, col_fp, thickness, cv2.LINE_AA)
    x_cur += w_fp
    cv2.putText(out, seg_fn, (x_cur, y), font, font_scale, col_fn, thickness, cv2.LINE_AA)

    return out


# =============================
# Core per-image routine
# =============================

def predict_one_image(model: YOLO,
                      img_path: Path,
                      predict_kwargs: Dict,
                      out_img_path: Path,
                      repeats: int,
                      eval_cfg: Dict,
                      vis_cfg: Dict) -> List[Dict]:
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    # Preload Ground Truth (GT) once
    gt_raw = load_gt(img_path, img.shape) if eval_cfg.get("enabled", True) else []

    rows = []
    mean_tp_iou_pct_first = ""
    std_tp_iou_pct_first = ""

    for rep in range(repeats):
        t0 = time.perf_counter()
        r = model.predict(source=img, **predict_kwargs)[0]
        t1 = time.perf_counter()

        speed = getattr(r, "speed", {})
        row = {
            "total_ms": round((t1 - t0) * 1000.0, 3),
            "preprocess_ms": round(float(speed.get("preprocess", float("nan"))), 3),
            "inference_ms": round(float(speed.get("inference", float("nan"))), 3),
            "postprocess_ms": round(float(speed.get("postprocess", float("nan"))), 3),
            "tp": "",
            "fp": "",
            "fn": "",
            "mean_tp_iou_pct": "",
            "std_tp_iou_pct": "",
        }

        if rep == 0 and eval_cfg.get("enabled", True):
            mode = eval_cfg.get("eval_mode")

            # --- prepare GT in requested mode
            if mode == "masks":
                gt_items = []
                for g in gt_raw:
                    if g["poly"] is not None:
                        gm = poly_mask(g["poly"], H, W)
                    else:
                        gm = rect_mask(g["xyxy"], H, W)
                    gt_items.append({"cls": g["cls"], "mask": gm, "xyxy": g["xyxy"]})
            elif mode == "boxes":
                gt_items = [{"cls": g["cls"], "xyxy": g["xyxy"]} for g in gt_raw]
            else:
                raise ValueError(f"Unsupported eval_mode: {mode}")

            # --- prepare predictions in requested mode
            if mode == "masks":
                preds = pred_masks(r, H, W)
                # Matching IoU over masks
                def iou_fn(g, p):
                    return iou_masks(g["mask"], p["mask"]) if ("mask" in p and "mask" in g) else 0.0
            else:
                preds = pred_boxes(r)
                def iou_fn(g, p):
                    return iou_boxes(g["xyxy"], p["xyxy"])            

            matches, ug, up = greedy_match(
                gt_items,
                preds,
                float(eval_cfg.get("iou_thr", 0.5)),
                bool(eval_cfg.get("match_by_class", False)),
                iou_fn,
            )

            tp = len(matches); fn = len(ug); fp = len(up)
            row["tp"], row["fp"], row["fn"] = tp, fp, fn

            # Mean TP IoU in percent
            if tp > 0:
                ious = np.array([m[2] for m in matches], dtype=np.float32)
                mean_iou = float(np.mean(ious) * 100.0)
                std_iou = float(np.std(ious) * 100.0)
                mean_tp_iou_pct_first = round(float(mean_iou), 2)
                std_tp_iou_pct_first = round(float(std_iou), 2)
                row["mean_tp_iou_pct"] = mean_tp_iou_pct_first
                row["std_tp_iou_pct"] = std_tp_iou_pct_first

            # Draw evaluation overlay
            if mode == "boxes":
                gt_boxes = [g["xyxy"] for g in gt_raw]
                pr_boxes = [p["xyxy"] for p in preds]
                overlay = draw_eval_overlay(
                    img,
                    gt_boxes,
                    pr_boxes,
                    matches,
                    ug,
                    up,
                    thickness=int(vis_cfg.get("thickness", 3))
                )
            else:
                # masks mode: draw translucent masks
                gt_masks = []
                for g in gt_items:
                    if "mask" in g and g["mask"] is not None:
                        gm = g["mask"]
                    else:
                        gm = rect_mask(g["xyxy"], H, W)
                    gt_masks.append(gm.astype(np.uint8))
                pr_masks = []
                for p in preds:
                    if p.get("mask") is not None:
                        pm = p["mask"].astype(np.uint8)
                    else:
                        pm = np.zeros((H, W), dtype=np.uint8)
                    pr_masks.append(pm)
                overlay = draw_eval_overlay_masks(
                    img,
                    gt_masks,
                    pr_masks,
                    matches,
                    ug,
                    up,
                    alpha=float(vis_cfg.get("alpha", 0.35)),
                    thickness=int(vis_cfg.get("thickness", 3))
                )
            # Add counts top-right
            overlay = draw_counts_top_right(
                overlay,
                tp=len(matches),
                fp=len(up),
                fn=len(ug),
                font_scale=0.75,
                thickness=2,
                pad=8
            )
            cv2.imwrite(str(out_img_path), overlay)
        elif rep == 0:
            # Save plain predictions overlay (no evaluation)
            overlay = img.copy()
            # Prefer masks if available
            if r.masks is not None and r.masks.data is not None:
                preds = pred_masks(r, H, W)
                for p in preds:
                    if p.get("mask") is not None:
                        overlay = _blend_mask(overlay, p["mask"], (255, 0, 0), float(vis_cfg.get("alpha", 0.35)))
            else:
                # draw boxes if any
                for b in pred_boxes(r):
                    x1, y1, x2, y2 = b["xyxy"].astype(int)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), int(vis_cfg.get("thickness", 3)))
            cv2.imwrite(str(out_img_path), overlay)

        rows.append(row)

    return rows


def save_csv(csv_path: Path, image_id: str, rows: List[Dict]):
    """Aggregate repeats into a single line per image.
    - Timing columns become the mean over repeats.
    - TP/FP/FN/IoU stats are taken from the first repeat (evaluation only done there).
    - Output is tab-separated.
    """
    if not rows:
        return

    # Collect timing means (ignore non-numeric / empty values)
    timing_keys = ["total_ms", "preprocess_ms", "inference_ms", "postprocess_ms"]
    means = {}
    for k in timing_keys:
        vals = []
        for r in rows:
            v = r.get(k, "")
            if isinstance(v, (int, float)) and not np.isnan(v):
                vals.append(float(v))
        means[k] = round(float(np.mean(vals)) if vals else float("nan"), 3)

    first = rows[0]
    tp = first.get("tp", "")
    fp = first.get("fp", "")
    fn = first.get("fn", "")
    mean_iou = first.get("mean_tp_iou_pct", "")
    std_iou = first.get("std_tp_iou_pct", "")

    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f, delimiter='\t')
        if new:
            w.writerow([
                "image_id",
                "total_ms",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "tp",
                "fp",
                "fn",
                "mean_tp_iou_pct",
                "std_tp_iou_pct",
            ])
        w.writerow([
            image_id,
            means["total_ms"],
            means["preprocess_ms"],
            means["inference_ms"],
            means["postprocess_ms"],
            tp,
            fp,
            fn,
            mean_iou,
            std_iou,
        ])


# =============================
# Modes
# =============================

def run_multi_models(cfg: dict):
    # Always use dataset root structure
    images_base = Path(cfg["images_dir"]) / cfg.get("images_subdir", "images")
    images = list_images(images_base)
    out_root = Path(cfg["out_root"]) ; ensure_dir(out_root)
    models_dir = Path(cfg["models_dir"]).resolve()
    model_list = cfg.get("models", {}).get("multi_models")
    if model_list:
        model_paths = [models_dir / fn for fn in model_list]
    else:
        # fallback: all .pt files
        model_paths = sorted([p for p in models_dir.iterdir() if p.suffix == ".pt"])

    for mpath in model_paths:
        model = YOLO(str(mpath))
        sub = out_root / mpath.stem
        ensure_dir(sub)
        csv_path = sub / "benchmark.csv"
        for img_path in images:
            # Use path relative to the images base (avoid including the "images" prefix)
            img_id = img_path.relative_to(images_base).with_suffix("").as_posix().replace("/", "__")
            out_img = sub / f"{img_id}{img_path.suffix.lower()}"
            rows = predict_one_image(model, img_path, cfg["predict_common"], out_img, cfg["repeats"], cfg["eval"], cfg["visual"])
            save_csv(csv_path, img_id, rows)


def run_single_model_multi_cfg(cfg: dict):
    # Always use dataset root structure
    images_base = Path(cfg["images_dir"]) / cfg.get("images_subdir", "images")
    images = list_images(images_base)
    out_root = Path(cfg["out_root"]) ; ensure_dir(out_root)

    single_rel = cfg.get("models", {}).get("single_model") or ""
    model_path = (Path(cfg["models_dir"]).resolve() / single_rel).resolve()
    model = YOLO(str(model_path))
    base = model_path.stem

    for variant in cfg["variants"]:
        pred_args = {**cfg["predict_common"], **variant}
        sub = out_root / f"{base}_CONF-{pred_args['conf']}_IOU-{pred_args['iou']}"
        ensure_dir(sub)
        csv_path = sub / "benchmark.csv"
        print(f"Evaluating {model_path.name} with {variant}")
        for img_path in images:
            # Use path relative to the images base (avoid including the "images" prefix)
            img_id = img_path.relative_to(images_base).with_suffix("").as_posix().replace("/", "__")
            out_img = sub / f"{img_id}{img_path.suffix.lower()}"
            rows = predict_one_image(model, img_path, pred_args, out_img, cfg["repeats"], cfg["eval"], cfg["visual"])
            save_csv(csv_path, img_id, rows)


if __name__ == "__main__":
    mode = str(CONFIG.get("mode", "")).upper()
    if mode == "MULTI_MODELS":
        run_multi_models(CONFIG)
    elif mode == "SINGLE_MODEL_MULTI_CFG":
        run_single_model_multi_cfg(CONFIG)
    else:
        raise ValueError(f"Unknown mode: {CONFIG.get('mode')}")
