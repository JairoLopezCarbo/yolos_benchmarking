"""
Interactive grader: shows the same image across model folders side-by-side and
prompts you to score each model 1–10. Results saved to a CSV in predictions/.

Refactored to use a single CONFIG dict at the top for readability.
"""
from pathlib import Path
import csv
import cv2
import numpy as np
import sys
import select
import time
import random


# =============================
# Centralized configuration
# =============================
CONFIG = {
    "predictions": {
        "root": "predictions",
        # Model folder names under predictions/ containing per-model images and benchmark.csv
        "models": [
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.25_IOU-0.25",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.25_IOU-0.5",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.25_IOU-0.75",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.25",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.5",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.75",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.55_IOU-0.25",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.55_IOU-0.5",
            "yolo11n-seg_epochs-50_imgsz-640_CONF-0.55_IOU-0.75",
        ],
    },
    # Output CSV filename (relative to predictions root)
    "output_csv": "graded_predictions.csv",
    "display": {
        # Shuffle display order each image to anonymize models
        "shuffle": False,
        # Overlay numeric labels (1..N) on each tile
        "numeric_labels": True,
        "native_scale": False,
    },
    "scoring": {
        "min": 1,
        "max": 10,
        # Guidance text shown when asking for a score
        "guidelines": [
            "[1-4] -> Missing labels",
            "[5-7] -> Misplaced labels",
            "[8-10] -> Correct labels",
        ],
    },
}

# --- helper: choose best interpolation for the direction of scaling ---
def _best_interp(scale: float) -> int:
    if scale < 1.0:
        return cv2.INTER_AREA       # best for downscale
    elif scale <= 2.0:
        return cv2.INTER_CUBIC      # good for mild upscales
    else:
        return cv2.INTER_LANCZOS4   # best for large upscales


def compute_grid(n: int, win_w: int, win_h: int) -> tuple[int, int]:
    """Compute an approximately square grid (cols, rows) adapted to window aspect."""
    if n <= 0:
        return (0, 0)
    if win_w <= 0 or win_h <= 0:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return cols, rows
    cols = int(np.ceil(np.sqrt(n * (win_w / max(1, win_h)))))
    cols = int(np.clip(cols, 1, n))
    rows = int(np.ceil(n / cols))
    return cols, rows

def compose_grid(images: list[np.ndarray], labels: list[str], win_w: int, win_h: int, native_scale: bool = False) -> np.ndarray:
    """
    If native_scale=True, images are placed at their original resolution and padded into cells.
    No cv2.resize is applied to the tiles. This preserves full detail; panning/fit is handled by the window.
    """
    n = len(images)
    if n == 0:
        return np.zeros((max(1, win_h), max(1, win_w), 3), dtype=np.uint8)

    cols, rows = compute_grid(n, win_w, win_h)

    if native_scale:
        # Cell size = global max image size -> pad only, never resize
        max_w = max(img.shape[1] for img in images)
        max_h = max(img.shape[0] for img in images)
        cell_w, cell_h = max_w, max_h
        canvas = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(cell_w, cell_h) / 400.0)
        thickness = 1

        for idx, img in enumerate(images):
            r = idx // cols
            c = idx % cols
            y0 = r * cell_h
            x0 = c * cell_w
            ih, iw = img.shape[:2]
            # center without scaling
            x_off = x0 + (cell_w - iw) // 2
            y_off = y0 + (cell_h - ih) // 2
            canvas[y_off:y_off + ih, x_off:x_off + iw] = img

            if labels and idx < len(labels) and labels[idx]:
                text = str(labels[idx])
                org = (x_off + 6, y_off + 20)
                cv2.putText(canvas, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(canvas, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Do NOT crop to (win_h, win_w) in native mode; keep full-res canvas
        return canvas

    # --- original "fit-to-window" path, but with higher-quality upscaling ---
    cell_w = max(1, win_w // max(1, cols))
    cell_h = max(1, win_h // max(1, rows))
    canvas = np.zeros((max(1, rows * cell_h), max(1, cols * cell_w), 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(cell_w, cell_h) / 400.0)
    thickness = 1

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0 = r * cell_h
        x0 = c * cell_w

        if img is None or img.size == 0:
            continue

        ih, iw = img.shape[:2]
        scale = min(cell_w / max(1, iw), cell_h / max(1, ih))
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        try:
            resized = cv2.resize(img, (new_w, new_h), interpolation=_best_interp(scale))
        except Exception:
            continue
        x_off = x0 + (cell_w - new_w) // 2
        y_off = y0 + (cell_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        if labels and idx < len(labels) and labels[idx]:
            text = str(labels[idx])
            org = (x_off + 6, y_off + 20)
            cv2.putText(canvas, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # keep exact window size
    return canvas[:win_h, :win_w]

ROOT_DIR = Path(__file__).resolve().parent
PREDICTIONS_ROOT = ROOT_DIR / CONFIG["predictions"]["root"]
MODEL_FOLDERS = list(CONFIG["predictions"]["models"])
OUTPUT_CSV = ROOT_DIR / CONFIG["output_csv"]


# Precompute timing summaries (avg and std) per model so we can save them
def summarize_benchmark(benchmark_path: Path) -> dict:
    """Read a benchmark.csv and return summary stats for columns of interest.

    Returns dict with keys: total_ms_avg, total_ms_std, preprocess_ms_avg,
    preprocess_ms_std, inference_ms_avg, inference_ms_std, postprocess_ms_avg,
    postprocess_ms_std. Missing file yields NaNs.
    """
    import math
    stats = {
        "total_ms_avg": math.nan,
        "total_ms_std": math.nan,
        "preprocess_ms_avg": math.nan,
        "preprocess_ms_std": math.nan,
        "inference_ms_avg": math.nan,
        "inference_ms_std": math.nan,
        "postprocess_ms_avg": math.nan,
        "postprocess_ms_std": math.nan,
    }
    if not benchmark_path.exists():
        return stats

    vals = {"total_ms": [], "preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    try:
        with open(benchmark_path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                for k in vals.keys():
                    v = row.get(k)
                    if v is None:
                        continue
                    try:
                        vals[k].append(float(v))
                    except Exception:
                        pass
    except Exception:
        return stats

    import statistics
    for k, arr in vals.items():
        if not arr:
            continue
        avg = statistics.mean(arr)
        # use population stdev if only one sample -> 0.0
        std = statistics.pstdev(arr) if len(arr) >= 1 else 0.0
        if k == 'total_ms':
            stats['total_ms_avg'] = avg
            stats['total_ms_std'] = std
        elif k == 'preprocess_ms':
            stats['preprocess_ms_avg'] = avg
            stats['preprocess_ms_std'] = std
        elif k == 'inference_ms':
            stats['inference_ms_avg'] = avg
            stats['inference_ms_std'] = std
        elif k == 'postprocess_ms':
            stats['postprocess_ms_avg'] = avg
            stats['postprocess_ms_std'] = std

    return stats


# compute timing summaries for every model folder
TIMING_SUMMARIES = {}
for m in MODEL_FOLDERS:
    TIMING_SUMMARIES[m] = summarize_benchmark(PREDICTIONS_ROOT / m / 'benchmark.csv')

# Collect image IDs that exist in ALL model folders
common = None
image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
for m in MODEL_FOLDERS:
    imgs = sorted([p.name for p in (PREDICTIONS_ROOT / m).iterdir() if p.suffix.lower() in image_exts])
    ids = [Path(x).stem for x in imgs]
    common = set(ids) if common is None else common.intersection(ids)

common = sorted(common) if common else []
print(f"Found {len(common)} common images across models: {MODEL_FOLDERS}")

# Prepare CSV header: first column is a row label (stat name or image id),
# remaining columns are models. First rows are timing stats; then one row per image id.
if not OUTPUT_CSV.exists():
    header = ["row", *MODEL_FOLDERS]
    stats_spec = [
        ("total_ms_avg", "total_avg"),
        ("total_ms_std", "total_std"),
        ("preprocess_ms_avg", "preprocess_avg"),
        ("preprocess_ms_std", "preprocess_std"),
        ("inference_ms_avg", "inference_avg"),
        ("inference_ms_std", "inference_std"),
        ("postprocess_ms_avg", "postprocess_avg"),
        ("postprocess_ms_std", "postprocess_std"),
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        # write one row per statistic, columns aligned with MODEL_FOLDERS
        for key, label in stats_spec:
            row = [label]
            for m in MODEL_FOLDERS:
                val = TIMING_SUMMARIES.get(m, {}).get(key)
                if val is None:
                    row.append("")
                else:
                    row.append(val)
            w.writerow(row)

for img_id in common:
    # Load each model's visualization
    imgs = []
    for m in MODEL_FOLDERS:
        # Find an image file starting with this id (try common extensions)
        folder = PREDICTIONS_ROOT / m
        cand = None
        for ext in ['.jpg','.jpeg','.png','.bmp']:
            p = folder / f"{img_id}{ext}"
            if p.exists():
                cand = p
                break
        if cand is None:
            print(f"Missing image {img_id} under {m}; skipping.")
            imgs = []
            break
        img = cv2.imread(str(cand))
        if img is None:
            print(f"Failed to read {cand}")
            imgs = []
            break
        imgs.append(img)

    if not imgs:
        continue

    # Determine a reasonable initial window size based on number of images
    n_imgs = len(imgs)
    # Shuffle display order to anonymize models if enabled; create numeric labels 1..n
    order = list(range(n_imgs))
    if CONFIG["display"].get("shuffle", True):
        random.shuffle(order)
    shuffled_imgs = [imgs[i] for i in order]
    numeric_labels = [str(i + 1) for i in range(n_imgs)] if CONFIG["display"].get("numeric_labels", True) else [""] * n_imgs
    base_cell_w = 480
    base_cell_h = 360
    cols0 = int(np.ceil(np.sqrt(n_imgs)))
    rows0 = int(np.ceil(n_imgs / cols0))
    min_w = 800
    min_h = 600
    max_w = 1600
    max_h = 1000
    start_w = min(max_w, max(min_w, cols0 * base_cell_w))
    start_h = min(max_h, max(min_h, rows0 * base_cell_h))

    # Create resizable window and show initial grid
    win_name = f"{img_id} :: {' | '.join(MODEL_FOLDERS)}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(win_name, start_w, start_h)
    except Exception:
        pass
    initial_grid = compose_grid(shuffled_imgs, numeric_labels, start_w, start_h)
    cv2.imshow(win_name, initial_grid)
    # Allow the window to refresh
    cv2.waitKey(1)
    guidelines = "\n".join(CONFIG["scoring"]["guidelines"]) if CONFIG["scoring"].get("guidelines") else ""
    print(
        f"Please enter a number from {CONFIG['scoring']['min']} to {CONFIG['scoring']['max']}:\n" +
        (guidelines + "\n" if guidelines else "") +
        f"Scoring image: {img_id}"
    )

    # Dynamic resizing loop + non-blocking stdin read so window stays responsive
    position_scores = []
    # track last window size so we only rescale when it changes
    try:
        _x, _y, last_w, last_h = cv2.getWindowImageRect(win_name)
    except Exception:
        last_w, last_h = start_w, start_h

    for pos in range(n_imgs):
        prompt = f"Score for image {pos + 1} ({CONFIG['scoring']['min']}–{CONFIG['scoring']['max']}): "
        print(prompt, end='', flush=True)
        last_render_ts = 0.0
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('1'):  # 1 = native (full-res, no resize)
                native_scale = True
            else:  # 0 = fit-to-window
                native_scale = False

            try:
                _x, _y, w, h = cv2.getWindowImageRect(win_name)
            except Exception:
                w, h = last_w, last_h

            now = time.time()
            if (w, h) != (last_w, last_h) or (now - last_render_ts) > 0.3:
                disp = compose_grid(shuffled_imgs, numeric_labels, max(1, w), max(1, h), native_scale=native_scale)
                cv2.imshow(win_name, disp)
                last_w, last_h = w, h
                last_render_ts = now

            # stdin handling unchanged ...
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                line = sys.stdin.readline().strip()
                try:
                    s = float(line)
                    if CONFIG['scoring']['min'] <= s <= CONFIG['scoring']['max']:
                        position_scores.append(s)
                        break
                except Exception:
                    pass
                print(f"Please enter a number from {CONFIG['scoring']['min']} to {CONFIG['scoring']['max']}:\n" +
                    ("\n".join(CONFIG["scoring"]["guidelines"]) if CONFIG["scoring"].get("guidelines") else ""),
                    flush=True)


    # Map position-based scores back to original model order and write a
    # single row with one score per model (columns align with MODEL_FOLDERS).
    scores_by_model = [''] * n_imgs
    for pos, s in enumerate(position_scores):
        model_idx = order[pos]
        scores_by_model[model_idx] = s

    with open(OUTPUT_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([img_id, *scores_by_model])

    # Close the image window between items
    cv2.destroyAllWindows()

print("\nDone. Scores saved to:", OUTPUT_CSV)