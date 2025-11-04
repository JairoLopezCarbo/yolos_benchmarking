"""
Interactive grader: shows the same image across model folders side-by-side and
prompts you to score each model 1–10. Results saved to human_scores.csv (wide format).
"""
from pathlib import Path
import csv
import cv2
import numpy as np
import sys
import select
import time
import random


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


def compose_grid(images: list[np.ndarray], labels: list[str], win_w: int, win_h: int) -> np.ndarray:
    """Compose a grid image preserving aspect ratio; draws a small label string
    at the top-left corner of each placed image when provided.

    - images: list of BGR images (np.uint8)
    - labels: list of strings per image (e.g., '1', '2', '3'); may be empty
    - win_w, win_h: current window dimensions
    Returns: BGR canvas of size (win_h, win_w, 3)
    """
    n = len(images)
    if n == 0:
        return np.zeros((max(1, win_h), max(1, win_w), 3), dtype=np.uint8)

    cols, rows = compute_grid(n, win_w, win_h)
    # Compute cell size (no margins). Images are aspect-preserving (letterboxed)
    cell_w = max(1, win_w // max(1, cols))
    cell_h = max(1, win_h // max(1, rows))

    canvas = np.zeros((max(1, rows * cell_h), max(1, cols * cell_w), 3), dtype=np.uint8)
    # Slightly dark background for any unused area
    canvas[:] = (20, 20, 20)
    # Text settings for overlaying small numbers
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

        # Resize image to fit inside the cell preserving aspect (letterbox)
        ih, iw = img.shape[:2]
        scale = min(cell_w / max(1, iw), cell_h / max(1, ih))
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        try:
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        except Exception:
            continue
        x_off = x0 + (cell_w - new_w) // 2
        y_off = y0 + (cell_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        # Draw label (number) at image top-left if provided
        if labels and idx < len(labels) and labels[idx]:
            text = str(labels[idx])
            org = (x_off + 6, y_off + 20)
            # simple outline for readability
            cv2.putText(canvas, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Ensure exact window size (crop if needed)
    canvas = canvas[:win_h, :win_w]
    return canvas

PREDICTIONS_ROOT = Path(__file__).resolve().parent
MODEL_FOLDERS = ["yolo11n-seg_epochs-50_imgsz-640","yolo11s-seg_epochs-50_imgsz-640"]
OUTPUT_CSV = PREDICTIONS_ROOT / "human_scores.csv"


# Precompute timing summaries (avg and std) per model so we can save them
def summarize_timings(timings_path: Path) -> dict:
    """Read a timings.csv and return summary stats for columns of interest.

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
    if not timings_path.exists():
        return stats

    vals = {"total_ms": [], "preprocess_ms": [], "inference_ms": [], "postprocess_ms": []}
    try:
        with open(timings_path, newline='') as f:
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
    TIMING_SUMMARIES[m] = summarize_timings(PREDICTIONS_ROOT / m / 'timings.csv')

# Collect image IDs that exist in ALL model folders
common = None
for m in MODEL_FOLDERS:
    imgs = sorted([p.name for p in (PREDICTIONS_ROOT / m).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])
    ids = [Path(x).stem for x in imgs]
    common = set(ids) if common is None else common.intersection(ids)

common = sorted(common) if common else []
print(f"Found {len(common)} common images across models: {MODEL_FOLDERS}")

# Prepare CSV header in new layout: columns = models. First rows will be
# timing summary statistics (total_avg, total_std, preprocess_avg, preprocess_std,
# inference_avg, inference_std, postprocess_avg, postprocess_std). After those
# stat rows, each subsequent row is a frame: one score per model (columns).
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
    # Shuffle display order to anonymize models; create numeric labels 1..n
    order = list(range(n_imgs))
    random.shuffle(order)
    shuffled_imgs = [imgs[i] for i in order]
    numeric_labels = [str(i + 1) for i in range(n_imgs)]
    base_cell_w, base_cell_h = 480, 360
    cols0 = int(np.ceil(np.sqrt(n_imgs)))
    rows0 = int(np.ceil(n_imgs / cols0))
    start_w = min(1600, max(800, cols0 * base_cell_w))
    start_h = min(1000, max(600, rows0 * base_cell_h))

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
    print("Please enter a number from 1 to 10:\n"+
                      "\t[1-4] -> Missing labels\n" +
                      "\t[5-7] -> Misplaced labels\n" +
                      "\t[8-10] -> Correct labels\n" +
        f"Scoring image: {img_id}")

    # Dynamic resizing loop + non-blocking stdin read so window stays responsive
    position_scores = []
    # track last window size so we only rescale when it changes
    try:
        _x, _y, last_w, last_h = cv2.getWindowImageRect(win_name)
    except Exception:
        last_w, last_h = start_w, start_h

    for pos in range(n_imgs):
        prompt = f"Score for image {pos + 1} (1–10): "
        print(prompt, end='', flush=True)
        # For each score, wait until a valid number is typed in the terminal.
        last_render_ts = 0.0
        while True:
            # Process GUI events and allow the user to resize the window
            cv2.waitKey(30)
            # detect window resize (or periodically re-render to keep responsive)
            try:
                _x, _y, w, h = cv2.getWindowImageRect(win_name)
            except Exception:
                w, h = last_w, last_h
            now = time.time()
            if (w, h) != (last_w, last_h) or (now - last_render_ts) > 0.3:
                # Recompose the grid to fit the new window size
                try:
                    disp = compose_grid(shuffled_imgs, numeric_labels, max(1, w), max(1, h))
                    cv2.imshow(win_name, disp)
                except Exception:
                    # Fallback: simple uniform resize
                    ih, iw = shuffled_imgs[0].shape[:2]
                    scale = min(max(1e-6, w / iw), max(1e-6, h / ih))
                    disp = cv2.resize(shuffled_imgs[0], (max(1, int(iw * scale)), max(1, int(ih * scale))))
                    cv2.imshow(win_name, disp[:h, :w])
                last_w, last_h = w, h
                last_render_ts = now

            # Non-blocking check for stdin input
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                line = sys.stdin.readline().strip()
                try:
                    s = float(line)
                    if 1 <= s <= 10:
                        position_scores.append(s)
                        break
                except Exception:
                    pass
                print("Please enter a number from 1 to 10:\n"+
                      "\t[1-4] -> Missing labels\n" +
                      "\t[5-7] -> Misplaced labels\n" +
                      "\t[8-10] -> Correct labels", flush=True)

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