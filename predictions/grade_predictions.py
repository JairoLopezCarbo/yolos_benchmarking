"""
Interactive grader: shows the same image across model folders side-by-side and
prompts you to score each model 1–10. Results saved to human_scores.csv (wide format).
"""
from pathlib import Path
import csv
import cv2
import numpy as np

PREDICTIONS_ROOT = Path(__file__).resolve().parent
MODEL_FOLDERS = "asdsad"
OUTPUT_CSV = PREDICTIONS_ROOT / "human_scores.csv"

# Collect image IDs that exist in ALL model folders
common = None
for m in MODEL_FOLDERS:
    imgs = sorted([p.name for p in (PREDICTIONS_ROOT / m).iterdir() if p.suffix.lower() in {{'.jpg','.jpeg','.png','.bmp'}}])
    ids = [Path(x).stem for x in imgs]
    common = set(ids) if common is None else common.intersection(ids)

common = sorted(common) if common else []
print(f"Found {len(common)} common images across models: {MODEL_FOLDERS}")

# Prepare CSV header
if not OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["image_id", *MODEL_FOLDERS])

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

    # Resize to same height and stack horizontally
    heights = [im.shape[0] for im in imgs]
    target_h = min(heights)
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        scale = target_h / h
        resized.append(cv2.resize(im, (int(w*scale), target_h)))

    canvas = np.hstack(resized)
    cv2.imshow(f"{img_id} :: {' | '.join(MODEL_FOLDERS)}", canvas)
    print(f"\nScoring image: {img_id}")

    scores = []
    for m in MODEL_FOLDERS:
        while True:
            try:
                s = float(input(f"Score for {m} (1–10): "))
                if 1 <= s <= 10:
                    scores.append(s)
                    break
            except Exception:
                pass
            print("Please enter a number from 1 to 10.")

    with open(OUTPUT_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([img_id, *scores])

    # Close the image window between items
    cv2.destroyAllWindows()

print("\nDone. Scores saved to:", OUTPUT_CSV)