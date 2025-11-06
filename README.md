# YOLO Segmentation Benchmarking

End-to-end pipeline to train YOLO segmentation models (Ultralytics), benchmark them on a fixed image set, collect human quality scores, and plot timing vs. quality comparisons.

This repo includes four scripts that form a simple workflow:

1) train_models.py → trains and exports .pt models
2) benchmark_models.py → runs inference, saves overlays, records timing
3) grade_predictions.py → interactive grader to assign 1–10 quality scores
4) plot_grades.py → plots timing breakdowns and human scores per model


## Requirements

- Python 3.10+ recommended
- Packages:
	- ultralytics
	- opencv-python or opencv-python-headless
	- numpy
	- pandas
	- matplotlib

Install (CPU example):

```bash
pip install ultralytics opencv-python-headless numpy pandas matplotlib
```

GPU tip: set `device="0"` (or the GPU index) in the configs below to use CUDA if available.


## Data & layout

- Dataset YAML for training: `containers_dataset/data.yaml`
- Benchmark images: place test images in `benchmark_images/`
- Trained models will be written to: `trained_models/`
- Predictions (overlays + timing CSVs) will be written to: `predictions/`
- Human scores CSV (output of the grader): `graded_predictions.csv`
- Final plot image: `plot.png`

Example repo structure (abridged):

```
containers_dataset/
	data.yaml
benchmark_images/
trained_models/
predictions/
train_models.py
benchmark_models.py
grade_predictions.py
plot_grades.py
```


## Quickstart: end-to-end

1) Train models (or skip if you already have .pt files)

```bash
python3 train_models.py
```

Notes:
- Configure at the top of `train_models.py`:
	- `CONFIG["mode"]`: `"MULTI_MODELS"` trains several pretrained checkpoints; `"SINGLE_MODEL_MULTI_CFG"` sweeps variants for one base checkpoint.
	- `CONFIG["train"]["common"]`: shared args like `imgsz`, `epochs`, `device`.
- Outputs (best checkpoints) are copied to `trained_models/<name>_epochs-<X>_imgsz-<Y>.pt`.

2) Benchmark and generate overlays

```bash
python3 benchmark_models.py
```

Notes:
- Configure at the top of `benchmark_models.py`:
	- `CONFIG["mode"]`:
		- `"MULTI_MODELS"` evaluates all `.pt` files under `trained_models/`.
		- `"SINGLE_MODEL_MULTI_CFG"` evaluates one `.pt` across a sweep of `conf`/`iou`.
	- `CONFIG["predict"]["common"]`: inference params (e.g., `imgsz`, `conf`, `iou`, `device`).
- Outputs per model/variant folder in `predictions/<subfolder>/`:
	- Overlay images for each input image
	- `benchmark.csv` with timing rows (total, preprocess, inference, postprocess in ms)

3) Grade visual quality (interactive)

```bash
python3 grade_predictions.py
```

Notes:
- This shows the same image across multiple model folders side-by-side and asks you to enter a score (1–10) for each position.
- Configure at the top of `grade_predictions.py`:
	- `CONFIG["predictions"]["models"]`: list of subfolders inside `predictions/` you want to compare.
	- `CONFIG["display"]`: shuffle and labeling behavior.
- Output: appends rows to `graded_predictions.csv`. It also writes timing summary rows (averages/std) per model at the top.
- Requires a GUI/display for OpenCV windows.

4) Plot timing vs. human scores

```bash
python3 plot_grades.py
```

Notes:
- Reads `graded_predictions.csv` and saves `plot.png`.
- Shows grouped timing bars (total with error bars; preprocess/inference/postprocess) and a separate bar for average human score with min/max whiskers.


## Script reference

### train_models.py
Purpose: Train YOLO (Ultralytics) segmentation models using a simple `CONFIG` dict.

- Modes:
	- `MULTI_MODELS`: loop over `CONFIG["models"]["pretrained"]` and train each with shared params.
	- `SINGLE_MODEL_MULTI_CFG`: train one base checkpoint across `CONFIG["train"]["variants"]`.
- Inputs: `containers_dataset/data.yaml`, Ultralytics checkpoints (e.g., `yolo11n-seg.pt`).
- Output: best checkpoints copied to `trained_models/` with a descriptive filename.
- Common knobs: `imgsz`, `epochs`, `batch`, `device`.

### benchmark_models.py
Purpose: Benchmark trained models on `benchmark_images/`, save visual overlays, and record timings.

- Modes:
	- `MULTI_MODELS`: evaluate all `.pt` in `trained_models/`.
	- `SINGLE_MODEL_MULTI_CFG`: evaluate one `.pt` with several `conf`/`iou` variants.
- Inputs: `.pt` models, images in `benchmark_images/`.
- Outputs: per model/variant folder in `predictions/` with images and `benchmark.csv`.
- Visual settings: `CONFIG["visual"]` lets you choose `mode` (`masks` or `boxes`), transparency, thickness, legend behavior.

### grade_predictions.py
Purpose: Human-in-the-loop grading of visual quality (1–10) per image across models.

- Inputs: list of model subfolders under `predictions/` containing the overlay images.
- Behavior: displays tiles of the same image from each model; you type a number (1–10) in the terminal per position.
- Output: `graded_predictions.csv` with timing summaries followed by one row per image with your scores.

### plot_grades.py
Purpose: Create a combined chart of timing breakdowns and human scores.

- Input: `graded_predictions.csv`.
- Output: `plot.png` (and also shows an interactive figure if a display is available).
- Options: `CONFIG["show_legend"]`, `CONFIG["log_ms_axis"]`.


## Tips & troubleshooting

- GPU usage: set `device="0"` in `train_models.py` and `benchmark_models.py` configs. Keep `half=True` for faster FP16 inference on supported GPUs.
- Matching images: `grade_predictions.py` expects the same image IDs across all selected prediction folders; missing images are skipped.
- OpenCV display: on headless servers, you may need X forwarding or a desktop session. The grader requires windows; use `opencv-python` (GUI). For non-GUI environments, `opencv-python-headless` is fine for the other scripts.
- File names: output files include epochs/imgsz in the name to keep variants clear.


## License

MIT (or your chosen license). Update this section if needed.

