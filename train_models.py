# ================================
# File: train_models.py
# ================================
"""
Train YOLO (Ultralytics) segmentation models in two modes:

1) MULTI_PRETRAIN: Train multiple pretrained models with the same train params.
2) SINGLE_PRETRAIN_MULTI_CFG: Train one pretrained model over a grid of train params
   (e.g., different epochs and imgsz). Trained weights are copied/renamed to
   OUTPUT_DIR as: <pretrain_name>_epochs-<x>_imgsz-<y>.pt

All paths and settings are hardcoded below as requested.
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# --------------------
# === CONSTANTS ===
# --------------------
MODE = "MULTI_PRETRAIN"  # "MULTI_PRETRAIN" | "SINGLE_PRETRAIN_MULTI_CFG"

# Dataset config
DATA_YAML = "containers_dataset/data.yaml"  # path to your dataset yaml

# Which pretrained models to use (filenames inside PRETRAIN_MODELS_DIR)
PRETRAIN_MODELS = [
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
]

# Where to put your final .pt exports after training
OUTPUT_DIR = Path("./trained_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common train args (applied in both modes unless overridden by a variant)
COMMON_TRAIN_ARGS = dict(
    imgsz=640,
    epochs=50,
    batch=-1,  # Auto batch
    device="cpu",  # change to "0" for first GPU
)

# For SINGLE_PRETRAIN_MULTI_CFG: pick one base pretrained model
SINGLE_PRETRAIN_MODEL = "yolo11n-seg.pt"

# Grid of variants for SINGLE_PRETRAIN_MULTI_CFG
TRAIN_PARAM_VARIANTS = [
    {"epochs": 50, "imgsz": 640},
    {"epochs": 50, "imgsz": 960},
    {"epochs": 100, "imgsz": 640},
]

# --------------------
# === HELPERS ===
# --------------------

def _copy_best_to_output(save_dir: Path, out_name: str) -> Path:
    """Copy runs/.../weights/best.pt to OUTPUT_DIR/out_name.pt"""
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found under {save_dir}")
    out_path = OUTPUT_DIR / f"{out_name}.pt"
    shutil.copy2(best, out_path)
    return out_path


# --------------------
# === MODES ===
# --------------------

def run_multi_pretrain():
    for prt_model in PRETRAIN_MODELS:
        print(f"\n>>> Training from pretrained: {prt_model}")
        model = YOLO(prt_model)
        results = model.train(data=DATA_YAML, **COMMON_TRAIN_ARGS)
        # Build name and export
        out_name = f"{prt_model}_epochs-{COMMON_TRAIN_ARGS['epochs']}_imgsz-{COMMON_TRAIN_ARGS['imgsz']}"
        final_path = _copy_best_to_output(Path(results.save_dir), out_name)
        print(f"Saved trained model -> {final_path}")


def run_single_pretrain_multi_cfg():
    if not SINGLE_PRETRAIN_MODEL.exists():
        raise FileNotFoundError(f"Base pretrained not found: {SINGLE_PRETRAIN_MODEL}")

    for variant in TRAIN_PARAM_VARIANTS:
        args = {**COMMON_TRAIN_ARGS, **variant}
        print(f"\n>>> Training {SINGLE_PRETRAIN_MODEL} with variant: {variant}")
        model = YOLO(SINGLE_PRETRAIN_MODEL)
        results = model.train(data=DATA_YAML, **args)
        out_name = f"{SINGLE_PRETRAIN_MODEL}_epochs-{args['epochs']}_imgsz-{args['imgsz']}"
        final_path = _copy_best_to_output(Path(results.save_dir), out_name)
        print(f"Saved trained model -> {final_path}")


if __name__ == "__main__":
    if MODE == "MULTI_PRETRAIN":
        run_multi_pretrain()
    elif MODE == "SINGLE_PRETRAIN_MULTI_CFG":
        run_single_pretrain_multi_cfg()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
