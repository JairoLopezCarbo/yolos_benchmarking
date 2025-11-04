# ================================
# File: train_models.py
# ================================
"""
Train YOLO (Ultralytics) segmentation models with a single, readable CONFIG dict.

Modes supported:
- MULTI_PRETRAIN: train several pretrained models with the same train params.
- SINGLE_PRETRAIN_MULTI_CFG: train one pretrained model over a grid of params
    (different epochs/imgsz, etc.).

Output filename format: <model_name>_epochs-<X>_imgsz-<Y>.pt (model_name without .pt)
"""
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# =============================
# Centralized configuration
# =============================
# Edit values here. No other code changes needed for common tweaks.
CONFIG = {
    "mode": "MULTI_PRETRAIN",  # "MULTI_PRETRAIN" | "SINGLE_PRETRAIN_MULTI_CFG"
    "data": {
        "yaml": "containers_dataset/data.yaml",  # path to dataset yaml
    },
    "models": {
        # Pretrained checkpoints from Ultralytics Hub (strings are fine; no local file required)
        "pretrained": [
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11m-seg.pt",
        ],
        # Base model for SINGLE_PRETRAIN_MULTI_CFG
        "single_pretrain": "yolo11n-seg.pt",
    },
    "output": {
        "dir": "trained_models",  # directory to save final .pt exports
    },
    "train": {
        # Common training args (applied to all runs unless overridden by a variant)
        "common": {
            "imgsz": 640,
            "epochs": 50,
            "batch": -1,     # Auto batch
            "device": "cpu", # Change to "0" for first GPU
        },
        # Variants for SINGLE_PRETRAIN_MULTI_CFG (merged over "common")
        "variants": [
            {"epochs": 35, "imgsz": 640},
            {"epochs": 50, "imgsz": 640},
            {"epochs": 65, "imgsz": 640},
        ],
    },
}

# --------------------
# === HELPERS ===
# --------------------

def _copy_best_to_output(save_dir: Path, out_dir: Path, out_name: str) -> Path:
    """Copy runs/.../weights/best.pt to out_dir/out_name.pt"""
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found under {save_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.pt"
    shutil.copy2(best, out_path)
    return out_path


# --------------------
# === MODES ===
# --------------------

def _model_stem(name: str) -> str:
    """Return the model base name without extension, e.g., 'yolo11n-seg.pt' -> 'yolo11n-seg'."""
    return Path(name).stem


def run_multi_pretrain(cfg: dict):
    data_yaml = cfg["data"]["yaml"]
    out_dir = Path(cfg["output"]["dir"])    
    common = dict(cfg["train"]["common"])

    for prt_model in cfg["models"]["pretrained"]:
        print(f"\n>>> Training from pretrained: {prt_model}")
        model = YOLO(prt_model)
        results = model.train(data=data_yaml, **common)

        # Build output name using stem (no double .pt in filename)
        out_name = f"{_model_stem(prt_model)}_epochs-{common['epochs']}_imgsz-{common['imgsz']}"
        final_path = _copy_best_to_output(Path(results.save_dir), out_dir, out_name)
        print(f"Saved trained model -> {final_path}")


def run_single_pretrain_multi_cfg(cfg: dict):
    data_yaml = cfg["data"]["yaml"]
    out_dir = Path(cfg["output"]["dir"])    
    base_model = cfg["models"]["single_pretrain"]
    common = dict(cfg["train"]["common"])
    variants = list(cfg["train"].get("variants", []))

    for variant in variants:
        args = {**common, **variant}
        print(f"\n>>> Training {base_model} with variant: {variant}")
        model = YOLO(base_model)
        results = model.train(data=data_yaml, **args)

        out_name = f"{_model_stem(base_model)}_epochs-{args['epochs']}_imgsz-{args['imgsz']}"
        final_path = _copy_best_to_output(Path(results.save_dir), out_dir, out_name)
        print(f"Saved trained model -> {final_path}")


if __name__ == "__main__":
    mode = (CONFIG.get("mode") or "").upper()
    if mode == "MULTI_PRETRAIN":
        run_multi_pretrain(CONFIG)
    elif mode == "SINGLE_PRETRAIN_MULTI_CFG":
        run_single_pretrain_multi_cfg(CONFIG)
    else:
        raise ValueError(f"Unknown CONFIG['mode']: {CONFIG.get('mode')} (expected 'MULTI_PRETRAIN' or 'SINGLE_PRETRAIN_MULTI_CFG')")
