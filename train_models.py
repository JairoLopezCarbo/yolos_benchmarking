"""
Train YOLO (Ultralytics) segmentation models with a single, readable CONFIG dict.

Modes supported:
- MULTI_MODELS: train several pretrained models with the same train params.
- SINGLE_MODEL_MULTI_CFG: train one pretrained model over a grid of params
    (different epochs/imgsz, etc.).

Output filename format: <model_name>_epochs-<X>_imgsz-<Y>.pt (model_name without .pt)
"""
import shutil
import subprocess
import json
from pathlib import Path

from ultralytics import YOLO

# =============================
# Centralized configuration
# =============================
CONFIG = {
    # Path to dataset YAML (replaces previous `data` block)
    "dataset_yaml": "in_out_data/containers_dataset/data.yaml",
    "mode": "MULTI_MODELS",  # "MULTI_MODELS" | "SINGLE_MODEL_MULTI_CFG"
    "models": { # Pretrained checkpoints from Ultralytics Hub (strings are fine; no local file required)
        "multi_models": [ # https://docs.ultralytics.com/tasks/segment/#val
            "yolo26n-seg.pt",
            "yolo26s-seg.pt",
            "yolo26m-seg.pt",        ],
        # Base model for SINGLE_MODEL_MULTI_CFG mode
        "single_model": "yolo26n-seg.pt",
    },
    "train": {
        # Common training args (applied to all runs unless overridden by a variant)
        "common": {
            "imgsz": 1024,
            "epochs": 250,
            "batch": -1,     # Auto batch
            "device": "0", # Change to "0" for first GPU
        },
        # Variants for SINGLE_PRETRAIN_MULTI_CFG (merged over "common")
        "variants": [
            {"epochs": 35, "imgsz": 640},
            {"epochs": 50, "imgsz": 640},
            {"epochs": 65, "imgsz": 640},
        ],
    },
    
    "engine": {
        "enabled": False,
        "params": {
            "precision": "fp16",  # 'fp16' or 'fp32' -> controls trtexec build precision (adds --fp16 flag)
            "workspace": 4096,     # workspace size for trtexec (MB)
            "half": True,          # export/convert model weights to FP16 before ONNX export
        },
    },

    "output": {
        "dir": "in_out_data/trained_models",  # directory to save final .pt exports
        "file_name": "TRG_containers",          # optional prefix for exported files, e.g. 'myexp' => myexp_yolo11n-seg_epochs-...pt
    },
    # Optional TensorRT engine export after each training run.
    # If enabled, the script will try to produce a .onnx (via Ultralytics export)
    # and then call `trtexec` to build a `.engine`. `trtexec` must be on PATH.
    
    
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


def _maybe_export_engine(pt_path: Path, out_dir: Path, out_name: str, engine_cfg: dict, imgsz: int) -> Path | None:
    """If enabled in engine_cfg, export the `.pt` to `.onnx` (attempt) and run trtexec to save a `.engine`.

    Returns the Path to the generated .engine or None if not enabled.
    Raises informative errors if required tools are missing or export fails.
    """
    if not engine_cfg or not engine_cfg.get("enabled"):
        return None

    params = engine_cfg.get("params", {})
    precision = (params.get("precision") or "fp16").lower()
    workspace = params.get("workspace")

    # Expect engine filename under out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = out_dir / f"{out_name}.engine"

    # Determine ONNX candidate path
    onnx_path = out_dir / f"{out_name}.onnx"

    if not onnx_path.exists():
        # Try to export ONNX via Ultralytics YOLO export API
        try:
            half_flag = bool(params.get("half"))
            print(f"Exporting {pt_path} -> ONNX (imgsz={imgsz}, half={half_flag})...")
            model = YOLO(str(pt_path))
            # If requested, attempt to convert internal weights to half precision before export.
            if half_flag:
                try:
                    if hasattr(model, "model") and hasattr(model.model, "half"):
                        model.model.half()
                except Exception:
                    # best-effort: continue even if half conversion not supported
                    pass

            # model.export returns path in newer versions; call and hope for creation
            try:
                model.export(format="onnx", imgsz=imgsz, simplify=True, half=half_flag)
            except TypeError:
                # Some ultralytics versions may not accept `half` arg; call without it
                model.export(format="onnx", imgsz=imgsz, simplify=True)
        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {e}")

        # Search for a produced onnx file near the pt or in out_dir
        candidates = list(out_dir.glob(f"{out_name}*.onnx")) + list(pt_path.parent.glob(f"{pt_path.stem}*.onnx"))
        if not candidates:
            raise FileNotFoundError("ONNX file not found after export. Check model.export behaviour.")
        onnx_path = candidates[0]

    # Ensure trtexec is available
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise FileNotFoundError("trtexec not found in PATH. Install TensorRT's trtexec to build .engine files.")

    cmd = [trtexec, f"--onnx={str(onnx_path)}", f"--saveEngine={str(engine_path)}"]
    if precision == "fp16":
        cmd.append("--fp16")
    if workspace:
        cmd.append(f"--workspace={int(workspace)}")

    print(f"Running trtexec to build engine: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"trtexec failed: {e}")

    if not engine_path.exists():
        raise FileNotFoundError("Engine file not produced by trtexec.")

    return engine_path


def run_multi_models(cfg: dict):
    data_yaml = cfg.get("dataset_yaml", cfg.get("data", {}).get("yaml"))
    out_dir = Path(cfg.get("output", {}).get("dir", "trained_models"))
    file_name = cfg.get("output", {}).get("file_name", "") or ""
    common = dict(cfg["train"]["common"])
    engine_cfg = cfg.get("engine", {})

    for prt_model in cfg["models"]["multi_models"]:
        # Print params being used for this run
        print(f"Multi Model: {prt_model} common: {json.dumps(common, indent=2)}")
        print(f"Dataset YAML: {data_yaml}")
        print(f"Output: dir={out_dir}, file_name={file_name}")
        if engine_cfg:
            print(f"Engine config: {json.dumps(engine_cfg, indent=2)}")

        print(f"\n>>> Training from pretrained: {prt_model}")
        model = YOLO(prt_model)
        results = model.train(data=data_yaml, **common)

        # Build output name using stem (no double .pt in filename)
        if file_name:
            out_name = f"{file_name}_{_model_stem(prt_model)}_epochs-{common['epochs']}_imgsz-{common['imgsz']}"
        else:
            out_name = f"{_model_stem(prt_model)}_epochs-{common['epochs']}_imgsz-{common['imgsz']}"
        final_path = _copy_best_to_output(Path(results.save_dir), out_dir, out_name)
        print(f"Saved trained model -> {final_path}")

        # Optionally export TensorRT engine
        try:
            engine = _maybe_export_engine(final_path, out_dir, out_name, engine_cfg, common.get("imgsz"))
            if engine:
                print(f"Saved TensorRT engine -> {engine}")
        except Exception as e:
            print(f"Engine export skipped/failed: {e}")


def run_single_model_multi_cfg(cfg: dict):
    data_yaml = cfg.get("dataset_yaml", cfg.get("data", {}).get("yaml"))
    out_dir = Path(cfg.get("output", {}).get("dir", "trained_models"))
    file_name = cfg.get("output", {}).get("file_name", "") or ""
    base_model = cfg["models"]["single_model"]
    common = dict(cfg["train"]["common"])
    variants = list(cfg["train"].get("variants", []))
    engine_cfg = cfg.get("engine", {})

    # Print base model and common params
    print(f"Single Model: {base_model} common: {json.dumps(common, indent=2)}")
    print(f"Dataset YAML: {data_yaml}")
    print(f"Output: dir={out_dir}, file_name={file_name}")
    if engine_cfg:
        print(f"Engine config: {json.dumps(engine_cfg, indent=2)}")

    for variant in variants:
        args = {**common, **variant}
        print(f"\n>>> Training {base_model} with variant: {variant}")
        # Print the merged args used for this variant run
        print(f"Using params: {json.dumps(args, indent=2)}")
        model = YOLO(base_model)
        results = model.train(data=data_yaml, **args)

        if file_name:
            out_name = f"{file_name}_{_model_stem(base_model)}_epochs-{args['epochs']}_imgsz-{args['imgsz']}"
        else:
            out_name = f"{_model_stem(base_model)}_epochs-{args['epochs']}_imgsz-{args['imgsz']}"
        final_path = _copy_best_to_output(Path(results.save_dir), out_dir, out_name)
        print(f"Saved trained model -> {final_path}")

        # Optionally export TensorRT engine
        try:
            engine = _maybe_export_engine(final_path, out_dir, out_name, engine_cfg, args.get("imgsz"))
            if engine:
                print(f"Saved TensorRT engine -> {engine}")
        except Exception as e:
            print(f"Engine export skipped/failed: {e}")


if __name__ == "__main__":
    mode = (CONFIG.get("mode") or "").upper()
    if mode == "MULTI_MODELS":
        run_multi_models(CONFIG)
    elif mode == "SINGLE_MODEL_MULTI_CFG":
        run_single_model_multi_cfg(CONFIG)
    else:
        raise ValueError(f"Unknown CONFIG['mode']: {CONFIG.get('mode')} (expected 'MULTI_MODELS' or 'SINGLE_MODEL_MULTI_CFG')")
