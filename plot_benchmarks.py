"""Grouped bar chart from per-model `predictions/<model>/benchmark.csv`.

Layout per model (grouped on x-axis):
 - Left: timing group
     - big wide bar = total_avg (with errorbar = total_std) [left y-axis ms]
     - three thin bars centered on the big bar = preprocess_avg, inference_avg, postprocess_avg
       with their std errorbars.
 - Right: metrics block (percent on right y-axis)
     - three small bars: found% (= TP/(TP+FN)), halluc% (= FP/(TP+FP)), IoU mean (%)
       IoU has an errorbar using pooled std across images weighted by TP.

Usage:
    python3 plot_benchmarks.py

Saves `plot.png` and shows an interactive window.
"""
from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================
# Centralized configuration
# =============================
CONFIG = {
    # List of model folders (under predictions/) to include in the plot
    # e.g. ["yolo11n-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.25", ...]
    "models": [
        # fill with your desired prediction subfolders
        "yolo11n-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.5",
        "yolo11s-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.5",
        "yolo11m-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.5",
        "yolo11l-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.5",
        "yolo11x-seg_epochs-50_imgsz-640_CONF-0.4_IOU-0.5",
    ],
    # Plot options
    "show_legend": True,       # whether to display the legend
    "log_ms_axis": False,      # time axis scale: True=logarithmic, False=linear
    # Paths (relative to this script's folder)
    "paths": {
        "predictions_root": "predictions",
        "out_png": "plot.png",
    },

    
}

ROOT = Path(__file__).resolve().parent
OUT_PNG = ROOT / CONFIG["paths"]["out_png"]
PRED_ROOT = ROOT / CONFIG["paths"]["predictions_root"]


def read_benchmarks(models: list[str]):
    """Read per-model benchmark.tsv files and compute timing stats and metrics.
    Returns:
      - models (same order)
      - timing_stats: dict[model] -> {total_avg,total_std,preprocess_avg,preprocess_std,inference_avg,inference_std,postprocess_avg,postprocess_std}
      - metric_stats: dict[model] -> {found_pct,halluc_pct,iou_mean_pct,iou_std_pct}
    """
    timing_stats = {m: {} for m in models}
    metric_stats = {m: {} for m in models}

    for m in models:
        csv_path = PRED_ROOT / m / 'benchmark.csv'
        if not csv_path.exists():
            print(f"Warning: missing {csv_path}")
            # leave NaNs
            for k in ['total','preprocess','inference','postprocess']:
                timing_stats[m][f'{k}_avg'] = math.nan
                timing_stats[m][f'{k}_std'] = math.nan
            metric_stats[m] = {
                'found_pct': math.nan,
                'halluc_pct': math.nan,
                'iou_mean_pct': math.nan,
                'iou_std_pct': math.nan,
            }
            continue

        try:
            df = pd.read_csv(csv_path, sep='\t')
        except Exception:
            # fallback to comma if user didn't switch delimiter
            df = pd.read_csv(csv_path)

        # Coerce expected columns
        cols = {
            'total_ms': 'total_ms',
            'preprocess_ms': 'preprocess_ms',
            'inference_ms': 'inference_ms',
            'postprocess_ms': 'postprocess_ms',
            'tp': 'tp', 'fp': 'fp', 'fn': 'fn',
            'mean_tp_iou_pct': 'mean_tp_iou_pct',
            'std_tp_iou_pct': 'std_tp_iou_pct',
        }
        for c in list(cols.values()):
            if c not in df.columns:
                df[c] = np.nan

        # numeric conversion
        for c in cols.values():
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # timing stats across images
        def nanmean(a):
            try:
                return float(np.nanmean(a))
            except Exception:
                return math.nan
        def nanstd(a):
            try:
                return float(np.nanstd(a))
            except Exception:
                return math.nan

        timing_stats[m]['total_avg'] = nanmean(df['total_ms'])
        timing_stats[m]['total_std'] = nanstd(df['total_ms'])
        timing_stats[m]['preprocess_avg'] = nanmean(df['preprocess_ms'])
        timing_stats[m]['preprocess_std'] = nanstd(df['preprocess_ms'])
        timing_stats[m]['inference_avg'] = nanmean(df['inference_ms'])
        timing_stats[m]['inference_std'] = nanstd(df['inference_ms'])
        timing_stats[m]['postprocess_avg'] = nanmean(df['postprocess_ms'])
        timing_stats[m]['postprocess_std'] = nanstd(df['postprocess_ms'])

        # detection metrics aggregated across images
        tp_sum = float(df['tp'].fillna(0).sum())
        fp_sum = float(df['fp'].fillna(0).sum())
        fn_sum = float(df['fn'].fillna(0).sum())
        found_pct = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else math.nan
        halluc_pct = fp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else math.nan

        # IoU pooled mean/std across images weighted by TP counts
        mi = df['mean_tp_iou_pct'].astype(float)
        si = df['std_tp_iou_pct'].astype(float)
        ni = df['tp'].astype(float).fillna(0.0)
        mask_valid = (~mi.isna()) & (ni > 0)
        if mask_valid.any():
            mi_v = mi[mask_valid].values
            si_v = si[mask_valid].fillna(0.0).values
            ni_v = ni[mask_valid].values
            N = np.sum(ni_v)
            M = float(np.sum(ni_v * mi_v) / N) if N > 0 else math.nan
            # Pooled variance formula
            if N > 1:
                num = np.sum((ni_v - 1) * (si_v ** 2) + ni_v * ((mi_v - M) ** 2))
                den = N - 1
                S = float(np.sqrt(num / den)) if den > 0 else math.nan
            else:
                S = float(si_v[0]) if si_v.size > 0 else math.nan
        else:
            M, S = math.nan, math.nan

        metric_stats[m] = {
            'found_pct': found_pct * 100.0 if not math.isnan(found_pct) else math.nan,
            'halluc_pct': halluc_pct * 100.0 if not math.isnan(halluc_pct) else math.nan,
            'iou_mean_pct': M,
            'iou_std_pct': S,
        }

    return models, timing_stats, metric_stats


def plot(models, benchmark, metrics, out_png=OUT_PNG):
    n = len(models)
    x = np.arange(n)
    # group layout parameters
    block_w = 0.5  # width of the big total bar block (contains 3 thin bars)
    thin_w = block_w / 3.0
    time_x = x  # center of timing block at x
    # place score bars just to the right of the timing block, but leave a small gap
    bar_w = 0.25
    # preferred small gap between timing block and score bar
    preferred_gap = 0.06
    # compute score_x so the score bar sits just to the right of the block
    score_x = x + block_w / 2.0 + preferred_gap + bar_w / 2.0

    # Figure size derived from config and number of models
    fig_w = max(8.0, n * 2.5)
    fig_h = 6.0
    fig, ax_time = plt.subplots(figsize=(fig_w, fig_h))
    ax_score = ax_time.twinx()

    # Plot timing groups on ax_time (ms)
    total_avgs = [benchmark[m].get('total_avg', math.nan) for m in models]
    total_stds = [benchmark[m].get('total_std', math.nan) for m in models]
    pre_avgs = [benchmark[m].get('preprocess_avg', math.nan) for m in models]
    inf_avgs = [benchmark[m].get('inference_avg', math.nan) for m in models]
    post_avgs = [benchmark[m].get('postprocess_avg', math.nan) for m in models]
    pre_stds = [benchmark[m].get('preprocess_std', math.nan) for m in models]
    inf_stds = [benchmark[m].get('inference_std', math.nan) for m in models]
    post_stds = [benchmark[m].get('postprocess_std', math.nan) for m in models]

    # Big total bars (semi-transparent) covering the whole block
    big_w = block_w
    ax_time.bar(time_x, total_avgs, width=big_w, color='C0', alpha=0.25, label='total avg (ms)')
    # errorbars for total (ensure positive lower bound for log scale later)
    ax_time.errorbar(time_x, total_avgs, yerr=total_stds, fmt='none', ecolor='C0', capsize=5)

    # thin internal bars tiled inside the block without gaps
    block_left = time_x - block_w / 2.0
    centers_pre = block_left + (0.5 * thin_w)
    centers_inf = block_left + (1.5 * thin_w)
    centers_post = block_left + (2.5 * thin_w)
    ax_time.bar(centers_pre, pre_avgs, width=thin_w, color='C4', label='preprocess avg (ms)')
    ax_time.bar(centers_inf, inf_avgs, width=thin_w, color='C5', label='inference avg (ms)')
    ax_time.bar(centers_post, post_avgs, width=thin_w, color='C6', label='postprocess avg (ms)')
    # Add std errorbars for thin bars, colored slightly darker than their avg bars
    from matplotlib import colors as mcolors

    def darker(col, factor=0.65):
        try:
            rgb = np.array(mcolors.to_rgb(col))
            dark = np.clip(rgb * factor, 0, 1)
            return tuple(dark)
        except Exception:
            return col

    ax_time.errorbar(centers_pre, pre_avgs, yerr=pre_stds, fmt='none', ecolor=darker('C4'), capsize=4)
    ax_time.errorbar(centers_inf, inf_avgs, yerr=inf_stds, fmt='none', ecolor=darker('C5'), capsize=4)
    ax_time.errorbar(centers_post, post_avgs, yerr=post_stds, fmt='none', ecolor=darker('C6'), capsize=4)

    ax_time.set_ylabel('Time (ms)')
    ax_time.set_xticks(x)
    # make labels horizontal and split long names at underscores for readability
    labels = [m.replace('_', '\n') for m in models]
    ax_time.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)

    # Set log scale for ms axis; avoid zero or negative values
    all_times = [v for v in total_avgs + pre_avgs + inf_avgs + post_avgs if (v is not None and not math.isnan(v))]
    if all_times:
        min_pos = min([v for v in all_times if v > 0]) if any(v > 0 for v in all_times) else 1e-3
        if CONFIG.get("log_ms_axis", True):
            ax_time.set_yscale('log')
        ax_time.set_ylim(max(min_pos * 0.5, 1e-3), max(all_times) * 1.6)

    # Plot metric stats on ax_score (percent values)
    found_vals = [metrics[m].get('found_pct', math.nan) for m in models]
    halluc_vals = [metrics[m].get('halluc_pct', math.nan) for m in models]
    iou_means = [metrics[m].get('iou_mean_pct', math.nan) for m in models]
    iou_stds = [metrics[m].get('iou_std_pct', math.nan) for m in models]

    # Create a small block for the three metric bars to the right of timing block
    metric_block_w = 0.35
    metric_thin_w = metric_block_w / 3.0
    metric_left = score_x - metric_block_w / 2.0
    centers_found = metric_left + (0.5 * metric_thin_w)
    centers_halluc = metric_left + (1.5 * metric_thin_w)
    centers_iou = metric_left + (2.5 * metric_thin_w)
    ax_score.bar(centers_found, found_vals, width=metric_thin_w, color='C2', alpha=0.9, label='found %')
    ax_score.bar(centers_halluc, halluc_vals, width=metric_thin_w, color='C3', alpha=0.9, label='hallucination %')
    ax_score.bar(centers_iou, iou_means, width=metric_thin_w, color='C1', alpha=0.9, label='IoU mean %')
    # errorbars for IoU
    ax_score.errorbar(centers_iou, iou_means, yerr=iou_stds, fmt='none', ecolor='C1', capsize=4)
    ax_score.set_ylabel('Percentage (%)')
    ax_score.set_ylim(0, 100)

    # Legends
    # Build combined legend with proxies so min/max entry is shown
    lines_time, labels_time = ax_time.get_legend_handles_labels()
    lines_score, labels_score = ax_score.get_legend_handles_labels()
    if CONFIG.get("show_legend", True):
        legend_handles = lines_time + lines_score
        legend_labels = labels_time + labels_score
        # place legend centered below the axes, inside the figure, so it appears in the PNG
        ax_time.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.16),
            ncol=min(4, len(legend_handles)),
            frameon=True,
        )

    plt.title('Model timing breakdown and detection metrics')
    plt.tight_layout()
    # make room at the bottom for the legend placed below the axes (if shown)
    plt.subplots_adjust(bottom=0.22 if CONFIG.get("show_legend", True) else 0.1)

    # Always save PNG
    try:
        plt.savefig(out_png, dpi=200)
        print(f"Saved plot to {out_png}")
    except Exception as e:
        print('Failed to save plot:', e)


def main():
    models = CONFIG.get('models', [])
    if not models:
        print('No models configured in CONFIG["models"].')
        return
    models, benchmark, metrics = read_benchmarks(models)
    plot(models, benchmark, metrics, out_png=OUT_PNG)


if __name__ == '__main__':
    main()
