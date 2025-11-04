"""Plot tuned grouped bar chart for `predictions/human_scores.csv`.

Layout per model (grouped on x-axis):
 - Left: timing group
     - big wide bar = total_avg (with errorbar = total_std) [left y-axis ms]
     - three thin bars centered on the big bar = preprocess_avg, inference_avg, postprocess_avg
       (no errorbars to avoid clutter; optionally add if desired)
 - Right: score column
     - plot average score as a bar and vertical error showing min..max (right y-axis)

The script auto-detects the CSV layout produced by the grader (either the
new 'row' labeled layout or older wide layout) and reads timing stats and
frame scores per model.

Usage:
    python3 predictions/plot_human_scores.py

Saves `predictions/human_scores_plot.png` and shows an interactive window.
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
    # Only two configurable options for the plot
    "show_legend": True,      # whether to display the legend
    "log_ms_axis": True,      # time axis scale: True=logarithmic, False=linear
    # Paths (relative to this script's folder)
    "paths": {
        "csv": "predictions/human_scores.csv",
        "out_png": "plot.png",
    },
}

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / CONFIG["paths"]["csv"]
OUT_PNG = ROOT / CONFIG["paths"]["out_png"]


def detect_model_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    # If first column is 'row' the rest are model columns
    if cols and cols[0].lower() == 'row':
        models = cols[1:]
        layout = 'row_labeled'
        return models, layout

    # If first column is 'image_id' or similar, try to detect model columns
    # by excluding timing-summary-like columns (those containing timing keys)
    timing_re = re.compile(r'.*(_total|total)(_avg|_std|_ms|_avg_ms|_std_ms)?$|.*_(preprocess|preprocess_ms|preprocess_avg|preprocess_std)$|.*_(inference|inference_ms|inference_avg|inference_std)$|.*_(postprocess|postprocess_ms|postprocess_avg|postprocess_std)$', re.I)
    candidate_models = [c for c in cols if not timing_re.match(c) and c.lower() not in ('image_id','frame','index')]
    if candidate_models:
        layout = 'wide_timing_cols'
        return candidate_models, layout

    # fallback: treat all columns except first as models
    if len(cols) > 1:
        return cols[1:], 'fallback'
    return [], 'empty'


def read_human_scores(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    models, layout = detect_model_columns(df)

    timing_stats = {m: {} for m in models}
    scores = {m: [] for m in models}

    if layout == 'row_labeled':
        # first rows are stat rows labeled in column 'row'
        # find stat rows by name
        stat_order = ['total_avg','total_std','preprocess_avg','preprocess_std','inference_avg','inference_std','postprocess_avg','postprocess_std']
        # build mapping from stat name to row index
        row_label = df.columns[0]
        # iterate rows and capture stats
        for _, r in df.iterrows():
            label = str(r[row_label]).strip()
            if label in stat_order:
                for m in models:
                    val = r.get(m)
                    timing_stats[m][label] = float(val) if pd.notna(val) and val != '' else math.nan
            else:
                # frame rows: label is frame id; values are scores per model
                for m in models:
                    v = r.get(m)
                    if pd.notna(v) and v != '':
                        try:
                            scores[m].append(float(v))
                        except Exception:
                            pass

    elif layout == 'wide_timing_cols':
        # timing stats are encoded in column names like <model>_total_avg_ms
        # extract for each model
        for m in models:
            # try several naming variants
            def find_col(patterns):
                for pat in patterns:
                    matches = [c for c in df.columns if c.lower().startswith((m.lower() + pat)) or c.lower().endswith(pat)]
                    if matches:
                        return matches[0]
                return None

            total_avg_col = find_col(['_total_avg_ms','_total_avg','_total_ms','_total_avg'])
            total_std_col = find_col(['_total_std_ms','_total_std','_total_std_ms'])
            pre_avg_col = find_col(['_preprocess_avg_ms','_preprocess_avg','_preprocess_ms','_preprocess'])
            pre_std_col = find_col(['_preprocess_std_ms','_preprocess_std'])
            inf_avg_col = find_col(['_inference_avg_ms','_inference_avg','_inference_ms','_inference'])
            inf_std_col = find_col(['_inference_std_ms','_inference_std'])
            post_avg_col = find_col(['_postprocess_avg_ms','_postprocess_avg','_postprocess_ms','_postprocess'])
            post_std_col = find_col(['_postprocess_std_ms','_postprocess_std'])

            def col_value(c):
                if c and c in df.columns:
                    try:
                        return float(df[c].dropna().iloc[0])
                    except Exception:
                        return math.nan
                return math.nan

            timing_stats[m]['total_avg'] = col_value(total_avg_col)
            timing_stats[m]['total_std'] = col_value(total_std_col)
            timing_stats[m]['preprocess_avg'] = col_value(pre_avg_col)
            timing_stats[m]['preprocess_std'] = col_value(pre_std_col)
            timing_stats[m]['inference_avg'] = col_value(inf_avg_col)
            timing_stats[m]['inference_std'] = col_value(inf_std_col)
            timing_stats[m]['postprocess_avg'] = col_value(post_avg_col)
            timing_stats[m]['postprocess_std'] = col_value(post_std_col)

        # scores are rows where image_id present; assume first column is image id
        score_cols = models
        for _, r in df.iterrows():
            for m in score_cols:
                v = r.get(m)
                if pd.notna(v) and v != '':
                    try:
                        scores[m].append(float(v))
                    except Exception:
                        pass

    else:
        # fallback: assume columns are models and rows are frame scores
        for m in models:
            col = df[m]
            scores[m] = [float(x) for x in col.dropna().values]

    # Normalize keys in timing_stats to unified names
    unified = {}
    for m, s in timing_stats.items():
        unified[m] = {
            'total_avg': s.get('total_avg') or s.get('total_ms_avg') or s.get('total_ms_avg'.replace('_','')) or s.get('total_ms_avg'),
            'total_std': s.get('total_std') or s.get('total_ms_std') or s.get('total_ms_std'),
            'preprocess_avg': s.get('preprocess_avg') or s.get('preprocess_ms_avg') or s.get('preprocess_ms_avg'),
            'preprocess_std': s.get('preprocess_std') or s.get('preprocess_ms_std') or s.get('preprocess_ms_std'),
            'inference_avg': s.get('inference_avg') or s.get('inference_ms_avg') or s.get('inference_ms_avg'),
            'inference_std': s.get('inference_std') or s.get('inference_ms_std') or s.get('inference_ms_std'),
            'postprocess_avg': s.get('postprocess_avg') or s.get('postprocess_ms_avg') or s.get('postprocess_ms_avg'),
            'postprocess_std': s.get('postprocess_std') or s.get('postprocess_ms_std') or s.get('postprocess_ms_std'),
        }

    # Compute per-model score stats
    score_stats = {}
    for m in models:
        arr = np.array(scores.get(m, []), dtype=float)
        if arr.size == 0:
            score_stats[m] = {'min': math.nan, 'max': math.nan, 'avg': math.nan}
        else:
            score_stats[m] = {'min': float(np.nanmin(arr)), 'max': float(np.nanmax(arr)), 'avg': float(np.nanmean(arr))}

    return models, unified, score_stats


def plot(models, benchmark, score_stats, out_png=OUT_PNG):
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
    ax_time.bar(centers_pre, pre_avgs, width=thin_w, color='C1', label='preprocess avg (ms)')
    ax_time.bar(centers_inf, inf_avgs, width=thin_w, color='C2', label='inference avg (ms)')
    ax_time.bar(centers_post, post_avgs, width=thin_w, color='C3', label='postprocess avg (ms)')
    # Add std errorbars for thin bars, colored slightly darker than their avg bars
    from matplotlib import colors as mcolors

    def darker(col, factor=0.65):
        try:
            rgb = np.array(mcolors.to_rgb(col))
            dark = np.clip(rgb * factor, 0, 1)
            return tuple(dark)
        except Exception:
            return col

    ax_time.errorbar(centers_pre, pre_avgs, yerr=pre_stds, fmt='none', ecolor=darker('C1'), capsize=4)
    ax_time.errorbar(centers_inf, inf_avgs, yerr=inf_stds, fmt='none', ecolor=darker('C2'), capsize=4)
    ax_time.errorbar(centers_post, post_avgs, yerr=post_stds, fmt='none', ecolor=darker('C3'), capsize=4)

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

    # Plot score stats on ax_score
    score_avgs = [score_stats[m]['avg'] for m in models]
    score_mins = [score_stats[m]['min'] for m in models]
    score_maxs = [score_stats[m]['max'] for m in models]

    # plot average score as bars
    bar_w = 0.25
    ax_score.bar(score_x, score_avgs, width=bar_w, color='C4', alpha=0.8, label='score avg')
    # show min/max as short discontinuous horizontal dashed lines centered at each score_x
    seg_w = 0.28
    for xi, mn, mx in zip(score_x, score_mins, score_maxs):
        if not (math.isnan(mn) or math.isnan(mx)):
            ax_score.hlines(mn, xi - seg_w / 2.0, xi + seg_w / 2.0, colors='k', linestyles='dashed', linewidth=1.8)
            ax_score.hlines(mx, xi - seg_w / 2.0, xi + seg_w / 2.0, colors='k', linestyles='dashed', linewidth=1.8)
    ax_score.set_ylabel('Score (1-10)')
    ax_score.set_ylim(0, 10)

    # Legends
    # Build combined legend with proxies so min/max entry is shown
    lines_time, labels_time = ax_time.get_legend_handles_labels()
    lines_score, labels_score = ax_score.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    proxy_minmax = Line2D([0], [0], color='k', marker='^', linestyle='None', label='score min/max')
    if CONFIG.get("show_legend", True):
        legend_handles = lines_time + lines_score + [proxy_minmax]
        legend_labels = labels_time + labels_score + ['score min/max']
        # place legend centered below the axes, inside the figure, so it appears in the PNG
        ax_time.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.16),
            ncol=min(4, len(legend_handles)),
            frameon=True,
        )

    plt.title('Model timing breakdown and human scores')
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
    models, benchmark, score_stats = read_human_scores(CSV_PATH)
    if not models:
        print('No model columns detected in', CSV_PATH)
        return
    plot(models, benchmark, score_stats, out_png=OUT_PNG)


if __name__ == '__main__':
    main()
