# --- Imports ---
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import ttest_ind, mannwhitneyu, gaussian_kde
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from tidepool_data_science_simulator.utils import DATA_DIR
from tidepool_data_science_simulator.evaluation.inspect_results import load_result
from tidepool_data_science_metrics.glucose.glucose import (
    percent_values_ge_70_le_180, percent_values_gt_180,
    percent_values_lt_70, blood_glucose_risk_index,
)

# --- Configuration ---
PROJECT_ROOT = Path(DATA_DIR) / "processed"
RESULT_DIR = PROJECT_ROOT / "autobolus_tempbasal_comparison_unannounced_meals_basal_cap2025_05_28_T_18_00_04"
HISTOGRAM_PATH = Path("/Users/mconn/Downloads/BG_Distribution_Histogram.csv")
METRIC_NAMES = [
    'Percent Time in Range (70 - 180 mg/dL)', 
    'Percent below Range (< 70 mg/dL)', 
    'Percent Time above Range (> 180 mg/dL)', 
    'Cumulative Insulin (U)', 
    'BGRI'
]
PAF_VALUES = ["paf=0.0", "paf=0.4"]

# --- Core Functions ---
def weighted_percentile(values, weights, percentiles):
    sorted_idx = np.argsort(values)
    cum_weights = np.cumsum(weights[sorted_idx])
    total = cum_weights[-1]
    pct = 100 * cum_weights / total
    return np.interp(percentiles, pct, values[sorted_idx])

def weighted_iqr(values, weights):
    return weighted_percentile(values, weights, [75])[0] - weighted_percentile(values, weights, [25])[0]

def weighted_mean_std(values, weights):
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return mean, np.sqrt(var)
 
def compare_kde_boxplot(
    data_1,
    data_2,
    weights=None,
    title="Comparison",
    ylabel="Value",
    label_1="Dataset 1",
    label_2="Dataset 2",
    color_1="#627cff",
    color_2="#271b45",
    bw_method=0.2,
    violin_width=0.4,
    box_width=0.05
):
    """
    Compare two datasets using KDEs and boxplots on a shared vertical axis.

    Parameters:
        data1: First dataset (1D array).
        data2: Second dataset (1D array).
        weights: Optional weights for computing KDE.
        title: Plot title.
        ylabel: Y-axis label.
        label1, label2: Labels for datasets.
        color1, color2: Colors for KDE and boxplot lines.
        bw_method: Bandwidth for KDE.
        violin_width: Max width of violin KDEs.
        box_width: Width of boxplots.
    """
    x_min = min(data_1.min(), data_2.min())
    x_max = max(data_1.max() , data_2.max())
    x_grid = np.linspace(x_min, x_max, 500)

    # KDEs
    kde_1 = gaussian_kde(data_1, weights=weights, bw_method=bw_method) 
    kde_2 = gaussian_kde(data_2, weights=weights, bw_method=bw_method)

    density_1 = kde_1(x_grid)
    density_2 = kde_2(x_grid)

    peaks_1, _ = find_peaks(density_1)
    modes_1 = x_grid[peaks_1]

    peaks_2, _ = find_peaks(density_2)
    modes_2 = x_grid[peaks_2]

    # Normalize KDEs
    density_1 /= density_1.max()
    density_2 /= density_2.max()

    # Add modes to labels
    print(f"{label_1} (mode(s): {', '.join([f'{m:.1f}' for m in modes_1])})" if len(modes_1) > 0 else label_1)
    print(f"{label_2} (mode(s): {', '.join([f'{m:.1f}' for m in modes_2])})" if len(modes_2) > 0 else label_2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.fill_betweenx(x_grid, (-density_1 * violin_width) - 0.1, -0.1, facecolor=color_1, alpha=1, label=label_1)
    ax.fill_betweenx(x_grid, 0.1, (density_2 * violin_width) + 0.1, facecolor=color_2, alpha=1, label=label_2)

    # Boxplots
    data_1_weighted = np.repeat(data_1, (weights * 10000).astype(int))
    data_2_weighted = np.repeat(data_2, (weights * 10000).astype(int))

    box_data = [data_1_weighted, data_2_weighted]
    positions = [-0.05, 0.05]
    box = ax.boxplot(box_data, vert=True, positions=positions, widths=box_width, patch_artist=True)

    line_width = 2.5
    for i, color in enumerate([color_1, color_2]):
        box['boxes'][i].set_facecolor('none')
        box['boxes'][i].set_edgecolor(color)
        box['boxes'][i].set_linewidth(line_width)

        box['medians'][i].set_color(color)
        box['medians'][i].set_linewidth(line_width)

        for j in [2*i, 2*i+1]:  # whiskers and caps
            box['whiskers'][j].set_color(color)
            box['whiskers'][j].set_linewidth(line_width)
            box['caps'][j].set_color(color)
            box['caps'][j].set_linewidth(line_width)

        if 'fliers' in box and i < len(box['fliers']):
            box['fliers'][i].set_markeredgecolor(color)
            box['fliers'][i].set_linewidth(line_width)

    ax.set_xlabel('Density')
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_title(title)
    ax.legend(frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.show()

def calculate_cumulative_insulin(df):
    return df['delivered_basal_insulin'].sum() + df['true_bolus'].sum()

def calculate_metrics(df):
    return (
        percent_values_ge_70_le_180(df['bg']),
        percent_values_lt_70(df['bg']),
        percent_values_gt_180(df['bg']),
        calculate_cumulative_insulin(df),
        blood_glucose_risk_index(df['bg'])[2],
    )

def group_files_by_user_ibg(files):
    grouped = defaultdict(lambda: {paf: [] for paf in PAF_VALUES})
    for file in files:
        parts = {kv.split("=")[0]: kv.split("=")[1] for kv in Path(file).stem.split("_") if "=" in kv}
        key = f"vp={parts['vp']}_patient_id={parts['id']}_ibg={parts['ibg']}"

        file_str = str(file) 
        for paf in PAF_VALUES:
            if paf in file_str:
                grouped[key][paf].append(file)
    return grouped

def load_histogram_weights(path):
    df = pd.read_csv(path)
    return {row['ibg']: row['proportion'] for _, row in df.iterrows()}

def process_pair_metrics(grouped_files, weights_dict, start_idx=137, hours=2):
    n = len(grouped_files)
    metrics_all = np.zeros((n, 5, 2))
    ibg_values = np.zeros(n)
    weights = np.zeros(n)
    insulin_diffs = []

    for idx, (key, files) in enumerate(grouped_files.items()):
        if not all(files[paf] for paf in PAF_VALUES):
            continue
        
        ibg = float(key.split("_ibg=")[1])
        if ibg < 70 or ibg > 180:
            continue

        df0 = pd.concat([load_result(f)[1] for f in files["paf=0.0"]])
        df1 = pd.concat([load_result(f)[1] for f in files["paf=0.4"]])

        slice_ = slice(start_idx, start_idx + hours * 12)
        m0 = calculate_metrics(df0.iloc[slice_])
        m1 = calculate_metrics(df1.iloc[slice_])

        for j in range(5):
            metrics_all[idx, j] = [m0[j], m1[j]]
        
        
        ibg_values[idx] = ibg
        weights[idx] = weights_dict.get(ibg, 0)
        insulin_diffs.append(m0[3] - m1[3])
    
    return metrics_all, ibg_values, weights, insulin_diffs

def summarize_and_plot(metrics_all, weights):
    summary = {}
    for i, name in enumerate(METRIC_NAMES):
        data_1 = metrics_all[:, i, 0]
        data_2 = metrics_all[:, i, 1]

        try:
            compare_kde_boxplot(data_1, data_2, weights, title=name,
                                ylabel=name, label_1="Temp Basal", label_2="Autobolus (0.4)")
        except Exception as e:
            print(f"Error plotting {name}: {e}")
            continue

        t_stat, p_val = ttest_ind(data_1, data_2, equal_var=False)
        u_stat, mw_p_val = mannwhitneyu(data_1, data_2, alternative='two-sided')
        med1, med2 = weighted_percentile(data_1, weights, 50), weighted_percentile(data_2, weights, 50)
        iqr1, iqr2 = weighted_iqr(data_1, weights), weighted_iqr(data_2, weights)
        mean1, std1 = weighted_mean_std(data_1, weights)
        mean2, std2 = weighted_mean_std(data_2, weights)

        summary[name] = {
            "Temp Basal": {"mean": mean1, "std": std1, "median": med1, "iqr": iqr1},
            "Autobolus (0.4)": {"mean": mean2, "std": std2, "median": med2, "iqr": iqr2},
            "t_stat": t_stat, "p_value": p_val,
            "mannwhitney_u_stat": u_stat, "mannwhitney_p_value": mw_p_val
        }
    return summary

def print_summary(summary):
    for metric, stats in summary.items():
        print(f"\nMetric: {metric}")
        for group, vals in stats.items():
            if isinstance(vals, dict):
                print(f"  {group}: Mean={vals['mean']:.2f}, Std={vals['std']:.2f}, Median={vals['median']:.2f}, IQR={vals['iqr']:.2f}")
            else:
                print(f"  {group}: {vals:.2f}")

# --- Run Pipeline ---
if __name__ == "__main__":
    all_files = list(RESULT_DIR.glob("*.tsv"))
    grouped = group_files_by_user_ibg(all_files)
    weights_dict = load_histogram_weights(HISTOGRAM_PATH)

    metrics_all, ibg_values, weights, insulin_diffs = process_pair_metrics(grouped, weights_dict)

    summary = summarize_and_plot(metrics_all, weights)
    print_summary(summary)