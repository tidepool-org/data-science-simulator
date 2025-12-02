from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from collections import defaultdict

# Column names for the metrics CSV file
# ----
# 'simulation_id', 'time_in_range_70_180', 'time_below_70',
# 'time_below_54', 'time_above_180', 'time_above_250', 'mean_glucose',
# 'median_glucose', 'std_glucose', 'cv_glucose', 'lbgi', 'hbgi', 'bgri',
# 'lbgi_risk_score', 'cumulative_insulin', 'basal_insulin',
# 'bolus_insulin', 'glucose_management_indicator',
# 'time_in_tight_range_70_140', 'coefficient_of_variation', 'alg',
# 'patient', 'ibg', 'meal', 'posvel', 'midisf', 'isf', 'cir', 'basal',
# 'paf'
# ----
METRICS_PATH = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/metrics_results.csv'

HISTOGRAM_PATH = Path("/Users/mconn/data/simulator/BG_Distribution_Histogram.csv")

# --- Weighted Statistics Functions ---
def weighted_percentile(values, weights, percentiles):
    """Calculate weighted percentiles."""
    sorted_idx = np.argsort(values)
    cum_weights = np.cumsum(weights[sorted_idx])
    total = cum_weights[-1]
    pct = 100 * cum_weights / total
    return np.interp(percentiles, pct, values[sorted_idx])

def weighted_mean_std(values, weights):
    """Calculate weighted mean and standard deviation."""
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return mean, np.sqrt(var)

def load_histogram_weights(path):
    """Load IBG distribution weights from histogram CSV."""
    df = pd.read_csv(path)
    return {row['ibg']: row['proportion'] for _, row in df.iterrows()}

def filter_and_weight_data(df, weights_dict, ibg_range=(70, 180)):
    """Filter data by IBG range and add weights."""
    # Filter to IBG range
    mask = (df['ibg'] >= ibg_range[0]) & (df['ibg'] <= ibg_range[1])
    filtered_df = df[mask].copy()
    
    # Add weights based on IBG values
    filtered_df['weight'] = filtered_df['ibg'].map(weights_dict).fillna(0)
    
    # Remove rows with zero weights
    filtered_df = filtered_df[filtered_df['weight'] > 0].copy()
    
    return filtered_df

def create_weighted_boxplot_comparison(df, metric_col='time_in_range_70_180', 
                                     paf_values=None, title=None):
    """Create grouped boxplots comparing posvel true vs false across PAF values with weighted data."""
    
    if paf_values is None:
        paf_values = sorted(df['paf'].unique())
    
    if title is None:
        title = f"Comparison of {metric_col.replace('_', ' ').title()} by PAF and RC Mitigation"
    
    # Prepare data for grouped boxplot
    box_data = []
    labels = []
    colors = []
    
    # Define colors for posvel true (blue) and false (red)
    color_true = '#627cff'  # Blue
    color_false = '#271b44'  # Dark Blue
    
    print(f"\nStatistical Summary for {metric_col} by PAF and RC Mitigation:")
    print("-" * 80)
    
    for paf in paf_values:
        # Get PAF data
        if pd.isna(paf):
            paf_data = df[df['paf'].isna()]
            paf_label = 'Temp Basal'
        else:
            paf_data = df[df['paf'] == paf]
            paf_label = f'PAF={paf}'
        
        if len(paf_data) == 0:
            continue
        
        # Split by posvel
        for posvel in [False, True]:
            posvel_data = paf_data[paf_data['posvel'] == posvel]
            
            if len(posvel_data) == 0:
                continue
                
            values = posvel_data[metric_col].values
            weights = posvel_data['weight'].values
            
            # Create weighted data for boxplot by replicating values based on weights
            # Scale weights to reasonable integers for replication
            scaled_weights = (weights * 10000).astype(int)
            weighted_values = np.repeat(values, scaled_weights)
            
            box_data.append(weighted_values)
            
            # Create labels
            posvel_label = 'True' if posvel else 'False'
            labels.append(f'{paf_label}')
            # labels.append(f'{paf_label}\nPosvel={posvel_label}')
            
            # Set colors
            colors.append(color_true if posvel else color_false)
            
            # Calculate and print statistics
            mean, std = weighted_mean_std(values, weights)
            median = weighted_percentile(values, weights, [50])[0]
            q25 = weighted_percentile(values, weights, [25])[0]
            q75 = weighted_percentile(values, weights, [75])[0]
            
            print(f"{paf_label:15s} RC Mitigation={posvel_label:5s}: Mean={mean:.1f}±{std:.1f}, Median={median:.1f} (IQR: {q25:.1f}-{q75:.1f}), n={len(values)}")
    
    # Create custom positions to group posvel pairs closer together
    positions = []
    group_centers = []  # Track center positions for group labels
    pos = 1
    for i in range(0, len(box_data), 2):  # Process in pairs
        if i + 1 < len(box_data):  # If we have both False and True
            positions.extend([pos, pos + 0.4])  # Close spacing within pair
            group_centers.append(pos + 0.2)  # Center between the two boxes
            pos += 1.5  # Larger gap between PAF groups
        else:  # If we only have one box (shouldn't happen normally)
            positions.append(pos)
            group_centers.append(pos)
            pos += 1.5
    
    # Create group labels (one per PAF value)
    group_labels = []
    for i in range(0, len(labels), 2):  # Take every other label (the PAF part)
        group_labels.append(labels[i])
    
    # Create the boxplot with custom positions
    fig, ax = plt.subplots(figsize=(6, 8))
    
    box_plot = ax.boxplot(box_data, patch_artist=True, 
                         showmeans=True, meanline=True, positions=positions,
                         widths=0.3)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1)
    
    # Style the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_col.replace('_', ' ').title() + ' (U)', fontsize=12)
    ax.set_xlabel('Algorithm Configuration', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set custom x-tick positions at group centers with group labels
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_false, alpha=0.9, label='RC Mitigation=False'),
                      Patch(facecolor=color_true, alpha=0.9, label='RC Mitigation=True')]
    # ax.legend(handles=legend_elements, loc='upper right')
    

    # KLUDGE REMOVE
    ax.set_xticks([1, 1.4])
    ax.set_xticklabels(["Temp Basal", "Autobolus"], rotation=45, ha='right')
    
    plt.tight_layout()
    return fig, ax

def perform_pairwise_statistical_tests(df, metric_col='time_in_range_70_180', 
                                     paf_values=None, reference_paf='none'):
    """Perform pairwise statistical tests comparing posvel groups within each PAF and across PAF values."""
    
    if paf_values is None:
        paf_values = sorted(df['paf'].unique())
    
    print(f"\nPairwise Statistical Tests for {metric_col}:")
    print("=" * 80)
    
    # Test 1: Compare posvel True vs False within each PAF
    print("\n1. RC Mitigation True vs False within each PAF:")
    print("-" * 50)
    
    for paf in paf_values:
        # Get PAF data
        if pd.isna(paf):
            paf_data = df[df['paf'].isna()]
            paf_label = 'Temp Basal'
        else:
            paf_data = df[df['paf'] == paf]
            paf_label = f'PAF={paf}'
        
        if len(paf_data) == 0:
            continue
        
        # Get posvel groups
        posvel_false = paf_data[paf_data['posvel'] == False]
        posvel_true = paf_data[paf_data['posvel'] == True]
        
        if len(posvel_false) == 0 or len(posvel_true) == 0:
            print(f"{paf_label:15s}: Insufficient data for comparison")
            continue
        
        # Create weighted arrays
        false_values = posvel_false[metric_col].values
        false_weights = posvel_false['weight'].values
        false_weighted = np.repeat(false_values, (false_weights * 10000).astype(int))
        
        true_values = posvel_true[metric_col].values
        true_weights = posvel_true['weight'].values
        true_weighted = np.repeat(true_values, (true_weights * 10000).astype(int))
        
        # Mann-Whitney U test
        try:
            u_stat, p_value = mannwhitneyu(false_weighted, true_weighted, 
                                         alternative='two-sided')
            
            false_median = np.median(false_weighted)
            true_median = np.median(true_weighted)
            effect_size = true_median - false_median
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"{paf_label:15s}: p={p_value:.4f}{significance:3s} | "
                  f"Δmedian={effect_size:+5.1f} | "
                  f"True={true_median:.1f} vs False={false_median:.1f}")
            
        except Exception as e:
            print(f"{paf_label:15s}: Error in statistical test: {e}")
    
    # Test 2: Compare each PAF+posvel combination to reference (Temp Basal + posvel False)
    print(f"\n2. All combinations vs Reference (Temp Basal + RC Mitigation=False):")
    print("-" * 70)
    
    # Get reference data
    if pd.isna(reference_paf) or reference_paf == 'none':
        ref_paf_data = df[df['paf'].isna()]
    else:
        ref_paf_data = df[df['paf'] == reference_paf]
    
    ref_data = ref_paf_data[ref_paf_data['posvel'] == False]
    
    if len(ref_data) == 0:
        print(f"Warning: No reference data found for PAF '{reference_paf}' with posvel=False")
        return
    
    ref_values = ref_data[metric_col].values
    ref_weights = ref_data['weight'].values
    ref_weighted = np.repeat(ref_values, (ref_weights * 10000).astype(int))
    ref_median = np.median(ref_weighted)
    
    for paf in paf_values:
        # Get PAF data
        if pd.isna(paf):
            paf_data = df[df['paf'].isna()]
            paf_label = 'Temp Basal'
        else:
            paf_data = df[df['paf'] == paf]
            paf_label = f'PAF={paf}'
        
        for posvel in [False, True]:
            # Skip reference combination
            if (pd.isna(paf) or paf == reference_paf) and posvel == False:
                continue
                
            test_data = paf_data[paf_data['posvel'] == posvel]
            if len(test_data) == 0:
                continue
                
            test_values = test_data[metric_col].values
            test_weights = test_data['weight'].values
            test_weighted = np.repeat(test_values, (test_weights * 10000).astype(int))
            
            # Mann-Whitney U test
            try:
                u_stat, p_value = mannwhitneyu(ref_weighted, test_weighted, 
                                             alternative='two-sided')
                
                test_median = np.median(test_weighted)
                effect_size = test_median - ref_median
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                posvel_label = 'True' if posvel else 'False'
                print(f"{paf_label:15s} P={posvel_label:5s}: p={p_value:.4f}{significance:3s} | "
                      f"Δmedian={effect_size:+5.1f} | "
                      f"median={test_median:.1f} vs {ref_median:.1f}")
                
            except Exception as e:
                print(f"{paf_label:15s} P={posvel_label:5s}: Error in statistical test: {e}")

def main():
    """Main analysis function."""
    
    # Load data
    print("Loading metrics data...")
    df = pd.read_csv(METRICS_PATH)
    print(f"Loaded {len(df)} simulation results")
    
    # Keep only rows where (paf == 0 and posvel == True) or (paf == 0.4 and posvel == False)
    df = df[
        ((df['paf'].isna()) & (df['posvel'] == True)) |
        ((df['paf'] == 0.4) & (df['posvel'] == True))
    ].copy()

    # Load weights
    print("Loading IBG distribution weights...")
    weights_dict = load_histogram_weights(HISTOGRAM_PATH)
    
    # Filter and weight data
    print("Filtering and weighting data...")
    df_weighted = filter_and_weight_data(df, weights_dict)
    print(f"After filtering: {len(df_weighted)} simulations with valid weights")

    # Get unique PAF values
    paf_values = sorted(df_weighted['paf'].unique())
    print(f"PAF values found: {paf_values}")
    
    # Create boxplot comparison for Time in Range
    print("\nCreating Time in Range comparison...")
    fig1, ax1 = create_weighted_boxplot_comparison(
        df_weighted, 
        metric_col='time_in_range_70_180',
        paf_values=paf_values,
        title='Time in Range (70-180 mg/dL)'
    )
    
    # Perform statistical tests
    # perform_pairwise_statistical_tests(
    #     df_weighted,
    #     metric_col='time_in_range_70_180',
    #     paf_values=paf_values,
    #     reference_paf='none'
    # )
    
    # Additional metrics comparison
    other_metrics = [
        ('time_below_70', 'Time Below Range (<70 mg/dL)'),
        ('time_above_180', 'Time Above Range (>180 mg/dL)'),
        ('mean_glucose', 'Mean Glucose'),
        ('cumulative_insulin', 'Total Insulin Delivered')
    ]

    other_metrics = [
        ('cumulative_insulin', 'Total Insulin Delivered')
    ]
    
    for metric_col, title in other_metrics:
        if metric_col in df_weighted.columns:
            print(f"\nCreating {title} comparison...")
            fig, ax = create_weighted_boxplot_comparison(
                df_weighted,
                metric_col=metric_col,
                paf_values=paf_values,
                title=f'{title}'
            )
            
            # perform_pairwise_statistical_tests(
            #     df_weighted,
            #     metric_col=metric_col,
            #     paf_values=paf_values,
            #     reference_paf='none'
            # )
    fig1.savefig("time_in_range_comparison.png", dpi=300, bbox_inches='tight')
    for metric_col, title in other_metrics:
        if metric_col in df_weighted.columns:
            fig.savefig(f"{metric_col}_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
