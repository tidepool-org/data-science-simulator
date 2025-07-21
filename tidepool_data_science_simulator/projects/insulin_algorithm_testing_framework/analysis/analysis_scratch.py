from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from collections import defaultdict

# Column names for the metrics CSV file
# ----
# simulation_id,time_in_range_70_180,time_below_70,time_below_54,time_above_180,time_above_250,
# mean_glucose,median_glucose,std_glucose,cv_glucose,lbgi,hbgi,bgri,lbgi_risk_score,
# cumulative_insulin,basal_insulin,bolus_insulin,glucose_management_indicator,time_in_tight_range_70_140,
# coefficient_of_variation,alg,patient,ibg,meal,isf,cir,basal,paf
# ----
METRICS_PATH = '/Users/mconn/Downloads/basic_comparison_metrics.csv'

HISTOGRAM_PATH = Path("/Users/mconn/Downloads/BG_Distribution_Histogram.csv")

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
    """Create boxplots comparing metrics across PAF values with weighted data."""
    
    if paf_values is None:
        paf_values = sorted(df['paf'].unique())
    
    if title is None:
        title = f"Comparison of {metric_col.replace('_', ' ').title()} by PAF"
    
    # Prepare data for boxplot
    box_data = []
    labels = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(paf_values)))
    
    print(f"\nStatistical Summary for {metric_col}:")
    print("-" * 60)
    
    for paf in paf_values:
        
        if np.isnan(paf):
            paf_data = df[df['paf'].isna()]
        else:
            paf_data = df[df['paf'] == paf]
        
        if len(paf_data) == 0:
            continue
            
        values = paf_data[metric_col].values
        weights = paf_data['weight'].values
        
        # Create weighted data for boxplot by replicating values based on weights
        # Scale weights to reasonable integers for replication
        scaled_weights = (weights * 10000).astype(int)
        weighted_values = np.repeat(values, scaled_weights)
        
        box_data.append(weighted_values)
        
        # Label formatting
        if paf == 'none':
            label = 'Temp Basal'
        else:
            label = f'PAF={paf}'
        labels.append(label)
        
        # Calculate and print statistics
        mean, std = weighted_mean_std(values, weights)
        median = weighted_percentile(values, weights, [50])[0]
        q25 = weighted_percentile(values, weights, [25])[0]
        q75 = weighted_percentile(values, weights, [75])[0]
        
        print(f"{label:15s}: Mean={mean:.1f}±{std:.1f}, Median={median:.1f} (IQR: {q25:.1f}-{q75:.1f}), n={len(values)}")
    
    # Create the boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box_plot = ax.boxplot(box_data, labels=labels, patch_artist=True, 
                         showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Style the plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_col.replace('_', ' ').title() + ' (%)', fontsize=12)
    ax.set_xlabel('Algorithm Configuration', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if there are many PAF values
    if len(labels) > 5:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig, ax

def perform_pairwise_statistical_tests(df, metric_col='time_in_range_70_180', 
                                     paf_values=None, reference_paf='none'):
    """Perform pairwise statistical tests comparing each PAF to reference."""
    
    if paf_values is None:
        paf_values = sorted(df['paf'].unique())
    
    # Get reference data (temp basal)
    ref_data = df[df['paf'] == reference_paf]
    if len(ref_data) == 0:
        print(f"Warning: No data found for reference PAF '{reference_paf}'")
        return
    
    ref_values = ref_data[metric_col].values
    ref_weights = ref_data['weight'].values
    ref_weighted = np.repeat(ref_values, (ref_weights * 10000).astype(int))
    
    print(f"\nPairwise Statistical Tests vs {reference_paf}:")
    print("-" * 50)
    
    for paf in paf_values:
        if paf == reference_paf:
            continue
            
        test_data = df[df['paf'] == paf]
        if len(test_data) == 0:
            continue
            
        test_values = test_data[metric_col].values
        test_weights = test_data['weight'].values
        test_weighted = np.repeat(test_values, (test_weights * 10000).astype(int))
        
        # Mann-Whitney U test
        try:
            u_stat, p_value = mannwhitneyu(ref_weighted, test_weighted, 
                                         alternative='two-sided')
            
            ref_median = np.median(ref_weighted)
            test_median = np.median(test_weighted)
            effect_size = test_median - ref_median
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"PAF={paf:>4s}: p={p_value:.4f}{significance:3s} | "
                  f"Δmedian={effect_size:+5.1f} | "
                  f"median={test_median:.1f} vs {ref_median:.1f}")
            
        except Exception as e:
            print(f"PAF={paf}: Error in statistical test: {e}")

def main():
    """Main analysis function."""
    
    # Load data
    print("Loading metrics data...")
    df = pd.read_csv(METRICS_PATH)
    print(f"Loaded {len(df)} simulation results")
    
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
        title='Time in Range (70-180 mg/dL) by PAF Value'
    )
    
    # Perform statistical tests
    perform_pairwise_statistical_tests(
        df_weighted,
        metric_col='time_in_range_70_180',
        paf_values=paf_values,
        reference_paf='none'
    )
    
    # Additional metrics comparison
    other_metrics = [
        ('time_below_70', 'Time Below Range (<70 mg/dL)'),
        ('time_above_180', 'Time Above Range (>180 mg/dL)'),
        ('mean_glucose', 'Mean Glucose'),
        ('cumulative_insulin', 'Total Insulin Delivered')
    ]
    
    for metric_col, title in other_metrics:
        if metric_col in df_weighted.columns:
            print(f"\nCreating {title} comparison...")
            fig, ax = create_weighted_boxplot_comparison(
                df_weighted,
                metric_col=metric_col,
                paf_values=paf_values,
                title=f'{title} by PAF Value'
            )
            
            perform_pairwise_statistical_tests(
                df_weighted,
                metric_col=metric_col,
                paf_values=paf_values,
                reference_paf='none'
            )
    
    plt.show()

if __name__ == "__main__":
    main()
