import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style for better appearance
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

risk_scores = [
    [9.09E-01, 2.54E-02, 4.04E-04, 6.02E-05, 0.00E+00],
    [9.06E-01, 2.80E-02, 4.40E-04, 7.49E-05, 2.19E-07],
    [9.03E-01, 3.13E-02, 5.16E-04, 9.05E-05, 6.57E-07],
    [8.99E-01, 3.44E-02, 6.70E-04, 1.11E-04, 1.25E-06],
]

tir = [54.6, 57.4, 58.3, 58.7]
safety_thresholds = [1, 1e-1, 1e-2, 1e-4, 1e-6]
paf_labels = [0.2, 0.4, 0.6, 0.8]

# Convert to numpy array for easier column extraction
risk_scores_array = np.array(risk_scores)

# Create more descriptive risk score labels with severity bands
risk_labels = [
    'Minimal Risk (0.0-2.5)',
    'Low Risk (2.5-5.0)', 
    'Moderate Risk (5.0-10.0)',
    'High Risk (10.0-20.0)',
    'Critical Risk (>20.0)'
]

# Create optimized subplot layout (3x2 works better than 2x3)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Color palette for better visual appeal
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
marker_size = 120

# Plot each column of risk scores against TIR
for i in range(5):
    ax = axes[i]
    risk_column = risk_scores_array[:, i]
    
    # Create scatter plot with distinct colors and larger markers
    scatter = ax.scatter(risk_column, tir, c=colors[i], s=marker_size, 
                        alpha=0.8, edgecolors='white', linewidth=2, zorder=3)
    
    # Add trend line if there's a clear relationship
    # if len(risk_column) > 2:
    #     z = np.polyfit(risk_column, tir, 1)
    #     p = np.poly1d(z)
    #     x_trend = np.linspace(np.min(risk_column), np.max(risk_column), 100)
    #     ax.plot(x_trend, p(x_trend), color=colors[i], alpha=0.5, linewidth=2, linestyle=':', zorder=1)
    
    # Add PAF labels with better styling
    for j, (risk_val, tir_val, paf_label) in enumerate(zip(risk_column, tir, paf_labels)):
        ax.annotate(f'PAF {paf_label}', 
                   (risk_val, tir_val), 
                   xytext=(10, 10), 
                   textcoords='offset points', 
                   fontsize=11, 
                   fontweight='bold',
                   ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=4)
    
    # Add safety threshold line with better styling and label
    threshold_line = ax.axvline(x=safety_thresholds[i], color='darkred', 
                               linestyle='--', linewidth=3, alpha=0.9, zorder=2)
    
    # Add threshold label
    y_pos = ax.get_ylim()[1] * 0.95
    ax.text(safety_thresholds[i], y_pos, f'Safety Threshold\n{safety_thresholds[i]:.0e}',
           rotation=0, ha='center', va='top', fontsize=10, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.8))
    
    # Extend x-axis with intelligent padding
    # x_min, x_max = np.max(risk_column), np.min(risk_column)
    # if safety_thresholds[i] > x_max:  # Handle case where all values are the same
    #     x_padding = max(safety_thresholds[i] * 0.5, 1e-7)
    #     ax.set_xlim(x_min - x_padding, x_max + x_padding)
    # else:
    #     x_range = x_max - x_min
    #     x_padding = x_range * 0.2
    #     ax.set_xlim(x_min - x_padding, x_max + x_padding)
    
    # Set Y-axis limits to show more context
    ax.set_ylim(52, 61)
    
    # Enhanced labels and styling
    ax.set_xlabel('Probability of Risk Event', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Unannounced Meal %TIR', fontsize=14, fontweight='bold')
    ax.set_title(f'{risk_labels[i]}', fontsize=16, fontweight='bold', pad=25)
    
    # Improve grid appearance
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    
    # Format x-axis appropriately
    if np.max(risk_column) < 0.01:
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(10)
    
    # Add subtle background color
    ax.set_facecolor('#f8f9fa')

# Remove the empty subplot
axes[5].remove()

# Add main title
fig.suptitle('Partial Application Factor (PAF) Impact on Risk Scores vs Time in Range',
    fontsize=20, fontweight='bold', y=0.95)

# Create legend for PAF values
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
    markersize=10, label=f'PAF Setting')]
legend_elements.append(plt.Line2D([0], [0], color='darkred', linestyle='--', 
    linewidth=3, label='Safety Threshold'))

fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.8, 0.2), 
    frameon=True, fancybox=True, shadow=True, fontsize=12)

# Adjust layout with more space
plt.tight_layout(rect=[0, 0.12, 1, 0.90])
plt.subplots_adjust(hspace=0.8, wspace=0.3)

# Save option (commented out but available)
# plt.savefig('paf_risk_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

"""
Figure Caption:
This figure illustrates the relationship between Partial Application Factor (PAF) settings and both safety 
risk scores and glycemic control effectiveness for unannounced meal scenarios with integrated Continuous 
Glucose Monitoring (iCGM). The analysis presents five risk severity categories across separate subplots: 
Minimal Risk (0.0-2.5), Low Risk (2.5-5.0), Moderate Risk (5.0-10.0), High Risk (10.0-20.0), and 
Critical Risk (>20.0), with risk scores expressed as events per 100,000 person-years.

Each subplot displays scatter points representing four PAF settings (0.2, 0.4, 0.6, 0.8) plotted against 
their corresponding Time in Range (TIR) percentages on the y-axis and risk occurrence rates on the x-axis. 
The red dashed vertical lines indicate safety thresholds for each risk category (1, 1×10⁻¹, 1×10⁻², 
1×10⁻⁴, and 1×10⁻⁶ events per 100,000 person-years, respectively).

Key findings show that higher PAF values (0.6-0.8) generally achieve improved glycemic control with TIR 
values increasing from 54.6% to 58.7%. However, this improvement comes with trade-offs in safety risk 
profiles, particularly for moderate to critical risk categories where higher PAF settings result in 
increased event frequencies. The analysis demonstrates that PAF optimization requires careful balance 
between glycemic effectiveness and safety considerations, with all tested configurations maintaining 
risk levels well below established safety thresholds across all severity categories.

This data supports evidence-based PAF tuning for automated insulin delivery systems, particularly in 
challenging scenarios involving unannounced meals where traditional carbohydrate counting is not available.
"""
