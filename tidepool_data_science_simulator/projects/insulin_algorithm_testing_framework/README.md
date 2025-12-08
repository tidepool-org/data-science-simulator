# Insulin Algorithm Testing Framework

A comprehensive framework for comparing insulin delivery algorithms (temp basal vs autobolus) through systematic simulation and statistical analysis.

## Overview

This framework provides tools for:
- Generating comprehensive test scenarios across patient parameters and conditions
- Running batch simulations comparing different insulin delivery algorithms
- Calculating standardized metrics for algorithm performance
- Performing statistical analysis and visualization of results
- Exporting results for regulatory submissions (FDA 510k)

## Features

### Core Functionality
- **Simulation Building**: Functional API for creating simulations from scenario configurations
- **iCGM Scenario Generation**: Grid-based generation of true BG × sensor BG scenarios
- **Metrics Calculation**: Standardized glucose control and safety metrics
- **Risk Scoring**: LBGI-based risk analysis with severity bands
- **Statistical Analysis**: Comprehensive statistical testing and comparison tools
- **Visualization**: Regulatory-compliant plotting capabilities

### Supported Algorithms
- **Temp Basal**: Traditional temporary basal rate adjustments
- **Autobolus**: Automated micro-bolus delivery with configurable partial application factors

### Key Metrics
- Time in Range (70-180 mg/dL, 70-140 mg/dL)
- Hypoglycemia metrics (time <70, <54 mg/dL)
- Hyperglycemia metrics (time >180, >250 mg/dL)
- Glucose variability (CV, standard deviation)
- Insulin delivery metrics
- Risk indices (LBGI, HBGI, BGRI)

## Installation

### Prerequisites
- Python 3.8+
- Required packages: numpy, pandas, matplotlib, seaborn, scipy, pyyaml

### Setup
```bash
# Clone the repository
git clone <repository_url>

# Navigate to the framework directory
cd tidepool_data_science_simulator/projects/insulin_algorithm_testing_framework

# Install dependencies (if using pip)
pip install numpy pandas matplotlib seaborn scipy pyyaml

# Or using conda
conda install numpy pandas matplotlib seaborn scipy pyyaml
```

## Quick Start

### Basic Usage Example
```python
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework import (
    ExperimentConfig,
    DataLoader,
    build_simulations,
    generate_simulations,
    calculate_point_metrics,
    metrics_to_dataframe
)
from tidepool_data_science_simulator.run import run_simulations

# Load configuration
config = ExperimentConfig('config/default_configs.yaml')

# Load patient data
data_loader = DataLoader(config)
patient_configs = data_loader.load_patient_configs(max_patients=5)

# Option 1: Build simulations from scenario dictionaries
scenarios = [
    {
        'algorithm_type': 'autobolus',
        'patient_config': patient_configs[0],
        'true_start_bg': 120,
        'partial_application_factor': 0.4,
        'gradual_transition_threshold': 50.0
    }
]
simulations = build_simulations(config, scenarios)

# Option 2: Generate iCGM scenarios directly (more efficient for large grids)
sim_generator, num_sims = generate_simulations(
    config,
    patient_configs,
    true_bg_range=(40, 405, 5),
    sensor_bg_range=(40, 405, 5)
)

# Run simulations
full_results, summary_results = run_simulations(
    sim_generator,
    save_dir='./results/simulation_results',
    save_results=True,
    num_procs=8,
    num_sims=num_sims
)

print("Simulations completed!")
```

### Running the iCGM 510k Analysis
```bash
# Full production run
python experiments/run_icgm_510k_analysis.py

# Quick test to validate pipeline
python experiments/run_icgm_510k_analysis.py --quick-test

# Custom config and output directory
python experiments/run_icgm_510k_analysis.py \
    --config config/510k_configs/icgm_sensitivity_analysis.yaml \
    --output-dir results/my_analysis
```

### Running Examples
```bash
# Basic comparison example
python examples/basic_comparison.py

# Parameter sweep analysis
python examples/parameter_sweep.py
```

## Configuration

The framework uses YAML configuration files for flexible parameter management:

```yaml
# config/default_configs.yaml
experiment:
  name: "insulin_algorithm_comparison"
  description: "Comparison of temp basal vs autobolus algorithms"
  output_dir: "./results"

scenarios:
  initial_bg:
    range: [80, 200]
    step: 20
  
  meal_scenarios:
    unannounced_meals: [20, 40, 60, 80]
    meal_timing: 120  # minutes after start
    absorption_time: 180  # minutes
  
  settings_mismatches:
    multipliers: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    apply_to: ["isf", "cir", "basal"]

algorithms:
  temp_basal:
    enabled: true
  
  autobolus:
    enabled: true
    partial_application_factors: [0.2, 0.3, 0.4, 0.5, 0.6]

processing:
  parallel_processes: 8
  chunk_size: 100
```

## Framework Structure

```
insulin_algorithm_testing_framework/
├── config/                 # Configuration management
│   ├── __init__.py
│   ├── experiment_config.py
│   ├── default_configs.yaml
│   └── 510k_configs/       # FDA 510k specific configs
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── data_loader.py      # Patient data loading
│   ├── simulation_builder.py  # Functional simulation building
│   ├── metrics_calculator.py  # Metrics calculation
│   └── risk_scoring.py     # LBGI-based risk analysis
├── analysis/               # Statistical analysis
│   ├── __init__.py
│   ├── statistical_analyzer.py
│   └── weighted_metrics.py
├── visualization/          # Plotting and visualization
│   ├── __init__.py
│   ├── comparison_plots.py
│   ├── regulatory_plots.py
│   └── plot_metrics_*.py
├── experiments/            # Experiment scripts
│   ├── __init__.py
│   ├── main_experiment.py
│   └── run_icgm_510k_analysis.py  # Turnkey 510k analysis
├── examples/               # Example scripts
│   ├── __init__.py
│   ├── basic_comparison.py
│   └── parameter_sweep.py
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_icgm_framework_comparison.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── data_utils.py
└── README.md
```

## API Reference

### Simulation Building Functions

```python
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework import (
    build_simulation,      # Build single simulation from scenario dict
    build_simulations,     # Build multiple simulations from scenarios
    generate_simulations,  # Generate iCGM grid simulations directly
    count_simulations      # Calculate total simulation count
)

# Build a single simulation
simulation = build_simulation(config, scenario_dict)

# Build multiple simulations from scenario dictionaries
simulations = build_simulations(config, scenarios)

# Generate iCGM simulations directly (returns generator + count)
sim_generator, num_sims = generate_simulations(
    config,
    patient_configs,
    true_bg_range=(40, 405, 5),
    sensor_bg_range=(40, 405, 5),
    algorithm='autobolus'
)
```

### Metrics Calculation Functions

```python
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework import (
    calculate_point_metrics,  # Calculate metrics for single simulation
    calculate_metrics_batch,  # Calculate metrics for multiple simulations
    metrics_to_dataframe,     # Convert metrics dict to DataFrame
    PointMetrics             # Dataclass containing all metrics
)

# Calculate metrics for a single simulation result
metrics = calculate_point_metrics(results_df, start_hours=0, duration_hours=8)

# Batch calculation
point_metrics, timeseries = calculate_metrics_batch(results_dict)

# Convert to DataFrame for analysis
metrics_df = metrics_to_dataframe(point_metrics, parse_sim_ids=True)
```

### Statistical Analysis

```python
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.analysis.statistical_analyzer import (
    StatisticalAnalyzer
)

analyzer = StatisticalAnalyzer(config)

# Compare algorithms
comparison = analyzer.compare_algorithms(
    metrics_df,
    reference_algorithm='tempbasal',
    comparison_algorithms=['autobolus']
)

# Non-inferiority analysis for safety metrics
ni_results = analyzer.perform_non_inferiority_analysis(
    reference_metrics,
    comparison_metrics,
    safety_metrics=['time_below_70', 'time_below_54', 'lbgi']
)
```

## Usage Examples

### 1. iCGM Sensitivity Analysis
```python
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework import (
    ExperimentConfig, DataLoader, generate_simulations, 
    calculate_point_metrics, metrics_to_dataframe
)
from tidepool_data_science_simulator.run import run_simulations
import pandas as pd

# Load configuration and patients
config = ExperimentConfig('config/510k_configs/icgm_sensitivity_analysis.yaml')
data_loader = DataLoader(config)
patient_configs = data_loader.load_patient_configs()

# Generate iCGM grid simulations
sim_generator, num_sims = generate_simulations(
    config,
    patient_configs,
    true_bg_range=(40, 405, 5),
    sensor_bg_range=(40, 405, 5)
)

# Run simulations with parallel processing
run_simulations(
    sim_generator,
    save_dir='./results/simulation_results',
    save_results=True,
    num_procs=8,
    num_sims=num_sims
)

# Calculate metrics from saved results
from pathlib import Path
results_dir = Path('./results/simulation_results')
point_metrics_dict = {}

for tsv_file in results_dir.glob("*.tsv"):
    sim_id = tsv_file.stem
    results_df = pd.read_csv(tsv_file, sep='\t')
    point_metrics_dict[sim_id] = calculate_point_metrics(results_df)

# Create summary DataFrame
summary_df = metrics_to_dataframe(point_metrics_dict, parse_sim_ids=True)
summary_df.to_csv('./results/simulation_summary.csv', index=False)
```

### 2. Risk Scoring Analysis
```python
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.risk_scoring import (
    analyze_icgm_risk, generate_risk_report
)

# Perform risk analysis
severity_df, analysis_arrays, report = analyze_icgm_risk(
    summary_df,
    population_type='adult'
)

# Save results
severity_df.to_csv('./results/risk_severity_analysis.csv', index=False)
print(report)
```

### 3. Regulatory Visualization
```python
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.visualization.regulatory_plots import (
    plot_risk_heatmap_grid, save_regulatory_figure
)

# Create risk heatmap grid
fig, axes = plot_risk_heatmap_grid(
    analysis_arrays['true_bg'],
    analysis_arrays['sensor_bg'],
    risk_data_dict,
    severity_bands,
    shared_z_scale=True
)

# Save in regulatory formats
save_regulatory_figure(fig, './results/risk_heatmap', dpi=300, formats=['png', 'pdf'])
```

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_icgm_framework_comparison.py

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Output Files

The framework generates several types of output:

### Results Files
- `simulation_summary.csv`: Detailed metrics for all simulations
- `risk_severity_analysis.csv`: Risk scores by severity band
- `risk_analysis_report.txt`: Human-readable risk report
- `scenario_summary.json`: Scenario generation metadata

### Visualizations
- `risk_heatmap_grid.png/pdf`: Risk heatmaps by severity band
- `algorithm_comparison.png`: Box plots comparing algorithms
- `glucose_traces.png`: Sample glucose traces

### Submission Package
For FDA 510k submissions, the `run_icgm_510k_analysis.py` script creates a ready-to-submit package:
```
submission_package/
├── risk_severity_analysis.csv
├── risk_analysis_report.txt
├── scenario_summary.json
├── risk_heatmap_grid.png
└── risk_heatmap_grid.pdf
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Simulations**
   - Reduce batch size: `config.set('processing.batch_size', 25)`
   - Use fewer parallel processes: `config.set('processing.parallel_processes', 4)`

2. **Missing Dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy pyyaml
   ```

3. **Configuration Errors**
   - Check YAML syntax in configuration files
   - Validate parameter ranges and types

4. **Simulation Failures**
   - Check patient configuration parameters
   - Verify scenario parameters are within valid ranges
   - Review error logs for detailed information

### Performance Optimization

- Use `generate_simulations()` for large iCGM grids (avoids creating scenario dicts)
- Adjust parallel processing based on CPU cores
- Filter scenarios before running large batches
- Use the `--quick-test` flag for initial validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```
Tidepool Data Science Team. (2024). Insulin Algorithm Testing Framework. 
GitHub repository: https://github.com/tidepool-org/data-science-simulator
```

## Support

For questions and support:
- Create an issue on GitHub
- Contact the Tidepool Data Science team
- Check the documentation and examples

## Changelog

### Version 1.1.0
- Refactored to functional API (removed class-based ScenarioGenerator and SimulationRunner)
- Added direct simulation generation with `generate_simulations()`
- Added turnkey `run_icgm_510k_analysis.py` script for regulatory submissions
- Improved metrics calculation with `PointMetrics` dataclass
- Added risk scoring module for LBGI-based analysis

### Version 1.0.0
- Initial release
- Core framework functionality
- Basic examples and tests
