# Insulin Algorithm Testing Framework

A comprehensive framework for comparing insulin delivery algorithms (temp basal vs autobolus) through systematic simulation and statistical analysis.

## Overview

This framework provides tools for:
- Generating comprehensive test scenarios across patient parameters and conditions
- Running batch simulations comparing different insulin delivery algorithms
- Calculating standardized metrics for algorithm performance
- Performing statistical analysis and visualization of results
- Exporting results for further analysis

## Features

### Core Functionality
- **Scenario Generation**: Systematic generation of test scenarios across multiple dimensions
- **Simulation Runner**: Batch execution of simulations with parallel processing support
- **Metrics Calculation**: Standardized glucose control and safety metrics
- **Statistical Analysis**: Comprehensive statistical testing and comparison tools
- **Visualization**: Rich plotting capabilities for results analysis

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

### Basic Comparison Example
```python
from config.experiment_config import ExperimentConfig
from core.data_loader import DataLoader
from core.scenario_generator import ScenarioGenerator
from core.simulation_runner import SimulationRunner
from core.metrics_calculator import MetricsCalculator
from analysis.statistical_analyzer import StatisticalAnalyzer

# Load configuration
config = ExperimentConfig()

# Load patient data
data_loader = DataLoader(config)
patient_configs = data_loader.load_patient_configs(max_patients=5)

# Generate scenarios
scenario_generator = ScenarioGenerator(config)
scenarios = list(scenario_generator.generate_all_scenarios(patient_configs))

# Run simulations
simulation_runner = SimulationRunner(config)
simulations = {}
for scenario in scenarios:
    simulation = simulation_runner.create_simulation_from_scenario(scenario)
    simulations[simulation.sim_id] = simulation

full_results, summary_results = simulation_runner.run_batch_simulations(simulations)

# Calculate metrics
metrics_calculator = MetricsCalculator(config)
metrics_dict = metrics_calculator.calculate_metrics_batch(full_results)
metrics_df = metrics_calculator.create_metrics_dataframe(metrics_dict)

# Statistical analysis
statistical_analyzer = StatisticalAnalyzer(config)
comparison_results = statistical_analyzer.compare_algorithms(
    metrics_df, 
    reference_algorithm='temp_basal',
    comparison_algorithms=['autobolus']
)

print("Analysis completed!")
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
│   └── default_configs.yaml
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── data_loader.py
│   ├── scenario_generator.py
│   ├── simulation_runner.py
│   └── metrics_calculator.py
├── analysis/               # Statistical analysis
│   ├── __init__.py
│   └── statistical_analyzer.py
├── visualization/          # Plotting and visualization
│   ├── __init__.py
│   ├── comparison_plots.py
│   ├── glucose_plots.py
│   └── statistical_plots.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── data_utils.py
│   ├── validation_utils.py
│   ├── export_utils.py
│   └── logging_utils.py
├── examples/               # Example scripts
│   ├── __init__.py
│   ├── basic_comparison.py
│   └── parameter_sweep.py
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_scenario_generator.py
│   ├── test_metrics_calculator.py
│   └── test_statistical_analyzer.py
└── README.md
```

## Usage Examples

### 1. Custom Scenario Generation
```python
from core.scenario_generator import ScenarioGenerator

# Create custom configuration
config = ExperimentConfig()
config.set('scenarios.initial_bg.range', [100, 150])
config.set('scenarios.meal_scenarios.unannounced_meals', [40])

generator = ScenarioGenerator(config)

# Generate scenarios for specific algorithm
autobolus_scenarios = list(generator.generate_scenarios_for_algorithm(
    patient_configs, 'autobolus'
))

# Filter scenarios
filtered_scenarios = list(generator.filter_scenarios_by_criteria(
    autobolus_scenarios, 
    {'initial_bg': 120, 'partial_application_factor': 0.4}
))
```

### 2. Metrics Analysis
```python
from core.metrics_calculator import MetricsCalculator

calculator = MetricsCalculator(config)

# Calculate metrics for single simulation
metrics = calculator.calculate_metrics(simulation_results)

# Batch calculation
metrics_dict = calculator.calculate_metrics_batch(all_results)

# Create summary DataFrame
metrics_df = calculator.create_metrics_dataframe(metrics_dict)

# Custom metric calculation
custom_metrics = calculator.calculate_custom_metrics(
    simulation_results, 
    custom_functions={'custom_tir': lambda bg: np.mean((bg >= 80) & (bg <= 160))}
)
```

### 3. Statistical Analysis
```python
from analysis.statistical_analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(config)

# Compare algorithms
comparison = analyzer.compare_algorithms(
    metrics_df,
    reference_algorithm='temp_basal',
    comparison_algorithms=['autobolus']
)

# Perform paired t-tests
paired_results = analyzer.perform_paired_tests(
    metrics_df,
    algorithm_pairs=[('temp_basal', 'autobolus')],
    metrics=['time_in_range_70_180', 'time_below_70']
)

# Effect size analysis
effect_sizes = analyzer.calculate_effect_sizes(
    metrics_df,
    reference_algorithm='temp_basal',
    comparison_algorithms=['autobolus']
)
```

### 4. Visualization
```python
from visualization.comparison_plots import ComparisonPlotter

plotter = ComparisonPlotter(config)

# Algorithm comparison plots
plotter.plot_algorithm_comparison(
    metrics_df, 
    save_path='algorithm_comparison.png'
)

# Glucose traces
plotter.plot_glucose_traces_sample(
    full_results, 
    n_samples=6,
    save_path='glucose_traces.png'
)

# Paired comparison
plotter.plot_paired_comparison(
    metrics_df,
    reference_algorithm='temp_basal',
    comparison_algorithm='autobolus',
    save_path='paired_comparison.png'
)
```

## Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_scenario_generator.py

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

## Output Files

The framework generates several types of output:

### Results Files
- `metrics.csv`: Detailed metrics for all simulations
- `summary_statistics.json`: Summary statistics by algorithm
- `statistical_tests.json`: Statistical test results
- `comparison_results.json`: Algorithm comparison results

### Visualizations
- `algorithm_comparison.png`: Box plots comparing algorithms
- `glucose_traces.png`: Sample glucose traces
- `paired_comparison.png`: Paired scatter plots
- `parameter_sweep.png`: Parameter sensitivity analysis

### Logs
- `experiment.log`: Detailed execution logs
- `errors.log`: Error and warning messages

## Advanced Usage

### Custom Patient Configurations
```python
# Define custom patient parameters
custom_patients = [
    {
        'patient_id': 'patient_001',
        'weight': 70,
        'isf': 50,
        'cir': 15,
        'basal_rate': 1.0,
        'target_bg': 120
    },
    # ... more patients
]

# Use with framework
scenarios = generator.generate_all_scenarios(custom_patients)
```

### Parallel Processing
```python
# Configure parallel processing
config.set('processing.parallel_processes', 16)
config.set('processing.chunk_size', 50)

# Run simulations in parallel
runner = SimulationRunner(config)
results = runner.run_batch_simulations(simulations)
```

### Custom Metrics
```python
# Define custom metric functions
def custom_time_in_tight_range(glucose_data):
    """Calculate time in tight range (80-140 mg/dL)."""
    return np.mean((glucose_data >= 80) & (glucose_data <= 140)) * 100

# Add to metrics calculator
calculator.add_custom_metric('time_in_tight_range', custom_time_in_tight_range)
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Simulations**
   - Reduce batch size: `config.set('processing.chunk_size', 25)`
   - Use fewer parallel processes: `config.set('processing.parallel_processes', 4)`

2. **Missing Dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy pyyaml
   ```

3. **Configuration Errors**
   - Check YAML syntax in configuration files
   - Validate parameter ranges and types
   - Use `ConfigValidator` for validation

4. **Simulation Failures**
   - Check patient configuration parameters
   - Verify scenario parameters are within valid ranges
   - Review error logs for detailed information

### Performance Optimization

- Use appropriate chunk sizes for your system memory
- Adjust parallel processing based on CPU cores
- Filter scenarios before running large batches
- Use sampling for initial exploration

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

### Version 1.0.0
- Initial release
- Core framework functionality
- Basic examples and tests
- Comprehensive documentation
