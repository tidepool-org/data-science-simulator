# Scenario validation

## Functionality

### Core Validation Module
Created a comprehensive pre-flight validation system in:
`tidepool_data_science_simulator/validation/`

**Components:**

1. **`value_validators.py`**
   - `ValidationError` class for error reporting
   - `ValueValidators` class with validation methods for all configuration parameters
   - Comprehensive validation for:
     - Basic metabolism settings (basal rate, ISF, ICR, target range)
     - Type 2 model parameters (GSF, BBG, IPR)
     - Physical activity entries (duration, intensity, expected_hr)
     - Metabolism parameters (w_hr, a, tau, n)
     - Max active insulin multiplier (NEW - properly validates this critical parameter)
     - Carb and bolus entries
     - Date/time format validation

2. **`config_validator.py`**
   - `ConfigValidator` class for orchestrating validation
   - Structure validation (required fields, types)
   - Recursive value validation through configuration tree
   - Basic reference validation for reusable configurations
   - Directory validation (single file or batch processing)
   - Support for recursive or non-recursive directory scanning

3. **`__init__.py`**
   - Package initialization and exports

### Command-Line Interface
User-friendly CLI script in:
`scripts/validate_configs.py

**Features:**
- Validate single file or entire directories
- Recursive or non-recursive scanning
- Auto-detect reusable configuration directory
- Multiple output modes (normal, verbose, quiet)
- Option to show valid configurations
- Save detailed reports to file
- Proper exit codes (0 = success, 1 = errors found)
- Detailed error statistics and grouping

**Usage Examples:**
```bash
# Validate a directory
python scripts/validate_configs.py --directory ./scenario_configs/loop_risk_v2_0

# Validate single file
python scripts/validate_configs.py --file ./config.json

# Save report
python scripts/validate_configs.py --directory ./configs --output report.txt

# Quick check (summary only)
python scripts/validate_configs.py --directory ./configs --quiet
```

### Testing and Documentation

4. **`test_validator.py`**
   - Test script with intentional errors
   - Validates both error detection and valid config acceptance
   - Demonstrates API usage

5. **`README.md`**
   - Comprehensive documentation
   - Usage examples (CLI and Python API)
   - Integration guide for simulation scripts
   - Troubleshooting section
   - Complete validation coverage details

## Validation Coverage

### What IS Validated ✅
- **Structural integrity**: Required fields, correct types
- **Metabolism settings**: All ranges checked
- **Physical activity**: All parameters including heart rate
- **Max active insulin**: Properly validated (0 < x ≤ 10, warning if ≠ 2.0)
- **Events**: Carb and bolus entries
- **Time formats**: Both full datetime and time-only
- **Target ranges**: Lower < upper, non-negative
- **Duration values**: Simulation, carbs, PA
- **Controller settings**: Including the new max_active_insulin_multiplier
- **Deep reference resolution**: Loads and validates referenced files' contents
- **JSON Schema validation**: Formal schema definitions

### What Is NOT Validated

- **Cross-field dependencies**: E.g., pump settings matching controller expectations
- **Temporal consistency**: Event ordering, overlaps
- **Circular dependencies**: In reusable references

## How to Use

### Quick Start

1. **Test the validator:**
   ```bash
   cd ./data-science-simulator-v2/data-science-simulator
   python scripts/test_validator.py
   ```

2. **Validate your configs:**
   ```bash
   python scripts/validate_configs.py \
       --directory ./scenario_configs/tidepool_risk_v2/loop_risk_v2_0
   ```

3. **Check a specific config:**
   ```bash
   python scripts/validate_configs.py \
       --file ./scenario_configs/tidepool_risk_v2/loop_risk_v2_0/my_scenario.json
   ```

## Testing Recommendations

1. **Run the test script** to verify installation:
   ```bash
   python scripts/test_validator.py
   ```

2. **Try on a small config directory first**:
   ```bash
   python scripts/validate_configs.py --directory ./scenario_configs/tidepool_risk_v2/loop_risk_v2_0/test --no-recursive
   ```

3. **Run on full config set** (may take a minute for large sets):
   ```bash
   python scripts/validate_configs.py --directory ./scenario_configs/tidepool_risk_v2/loop_risk_v2_0 --output validation_report.txt
   ```

4. **Review the report** to see what errors exist in your current configs
