# iCGM Regression Test Baseline Data

This directory contains baseline data for regression testing the iCGM analysis pipeline. These tests ensure that code changes don't unexpectedly alter simulation outputs.

## Baseline Configuration

- **Parameters:**
  - PAF (Partial Application Factor): 0.4
  - Positive RC: False
  - Virtual Patients: 3
  - True BG values: [40, 50, 60, 70, 80] (5 values)
  - Sensor BG values: [45, 55, 65, 75, 85] (5 values)
  - Total simulations: 75 (5×5×3)

- **Files:**
  - `baseline_summary.csv` - Summary DataFrame with all simulation results (75 rows)
  - `baseline_risk_table.csv` - Risk table with 5 severity bands
  - `baseline_aux_data.npz` - Auxiliary numpy arrays (axes, probabilities)
  - `test_params.json` - Test configuration parameters

## Workflow

### Running Regression Tests

```bash
# Run all regression tests
pytest tests/test_icgm_regression.py -v

# Run specific test
pytest tests/test_icgm_regression.py::test_regression_summary_df -v
```

**Expected Result:** All tests should pass if the code hasn't changed the outputs.

### Updating Baseline Data

When you intentionally change code that affects outputs:

1. **Remove the skip decorator** in `tests/test_icgm_regression.py`:
   ```python
   # Comment out or remove this line:
   # @pytest.mark.skip(reason="Manual baseline generation - remove skip to regenerate")
   def test_generate_baseline():
       ...
   ```

2. **Generate new baseline:**
   ```bash
   pytest tests/test_icgm_regression.py::test_generate_baseline -v -s
   ```

3. **Re-add the skip decorator** to prevent accidental regeneration:
   ```python
   @pytest.mark.skip(reason="Manual baseline generation - remove skip to regenerate")
   def test_generate_baseline():
       ...
   ```

4. **Verify the new baseline:**
   ```bash
   pytest tests/test_icgm_regression.py -v
   ```

5. **Commit with a clear message:**
   ```bash
   git add tests/test_data/regression/icgm/
   git commit -m "Update iCGM regression baseline: [reason for change]"
   ```

## What Gets Tested

### 1. Summary DataFrame (`test_regression_summary_df`)
Compares all key metrics from simulation results:
- LBGI values (start and valid)
- Blood glucose values (true and sensor)
- Bolus delivery (max, traditional, difference)
- Patient parameters (SBR, ISF, CIR)

**Tolerance:** 1e-10 (strict exact matching)

### 2. Risk Table (`test_regression_risk_table`)
Compares the 5-band severity risk table:
- Probability distribution across severity bands
- Derived from LBGI calculations

**Tolerance:** 1e-10 (strict exact matching)

### 3. Auxiliary Data (`test_regression_auxiliary_data`)
Compares auxiliary arrays:
- `low_icgm_axis` - iCGM BG axis values
- `low_true_axis` - True BG axis values
- `mean_lbgi_start` - Mean LBGI probabilities per severity band
- `joint_prob` - Joint probability distribution

**Tolerance:** 1e-10 (strict exact matching)

## Troubleshooting

### Test Fails After Code Change

1. **Review the failure message** - it will show which metrics differ
2. **Determine if change is intentional:**
   - ✅ If expected: Update baseline (see "Updating Baseline Data")
   - ❌ If unexpected: Fix the bug causing the change

### Baseline Data Missing

If you get: `Baseline data not found. Run test_generate_baseline first.`

1. Generate baseline data (see "Updating Baseline Data" section)
2. Or restore from git if accidentally deleted:
   ```bash
   git checkout tests/test_data/regression/icgm/
   ```

### Floating Point Differences

If you see very small differences (< 1e-10):
- This may be due to platform/compiler differences
- Consider adjusting `TOLERANCE` in `test_icgm_regression.py` if necessary
- Document the reason in commit message

## Best Practices

1. **Always review changes before updating baseline**
   - Compare old vs new baseline files
   - Understand why outputs changed

2. **Keep baseline data small but comprehensive**
   - Current: 75 simulations (fast enough, good coverage)
   - Avoid making it too large (slow tests)

3. **Clear commit messages**
   - Explain why baseline was updated
   - Reference related code changes

4. **Run tests before pushing**
   ```bash
   pytest tests/test_icgm_regression.py -v
   ```

5. **Keep baseline in sync with code**
   - Update baseline in the same PR as code changes
   - Don't commit broken baselines

## Notes

- Tests use **fixed random seeds** for determinism
- Tests run in **temporary directories** (no cleanup needed)
- Tests take approximately **2-3 minutes** to run
- Baseline generation takes approximately **3-5 minutes**
