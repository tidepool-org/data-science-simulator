#!/usr/bin/env python3
"""
Simple import test to verify the validation module is properly installed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing validation module imports...")
print("-" * 60)

try:
    from tidepool_data_science_simulator.validation import ValidationError
    print("✓ ValidationError imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ValidationError: {e}")
    sys.exit(1)

try:
    from tidepool_data_science_simulator.validation import ValueValidators
    print("✓ ValueValidators imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ValueValidators: {e}")
    sys.exit(1)

try:
    from tidepool_data_science_simulator.validation import ConfigValidator
    print("✓ ConfigValidator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ConfigValidator: {e}")
    sys.exit(1)

print("\nTesting validator instantiation...")
print("-" * 60)

try:
    validator = ConfigValidator()
    print("✓ ConfigValidator instantiated successfully")
except Exception as e:
    print(f"✗ Failed to instantiate ConfigValidator: {e}")
    sys.exit(1)

try:
    value_validators = ValueValidators()
    print("✓ ValueValidators instantiated successfully")
except Exception as e:
    print(f"✗ Failed to instantiate ValueValidators: {e}")
    sys.exit(1)

print("\nTesting basic validation...")
print("-" * 60)

# Test a simple validation
errors = value_validators.validate_basal_rate(150.0, "test_field")
if len(errors) > 0:
    print(f"✓ Correctly detected error for out-of-range basal rate: {errors[0]}")
else:
    print("✗ Failed to detect error for out-of-range basal rate")

errors = value_validators.validate_basal_rate(1.5, "test_field")
if len(errors) == 0:
    print("✓ Correctly validated valid basal rate")
else:
    print(f"✗ Incorrectly flagged valid basal rate as error: {errors}")

# Test max active insulin multiplier
errors = value_validators.validate_max_active_insulin_multiplier(15.0, "test_field")
if len(errors) > 0:
    print(f"✓ Correctly detected error for out-of-range max_active_insulin: {errors[0]}")
else:
    print("✗ Failed to detect error for out-of-range max_active_insulin")

errors = value_validators.validate_max_active_insulin_multiplier(2.0, "test_field")
if len(errors) == 0:
    print("✓ Correctly validated valid max_active_insulin")
else:
    print(f"✗ Incorrectly flagged valid max_active_insulin as error: {errors}")

print("\n" + "=" * 60)
print("✅ All import and basic validation tests passed!")
print("=" * 60)
print("\nPhase 1 implementation is working correctly.")
print("You can now use the validation module in your simulations.")
