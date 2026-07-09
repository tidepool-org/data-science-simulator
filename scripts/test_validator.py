"""
Quick test of the validation module.

This creates a test configuration with known errors to verify the validator works.
"""

import json
import tempfile
import os
from pathlib import Path

# Import the validation module
from tidepool_data_science_simulator.validation import ConfigValidator, ValidationError

def create_test_config_with_errors():
    """Create a test config with various types of errors"""
    config = {
        "metadata": {
            "simulation_id": "test_validation",
            "description": "Test config with intentional errors"
        },
        "base_config": {
            "sim_id": "test_sim",
            "time_to_calculate_at": "08/15/2019 12:00:00",
            "duration_hours": 8.0,
            "patient": {
                "pump": {
                    "metabolism_settings": {
                        "basal_rate": {
                            "start_times": ["00:00:00"],
                            "values": [150.0]  # ERROR: Out of range (max 100)
                        },
                        "carb_insulin_ratio": {
                            "start_times": ["00:00:00"],
                            "values": [10.0]
                        },
                        "insulin_sensitivity_factor": {
                            "start_times": ["00:00:00"],
                            "values": [50.0]
                        }
                    },
                    "target_range": {
                        "start_times": ["00:00:00"],
                        "lower_values": [100.0],
                        "upper_values": [120.0]
                    },
                    "carb_entries": [],
                    "bolus_entries": []
                },
                "patient_model": {
                    "metabolism_settings": {
                        "basal_rate": {
                            "start_times": ["00:00:00"],
                            "values": [1.0]
                        },
                        "carb_insulin_ratio": {
                            "start_times": ["00:00:00"],
                            "values": [10.0]
                        },
                        "insulin_sensitivity_factor": {
                            "start_times": ["00:00:00"],
                            "values": [50.0]
                        }
                    },
                    "glucose_history": {
                        "datetime": {"0": "08/15/2019 11:00:00"},
                        "value": {"0": 120.0}
                    },
                    "carb_entries": [
                        {
                            "start_time": "08/15/2019 12:00:00",
                            "value": 600  # ERROR: Out of range (max 500)
                        }
                    ],
                    "bolus_entries": [],
                    "physical_activity_entries": [
                        {
                            "start_time": "08/15/2019 13:00:00",
                            "activity": "running",
                            "duration": 30,
                            "expected_hr": 250  # ERROR: Out of range (max 220)
                        }
                    ],
                    "w_hr": 0.5,
                    "a": 1.0,
                    "tau": 60.0,
                    "n": 1.0
                },
                "sensor": {
                    "glucose_history": {
                        "datetime": {"0": "08/15/2019 11:00:00"},
                        "value": {"0": 120.0}
                    }
                }
            },
            "controller": {
                "id": "py_loop",
                "settings": {
                    "model": "rapid_acting_adult",
                    "dynamic_carb_absorption_enabled": True,
                    "retrospective_correction_integration_interval": 30,
                    "recency_interval": 15,
                    "retrospective_correction_grouping_interval": 30,
                    "insulin_delay": 10,
                    "carb_delay": 10,
                    "default_absorption_times": [120, 180, 240],
                    "max_basal_rate": 6.0,
                    "max_bolus": 10.0,
                    "max_active_insulin_multiplier": 15.0  # ERROR: Out of range (max 10)
                }
            }
        },
        "override_config": [
            {}
        ]
    }
    return config


def create_test_config_valid():
    """Create a valid test config"""
    config = {
        "metadata": {
            "simulation_id": "test_validation_valid",
            "description": "Valid test config"
        },
        "base_config": {
            "sim_id": "test_sim_valid",
            "time_to_calculate_at": "08/15/2019 12:00:00",
            "duration_hours": 8.0,
            "patient": {
                "pump": {
                    "metabolism_settings": {
                        "basal_rate": {
                            "start_times": ["00:00:00"],
                            "values": [1.0]  # Valid
                        },
                        "carb_insulin_ratio": {
                            "start_times": ["00:00:00"],
                            "values": [10.0]
                        },
                        "insulin_sensitivity_factor": {
                            "start_times": ["00:00:00"],
                            "values": [50.0]
                        }
                    },
                    "target_range": {
                        "start_times": ["00:00:00"],
                        "lower_values": [100.0],
                        "upper_values": [120.0]
                    },
                    "carb_entries": [],
                    "bolus_entries": []
                },
                "patient_model": {
                    "metabolism_settings": {
                        "basal_rate": {
                            "start_times": ["00:00:00"],
                            "values": [1.0]
                        },
                        "carb_insulin_ratio": {
                            "start_times": ["00:00:00"],
                            "values": [10.0]
                        },
                        "insulin_sensitivity_factor": {
                            "start_times": ["00:00:00"],
                            "values": [50.0]
                        }
                    },
                    "glucose_history": {
                        "datetime": {"0": "08/15/2019 11:00:00"},
                        "value": {"0": 120.0}
                    },
                    "carb_entries": [],
                    "bolus_entries": [],
                    "physical_activity_entries": [],
                    "w_hr": 0.5,
                    "a": 1.0,
                    "tau": 60.0,
                    "n": 1.0
                },
                "sensor": {
                    "glucose_history": {
                        "datetime": {"0": "08/15/2019 11:00:00"},
                        "value": {"0": 120.0}
                    }
                }
            },
            "controller": {
                "id": "py_loop",
                "settings": {
                    "model": "rapid_acting_adult",
                    "dynamic_carb_absorption_enabled": True,
                    "retrospective_correction_integration_interval": 30,
                    "recency_interval": 15,
                    "retrospective_correction_grouping_interval": 30,
                    "insulin_delay": 10,
                    "carb_delay": 10,
                    "default_absorption_times": [120, 180, 240],
                    "max_basal_rate": 6.0,
                    "max_bolus": 10.0,
                    "max_active_insulin_multiplier": 2.0  # Valid
                }
            }
        },
        "override_config": [
            {}
        ]
    }
    return config


def create_test_config_pump_sentinel():
    """
    Config that places the ``accept_recommendation`` sentinel on the pump's
    bolus_entries. The patient acceptance logic only resolves sentinels on
    ``patient.patient_model.bolus_entries``; placing it on the pump leaks the
    literal string into the Loop input JSON and crashes the Swift bridge.
    The validator must reject this placement.
    """
    config = create_test_config_valid()
    config["metadata"]["simulation_id"] = "test_pump_sentinel"
    config["base_config"]["patient"]["pump"]["bolus_entries"] = [
        {"time": "08/15/2019 12:00:00", "value": "accept_recommendation"}
    ]
    return config


def create_test_config_patient_model_sentinel():
    """
    Config that places the sentinel on patient_model — the only valid slot.
    Must validate cleanly.
    """
    config = create_test_config_valid()
    config["metadata"]["simulation_id"] = "test_patient_model_sentinel"
    config["base_config"]["patient"]["patient_model"]["bolus_entries"] = [
        {"time": "08/15/2019 12:00:00", "value": "accept_recommendation"}
    ]
    return config


def test_validator():
    """Test the configuration validator"""
    
    print("=" * 80)
    print("Testing Configuration Validator")
    print("=" * 80 + "\n")
    
    # Create temporary directory for test configs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test config with errors
        error_config_path = os.path.join(temp_dir, "config_with_errors.json")
        with open(error_config_path, 'w') as f:
            json.dump(create_test_config_with_errors(), f, indent=2)
        
        # Create valid test config
        valid_config_path = os.path.join(temp_dir, "config_valid.json")
        with open(valid_config_path, 'w') as f:
            json.dump(create_test_config_valid(), f, indent=2)
        
        # Create validator
        validator = ConfigValidator()
        
        # Test 1: Validate config with errors
        print("Test 1: Validating config with intentional errors")
        print("-" * 80)
        is_valid, errors, warnings = validator.validate_config_file(error_config_path)
        
        if warnings:
            print(f"  {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"  {warning}")
            print()

        if not is_valid:
            print(f"✓ Correctly detected {len(errors)} errors:")
            for error in errors:
                print(f"  {error}")
            print()
        else:
            print("✗ FAILED: Should have detected errors but validation passed!")
            print()
        
        # Test 2: Validate valid config
        print("Test 2: Validating valid config")
        print("-" * 80)
        is_valid, errors, warnings = validator.validate_config_file(valid_config_path)
        
        if is_valid:
            print("✓ Correctly validated as valid (no errors)")
            if warnings:
                print(f"  {len(warnings)} warning(s):")
                for warning in warnings:
                    print(f"  {warning}")
            print()
        else:
            print(f"✗ FAILED: Valid config reported as invalid with {len(errors)} errors:")
            for error in errors:
                print(f"  {error}")
            print()
        
        # Test 3: Sentinel on pump bolus_entries must be rejected
        print("Test 3: Rejecting 'accept_recommendation' on pump.bolus_entries")
        print("-" * 80)
        pump_sentinel_path = os.path.join(temp_dir, "config_pump_sentinel.json")
        with open(pump_sentinel_path, 'w') as f:
            json.dump(create_test_config_pump_sentinel(), f, indent=2)

        is_valid, errors, warnings = validator.validate_config_file(pump_sentinel_path)
        sentinel_errors = [
            e for e in errors
            if "accept_recommendation" in e.error_message
            and "pump.bolus_entries" in e.field_path
        ]
        test3_passed = not is_valid and len(sentinel_errors) >= 1
        if test3_passed:
            print("✓ Correctly rejected pump-placed sentinel:")
            for e in sentinel_errors:
                print(f"  {e}")
        else:
            print("✗ FAILED: pump-placed sentinel should be rejected")
            print(f"  is_valid={is_valid}, sentinel_errors={len(sentinel_errors)}")
            for e in errors:
                print(f"  {e}")
        print()

        # Test 4: Sentinel on patient_model bolus_entries must remain valid
        print("Test 4: Accepting 'accept_recommendation' on patient_model.bolus_entries")
        print("-" * 80)
        pm_sentinel_path = os.path.join(temp_dir, "config_patient_model_sentinel.json")
        with open(pm_sentinel_path, 'w') as f:
            json.dump(create_test_config_patient_model_sentinel(), f, indent=2)

        is_valid, errors, warnings = validator.validate_config_file(pm_sentinel_path)
        test4_passed = is_valid
        if test4_passed:
            print("✓ Correctly accepted patient_model-placed sentinel")
        else:
            print(f"✗ FAILED: patient_model-placed sentinel should be valid; got {len(errors)} errors:")
            for e in errors:
                print(f"  {e}")
        print()

        # Test 5: Validate directory
        print("Test 5: Validating directory")
        print("-" * 80)
        results = validator.validate_directory(temp_dir, recursive=False)

        print(f"Validated {len(results)} files:")
        for file_path, (is_valid, errors, warnings) in results.items():
            status = "✓ VALID" if is_valid else f"✗ INVALID ({len(errors)} errors)"
            warn_str = f", {len(warnings)} warning(s)" if warnings else ""
            print(f"  {os.path.basename(file_path)}: {status}{warn_str}")
        print()

        # Summary
        print("=" * 80)
        print("Test Summary")
        print("=" * 80)

        total_valid = sum(1 for (is_valid, _, _w) in results.values() if is_valid)
        total_invalid = len(results) - total_valid

        print(f"Total files validated: {len(results)}")
        print(f"Valid: {total_valid}")
        print(f"Invalid: {total_invalid}")
        print()

        # Expect: valid + patient_model_sentinel = 2 valid; errors + pump_sentinel = 2 invalid.
        if total_valid == 2 and total_invalid == 2 and test3_passed and test4_passed:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")


if __name__ == "__main__":
    test_validator()
