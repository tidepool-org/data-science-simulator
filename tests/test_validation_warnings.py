"""
Unit tests for the structured ValidationWarning system.

Covers:
- ValidationWarning class construction and string representation
- validate_max_active_insulin_multiplier logic:
    - standard value (2.0) → no issues
    - valid but non-standard value → warning only, no error
    - out-of-range value → error only, no warning
    - non-numeric value → error only, no warning
- ConfigValidator.validate_config_file returns 3-tuple (is_valid, errors, warnings)
- Warnings do not affect is_valid
- ConfigValidator.validate_directory returns 3-tuple per file
- No bare print() calls leak through validate_max_active_insulin_multiplier
- validate_directory returns empty dict for a directory with no JSON configs
"""

import json
import os
import tempfile

import pytest

from tidepool_data_science_simulator.validation import (
    ConfigValidator,
    ValidationError,
    ValidationWarning,
)
from tidepool_data_science_simulator.validation.value_validators import ValueValidators


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(max_active_insulin_multiplier=2.0):
    """Return a minimal valid config with the given multiplier value."""
    return {
        "metadata": {
            "simulation_id": "test-warnings",
            "description": "Warning system test",
        },
        "base_config": {
            "sim_id": "test_warn_sim",
            "time_to_calculate_at": "08/15/2019 12:00:00",
            "duration_hours": 8.0,
            "patient": {
                "pump": {
                    "metabolism_settings": {
                        "basal_rate": {"start_times": ["00:00:00"], "values": [1.0]},
                        "carb_insulin_ratio": {"start_times": ["00:00:00"], "values": [10.0]},
                        "insulin_sensitivity_factor": {"start_times": ["00:00:00"], "values": [50.0]},
                    },
                    "target_range": {
                        "start_times": ["00:00:00"],
                        "lower_values": [100.0],
                        "upper_values": [120.0],
                    },
                    "carb_entries": [],
                    "bolus_entries": [],
                },
                "patient_model": {
                    "metabolism_settings": {
                        "basal_rate": {"start_times": ["00:00:00"], "values": [1.0]},
                        "carb_insulin_ratio": {"start_times": ["00:00:00"], "values": [10.0]},
                        "insulin_sensitivity_factor": {"start_times": ["00:00:00"], "values": [50.0]},
                    },
                    "glucose_history": {
                        "datetime": {"0": "08/15/2019 11:00:00"},
                        "value": {"0": 120.0},
                    },
                    "carb_entries": [],
                    "bolus_entries": [],
                    "physical_activity_entries": [],
                },
                "sensor": {
                    "glucose_history": {
                        "datetime": {"0": "08/15/2019 11:00:00"},
                        "value": {"0": 120.0},
                    }
                },
            },
            "controller": {
                "id": "py_loop",
                "settings": {
                    "max_active_insulin_multiplier": max_active_insulin_multiplier,
                },
            },
        },
        "override_config": [{}],
    }


def _write_config(config: dict, tmp_dir: str, filename: str = "config.json") -> str:
    """Write *config* to *tmp_dir/filename* and return the full path."""
    path = os.path.join(tmp_dir, filename)
    with open(path, "w") as f:
        json.dump(config, f)
    return path


# ---------------------------------------------------------------------------
# ValidationWarning class
# ---------------------------------------------------------------------------

class TestValidationWarningClass:
    """Unit tests for the ValidationWarning data class itself."""

    def test_construction_with_value(self):
        w = ValidationWarning("some.field", "A warning message", value=3.5)
        assert w.field_path == "some.field"
        assert w.warning_message == "A warning message"
        assert w.value == 3.5

    def test_construction_without_value(self):
        w = ValidationWarning("some.field", "A warning message")
        assert w.value is None

    def test_str_with_value(self):
        w = ValidationWarning("some.field", "Non-standard value", value=3.5)
        s = str(w)
        assert "⚠️" in s
        assert "some.field" in s
        assert "Non-standard value" in s
        assert "3.5" in s

    def test_str_without_value(self):
        w = ValidationWarning("some.field", "A plain warning")
        s = str(w)
        assert "⚠️" in s
        assert "some.field" in s
        assert "A plain warning" in s
        # Should not end with "(value: None)"
        assert "None" not in s

    def test_repr_contains_class_name(self):
        w = ValidationWarning("p.q", "msg", value=1)
        r = repr(w)
        assert "ValidationWarning" in r
        assert "p.q" in r
        assert "msg" in r

    def test_is_distinct_from_validation_error(self):
        w = ValidationWarning("f", "msg")
        e = ValidationError("f", "msg")
        assert not isinstance(w, ValidationError)
        assert not isinstance(e, ValidationWarning)


# ---------------------------------------------------------------------------
# ValueValidators.validate_max_active_insulin_multiplier
# ---------------------------------------------------------------------------

class TestValidateMaxActiveInsulinMultiplier:
    """Tests for the updated validate_max_active_insulin_multiplier method."""

    # ---- standard value ----

    def test_standard_value_returns_empty(self):
        """2.0 is standard — no errors, no warnings."""
        results = ValueValidators.validate_max_active_insulin_multiplier(2.0, "f")
        assert results == []

    def test_standard_value_as_string_returns_empty(self):
        """'2.0' coerced to float is still standard."""
        results = ValueValidators.validate_max_active_insulin_multiplier("2.0", "f")
        assert results == []

    # ---- valid but non-standard → warning only ----

    def test_non_standard_valid_value_returns_warning(self):
        """3.5 is in (0, 10] but not 2.0 → one warning, no errors."""
        results = ValueValidators.validate_max_active_insulin_multiplier(3.5, "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationWarning)
        assert not any(isinstance(r, ValidationError) for r in results)

    def test_warning_contains_value(self):
        results = ValueValidators.validate_max_active_insulin_multiplier(3.5, "my.field")
        assert results[0].value == 3.5

    def test_warning_contains_field_path(self):
        results = ValueValidators.validate_max_active_insulin_multiplier(5.0, "controller.multiplier")
        assert results[0].field_path == "controller.multiplier"

    def test_warning_message_mentions_standard(self):
        results = ValueValidators.validate_max_active_insulin_multiplier(5.0, "f")
        assert "2.0" in results[0].warning_message
        assert "standard" in results[0].warning_message.lower()

    def test_boundary_value_10_returns_warning(self):
        """10.0 is valid (at the inclusive boundary) but not standard → warning."""
        results = ValueValidators.validate_max_active_insulin_multiplier(10.0, "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationWarning)

    def test_small_valid_value_returns_warning(self):
        """0.5 is valid (above 0) but not standard → warning."""
        results = ValueValidators.validate_max_active_insulin_multiplier(0.5, "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationWarning)

    # ---- out-of-range → error only, no warning ----

    def test_out_of_range_high_returns_error_not_warning(self):
        """15.0 > 10 → one error, no warning (the original bug)."""
        results = ValueValidators.validate_max_active_insulin_multiplier(15.0, "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationError)
        assert not any(isinstance(r, ValidationWarning) for r in results)

    def test_out_of_range_zero_returns_error(self):
        """0 is not in (0, 10] → error."""
        results = ValueValidators.validate_max_active_insulin_multiplier(0.0, "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationError)

    def test_out_of_range_negative_returns_error(self):
        results = ValueValidators.validate_max_active_insulin_multiplier(-1.0, "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationError)

    def test_out_of_range_error_message(self):
        results = ValueValidators.validate_max_active_insulin_multiplier(15.0, "f")
        assert "range" in results[0].error_message.lower()

    # ---- non-numeric → error only ----

    def test_non_numeric_string_returns_error(self):
        results = ValueValidators.validate_max_active_insulin_multiplier("abc", "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationError)

    def test_none_returns_error(self):
        results = ValueValidators.validate_max_active_insulin_multiplier(None, "f")
        assert len(results) == 1
        assert isinstance(results[0], ValidationError)

    # ---- no bare print() side-effects ----

    def test_no_print_on_out_of_range(self, capsys):
        """Out-of-range values must NOT trigger a bare print() call."""
        ValueValidators.validate_max_active_insulin_multiplier(15.0, "f")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_print_on_non_standard(self, capsys):
        """Non-standard-but-valid values must NOT trigger a bare print() call."""
        ValueValidators.validate_max_active_insulin_multiplier(5.0, "f")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_print_on_standard(self, capsys):
        """Standard value (2.0) must NOT trigger any print() call."""
        ValueValidators.validate_max_active_insulin_multiplier(2.0, "f")
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# ConfigValidator public API — 3-tuple return
# ---------------------------------------------------------------------------

class TestValidateConfigFileReturnsTuple:
    """Tests that validate_config_file returns (bool, errors, warnings) correctly."""

    def setup_method(self):
        self.validator = ConfigValidator(pointer_object_dir=None)

    def test_returns_three_tuple(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(2.0), d)
            result = self.validator.validate_config_file(path)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_errors_is_list(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(2.0), d)
            _, errors, _ = self.validator.validate_config_file(path)
        assert isinstance(errors, list)

    def test_warnings_is_list(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(2.0), d)
            _, _, warnings = self.validator.validate_config_file(path)
        assert isinstance(warnings, list)

    # ---- standard multiplier: valid, no warnings ----

    def test_standard_multiplier_is_valid_no_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(2.0), d)
            is_valid, errors, warnings = self.validator.validate_config_file(path)
        assert is_valid
        assert errors == []
        assert warnings == []

    # ---- non-standard multiplier: valid, warning present ----

    def test_non_standard_multiplier_is_valid_with_warning(self):
        """A non-standard-but-in-range multiplier should be valid with one warning."""
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(5.0), d)
            is_valid, errors, warnings = self.validator.validate_config_file(path)
        assert is_valid, f"Expected valid, got errors: {errors}"
        assert errors == []
        assert len(warnings) == 1
        assert isinstance(warnings[0], ValidationWarning)

    def test_non_standard_multiplier_warning_value(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(5.0), d)
            _, _, warnings = self.validator.validate_config_file(path)
        assert warnings[0].value == 5.0

    def test_non_standard_multiplier_warning_field_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(5.0), d)
            _, _, warnings = self.validator.validate_config_file(path)
        assert "max_active_insulin_multiplier" in warnings[0].field_path

    # ---- out-of-range multiplier: invalid, error present, no warning ----

    def test_out_of_range_multiplier_is_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(15.0), d)
            is_valid, errors, warnings = self.validator.validate_config_file(path)
        assert not is_valid
        assert any("max_active_insulin_multiplier" in e.field_path for e in errors)

    def test_out_of_range_multiplier_no_warning(self):
        """Out-of-range values must NOT also produce a warning."""
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(15.0), d)
            _, _, warnings = self.validator.validate_config_file(path)
        assert warnings == [], (
            f"Out-of-range multiplier should produce no warning, got: {warnings}"
        )

    # ---- warnings do not affect is_valid ----

    def test_warnings_do_not_affect_is_valid(self):
        """is_valid is True even when warnings are present."""
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(5.0), d)
            is_valid, errors, warnings = self.validator.validate_config_file(path)
        assert is_valid
        assert len(warnings) > 0

    # ---- no bare print() leaks ----

    def test_no_print_for_out_of_range(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(15.0), d)
            self.validator.validate_config_file(path)
        captured = capsys.readouterr()
        assert "⚠️" not in captured.out, (
            "validate_config_file should not bare-print warnings; "
            f"stdout was: {captured.out!r}"
        )

    def test_no_print_for_non_standard(self, capsys):
        with tempfile.TemporaryDirectory() as d:
            path = _write_config(_minimal_config(5.0), d)
            self.validator.validate_config_file(path)
        captured = capsys.readouterr()
        assert "⚠️" not in captured.out, (
            "validate_config_file should not bare-print warnings; "
            f"stdout was: {captured.out!r}"
        )


# ---------------------------------------------------------------------------
# ConfigValidator.validate_directory — 3-tuple per entry
# ---------------------------------------------------------------------------

class TestValidateDirectoryReturnsTuple:
    """Tests that validate_directory returns (bool, errors, warnings) per file."""

    def setup_method(self):
        self.validator = ConfigValidator(pointer_object_dir=None)

    def test_nonexistent_directory_raises_value_error(self):
        """validate_directory must raise ValueError for a path that does not exist."""
        with pytest.raises(ValueError, match="is not an existing directory"):
            self.validator.validate_directory("/no/such/directory/xyz", recursive=True)

    def test_empty_directory_returns_empty_dict(self):
        """A directory with no JSON files should return an empty dict, not crash."""
        with tempfile.TemporaryDirectory() as d:
            results = self.validator.validate_directory(d, recursive=True)
        assert results == {}

    def test_directory_with_only_non_json_files_returns_empty(self):
        """Non-JSON files should be ignored; result should be empty."""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "not_a_config.txt"), "w").close()
            open(os.path.join(d, "also_not.csv"), "w").close()
            results = self.validator.validate_directory(d, recursive=False)
        assert results == {}

    def test_directory_result_values_are_three_tuples(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(_minimal_config(2.0), d, "valid.json")
            results = self.validator.validate_directory(d, recursive=False)
        for entry in results.values():
            assert len(entry) == 3

    def test_directory_valid_file_has_empty_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(_minimal_config(2.0), d, "valid.json")
            results = self.validator.validate_directory(d, recursive=False)
        for is_valid, errors, warnings in results.values():
            assert is_valid
            assert warnings == []

    def test_directory_non_standard_file_has_warning(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(_minimal_config(5.0), d, "warn.json")
            results = self.validator.validate_directory(d, recursive=False)
        for is_valid, errors, warnings in results.values():
            assert is_valid
            assert len(warnings) == 1
            assert isinstance(warnings[0], ValidationWarning)

    def test_directory_invalid_file_has_error_no_warning(self):
        with tempfile.TemporaryDirectory() as d:
            _write_config(_minimal_config(15.0), d, "bad.json")
            results = self.validator.validate_directory(d, recursive=False)
        for is_valid, errors, warnings in results.values():
            assert not is_valid
            assert len(errors) >= 1
            assert warnings == []

    def test_directory_mixed_files(self):
        """Valid, non-standard, and invalid configs processed together."""
        with tempfile.TemporaryDirectory() as d:
            _write_config(_minimal_config(2.0), d, "standard.json")
            _write_config(_minimal_config(5.0), d, "nonstandard.json")
            _write_config(_minimal_config(15.0), d, "invalid.json")
            results = self.validator.validate_directory(d, recursive=False)

        assert len(results) == 3

        valid_count = sum(1 for e in results.values() if e[0])
        warning_count = sum(1 for e in results.values() if e[2])
        invalid_count = sum(1 for e in results.values() if not e[0])

        assert valid_count == 2    # standard + nonstandard are both valid
        assert warning_count == 1  # only nonstandard has a warning
        assert invalid_count == 1  # only invalid has an error

    def test_reusable_subdirectory_files_are_skipped(self):
        """Files inside a 'reusable' subdirectory must not appear in results."""
        with tempfile.TemporaryDirectory() as d:
            reusable_dir = os.path.join(d, "reusable")
            os.makedirs(reusable_dir)
            _write_config(_minimal_config(2.0), d, "real_config.json")
            _write_config(_minimal_config(2.0), reusable_dir, "template.json")
            results = self.validator.validate_directory(d, recursive=True)

        assert len(results) == 1
        assert all("reusable" not in path for path in results)
