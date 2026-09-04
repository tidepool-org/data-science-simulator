"""
Tests for two validator bug fixes:

1. base_config as a reusable reference string should not fail structure validation.
2. Bolus value "accept_recommendation" should not fail value validation.
"""

import pytest
from tidepool_data_science_simulator.validation.config_validator import ConfigValidator
from tidepool_data_science_simulator.validation.value_validators import ValueValidators, ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_config(base_config_value):
    """Return a minimal valid config dict with a custom base_config value."""
    return {
        "metadata": {
            "risk-id": "TEST-000",
            "simulation_id": "test",
            "risk_description": "test",
            "config_format_version": "v1.0"
        },
        "base_config": base_config_value,
        "override_config": []
    }


# ---------------------------------------------------------------------------
# Fix 1 – base_config as a reusable reference string
# ---------------------------------------------------------------------------

class TestBaseConfigStructureValidation:

    def setup_method(self):
        self.validator = ConfigValidator(pointer_object_dir=None)

    def _structure_errors(self, config):
        """Return only structure-validation errors for a config dict."""
        return self.validator._validate_structure(config, "test_file.json")

    # --- should PASS (no errors) ---

    def test_base_config_as_dict_is_valid(self):
        config = _make_base_config({"some_key": "some_value"})
        errors = self._structure_errors(config)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_base_config_as_reusable_string_is_valid(self):
        config = _make_base_config("reusable.simulations.custom_presets.preset_130_resistant_2_0_v1")
        errors = self._structure_errors(config)
        assert errors == [], f"Unexpected errors for reusable base_config: {errors}"

    def test_base_config_various_reusable_prefixes_are_valid(self):
        reusable_refs = [
            "reusable.simulations.base.standard",
            "reusable.simulations.base_median_2_0_v1",
            "reusable.some.deep.nested.reference",
        ]
        for ref in reusable_refs:
            config = _make_base_config(ref)
            errors = self._structure_errors(config)
            assert errors == [], f"Expected no errors for '{ref}', got: {errors}"

    # --- should FAIL (errors expected) ---

    def test_base_config_as_arbitrary_string_is_invalid(self):
        config = _make_base_config("some_random_string")
        errors = self._structure_errors(config)
        assert len(errors) == 1
        assert "base_config" in errors[0].field_path

    def test_base_config_as_integer_is_invalid(self):
        config = _make_base_config(42)
        errors = self._structure_errors(config)
        assert len(errors) == 1
        assert "base_config" in errors[0].field_path

    def test_base_config_as_none_is_invalid(self):
        config = _make_base_config(None)
        errors = self._structure_errors(config)
        assert len(errors) == 1
        assert "base_config" in errors[0].field_path


# ---------------------------------------------------------------------------
# Fix 2 – accept_recommendation as a valid bolus value
# ---------------------------------------------------------------------------

class TestBolusEntryValidation:

    # --- should PASS (no errors) ---

    def test_numeric_bolus_value_is_valid(self):
        entry = {"time": "8/15/2019 12:00:00", "value": 5.0}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert errors == [], f"Unexpected errors: {errors}"

    def test_accept_recommendation_is_valid(self):
        entry = {"time": "8/15/2019 12:00:00", "value": "accept_recommendation"}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert errors == [], (
            f"'accept_recommendation' should be a valid bolus value, got: {errors}"
        )

    def test_numeric_string_bolus_is_valid(self):
        """Numeric values stored as strings (e.g. from JSON) should still pass."""
        entry = {"time": "8/15/2019 12:00:00", "value": "3.5"}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert errors == [], f"Unexpected errors: {errors}"

    # --- should FAIL (errors expected) ---

    def test_unknown_string_bolus_value_is_invalid(self):
        entry = {"time": "8/15/2019 12:00:00", "value": "some_other_string"}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert len(errors) == 1
        assert "bolus[0].value" in errors[0].field_path
        # Error message should hint at valid sentinels
        assert "accept_recommendation" in errors[0].error_message

    def test_bolus_value_out_of_range_is_invalid(self):
        entry = {"time": "8/15/2019 12:00:00", "value": 999}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert len(errors) == 1
        assert "bolus[0].value" in errors[0].field_path

    def test_zero_bolus_value_is_valid(self):
        """0.0 is a valid bolus dose (e.g. Loop recommends no bolus)."""
        entry = {"time": "8/15/2019 12:00:00", "value": 0}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert errors == [], f"Expected 0 to be valid, got: {errors}"

    def test_zero_float_bolus_value_is_valid(self):
        entry = {"time": "8/15/2019 12:00:00", "value": 0.0}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert errors == [], f"Expected 0.0 to be valid, got: {errors}"

    def test_negative_bolus_value_is_invalid(self):
        entry = {"time": "8/15/2019 12:00:00", "value": -1.0}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        assert len(errors) == 1

    def test_missing_value_field_is_invalid(self):
        entry = {"time": "8/15/2019 12:00:00"}
        errors = ValueValidators.validate_bolus_entry(entry, "bolus[0]")
        field_paths = [e.field_path for e in errors]
        assert "bolus[0]" in field_paths  # missing-field error reported on parent path

    # --- sentinel set integrity ---

    def test_valid_bolus_sentinels_contains_accept_recommendation(self):
        assert "accept_recommendation" in ValueValidators.VALID_BOLUS_SENTINELS

    def test_valid_bolus_sentinels_is_frozenset(self):
        assert isinstance(ValueValidators.VALID_BOLUS_SENTINELS, frozenset)
