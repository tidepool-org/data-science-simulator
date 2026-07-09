"""
Integration tests for validate_configs.py and ConfigValidator.

These tests exercise the full validation pipeline against real scenario
config files and real reusable component files that live in the
``scenario_configs/`` directory tree.

All tests skip gracefully when the expected config directories are not
present (e.g. in a fresh checkout without large data files).
"""

import json
import pytest
from pathlib import Path

from tidepool_data_science_simulator.validation.config_validator import ConfigValidator
from tidepool_data_science_simulator.validation.value_validators import ValidationError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SCENARIO_CONFIGS = REPO_ROOT / "scenario_configs"
RISK_TEST_DIR = SCENARIO_CONFIGS / "tidepool_risk_test"
RISK_V2_DIR = SCENARIO_CONFIGS / "tidepool_risk_v2"
LOOP_RISK_TEST_DIR = RISK_TEST_DIR / "loop_risk_test"

# A single well-known valid config file used across multiple tests
_KNOWN_VALID_FILE = (
    LOOP_RISK_TEST_DIR
    / "TLR-000-T2-test"
    / "Simulation-Configuration-TLR-000-t2-controller-low_insulin.json"
)


def _require_dir(path: Path):
    """Skip the test if *path* does not exist."""
    if not path.exists():
        pytest.skip(f"Required directory not found: {path}")


def _require_file(path: Path):
    """Skip the test if *path* does not exist."""
    if not path.is_file():
        pytest.skip(f"Required file not found: {path}")


# ---------------------------------------------------------------------------
# TestSingleFileValidation
# ---------------------------------------------------------------------------


class TestSingleFileValidation:
    """File-level integration tests: known files, non-existent paths."""

    def setup_method(self):
        _require_file(_KNOWN_VALID_FILE)
        # Validator without a pointer directory (no reference resolution)
        self.validator = ConfigValidator(pointer_object_dir=None)

    def test_known_valid_config_passes_without_pointer_dir(self):
        """A structurally sound config should produce no errors (structure + values only)."""
        is_valid, errors, warnings = self.validator.validate_config_file(str(_KNOWN_VALID_FILE))
        assert is_valid, (
            f"Expected valid, got {len(errors)} error(s):\n"
            + "\n".join(str(e) for e in errors)
        )
        assert errors == []

    def test_known_valid_config_passes_with_pointer_dir(self):
        """The same file should still pass when reference resolution is enabled."""
        _require_dir(RISK_TEST_DIR)
        validator = ConfigValidator(pointer_object_dir=str(RISK_TEST_DIR))
        is_valid, errors, warnings = validator.validate_config_file(str(_KNOWN_VALID_FILE))
        assert is_valid, (
            f"Expected valid with reference resolution, got {len(errors)} error(s):\n"
            + "\n".join(str(e) for e in errors)
        )

    def test_missing_file_returns_error(self):
        """Passing a non-existent path should produce a 'not found' error."""
        is_valid, errors, warnings = self.validator.validate_config_file("/no/such/file.json")
        assert not is_valid
        assert len(errors) >= 1
        assert any("not found" in e.error_message.lower() for e in errors)

    def test_result_is_tuple_of_bool_and_list(self):
        """validate_config_file must return (bool, list, list)."""
        result = self.validator.validate_config_file(str(_KNOWN_VALID_FILE))
        assert isinstance(result, tuple)
        assert len(result) == 3
        is_valid, errors, warnings = result
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# TestDirectoryValidation
# ---------------------------------------------------------------------------


class TestDirectoryValidation:
    """Directory-level integration tests."""

    def setup_method(self):
        _require_dir(LOOP_RISK_TEST_DIR)

    def test_all_risk_test_configs_pass_structure_and_values(self):
        """
        Every scenario config in loop_risk_test/ should be structurally valid
        and pass value checks.  Reference resolution is intentionally disabled
        here because some TLR-000-T2-temp files reference reusable files that
        have not yet been created (e.g. t2_test_simulation_no_controller_v1).
        """
        validator = ConfigValidator(pointer_object_dir=None)
        results = validator.validate_directory(str(LOOP_RISK_TEST_DIR), recursive=True)

        assert len(results) > 0, "No config files were discovered — check the directory path."

        failures = {
            path: errors
            for path, (is_valid, errors, warnings) in results.items()
            if not is_valid
        }
        assert failures == {}, (
            f"{len(failures)} config(s) failed validation:\n"
            + "\n".join(
                f"  {p}: " + "; ".join(str(e) for e in errs)
                for p, errs in failures.items()
            )
        )

    def test_complete_configs_pass_with_reference_resolution(self):
        """
        Configs in TLR-000-T2-test/ have all their reusable files present and
        should pass full validation including reference resolution.
        """
        tlr_test_subdir = LOOP_RISK_TEST_DIR / "TLR-000-T2-test"
        _require_dir(tlr_test_subdir)
        _require_dir(RISK_TEST_DIR)
        validator = ConfigValidator(pointer_object_dir=str(RISK_TEST_DIR))
        results = validator.validate_directory(str(tlr_test_subdir), recursive=True)

        assert len(results) > 0, "No config files found in TLR-000-T2-test/."

        failures = {
            path: errors
            for path, (is_valid, errors, warnings) in results.items()
            if not is_valid
        }
        assert failures == {}, (
            f"{len(failures)} config(s) failed validation with references:\n"
            + "\n".join(
                f"  {p}: " + "; ".join(str(e) for e in errs)
                for p, errs in failures.items()
            )
        )

    def test_result_keys_are_file_paths(self):
        """Result dict keys should be absolute path strings pointing to JSON files."""
        validator = ConfigValidator(pointer_object_dir=None)
        results = validator.validate_directory(str(LOOP_RISK_TEST_DIR), recursive=True)
        for path_str in results:
            assert path_str.endswith(".json"), f"Expected .json path, got: {path_str}"
            assert Path(path_str).is_file(), f"Result key is not a real file: {path_str}"

    def test_non_recursive_finds_fewer_files_than_recursive(self):
        """Non-recursive mode should find ≤ as many files as recursive mode."""
        validator = ConfigValidator(pointer_object_dir=None)
        recursive_results = validator.validate_directory(
            str(LOOP_RISK_TEST_DIR), recursive=True
        )
        non_recursive_results = validator.validate_directory(
            str(LOOP_RISK_TEST_DIR), recursive=False
        )
        assert len(non_recursive_results) <= len(recursive_results)

    def test_reusable_files_are_excluded(self):
        """Files inside 'reusable' directories must be excluded from results."""
        validator = ConfigValidator(pointer_object_dir=None)
        results = validator.validate_directory(str(RISK_TEST_DIR), recursive=True)
        for path_str in results:
            assert "reusable" not in path_str, (
                f"A reusable file was included in results: {path_str}"
            )


# ---------------------------------------------------------------------------
# TestReferenceValidation
# ---------------------------------------------------------------------------


class TestReferenceValidation:
    """Reference resolver integration tests using real reusable files."""

    def setup_method(self):
        _require_dir(RISK_TEST_DIR)
        self.validator = ConfigValidator(pointer_object_dir=str(RISK_TEST_DIR))

    def _load_known_config(self) -> dict:
        """Load the known-valid config as a dict."""
        _require_file(_KNOWN_VALID_FILE)
        with open(_KNOWN_VALID_FILE) as fh:
            return json.load(fh)

    def test_valid_references_produce_no_errors(self):
        """Real references in a known-valid config should resolve without errors."""
        _require_file(_KNOWN_VALID_FILE)
        is_valid, errors, warnings = self.validator.validate_config_file(str(_KNOWN_VALID_FILE))
        reference_errors = [
            e for e in errors
            if "not found" in e.error_message.lower()
            or "reference" in e.error_message.lower()
        ]
        assert reference_errors == [], (
            "Unexpected reference errors on a known-valid config:\n"
            + "\n".join(str(e) for e in reference_errors)
        )

    def test_nonexistent_reference_produces_error(self):
        """A config referencing a file that does not exist must fail."""
        config = {
            "metadata": {
                "simulation_id": "test-ref-missing",
                "risk_description": "test",
                "config_format_version": "v1.0",
            },
            "base_config": "reusable.simulations.this_file_does_not_exist_xyz",
            "override_config": [],
        }
        # Write to a temp file so the validator can process it
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(config, tmp)
            tmp_path = tmp.name
        try:
            is_valid, errors, warnings = self.validator.validate_config_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        assert not is_valid
        assert any(
            "not found" in e.error_message.lower() for e in errors
        ), f"Expected a 'not found' reference error, got: {[str(e) for e in errors]}"

    def test_resolved_reference_files_pass_structure_validation(self):
        """
        When references resolve successfully, the referenced files themselves
        should have no Pydantic structure errors.
        """
        _require_file(_KNOWN_VALID_FILE)
        is_valid, errors, warnings = self.validator.validate_config_file(str(_KNOWN_VALID_FILE))
        structure_errors_in_refs = [
            e for e in errors
            if "structure error in referenced file" in e.error_message.lower()
        ]
        assert structure_errors_in_refs == [], (
            "Structure errors found inside referenced reusable files:\n"
            + "\n".join(str(e) for e in structure_errors_in_refs)
        )

    def test_reference_cache_is_populated_after_directory_validation(self):
        """After validating a directory, the internal reference cache should be non-empty."""
        _require_dir(LOOP_RISK_TEST_DIR)
        self.validator.validate_directory(str(LOOP_RISK_TEST_DIR), recursive=True)
        # Cache should have entries (references were loaded and reused)
        assert len(self.validator._ref_file_cache) > 0


# ---------------------------------------------------------------------------
# TestPydanticStructureIntegration
# ---------------------------------------------------------------------------


class TestPydanticStructureIntegration:
    """
    Tests that exercise the Pydantic structural validation layer using a mix
    of real config data and deliberately broken configs.
    """

    def setup_method(self):
        self.validator = ConfigValidator(pointer_object_dir=None)

    def _validate_structure(self, config: dict) -> list:
        """Run only the Pydantic structural layer and return errors."""
        import tempfile, os, json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(config, tmp)
            tmp_path = tmp.name
        try:
            _, errors = self.validator.validate_config_file(tmp_path)
        finally:
            os.unlink(tmp_path)
        return errors

    def test_valid_scenario_config_passes_pydantic(self):
        """A complete, valid config dict should pass Pydantic structural validation."""
        _require_file(_KNOWN_VALID_FILE)
        with open(_KNOWN_VALID_FILE) as fh:
            config = json.load(fh)
        errors = self.validator._validate_pydantic_structure(
            config, "test_valid.json"
        )
        assert errors == [], (
            f"Unexpected Pydantic errors on a known-valid config:\n"
            + "\n".join(str(e) for e in errors)
        )

    def test_missing_simulation_id_raises_error_with_suggestion(self):
        """Removing 'simulation_id' from metadata should surface a clear error."""
        _require_file(_KNOWN_VALID_FILE)
        with open(_KNOWN_VALID_FILE) as fh:
            config = json.load(fh)
        # Remove the required field
        config["metadata"].pop("simulation_id", None)

        errors = self.validator._validate_pydantic_structure(config, "test.json")
        assert len(errors) >= 1
        paths = [e.field_path for e in errors]
        assert any("simulation_id" in p for p in paths), (
            f"Expected an error mentioning 'simulation_id', got: {paths}"
        )
        # The error message should contain a suggestion
        messages = [e.error_message for e in errors]
        assert any("simulation_id" in m or "suggestion" in m.lower() for m in messages), (
            f"Expected a suggestion in error messages, got: {messages}"
        )

    def test_override_config_as_dict_raises_list_error(self):
        """override_config must be a list; passing a dict should raise an error."""
        config = {
            "metadata": {"simulation_id": "test-001"},
            "base_config": "reusable.simulations.foo",
            "override_config": {"not": "a list"},
        }
        errors = self.validator._validate_pydantic_structure(config, "test.json")
        assert len(errors) >= 1
        assert any("override_config" in e.field_path for e in errors)

    def test_base_config_wrong_string_raises_error_with_suggestion(self):
        """A base_config string not starting with 'reusable.' should raise an error."""
        config = {
            "metadata": {"simulation_id": "test-001"},
            "base_config": "this_is_not_a_reusable_ref",
            "override_config": [],
        }
        errors = self.validator._validate_pydantic_structure(config, "test.json")
        assert len(errors) >= 1
        assert any("base_config" in e.field_path for e in errors)
        # Error message should mention reusable
        assert any("reusable" in e.error_message.lower() for e in errors)

    def test_missing_entire_metadata_block_raises_error(self):
        """Omitting 'metadata' entirely should surface an error mentioning metadata."""
        config = {
            "base_config": "reusable.simulations.foo",
            "override_config": [],
        }
        errors = self.validator._validate_pydantic_structure(config, "test.json")
        assert len(errors) >= 1
        assert any("metadata" in e.field_path for e in errors)

    def test_metadata_as_wrong_type_raises_error(self):
        """metadata must be a dict; a string value should raise an error."""
        config = {
            "metadata": "not-a-dict",
            "base_config": "reusable.simulations.foo",
            "override_config": [],
        }
        errors = self.validator._validate_pydantic_structure(config, "test.json")
        assert len(errors) >= 1
        assert any("metadata" in e.field_path for e in errors)

    def test_pydantic_not_available_returns_empty(self, monkeypatch):
        """If Pydantic is not available, _validate_pydantic_structure returns []."""
        import tidepool_data_science_simulator.validation.config_validator as cv_module
        original = cv_module._PYDANTIC_AVAILABLE
        monkeypatch.setattr(cv_module, "_PYDANTIC_AVAILABLE", False)
        try:
            result = self.validator._validate_pydantic_structure({}, "test.json")
            assert result == []
        finally:
            monkeypatch.setattr(cv_module, "_PYDANTIC_AVAILABLE", original)


# ---------------------------------------------------------------------------
# TestSchemaModels (unit-level, but lives here as it tests integration of models)
# ---------------------------------------------------------------------------


class TestSchemaModels:
    """Tests for the Pydantic schema models and helper functions in schema_models.py."""

    def test_scenario_config_valid_reusable_base_config(self):
        """ScenarioConfig accepts a reusable reference string for base_config."""
        from tidepool_data_science_simulator.validation.schema_models import ScenarioConfig
        config = ScenarioConfig.model_validate({
            "metadata": {"simulation_id": "test-001"},
            "base_config": "reusable.simulations.foo_v1",
            "override_config": [],
        })
        assert config.base_config == "reusable.simulations.foo_v1"

    def test_scenario_config_valid_inline_base_config(self):
        """ScenarioConfig accepts an inline dict for base_config."""
        from tidepool_data_science_simulator.validation.schema_models import ScenarioConfig
        config = ScenarioConfig.model_validate({
            "metadata": {"simulation_id": "test-001"},
            "base_config": {"sim_id": "my_sim", "duration_hours": 8.0},
            "override_config": [],
        })
        assert config.base_config.sim_id == "my_sim"

    def test_scenario_config_rejects_non_reusable_string_base_config(self):
        """ScenarioConfig rejects base_config strings not starting with 'reusable.'."""
        from pydantic import ValidationError as PydanticValidationError
        from tidepool_data_science_simulator.validation.schema_models import ScenarioConfig
        with pytest.raises(PydanticValidationError):
            ScenarioConfig.model_validate({
                "metadata": {"simulation_id": "test-001"},
                "base_config": "not_a_reusable_ref",
                "override_config": [],
            })

    def test_scenario_config_requires_simulation_id(self):
        """ScenarioConfig raises ValidationError when simulation_id is absent."""
        from pydantic import ValidationError as PydanticValidationError
        from tidepool_data_science_simulator.validation.schema_models import ScenarioConfig
        with pytest.raises(PydanticValidationError) as exc_info:
            ScenarioConfig.model_validate({
                "metadata": {},  # missing simulation_id
                "base_config": "reusable.simulations.foo",
                "override_config": [],
            })
        assert "simulation_id" in str(exc_info.value)

    def test_carb_doses_adapter_validates_list(self):
        """CarbDosesAdapter should validate a list of carb entries."""
        from tidepool_data_science_simulator.validation.schema_models import CarbDosesAdapter
        data = [{"start_time": "8/15/2019 12:00:00", "value": 45.0, "type": "carb"}]
        result = CarbDosesAdapter.validate_python(data)
        assert len(result) == 1
        assert result[0].value == 45.0

    def test_insulin_doses_adapter_validates_list(self):
        """InsulinDosesAdapter should validate a list of bolus entries."""
        from tidepool_data_science_simulator.validation.schema_models import InsulinDosesAdapter
        data = [{"time": "8/15/2019 12:00:00", "value": 3.5, "type": "bolus"}]
        result = InsulinDosesAdapter.validate_python(data)
        assert len(result) == 1

    def test_pydantic_errors_to_validation_errors_missing_field(self):
        """pydantic_errors_to_validation_errors converts a 'missing' error correctly."""
        from pydantic import ValidationError as PydanticValidationError
        from tidepool_data_science_simulator.validation.schema_models import (
            ScenarioConfig,
            pydantic_errors_to_validation_errors,
        )
        try:
            ScenarioConfig.model_validate({
                "metadata": {},  # missing simulation_id
                "base_config": "reusable.simulations.foo",
                "override_config": [],
            })
        except PydanticValidationError as exc:
            our_errors = pydantic_errors_to_validation_errors(exc, "test_file.json")

        assert len(our_errors) >= 1
        assert isinstance(our_errors[0], ValidationError)
        assert "simulation_id" in our_errors[0].field_path
        assert "Suggestion:" in our_errors[0].error_message or "simulation_id" in our_errors[0].error_message
