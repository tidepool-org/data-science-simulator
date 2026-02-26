"""
Main configuration validator for simulation configuration files.

This module orchestrates validation of entire configuration files,
including structure, values, and references.
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

from .value_validators import ValueValidators, ValidationError

try:
    from pydantic import ValidationError as PydanticValidationError
    from .schema_models import (
        ScenarioConfig,
        SimulationConfig,
        MetabolismSettings,
        LoopSettings,
        TargetRange,
        CarbDosesAdapter,
        InsulinDosesAdapter,
        pydantic_errors_to_validation_errors,
    )
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False


class ConfigValidator:
    """Main configuration validator"""
    
    def __init__(self, pointer_object_dir: str = None):
        """
        Initialize the configuration validator.

        Parameters
        ----------
        pointer_object_dir : str, optional
            Directory containing reusable configuration files
        """
        self.pointer_object_dir = pointer_object_dir
        self.value_validators = ValueValidators()
        # Cache of already-loaded reusable reference files: path -> parsed JSON
        self._ref_file_cache: Dict[str, Any] = {}
        
    def validate_config_file(self, config_path: str) -> Tuple[bool, List[ValidationError]]:
        """
        Validate a single configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to the configuration file
            
        Returns
        -------
        Tuple[bool, List[ValidationError]]
            (is_valid, list of errors)
        """
        errors = []
        
        # Check file exists
        if not os.path.isfile(config_path):
            errors.append(ValidationError(
                config_path,
                "Configuration file not found"
            ))
            return False, errors
        
        # Load JSON
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(ValidationError(
                config_path,
                f"Invalid JSON: {str(e)}"
            ))
            return False, errors
        except Exception as e:
            errors.append(ValidationError(
                config_path,
                f"Failed to load config: {str(e)}"
            ))
            return False, errors
        
        # Validate structure (quick required-field check)
        errors.extend(self._validate_structure(config, config_path))

        # Validate structure with Pydantic (deeper structural check + suggestions)
        errors.extend(self._validate_pydantic_structure(config, os.path.basename(config_path)))

        # Validate values (range / format checks)
        errors.extend(self._validate_values(config, os.path.basename(config_path)))

        # Validate reusable references (existence + structure)
        errors.extend(self._validate_references(config, os.path.basename(config_path)))
        
        return len(errors) == 0, errors
    
    def _validate_structure(self, config: dict, config_path: str) -> List[ValidationError]:
        """
        Validate required top-level structure.
        
        Parameters
        ----------
        config : dict
            The loaded configuration
        config_path : str
            Path to the config file (for error reporting)
            
        Returns
        -------
        List[ValidationError]
            List of structural validation errors
        """
        errors = []
        
        required_fields = ['metadata', 'base_config', 'override_config']
        for field in required_fields:
            if field not in config:
                errors.append(ValidationError(
                    f"{config_path}.{field}",
                    f"Missing required top-level field"
                ))
        
        # Validate metadata
        if 'metadata' in config:
            if not isinstance(config['metadata'], dict):
                errors.append(ValidationError(
                    f"{config_path}.metadata",
                    "metadata must be a dictionary"
                ))
        
        # Validate base_config — must be a dict of inline settings or a reusable reference string
        if 'base_config' in config:
            base = config['base_config']
            is_reusable_ref = isinstance(base, str) and base.startswith('reusable.')
            if not isinstance(base, dict) and not is_reusable_ref:
                errors.append(ValidationError(
                    f"{config_path}.base_config",
                    "base_config must be a dictionary or a reusable reference string (e.g. 'reusable.simulations...')",
                    base
                ))
        
        # Validate override_config is a list
        if 'override_config' in config:
            if not isinstance(config['override_config'], list):
                errors.append(ValidationError(
                    f"{config_path}.override_config",
                    "override_config must be a list"
                ))
        
        return errors

    def _validate_pydantic_structure(self, config: dict, path_prefix: str) -> List[ValidationError]:
        """
        Validate the scenario config structure using Pydantic models.

        Produces richer error messages than the basic structural check and
        includes actionable fix suggestions. Requires pydantic>=2.0 to be
        installed; silently skips if pydantic is unavailable.

        Parameters
        ----------
        config : dict
            The loaded configuration dictionary.
        path_prefix : str
            Basename of the config file, used as the error path prefix.

        Returns
        -------
        List[ValidationError]
            Structural validation errors with suggestions.
        """
        if not _PYDANTIC_AVAILABLE:
            return []

        try:
            ScenarioConfig.model_validate(config)
        except PydanticValidationError as exc:
            return pydantic_errors_to_validation_errors(exc, path_prefix)

        return []

    def _validate_values(self, config: dict, path_prefix: str) -> List[ValidationError]:
        """
        Validate all field values recursively.
        
        Parameters
        ----------
        config : dict
            The configuration to validate
        path_prefix : str
            Current path in the configuration tree
            
        Returns
        -------
        List[ValidationError]
            List of value validation errors
        """
        errors = []
        
        # Validate base_config
        if 'base_config' in config and isinstance(config['base_config'], dict):
            errors.extend(self._validate_config_section(
                config['base_config'], 
                f"{path_prefix}.base_config"
            ))
        
        # Validate each override_config
        if 'override_config' in config and isinstance(config['override_config'], list):
            for i, override in enumerate(config['override_config']):
                if isinstance(override, dict):
                    errors.extend(self._validate_config_section(
                        override, 
                        f"{path_prefix}.override_config[{i}]"
                    ))
        
        return errors
    
    def _validate_config_section(self, section: Any, path: str) -> List[ValidationError]:
        """
        Recursively validate configuration sections.
        
        Parameters
        ----------
        section : Any
            The configuration section to validate
        path : str
            Current path in the configuration tree
            
        Returns
        -------
        List[ValidationError]
            List of validation errors
        """
        errors = []
        
        if not isinstance(section, dict):
            return errors
        
        for key, value in section.items():
            current_path = f"{path}.{key}"
            
            # Skip reusable references (would need pointer_object_dir to validate)
            if isinstance(value, str) and value.startswith('reusable.'):
                continue
            
            # Validate based on key name
            if key == "basal_rate" and isinstance(value, dict):
                if 'values' in value and isinstance(value['values'], list):
                    for i, v in enumerate(value['values']):
                        errors.extend(self.value_validators.validate_basal_rate(
                            v, f"{current_path}.values[{i}]"
                        ))
                    
                    # Validate start_times if present
                    if 'start_times' in value and isinstance(value['start_times'], list):
                        for i, time_str in enumerate(value['start_times']):
                            errors.extend(self.value_validators.validate_time_format(
                                time_str, f"{current_path}.start_times[{i}]"
                            ))
            
            elif key == "carb_insulin_ratio" and isinstance(value, dict):
                if 'values' in value and isinstance(value['values'], list):
                    for i, v in enumerate(value['values']):
                        errors.extend(self.value_validators.validate_carb_ratio(
                            v, f"{current_path}.values[{i}]"
                        ))
                    
                    # Validate start_times
                    if 'start_times' in value and isinstance(value['start_times'], list):
                        for i, time_str in enumerate(value['start_times']):
                            errors.extend(self.value_validators.validate_time_format(
                                time_str, f"{current_path}.start_times[{i}]"
                            ))
            
            elif key == "insulin_sensitivity_factor" and isinstance(value, dict):
                if 'values' in value and isinstance(value['values'], list):
                    for i, v in enumerate(value['values']):
                        errors.extend(self.value_validators.validate_insulin_sensitivity(
                            v, f"{current_path}.values[{i}]"
                        ))
                    
                    # Validate start_times
                    if 'start_times' in value and isinstance(value['start_times'], list):
                        for i, time_str in enumerate(value['start_times']):
                            errors.extend(self.value_validators.validate_time_format(
                                time_str, f"{current_path}.start_times[{i}]"
                            ))
            
            elif key == "glucose_sensitivity_factor" and isinstance(value, dict):
                if 'values' in value and isinstance(value['values'], list):
                    for i, v in enumerate(value['values']):
                        errors.extend(self.value_validators.validate_glucose_sensitivity_factor(
                            v, f"{current_path}.values[{i}]"
                        ))
            
            elif key == "basal_blood_glucose" and isinstance(value, dict):
                if 'values' in value and isinstance(value['values'], list):
                    for i, v in enumerate(value['values']):
                        errors.extend(self.value_validators.validate_basal_blood_glucose(
                            v, f"{current_path}.values[{i}]"
                        ))
            
            elif key == "insulin_production_rate" and isinstance(value, dict):
                if 'values' in value and isinstance(value['values'], list):
                    for i, v in enumerate(value['values']):
                        errors.extend(self.value_validators.validate_insulin_production_rate(
                            v, f"{current_path}.values[{i}]"
                        ))
            
            elif key == "target_range" and isinstance(value, dict):
                if 'lower_values' in value and 'upper_values' in value:
                    lower_values = value['lower_values']
                    upper_values = value['upper_values']
                    
                    if isinstance(lower_values, list) and isinstance(upper_values, list):
                        min_len = min(len(lower_values), len(upper_values))
                        for i in range(min_len):
                            errors.extend(self.value_validators.validate_target_range(
                                lower_values[i], upper_values[i], f"{current_path}[{i}]"
                            ))
                        
                        # Check for mismatched lengths
                        if len(lower_values) != len(upper_values):
                            errors.append(ValidationError(
                                current_path,
                                f"Mismatched target_range array lengths: lower={len(lower_values)}, upper={len(upper_values)}"
                            ))
            
            elif key == "max_active_insulin_multiplier":
                errors.extend(self.value_validators.validate_max_active_insulin_multiplier(
                    value, current_path
                ))
            
            elif key == "physical_activity_entries" and isinstance(value, list):
                for i, entry in enumerate(value):
                    if isinstance(entry, dict):
                        errors.extend(self.value_validators.validate_physical_activity_entry(
                            entry, f"{current_path}[{i}]"
                        ))
                    elif isinstance(entry, str) and entry.startswith('reusable.'):
                        # Skip validation of reusable references (need pointer_object_dir)
                        continue
            
            elif key == "carb_entries" and isinstance(value, list):
                for i, entry in enumerate(value):
                    if isinstance(entry, dict):
                        errors.extend(self.value_validators.validate_carb_entry(
                            entry, f"{current_path}[{i}]"
                        ))
            
            elif key == "bolus_entries" and isinstance(value, list):
                for i, entry in enumerate(value):
                    if isinstance(entry, dict):
                        errors.extend(self.value_validators.validate_bolus_entry(
                            entry, f"{current_path}[{i}]"
                        ))
            
            elif key == "time_to_calculate_at":
                errors.extend(self.value_validators.validate_datetime_format(
                    value, current_path
                ))
            
            elif key == "duration_hours":
                try:
                    duration = float(value)
                    if not 0 < duration <= 168:  # Max 1 week
                        errors.append(ValidationError(
                            current_path,
                            "Simulation duration must be between 0 and 168 hours",
                            value
                        ))
                except (ValueError, TypeError):
                    errors.append(ValidationError(
                        current_path,
                        "duration_hours must be numeric",
                        value
                    ))
            
            # Validate metabolism parameters
            elif key in ["w_hr", "a", "tau", "n"]:
                param_dict = {key: value}
                errors.extend(self.value_validators.validate_metabolism_parameters(
                    param_dict, path
                ))
            
            # Recurse into nested dictionaries
            elif isinstance(value, dict):
                errors.extend(self._validate_config_section(value, current_path))
            
            # Recurse into lists of dictionaries
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        errors.extend(self._validate_config_section(item, f"{current_path}[{i}]"))
        
        return errors
    
    def _validate_references(self, config: dict, path_prefix: str) -> List[ValidationError]:
        """
        Validate reusable references (basic check without loading).
        
        Parameters
        ----------
        config : dict
            The configuration to validate
        path_prefix : str
            Current path in the configuration tree
            
        Returns
        -------
        List[ValidationError]
            List of reference validation errors
        """
        errors = []
        
        # If pointer_object_dir is not set, skip reference validation
        if not self.pointer_object_dir:
            return errors
        
        # Recursively find all reusable references
        errors.extend(self._find_and_validate_references(config, path_prefix))
        
        return errors
    
    def _find_and_validate_references(self, obj: Any, path: str) -> List[ValidationError]:
        """
        Recursively find and validate reusable references.

        For each reference found this method checks:
        1. That the referenced file exists on disk.
        2. That the referenced file's structure matches its expected Pydantic model.

        Parameters
        ----------
        obj : Any
            Object to search for references
        path : str
            Current path in the configuration tree

        Returns
        -------
        List[ValidationError]
            List of reference validation errors
        """
        errors = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                if isinstance(value, str) and value.startswith('reusable.'):
                    errors.extend(self._validate_reference(value, current_path))
                elif isinstance(value, (dict, list)):
                    errors.extend(self._find_and_validate_references(value, current_path))

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"

                if isinstance(item, str) and item.startswith('reusable.'):
                    errors.extend(self._validate_reference(item, current_path))
                elif isinstance(item, (dict, list)):
                    errors.extend(self._find_and_validate_references(item, current_path))

        return errors

    def _validate_reference(self, ref_string: str, field_path: str) -> List[ValidationError]:
        """
        Validate a single reusable reference: check existence then check structure.

        Parameters
        ----------
        ref_string : str
            The reusable reference string (e.g. ``"reusable.simulations.base.standard"``).
        field_path : str
            Dot-notation path to the field containing this reference (for error reporting).

        Returns
        -------
        List[ValidationError]
            Existence and/or structural validation errors.
        """
        errors = []

        resolved_path = self._resolve_reference_to_path(ref_string)
        if resolved_path is None:
            # File not found — build error with searched locations
            searched = self._describe_search_paths(ref_string)
            errors.append(ValidationError(
                field_path,
                f"Reference file not found. Searched: {searched}",
                ref_string,
            ))
            return errors

        # File exists — validate its structure if Pydantic is available
        if _PYDANTIC_AVAILABLE:
            errors.extend(self._validate_reference_structure(ref_string, resolved_path, field_path))

        return errors

    def _resolve_reference_to_path(self, ref_string: str) -> Optional[str]:
        """
        Resolve a reusable reference string to an absolute file path.

        Returns the resolved path, or ``None`` if not found.

        Parameters
        ----------
        ref_string : str
            The reusable reference string.

        Returns
        -------
        Optional[str]
            Absolute path to the referenced file, or ``None``.
        """
        if not self.pointer_object_dir:
            return None  # Cannot resolve without a pointer directory

        pointer_segments = ref_string.split(".")
        folder_path = "/".join(pointer_segments[:-1])
        filename_no_ext = pointer_segments[-1]
        json_filename = f"{filename_no_ext}.json"
        csv_filename = f"{filename_no_ext}.csv"

        subdirectories = self._subdirectories_for_ref(folder_path)
        search_paths = [folder_path] + [os.path.join(folder_path, sub) for sub in subdirectories]

        for search_path in search_paths:
            json_path = os.path.join(self.pointer_object_dir, search_path, json_filename)
            csv_path = os.path.join(self.pointer_object_dir, search_path, csv_filename)

            if os.path.isfile(json_path):
                return json_path
            if os.path.isfile(csv_path):
                return csv_path

        return None

    def _describe_search_paths(self, ref_string: str) -> str:
        """Return a human-readable list of paths searched for *ref_string*."""
        if not self.pointer_object_dir:
            return "(no pointer directory configured)"

        pointer_segments = ref_string.split(".")
        folder_path = "/".join(pointer_segments[:-1])
        subdirectories = self._subdirectories_for_ref(folder_path)
        search_paths = [folder_path] + [os.path.join(folder_path, sub) for sub in subdirectories]
        return ", ".join(os.path.join(self.pointer_object_dir, p) for p in search_paths)

    @staticmethod
    def _subdirectories_for_ref(folder_path: str) -> List[str]:
        """Return the list of subdirectories to search for a given reference folder path."""
        if "simulations" in folder_path:
            return ["base", "suspend", "loop_versions", "specialized", "versions",
                    "1xComparator", "custom_presets"]
        if "metabolism_settings" in folder_path:
            return ["profiles", "suspensions", "presets", "versions", "types"]
        if "physical_activities" in folder_path:
            return ["profiles", "activities", "templates"]
        return []

    def _validate_reference_structure(
        self,
        ref_string: str,
        file_path: str,
        field_path: str,
    ) -> List[ValidationError]:
        """
        Load *file_path* and validate its structure with the appropriate Pydantic model.

        Results are cached by file path so the same reusable file is not loaded
        and re-validated for every config that references it.

        Parameters
        ----------
        ref_string : str
            The original reference string (used to infer expected model).
        file_path : str
            Absolute path to the resolved file.
        field_path : str
            Dot-notation path to the referencing field (for error path prefixing).

        Returns
        -------
        List[ValidationError]
            Structural errors found in the referenced file.
        """
        # Only JSON files have structure we can validate with Pydantic
        if not file_path.endswith(".json"):
            return []

        # Load (with caching)
        if file_path not in self._ref_file_cache:
            try:
                with open(file_path, "r") as fh:
                    self._ref_file_cache[file_path] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                return []  # Cannot load — skip structural validation

        ref_data = self._ref_file_cache[file_path]
        model, adapter = self._get_model_for_reference(ref_string)

        if model is None and adapter is None:
            return []  # No model defined for this reference type

        try:
            if adapter is not None:
                adapter.validate_python(ref_data)
            else:
                model.model_validate(ref_data)
        except PydanticValidationError as exc:
            # Prefix errors with the reference string so the caller knows which file
            raw_errors = pydantic_errors_to_validation_errors(exc, ref_string)
            # Annotate each error's path to show it originated from a referenced file
            annotated = []
            for err in raw_errors:
                annotated.append(ValidationError(
                    err.field_path,
                    f"Structure error in referenced file '{file_path}': {err.error_message}",
                ))
            return annotated

        return []

    @staticmethod
    def _get_model_for_reference(ref_string: str):
        """
        Return ``(PydanticModel, TypeAdapter)`` appropriate for *ref_string*.

        Exactly one of the two return values will be non-``None``.
        Both are ``None`` when no model is defined for the reference type.

        Parameters
        ----------
        ref_string : str
            The reusable reference string.

        Returns
        -------
        tuple
            ``(model_class_or_None, type_adapter_or_None)``
        """
        parts = ref_string.split(".")
        filename = parts[-1].lower() if parts else ""

        if "simulations" in parts:
            return SimulationConfig, None
        if "metabolism_settings" in parts:
            return MetabolismSettings, None
        if "loop_settings" in parts:
            return LoopSettings, None
        if "carb_doses" in parts:
            return None, CarbDosesAdapter
        if "insulin_doses" in parts:
            return None, InsulinDosesAdapter
        if "mitigations" in parts and "guardrails" in parts:
            if filename.startswith("target_range"):
                return TargetRange, None
            if filename.startswith("controller_settings"):
                return LoopSettings, None
        # glucose files are CSV; physical_activities have variable structure
        return None, None

    def _check_reference_exists(self, ref_string: str) -> Tuple[bool, str]:
        """
        Check if a reusable reference points to an existing file.

        .. deprecated::
            Prefer :meth:`_validate_reference` which also checks structure.
            Kept for backwards compatibility.

        Parameters
        ----------
        ref_string : str
            The reusable reference string.

        Returns
        -------
        Tuple[bool, str]
            ``(exists, error_message)``
        """
        resolved = self._resolve_reference_to_path(ref_string)
        if resolved is not None:
            return True, ""
        searched = self._describe_search_paths(ref_string)
        return False, f"Reference file not found. Searched: {searched}"
    
    def validate_directory(self, directory: str, recursive: bool = True) -> Dict[str, Tuple[bool, List[ValidationError]]]:
        """
        Validate all configuration files in a directory.
        
        Parameters
        ----------
        directory : str
            Directory containing configuration files
        recursive : bool
            Whether to search subdirectories recursively
            
        Returns
        -------
        Dict[str, Tuple[bool, List[ValidationError]]]
            Dictionary mapping file paths to (is_valid, errors) tuples
        """
        results = {}
        
        if recursive:
            config_files = list(Path(directory).rglob("*.json"))
        else:
            config_files = list(Path(directory).glob("*.json"))
        
        for config_file in config_files:
            # Skip files in reusable directories (these are templates)
            if "reusable" in str(config_file):
                continue
            
            is_valid, errors = self.validate_config_file(str(config_file))
            results[str(config_file)] = (is_valid, errors)
        
        return results
