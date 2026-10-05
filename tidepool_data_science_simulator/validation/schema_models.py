"""
Pydantic models for simulation configuration file structure validation.

These models mirror the JSON schema of scenario config files and reusable
component files. They are used to:
  1. Validate the structural correctness of scenario configs
  2. Validate the structure of referenced reusable files
  3. Produce actionable error messages with fix suggestions

All models use ``extra='allow'`` so that unknown fields do not cause
validation failures — the config format evolves frequently.

Union fields that accept either a reusable reference string (starting with
``"reusable."``) or an inline config object are typed as ``Union[str, Model]``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_reusable_ref(value: Any) -> bool:
    """Return True if *value* is a valid reusable reference string."""
    return isinstance(value, str) and value.startswith("reusable.")


# ---------------------------------------------------------------------------
# Leaf / primitive models
# ---------------------------------------------------------------------------

class TimeSeriesParam(BaseModel):
    """A time-scheduled parameter (basal rate, ISF, CIR, …)."""

    model_config = ConfigDict(extra="allow")

    start_times: List[str]
    values: List[Union[float, int]]


class TargetRange(BaseModel):
    """Glucose target range schedule."""

    model_config = ConfigDict(extra="allow")

    start_times: List[str]
    lower_values: List[Union[float, int]]
    upper_values: List[Union[float, int]]


class CarbEntry(BaseModel):
    """A single carbohydrate dose entry."""

    model_config = ConfigDict(extra="allow")

    start_time: str
    value: Union[float, int]
    type: Optional[str] = None
    duration: Optional[Union[float, int]] = None


class BolusEntry(BaseModel):
    """A single bolus dose entry.

    ``value`` may be a numeric dose or the sentinel string
    ``"accept_recommendation"``.
    """

    model_config = ConfigDict(extra="allow")

    time: str
    value: Union[float, int, str]
    type: Optional[str] = None


# ---------------------------------------------------------------------------
# Parameter group models
# ---------------------------------------------------------------------------

class MetabolismSettings(BaseModel):
    """Patient metabolism parameters (type 1 and type 2)."""

    model_config = ConfigDict(extra="allow")

    insulin_sensitivity_factor: Optional[Union[str, TimeSeriesParam]] = None
    carb_insulin_ratio: Optional[Union[str, TimeSeriesParam]] = None
    basal_rate: Optional[Union[str, TimeSeriesParam]] = None
    glucose_sensitivity_factor: Optional[Union[str, TimeSeriesParam]] = None
    basal_blood_glucose: Optional[Union[str, TimeSeriesParam]] = None
    insulin_production_rate: Optional[Union[str, TimeSeriesParam]] = None
    patient_insulin_type: Optional[str] = None


class LoopSettings(BaseModel):
    """Loop / Swift algorithm controller settings."""

    model_config = ConfigDict(extra="allow")

    max_basal_rate: Optional[Union[float, int]] = None
    max_bolus: Optional[Union[float, int]] = None
    suspend_threshold: Optional[Union[float, int]] = None
    model: Optional[str] = None
    momentum_data_interval: Optional[Union[float, int]] = None
    dynamic_carb_absorption_enabled: Optional[bool] = None
    retrospective_correction_integration_interval: Optional[Union[float, int]] = None
    recency_interval: Optional[Union[float, int]] = None
    retrospective_correction_grouping_interval: Optional[Union[float, int]] = None
    rate_rounder: Optional[Union[float, int]] = None
    insulin_delay: Optional[Union[float, int]] = None
    carb_delay: Optional[Union[float, int]] = None
    minimum_autobolus: Optional[Union[float, int]] = None
    maximum_autobolus: Optional[Union[float, int]] = None
    partial_application_factor: Optional[Union[float, int]] = None
    default_absorption_times: Optional[List[Union[float, int]]] = None
    retrospective_correction_enabled: Optional[bool] = None
    use_mid_absorption_isf: Optional[bool] = None
    carb_absorption_model: Optional[str] = None
    max_active_insulin_modifier: Optional[Union[float, int]] = None
    max_active_insulin_multiplier: Optional[Union[float, int]] = None


# ---------------------------------------------------------------------------
# Patient sub-component models
# ---------------------------------------------------------------------------

class SensorConfig(BaseModel):
    """CGM sensor configuration."""

    model_config = ConfigDict(extra="allow")

    glucose_history: Optional[Any] = None
    type: Optional[str] = None
    parameters: Optional[Any] = None


class PumpConfig(BaseModel):
    """Insulin pump configuration."""

    model_config = ConfigDict(extra="allow")

    metabolism_settings: Optional[Union[str, MetabolismSettings]] = None
    bolus_entries: Optional[Union[str, List[BolusEntry]]] = None
    carb_entries: Optional[Union[str, List[CarbEntry]]] = None
    target_range: Optional[Union[str, TargetRange]] = None


class PatientModelConfig(BaseModel):
    """Internal patient simulation model configuration."""

    model_config = ConfigDict(extra="allow")

    metabolism_settings: Optional[Union[str, MetabolismSettings]] = None
    glucose_history: Optional[Any] = None
    bolus_entries: Optional[Union[str, List[BolusEntry]]] = None
    carb_entries: Optional[Union[str, List[CarbEntry]]] = None
    physical_activity_entries: Optional[Union[str, List[Any]]] = None


class PatientConfig(BaseModel):
    """Top-level patient configuration grouping sensor, pump, and model."""

    model_config = ConfigDict(extra="allow")

    sensor: Optional[Union[str, SensorConfig]] = None
    pump: Optional[Union[str, PumpConfig]] = None
    patient_model: Optional[Union[str, PatientModelConfig]] = None


class ControllerConfig(BaseModel):
    """Closed-loop controller configuration."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    settings: Optional[Union[str, LoopSettings]] = None
    automation_control_timeline: Optional[List[Any]] = None


# ---------------------------------------------------------------------------
# Top-level simulation / scenario models
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """Full simulation configuration (used for base_config dicts and reusable simulation files)."""

    model_config = ConfigDict(extra="allow")

    sim_id: Optional[str] = None
    time_to_calculate_at: Optional[str] = None
    duration_hours: Optional[Union[float, int]] = None
    offset_applied_to_dates: Optional[Union[float, int]] = None
    patient: Optional[Union[str, PatientConfig]] = None
    controller: Optional[Union[str, ControllerConfig]] = None


class OverrideItem(BaseModel):
    """A single entry in ``override_config``.

    Each override is merged on top of the base config to produce one simulation run.
    The ``controller`` field may be ``None`` to disable the controller entirely.
    """

    model_config = ConfigDict(extra="allow")

    sim_id: Optional[str] = None
    duration_hours: Optional[Union[float, int]] = None
    patient: Optional[Union[str, PatientConfig]] = None
    controller: Optional[Union[str, ControllerConfig]] = None


class ScenarioMetadata(BaseModel):
    """Scenario metadata block.

    Accepts both ``risk_id`` and ``risk-id`` spellings via ``extra='allow'``.
    Only ``simulation_id`` is required.
    """

    model_config = ConfigDict(extra="allow")

    simulation_id: str


class ScenarioConfig(BaseModel):
    """Top-level scenario configuration file schema."""

    model_config = ConfigDict(extra="allow")

    metadata: ScenarioMetadata
    base_config: Union[str, SimulationConfig]
    override_config: List[OverrideItem]

    @field_validator("base_config", mode="before")
    @classmethod
    def validate_base_config_ref(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.startswith("reusable."):
            raise ValueError(
                f"base_config string must start with 'reusable.' (got: '{v}'). "
                f"Use a reusable reference like 'reusable.simulations.my_sim_v1' "
                f"or provide a full simulation config object."
            )
        return v


# ---------------------------------------------------------------------------
# TypeAdapters for list-valued reusable files
# ---------------------------------------------------------------------------

CarbDosesAdapter: TypeAdapter = TypeAdapter(List[CarbEntry])
InsulinDosesAdapter: TypeAdapter = TypeAdapter(List[BolusEntry])


# ---------------------------------------------------------------------------
# Suggestion system
# ---------------------------------------------------------------------------

# Per-field suggestions for known important fields.
_FIELD_SUGGESTIONS: dict[str, str] = {
    "simulation_id": (
        'Add \'simulation_id\' to the metadata section. '
        'Example: "simulation_id": "MyRisk-001"'
    ),
    "metadata": (
        'Add a "metadata" section at the top level of the config with at least '
        '"simulation_id". Example: "metadata": {"simulation_id": "MyRisk-001"}'
    ),
    "base_config": (
        'Set "base_config" to a reusable reference string '
        '(e.g. "reusable.simulations.my_sim_v1") or an inline simulation '
        'config object.'
    ),
    "override_config": (
        '"override_config" must be a JSON array of simulation overrides. '
        'Example: "override_config": [{"sim_id": "run_1", "patient": {...}}]'
    ),
}

# Generic suggestions by Pydantic v2 error type.
_TYPE_SUGGESTIONS: dict[str, str] = {
    "missing": "Add the required field to your config.",
    "string_type": "This field must be a string value.",
    "int_type": "This field must be an integer.",
    "float_type": "This field must be a number.",
    "bool_type": "This field must be true or false.",
    "dict_type": "This field must be a JSON object {...}.",
    "list_type": "This field must be a JSON array [...].",
    "value_error": "",  # message comes from the validator itself
    "union_tag_invalid": (
        "This field does not match any valid type. "
        "Check that you are using the correct structure for this field."
    ),
    "model_type": "This field must be a JSON object {...}.",
}


def _build_suggestion(error_type: str, loc: tuple, msg: str) -> str:
    """Return a human-readable fix suggestion for a Pydantic error."""
    field_name = str(loc[-1]) if loc else ""

    # Field-specific suggestion takes priority
    if field_name in _FIELD_SUGGESTIONS:
        return _FIELD_SUGGESTIONS[field_name]

    # value_error — the validator message IS the suggestion
    if error_type == "value_error":
        return msg

    # Generic suggestion by error type
    suggestion = _TYPE_SUGGESTIONS.get(error_type, "")
    if suggestion and field_name:
        return f"Field '{field_name}': {suggestion}"
    return suggestion


def pydantic_errors_to_validation_errors(
    pydantic_exc: "pydantic.ValidationError",  # noqa: F821 — avoid hard import at module level
    path_prefix: str,
) -> list:
    """Convert a Pydantic ``ValidationError`` into a list of our ``ValidationError`` objects.

    Parameters
    ----------
    pydantic_exc:
        The Pydantic ``ValidationError`` raised during ``model_validate()``.
    path_prefix:
        Dot-notation prefix to prepend to each error's field path
        (typically the config file's basename).

    Returns
    -------
    list of ValidationError
        Our custom ``ValidationError`` instances with field paths and suggestions.
    """
    # Import here to avoid circular dependency at module load time
    from tidepool_data_science_simulator.validation.value_validators import (
        ValidationError as OurValidationError,
    )

    our_errors = []
    for err in pydantic_exc.errors(include_url=False):
        loc: tuple = err.get("loc", ())
        error_type: str = err.get("type", "")
        msg: str = err.get("msg", "")

        # Build dot-notation field path
        parts = [str(p) for p in loc]
        if parts:
            field_path = f"{path_prefix}.{'.'.join(parts)}"
        else:
            field_path = path_prefix

        suggestion = _build_suggestion(error_type, loc, msg)

        # Compose the full error message: Pydantic's message + fix suggestion
        full_message = msg
        if suggestion and suggestion not in msg:
            full_message = f"{msg}. Suggestion: {suggestion}"

        our_errors.append(OurValidationError(field_path, full_message))

    return our_errors
