"""
Tests that SwiftLoopController correctly passes the controller insulin model
(recommendationInsulinType) to the Loop Algorithm API payload.

Bug: recommendationInsulinType was hardcoded to 'novolog', ignoring the
'model' field set in controller_config.controller_settings by the scenario
parser.  This prevented ab_URAI_pump configs (fiasp controller) from
actually exercising the fiasp insulin model in the Loop algorithm.

Run with:
    pytest tests/test_swift_controller_insulin_type.py -v
"""

import datetime
import pytest
from unittest.mock import MagicMock, patch

from tidepool_data_science_simulator.models.controller import AutomationControlTimeline
from tidepool_data_science_simulator.makedata.scenario_parser import ControllerConfig
from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller_config(model: str) -> ControllerConfig:
    """Return a minimal ControllerConfig with the given insulin model name."""
    settings = {
        "model": model,
        "max_basal_rate": 3.5,
        "max_bolus": 10.0,
        "suspend_threshold": 70.0,
        "partial_application_factor": 0.0,
        "use_mid_absorption_isf": False,
    }
    return ControllerConfig(
        bolus_event_timeline=BolusTimeline(),
        carb_event_timeline=CarbTimeline(),
        controller_settings=settings,
    )


def _make_minimal_virtual_patient():
    """
    Return a MagicMock that satisfies the attribute lookups inside
    SwiftLoopController.prepare_inputs without triggering real I/O.
    """
    vp = MagicMock()

    # sensor.get_loop_inputs() -> (dates, values)
    vp.sensor.get_loop_inputs.return_value = ([], [])

    # Timelines returned by get_dose_event_timelines
    _empty_timeline = MagicMock()
    _empty_timeline.get_loop_inputs.return_value = ([], [], [], [], [])
    vp.__class__ = MagicMock  # silence isinstance checks

    # basal / isf / cir / target schedules – each returns three empty lists
    for attr in [
        "pump.pump_config.basal_schedule",
        "pump.pump_config.insulin_sensitivity_schedule",
        "pump.pump_config.carb_ratio_schedule",
    ]:
        sched = MagicMock()
        sched.get_loop_swift_inputs.return_value = ([], [], [])
        _set_nested(vp, attr, sched)

    target_sched = MagicMock()
    target_sched.get_loop_swift_inputs.return_value = ([], [], [], [])
    _set_nested(vp, "pump.pump_config.target_range_schedule", target_sched)

    return vp


def _set_nested(obj, dotted_attr, value):
    """Set a deeply nested attribute on a MagicMock via dot notation."""
    parts = dotted_attr.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _make_controller(model: str):
    """Build a SwiftLoopController with the given insulin model, mocking the Swift lib."""
    # Import here to avoid import-time errors if the Swift dylib is not present
    from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController

    t0 = datetime.datetime(2019, 8, 15, 12, 0, 0)
    cfg = _make_controller_config(model)
    ctrl = SwiftLoopController(t0, cfg, AutomationControlTimeline([], []))
    ctrl.time = t0
    return ctrl


def _prepare_inputs_with_mocked_timelines(controller):
    """
    Call prepare_inputs() with a fully-mocked virtual patient and patch the
    get_dose_event_timelines method so we don't need real timeline objects.
    """
    vp = _make_minimal_virtual_patient()

    # Each of the three timelines returned by get_dose_event_timelines needs
    # its own get_loop_inputs mock returning the 5-tuple expected by prepare_inputs.
    empty5 = ([], [], [], [], [])
    bolus_tl = MagicMock(); bolus_tl.get_loop_inputs.return_value = empty5
    carb_tl  = MagicMock(); carb_tl.get_loop_inputs.return_value  = ([], [], [])
    temp_tl  = MagicMock(); temp_tl.get_loop_inputs.return_value  = empty5

    with patch.object(controller, "get_dose_event_timelines",
                      return_value=(bolus_tl, carb_tl, temp_tl)):
        return controller.prepare_inputs(vp)


# ---------------------------------------------------------------------------
# Tests: recommendationInsulinType is read from controller_config
# ---------------------------------------------------------------------------

class TestRecommendationInsulinType:

    def test_novolog_model_sets_novolog(self):
        """Standard URAI scenario: controller model 'novolog' → 'novolog' in payload."""
        ctrl = _make_controller("novolog")
        payload = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload["recommendationInsulinType"] == "novolog"

    def test_fiasp_model_sets_fiasp(self):
        """ab_URAI_pump scenario: controller model 'fiasp' → 'fiasp' in payload."""
        ctrl = _make_controller("fiasp")
        payload = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload["recommendationInsulinType"] == "fiasp"

    def test_rapid_acting_adult_model_passes_through(self):
        """
        'rapid_acting_adult' is mapped to 'novolog' by the scenario parser
        before being stored in controller_settings['model'].  After mapping,
        the stored value is 'novolog', which should appear in the payload.
        """
        ctrl = _make_controller("novolog")   # post-parser value
        payload = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload["recommendationInsulinType"] == "novolog"

    def test_missing_model_key_falls_back_to_novolog(self):
        """
        Configs that pre-date the 'model' setting don't have the key.
        The fallback must be 'novolog' for backward compatibility.
        """
        from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController

        t0 = datetime.datetime(2019, 8, 15, 12, 0, 0)
        # Build settings dict WITHOUT a 'model' key
        settings_no_model = {
            "max_basal_rate": 3.5,
            "max_bolus": 10.0,
            "suspend_threshold": 70.0,
            "partial_application_factor": 0.0,
            "use_mid_absorption_isf": False,
        }
        cfg = ControllerConfig(
            bolus_event_timeline=BolusTimeline(),
            carb_event_timeline=CarbTimeline(),
            controller_settings=settings_no_model,
        )
        ctrl = SwiftLoopController(t0, cfg, AutomationControlTimeline([], []))
        ctrl.time = t0

        payload = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload["recommendationInsulinType"] == "novolog"

    def test_model_change_reflected_in_new_payload(self):
        """
        Mutating controller_settings['model'] after construction should be
        immediately reflected in the next prepare_inputs() call, confirming
        there is no caching of the old hardcoded value.
        """
        ctrl = _make_controller("novolog")
        payload_before = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload_before["recommendationInsulinType"] == "novolog"

        ctrl.controller_config.controller_settings["model"] = "fiasp"
        payload_after = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload_after["recommendationInsulinType"] == "fiasp"


# ---------------------------------------------------------------------------
# Tests: optional vs required controller settings in the Swift payload
#
# Regression coverage for a cherry-pick conflict (commit b1d7b92d) that turned
# tolerant `.get(...)` accessors into hard `settings_dictionary[...]` lookups,
# raising KeyError for configs that omit optional settings.  The correct
# semantics are dictated by the Swift AlgorithmInputFixture decoder:
#   - maxBolus / maxBasalRate      -> required (non-optional `decode`)
#   - automaticBolusApplicationFactor -> optional (Double?, nil default)
#   - useMidAbsorptionISF          -> optional (decodeIfPresent ... ?? false)
# ---------------------------------------------------------------------------

def _make_controller_from_settings(settings: dict):
    """Build a SwiftLoopController from an arbitrary controller_settings dict."""
    from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController

    t0 = datetime.datetime(2019, 8, 15, 12, 0, 0)
    cfg = ControllerConfig(
        bolus_event_timeline=BolusTimeline(),
        carb_event_timeline=CarbTimeline(),
        controller_settings=settings,
    )
    ctrl = SwiftLoopController(t0, cfg, AutomationControlTimeline([], []))
    ctrl.time = t0
    return ctrl


# A minimal config carrying only the two genuinely-required safety limits.
_REQUIRED_ONLY = {"max_basal_rate": 3.5, "max_bolus": 10.0, "suspend_threshold": 70.0}


class TestOptionalControllerSettings:

    def test_missing_partial_application_factor_omits_field(self):
        """
        No partial_application_factor -> automaticBolusApplicationFactor is
        omitted (Swift falls back to its nil default) and the controller runs
        in tempBasal mode.  This is the exact path exercised by the
        NoisySensor override tests.
        """
        ctrl = _make_controller_from_settings(dict(_REQUIRED_ONLY))
        payload = _prepare_inputs_with_mocked_timelines(ctrl)

        assert "automaticBolusApplicationFactor" not in payload
        assert payload["recommendationType"] == "tempBasal"
        assert payload["includePositiveVelocityAndRC"] is True

    def test_truthy_partial_application_factor_unchanged(self):
        """Configs that DO set a truthy factor keep the automaticBolus path."""
        settings = dict(_REQUIRED_ONLY, partial_application_factor=0.4)
        ctrl = _make_controller_from_settings(settings)
        payload = _prepare_inputs_with_mocked_timelines(ctrl)

        assert payload["automaticBolusApplicationFactor"] == 0.4
        assert payload["recommendationType"] == "automaticBolus"
        assert payload["includePositiveVelocityAndRC"] is False

    def test_zero_partial_application_factor_still_included(self):
        """
        A factor explicitly set to 0.0 is falsy (so recommendationType stays
        tempBasal) but the key IS present, so the field is still emitted --
        preserving the pre-regression payload for configs that set it to 0.0.
        """
        settings = dict(_REQUIRED_ONLY, partial_application_factor=0.0)
        ctrl = _make_controller_from_settings(settings)
        payload = _prepare_inputs_with_mocked_timelines(ctrl)

        assert payload["automaticBolusApplicationFactor"] == 0.0
        assert payload["recommendationType"] == "tempBasal"

    def test_missing_use_mid_absorption_isf_defaults_false(self):
        """Absent use_mid_absorption_isf mirrors the Swift default of false."""
        ctrl = _make_controller_from_settings(dict(_REQUIRED_ONLY))
        payload = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload["useMidAbsorptionISF"] is False

    def test_use_mid_absorption_isf_true_preserved(self):
        """An explicit True is passed through (the old `or True` masked False)."""
        settings = dict(_REQUIRED_ONLY, use_mid_absorption_isf=True)
        ctrl = _make_controller_from_settings(settings)
        payload = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload["useMidAbsorptionISF"] is True

    def test_use_mid_absorption_isf_false_preserved(self):
        """An explicit False must survive (regression guard on the `or True` bug)."""
        settings = dict(_REQUIRED_ONLY, use_mid_absorption_isf=False)
        ctrl = _make_controller_from_settings(settings)
        payload = _prepare_inputs_with_mocked_timelines(ctrl)
        assert payload["useMidAbsorptionISF"] is False

    def test_missing_required_max_bolus_raises(self):
        """
        maxBolus is required by the Swift decoder; a config missing it is a
        genuine error and must surface clearly rather than being silently
        defaulted.
        """
        settings = {"max_basal_rate": 3.5, "suspend_threshold": 70.0}
        ctrl = _make_controller_from_settings(settings)
        with pytest.raises(KeyError):
            _prepare_inputs_with_mocked_timelines(ctrl)


# ---------------------------------------------------------------------------
# Tests: SWIFT_CONTROLLER_MODEL_NAME_MAP entries
# ---------------------------------------------------------------------------

class TestSwiftControllerModelNameMap:

    def test_map_contains_fiasp(self):
        from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import (
            SWIFT_CONTROLLER_MODEL_NAME_MAP,
        )
        assert "fiasp" in SWIFT_CONTROLLER_MODEL_NAME_MAP

    def test_fiasp_maps_to_fiasp(self):
        from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import (
            SWIFT_CONTROLLER_MODEL_NAME_MAP,
        )
        assert SWIFT_CONTROLLER_MODEL_NAME_MAP["fiasp"] == "fiasp"

    def test_rapid_acting_adult_maps_to_novolog(self):
        from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import (
            SWIFT_CONTROLLER_MODEL_NAME_MAP,
        )
        assert SWIFT_CONTROLLER_MODEL_NAME_MAP["rapid_acting_adult"] == "novolog"

    def test_all_map_values_are_strings(self):
        """Values must be strings (not legacy PyLoopKit [duration, peak] lists)."""
        from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import (
            SWIFT_CONTROLLER_MODEL_NAME_MAP,
        )
        for key, val in SWIFT_CONTROLLER_MODEL_NAME_MAP.items():
            assert isinstance(val, str), (
                f"SWIFT_CONTROLLER_MODEL_NAME_MAP['{key}'] should be a string, got {type(val)}"
            )
