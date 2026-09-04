"""
Tests for configure_insulin_model_compare.py

Run with:
    pytest tests/test_configure_insulin_model_compare.py -v
"""

import copy
import json
import pytest
from pathlib import Path

# Adjust sys.path so pytest can import the script from scripts/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from configure_insulin_model_compare import (
    swap_profile_ref_to_fiasp,
    apply_patient_fiasp,
    apply_controller_fiasp,
    process_scenario_file,
    get_matched_dirs,
    PROFILE_FIASP_MAP,
    _REUSABLE_ROOT,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reusable_root():
    return _REUSABLE_ROOT


@pytest.fixture
def override_with_profile_ref():
    """Override item where patient_model.metabolism_settings is a known profile ref."""
    return {
        "sim_id": "pre-Loop_t1_median",
        "patient": {
            "patient_model": {
                "metabolism_settings": "reusable.metabolism_settings.profiles.adolescent_v1",
                "glucose_history": "reusable.glucose.flat_110_12hr",
            }
        },
    }


@pytest.fixture
def override_with_inline_ms():
    """Override item where patient_model.metabolism_settings is an inline dict."""
    return {
        "sim_id": "pre-Loop_t1_median",
        "patient": {
            "patient_model": {
                "metabolism_settings": {
                    "basal_rate": {"start_times": ["00:00:00"], "values": [0.5]},
                    "patient_insulin_type": "rapid_acting_adult",
                }
            }
        },
    }


@pytest.fixture
def override_missing_ms():
    """Override item where patient_model exists but metabolism_settings is absent."""
    return {
        "sim_id": "pre-Loop_t1_adolescent",
        "patient": {
            "patient_model": {
                "glucose_history": "reusable.glucose.flat_110_12hr",
            }
        },
    }


@pytest.fixture
def override_missing_patient_model():
    """Override item where patient key exists but patient_model is absent."""
    return {
        "sim_id": "pre-Loop_t1_resistant",
        "patient": {
            "pump": {"bolus_entries": []},
        },
    }


@pytest.fixture
def override_missing_patient():
    """Override item with no patient key at all."""
    return {
        "sim_id": "pre-Loop_t1_sensitive",
        "duration_hours": 8.0,
    }


@pytest.fixture
def override_controller_null():
    """Override item with controller: null (no-Loop simulation)."""
    return {
        "sim_id": "pre-noLoop_t1_median",
        "patient": {},
        "controller": None,
    }


@pytest.fixture
def override_controller_absent():
    """Override item with no controller key (pre-Loop sim — comes from base)."""
    return {
        "sim_id": "pre-Loop_t1_median",
        "patient": {},
    }


@pytest.fixture
def override_controller_string_ref():
    """Override item where controller.settings is a guardrails string ref."""
    return {
        "sim_id": "post-Loop_WithMitigations_t1_adolescent",
        "patient": {},
        "controller": {
            "settings": "reusable.mitigations.guardrails.controller_settings_adolescent_wmax"
        },
    }


@pytest.fixture
def override_controller_inline_dict():
    """Override item where controller.settings is an inline dict (no model key)."""
    return {
        "sim_id": "pre-Loop_NoMitigations_t1_median",
        "patient": {},
        "controller": {
            "settings": {
                "max_basal_rate": 0.0,
                "partial_application_factor": 0.0,
            }
        },
    }


@pytest.fixture
def override_controller_inline_with_model():
    """Override item where controller.settings already has a model key."""
    return {
        "sim_id": "pre-Loop_NoMitigations_t1_median",
        "patient": {},
        "controller": {
            "settings": {
                "max_basal_rate": 35.0,
                "model": "rapid_acting_adult",
            }
        },
    }


# ---------------------------------------------------------------------------
# swap_profile_ref_to_fiasp
# ---------------------------------------------------------------------------

class TestSwapProfileRefToFiasp:

    @pytest.mark.parametrize("base_name", list(PROFILE_FIASP_MAP.keys()))
    def test_known_profiles_swapped(self, base_name):
        ref = f"reusable.metabolism_settings.profiles.{base_name}"
        result = swap_profile_ref_to_fiasp(ref)
        expected_suffix = PROFILE_FIASP_MAP[base_name]
        assert result == f"reusable.metabolism_settings.profiles.{expected_suffix}"

    def test_unknown_profile_returns_none(self):
        ref = "reusable.metabolism_settings.median_preset_10_v1"
        assert swap_profile_ref_to_fiasp(ref) is None

    def test_path_prefix_preserved(self):
        ref = "reusable.metabolism_settings.profiles.median_v1"
        result = swap_profile_ref_to_fiasp(ref)
        assert result.startswith("reusable.metabolism_settings.profiles.")

    def test_fiasp_ref_unchanged_because_not_in_map(self):
        # fiasp variants are not in PROFILE_FIASP_MAP, so the ref returns None
        ref = "reusable.metabolism_settings.profiles.median_fiasp_v1"
        assert swap_profile_ref_to_fiasp(ref) is None


# ---------------------------------------------------------------------------
# apply_patient_fiasp
# ---------------------------------------------------------------------------

class TestApplyPatientFiasp:

    def test_known_profile_ref_swapped(self, override_with_profile_ref, reusable_root):
        item = copy.deepcopy(override_with_profile_ref)
        apply_patient_fiasp(item, reusable_root)
        ms = item["patient"]["patient_model"]["metabolism_settings"]
        assert isinstance(ms, str)
        assert ms.endswith("adolescent_fiasp_v1")

    def test_inline_dict_with_existing_type_updated(self, override_with_inline_ms, reusable_root):
        item = copy.deepcopy(override_with_inline_ms)
        apply_patient_fiasp(item, reusable_root)
        ms = item["patient"]["patient_model"]["metabolism_settings"]
        assert ms["patient_insulin_type"] == "fiasp"

    def test_inline_dict_other_fields_preserved(self, override_with_inline_ms, reusable_root):
        item = copy.deepcopy(override_with_inline_ms)
        apply_patient_fiasp(item, reusable_root)
        ms = item["patient"]["patient_model"]["metabolism_settings"]
        assert "basal_rate" in ms  # original field preserved

    def test_missing_ms_adds_inline_dict(self, override_missing_ms, reusable_root):
        item = copy.deepcopy(override_missing_ms)
        apply_patient_fiasp(item, reusable_root)
        ms = item["patient"]["patient_model"]["metabolism_settings"]
        assert isinstance(ms, dict)
        assert ms["patient_insulin_type"] == "fiasp"

    def test_missing_patient_model_creates_it(self, override_missing_patient_model, reusable_root):
        item = copy.deepcopy(override_missing_patient_model)
        apply_patient_fiasp(item, reusable_root)
        ms = item["patient"]["patient_model"]["metabolism_settings"]
        assert ms["patient_insulin_type"] == "fiasp"

    def test_missing_patient_creates_it(self, override_missing_patient, reusable_root):
        item = copy.deepcopy(override_missing_patient)
        apply_patient_fiasp(item, reusable_root)
        ms = item["patient"]["patient_model"]["metabolism_settings"]
        assert ms["patient_insulin_type"] == "fiasp"

    def test_all_known_profiles_map_correctly(self, reusable_root):
        for base_name, fiasp_name in PROFILE_FIASP_MAP.items():
            item = {
                "sim_id": "test",
                "patient": {
                    "patient_model": {
                        "metabolism_settings": f"reusable.metabolism_settings.profiles.{base_name}"
                    }
                },
            }
            apply_patient_fiasp(item, reusable_root)
            ms = item["patient"]["patient_model"]["metabolism_settings"]
            assert ms.endswith(fiasp_name), f"Expected {fiasp_name}, got {ms}"


# ---------------------------------------------------------------------------
# apply_controller_fiasp
# ---------------------------------------------------------------------------

class TestApplyControllerFiasp:

    def test_null_controller_skipped(self, override_controller_null, reusable_root):
        item = copy.deepcopy(override_controller_null)
        result = apply_controller_fiasp(item, reusable_root)
        assert item["controller"] is None
        assert "skipped" in result

    def test_absent_controller_added(self, override_controller_absent, reusable_root):
        item = copy.deepcopy(override_controller_absent)
        apply_controller_fiasp(item, reusable_root)
        assert item["controller"]["settings"]["model"] == "fiasp"

    def test_string_ref_resolved_and_inlined(self, override_controller_string_ref, reusable_root):
        item = copy.deepcopy(override_controller_string_ref)
        apply_controller_fiasp(item, reusable_root)
        settings = item["controller"]["settings"]
        assert isinstance(settings, dict), "settings should be inlined dict, not string"
        assert settings["model"] == "fiasp"

    def test_string_ref_other_guardrails_fields_preserved(self, override_controller_string_ref, reusable_root):
        item = copy.deepcopy(override_controller_string_ref)
        apply_controller_fiasp(item, reusable_root)
        settings = item["controller"]["settings"]
        # guardrails file has max_basal_rate and max_bolus
        assert "max_basal_rate" in settings
        assert "max_bolus" in settings

    def test_inline_dict_without_model_gets_model(self, override_controller_inline_dict, reusable_root):
        item = copy.deepcopy(override_controller_inline_dict)
        apply_controller_fiasp(item, reusable_root)
        assert item["controller"]["settings"]["model"] == "fiasp"

    def test_inline_dict_existing_model_overwritten(self, override_controller_inline_with_model, reusable_root):
        item = copy.deepcopy(override_controller_inline_with_model)
        apply_controller_fiasp(item, reusable_root)
        assert item["controller"]["settings"]["model"] == "fiasp"

    def test_inline_dict_other_fields_preserved(self, override_controller_inline_dict, reusable_root):
        item = copy.deepcopy(override_controller_inline_dict)
        apply_controller_fiasp(item, reusable_root)
        settings = item["controller"]["settings"]
        assert settings["max_basal_rate"] == 0.0
        assert settings["partial_application_factor"] == 0.0


# ---------------------------------------------------------------------------
# process_scenario_file  (integration — reads real TLR-549 source files)
# ---------------------------------------------------------------------------

_TLR549_DIR = (
    Path(__file__).resolve().parent.parent
    / "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/loop_risk_v2_2_0_full/TLR-549"
)
_ADOLESCENT_FILE = _TLR549_DIR / "Simulation-Configuration-TLR-549_30_adolescent_profile_v1.json"
_MEDIAN_FILE     = _TLR549_DIR / "Simulation-Configuration-TLR-549_30_median_profile_v1.json"


@pytest.mark.skipif(not _ADOLESCENT_FILE.exists(), reason="TLR-549 source files not present")
class TestProcessScenarioFile:

    def _read_dest(self, tmp_path, src_file, mode, reusable_root):
        dest = tmp_path / src_file.name
        process_scenario_file(src_file, dest, mode, reusable_root)
        with open(dest) as f:
            return json.load(f)

    # --- patient mode (adolescent file: patient_model.metabolism_settings absent) ---

    def test_patient_mode_adds_ms_when_absent(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _ADOLESCENT_FILE, "patient", reusable_root)
        for item in data["override_config"]:
            if item.get("controller") is None:
                continue  # null-controller items may still have patient set
            ms = item["patient"]["patient_model"].get("metabolism_settings")
            assert ms is not None, f"metabolism_settings missing in {item.get('sim_id')}"
            if isinstance(ms, dict):
                assert ms["patient_insulin_type"] == "fiasp"
            elif isinstance(ms, str):
                assert "fiasp" in ms

    def test_patient_mode_controller_unchanged(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _ADOLESCENT_FILE, "patient", reusable_root)
        post_item = next(
            i for i in data["override_config"]
            if i.get("sim_id", "").startswith("post-")
        )
        # controller.settings should still be a string ref (not inlined)
        assert isinstance(post_item["controller"]["settings"], str)

    # --- controller mode (median file: patient_model.metabolism_settings present as string ref) ---

    def test_controller_mode_patient_ms_unchanged(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _MEDIAN_FILE, "controller", reusable_root)
        pre_item = next(
            i for i in data["override_config"]
            if i.get("sim_id", "").startswith("pre-Loop_")
        )
        ms = pre_item["patient"]["patient_model"]["metabolism_settings"]
        # Should still be the original profile ref, no fiasp in it
        assert isinstance(ms, str)
        assert "fiasp" not in ms

    def test_controller_mode_post_loop_model_set(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _MEDIAN_FILE, "controller", reusable_root)
        post_item = next(
            i for i in data["override_config"]
            if i.get("sim_id", "").startswith("post-")
        )
        settings = post_item["controller"]["settings"]
        assert isinstance(settings, dict)
        assert settings["model"] == "fiasp"

    def test_controller_mode_null_controller_preserved(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _MEDIAN_FILE, "controller", reusable_root)
        null_item = next(
            i for i in data["override_config"]
            if i.get("controller") is None
        )
        assert null_item["controller"] is None

    # --- both mode ---

    def test_both_mode_patient_ms_fiasp(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _MEDIAN_FILE, "both", reusable_root)
        pre_item = next(
            i for i in data["override_config"]
            if i.get("sim_id", "").startswith("pre-Loop_")
        )
        ms = pre_item["patient"]["patient_model"]["metabolism_settings"]
        assert "fiasp" in (ms if isinstance(ms, str) else ms.get("patient_insulin_type", ""))

    def test_both_mode_controller_model_fiasp(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _MEDIAN_FILE, "both", reusable_root)
        post_item = next(
            i for i in data["override_config"]
            if i.get("sim_id", "").startswith("post-")
        )
        assert post_item["controller"]["settings"]["model"] == "fiasp"

    def test_both_mode_null_controller_preserved(self, tmp_path, reusable_root):
        data = self._read_dest(tmp_path, _MEDIAN_FILE, "both", reusable_root)
        null_item = next(
            i for i in data["override_config"]
            if i.get("controller") is None
        )
        assert null_item["controller"] is None


# ---------------------------------------------------------------------------
# get_matched_dirs
# ---------------------------------------------------------------------------

class TestGetMatchedDirs:

    def test_exact_match(self, tmp_path):
        (tmp_path / "TLR-549").mkdir()
        result = get_matched_dirs(tmp_path, ["TLR-549"])
        assert len(result) == 1
        assert result[0].name == "TLR-549"

    def test_suffix_match(self, tmp_path):
        (tmp_path / "TLR-845_10").mkdir()
        (tmp_path / "TLR-845_20").mkdir()
        result = get_matched_dirs(tmp_path, ["TLR-845"])
        names = [d.name for d in result]
        assert "TLR-845_10" in names
        assert "TLR-845_20" in names

    def test_no_false_positives(self, tmp_path):
        (tmp_path / "TLR-54").mkdir()  # should NOT match TLR-549
        result = get_matched_dirs(tmp_path, ["TLR-549"])
        assert len(result) == 0

    def test_missing_tlr_not_in_results(self, tmp_path):
        result = get_matched_dirs(tmp_path, ["TLR-9999"])
        assert len(result) == 0
