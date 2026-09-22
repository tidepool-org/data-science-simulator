"""
TRSET-40 regression tests: physical activity (PA) metabolism parameters
(w_hr, a, tau, n) must flow through ScenarioParserV2's `reusable.` pointer
resolution instead of being silently dropped.

Unlike tests/test_physical_activity_model.py, these tests exercise the actual
parser path (ScenarioParserV2 -> get_sims()), since that is the code path the
original defect lived in and a hand-built PatientConfig would not catch it.
"""

import datetime
import json
import os
import shutil
import tempfile

import pytest

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


class TestPAReferenceFormsResolveEquivalently:
    """AC 1, 3: bare string, list-wrapped, and activity_ref profile references
    must all resolve to the same PA timeline and the same four metabolism
    parameters -- the values an author would get by typing them inline.
    """

    PROFILE_METABOLISM_PARAMS = {"w_hr": 0.8, "a": -0.0015, "tau": 0.995, "n": 20}
    PROFILE_ENTRY = {
        "start_time": "8/15/2019 13:00:00",
        "activity": "running",
        "duration": 20,
        "intensity": "moderate",
        "expected_hr": 140,
    }

    @pytest.fixture
    def temp_config_dir(self):
        """Reusable-file tree, following the tests/test_noisy_sensor_override.py
        convention, plus a PA profile carrying metabolism_parameters."""
        temp_dir = tempfile.mkdtemp()

        reusable_dir = os.path.join(temp_dir, "reusable")
        simulations_dir = os.path.join(reusable_dir, "simulations")
        glucose_dir = os.path.join(reusable_dir, "glucose")
        metabolism_dir = os.path.join(reusable_dir, "metabolism_settings")
        loop_settings_dir = os.path.join(reusable_dir, "loop_settings")
        pa_profiles_dir = os.path.join(reusable_dir, "physical_activities", "profiles")

        for d in (simulations_dir, glucose_dir, metabolism_dir, loop_settings_dir, pa_profiles_dir):
            os.makedirs(d)

        # History must end at base_config's time_to_calculate_at (8/15/2019 12:00:00).
        _write_json(os.path.join(glucose_dir, "flat_110.json"), {
            "datetime": {"0": "8/15/2019 11:55:00", "1": "8/15/2019 12:00:00"},
            "value": {"0": 110, "1": 110},
        })

        # No w_hr/a/tau/n here: PA parameters must come from the profile, not
        # from this tier, so this test proves resolution through the profile.
        _write_json(os.path.join(metabolism_dir, "test_v1.json"), {
            "patient_insulin_type": "rapid_acting_adult",
            "basal_rate": {"start_times": ["0:00:00"], "values": [1.0]},
            "carb_insulin_ratio": {"start_times": ["0:00:00"], "values": [10.0]},
            "insulin_sensitivity_factor": {"start_times": ["0:00:00"], "values": [50.0]},
        })

        _write_json(os.path.join(loop_settings_dir, "test_v1.json"), {
            "model": "rapid_acting_adult",
            "momentum_data_interval": 15,
            "suspend_threshold": 70,
            "max_basal_rate": 4.0,
            "max_bolus": 10.0,
        })

        _write_json(os.path.join(pa_profiles_dir, "test_pa_profile_v1.json"), {
            "metabolism_parameters": self.PROFILE_METABOLISM_PARAMS,
            "physical_activity_entries": [self.PROFILE_ENTRY],
        })
        _write_json(os.path.join(pa_profiles_dir, "no_params_v1.json"), {
            "physical_activity_entries": [self.PROFILE_ENTRY],
        })

        _write_json(os.path.join(simulations_dir, "base_test.json"), {
            "sim_id": "base_test",
            "time_to_calculate_at": "8/15/2019 12:00:00",
            "duration_hours": 2.0,
            "patient": {
                "sensor": {"glucose_history": "reusable.glucose.flat_110", "type": "IdealSensor", "parameters": {}},
                "pump": {
                    "metabolism_settings": "reusable.metabolism_settings.test_v1",
                    "bolus_entries": [],
                    "carb_entries": [],
                    "target_range": {"start_times": ["0:00:00"], "lower_values": [70], "upper_values": [90]},
                },
                "patient_model": {
                    "metabolism_settings": "reusable.metabolism_settings.test_v1",
                    "glucose_history": "reusable.glucose.flat_110",
                    "bolus_entries": [],
                    "carb_entries": [],
                    "physical_activity_entries": [],
                },
            },
            "controller": {"id": "swift", "settings": "reusable.loop_settings.test_v1", "automation_control_timeline": []},
        })

        yield temp_dir

        shutil.rmtree(temp_dir)

    def _build_sim(self, temp_config_dir, sim_id, physical_activity_entries):
        scenario_config = {
            "metadata": {
                "risk-id": "TRSET-40-TEST",
                "simulation_id": sim_id,
                "risk_description": "TRSET-40 PA reference form test",
                "config_format_version": "v1.0",
            },
            "base_config": "reusable.simulations.base_test",
            "override_config": [
                {
                    "sim_id": sim_id,
                    "patient": {"patient_model": {"physical_activity_entries": physical_activity_entries}},
                }
            ],
        }
        scenario_path = os.path.join(temp_config_dir, f"scenario_{sim_id}.json")
        _write_json(scenario_path, scenario_config)

        parser = ScenarioParserV2(path_to_json_config=scenario_path, pointer_object_dir=temp_config_dir)
        sims = parser.get_sims()
        return sims[sim_id]

    def test_bare_string_list_wrapped_and_activity_ref_forms_match(self, temp_config_dir):
        sim_bare = self._build_sim(
            temp_config_dir, "pa_bare_string", "reusable.physical_activities.profiles.test_pa_profile_v1")
        sim_list = self._build_sim(
            temp_config_dir, "pa_list_wrapped", ["reusable.physical_activities.profiles.test_pa_profile_v1"])
        sim_ref = self._build_sim(
            temp_config_dir, "pa_activity_ref",
            [{"start_time": "8/15/2019 13:00:00", "activity_ref": "reusable.physical_activities.profiles.test_pa_profile_v1"}])

        for sim in (sim_bare, sim_list, sim_ref):
            patient_config = sim.virtual_patient.patient_config

            assert patient_config.w_hr == self.PROFILE_METABOLISM_PARAMS["w_hr"]
            assert patient_config.a == self.PROFILE_METABOLISM_PARAMS["a"]
            assert patient_config.tau == self.PROFILE_METABOLISM_PARAMS["tau"]
            assert patient_config.n == self.PROFILE_METABOLISM_PARAMS["n"]

            pa_event = patient_config.pa_timeline.get_only_event()
            assert pa_event.activity == self.PROFILE_ENTRY["activity"]
            assert pa_event.duration == self.PROFILE_ENTRY["duration"]
            assert pa_event.expected_hr == self.PROFILE_ENTRY["expected_hr"]

    def test_entries_present_with_no_resolvable_parameters_raises(self, temp_config_dir):
        """AC 4: a profile with PA entries but no metabolism_parameters, and no
        model_config/metabolism_settings override, must raise -- never fall
        through to w_hr=0.0."""
        with pytest.raises(ValueError, match="w_hr"):
            self._build_sim(
                temp_config_dir, "pa_no_params", "reusable.physical_activities.profiles.no_params_v1")


class TestJoggingProfileValuesFlowThroughParser:
    """AC 2: the real jogging_v1.json profile's metabolism parameters must
    reach the patient config through the bare-string pointer form."""

    def test_jogging_profile_values(self, tmp_path):
        scenario_config = {
            "metadata": {
                "risk-id": "TRSET-40-TEST",
                "simulation_id": "jog_ac2",
                "risk_description": "TRSET-40 jogging profile values",
                "config_format_version": "v1.0",
            },
            "base_config": "reusable.simulations.activity_presets.ap_jog_median_2_0_v1",
            "override_config": [
                {
                    "sim_id": "jog_ac2",
                    "patient": {
                        "patient_model": {
                            "physical_activity_entries": "reusable.physical_activities.profiles.jogging_v1",
                        }
                    },
                    "controller": None,
                }
            ],
        }
        scenario_path = tmp_path / "scenario.json"
        _write_json(str(scenario_path), scenario_config)

        # No pointer_object_dir override: resolves against the repo's real
        # scenario_configs/tidepool_risk_v2/ tree (the affected configs).
        parser = ScenarioParserV2(path_to_json_config=str(scenario_path))
        sims = parser.get_sims()
        patient_config = sims["jog_ac2"].virtual_patient.patient_config

        assert patient_config.w_hr == 1.0
        assert patient_config.a == -0.002462
        assert patient_config.tau == 0.9989
        assert patient_config.n == 28


class TestGetPatientConfigHasSingleResolutionPoint:
    """AC 5: get_patient_config must carry no independent fallback for
    w_hr/a/tau/n -- it must raise (not silently default) if build_model_from_config
    did not resolve one, proving there is exactly one place parameters are set."""

    def test_get_patient_config_raises_if_w_hr_was_never_resolved(self):
        parser = ScenarioParserV2()
        parser.patient_model_glucose_history = None
        parser.patient_model = {
            "basal_rate_schedule": None,
            "carb_ratio_schedule": None,
            "insulin_sensitivity_schedule": None,
            "glucose_sensitivity_factor_schedule": None,
            "basal_blood_glucose_schedule": None,
            "insulin_production_rate_schedule": None,
            "carb_timeline": None,
            "bolus_timeline": None,
            "action_timeline": None,
            "pa_timeline": None,
            # w_hr deliberately absent -- build_model_from_config always sets
            # it (or raises); get_patient_config must not paper over that.
            "a": -0.002462,
            "tau": 0.9989,
            "n": 28,
        }

        with pytest.raises(KeyError):
            parser.get_patient_config()


class TestExtractMetabolismParamsSwallowsNothing:
    """AC 6: any failure loading or reading a referenced PA profile must raise,
    naming the entry -- not be logged and skipped."""

    def test_unreadable_profile_raises_naming_the_entry(self):
        parser = ScenarioParserV2()
        missing_ref = "reusable.physical_activities.profiles.nonexistent_profile_xyz"

        with pytest.raises(ValueError, match="nonexistent_profile_xyz"):
            parser.extract_metabolism_params_from_pa_profiles([missing_ref])


class TestTLR000PARegressionProof:
    """AC 7: using the real loop_risk_v2_0/test/TLR-000-pa/ configs, "Activity
    preset with exercise" and "Activity preset no exercise" must produce
    materially different glucose traces.

    The controller is overridden to None (DoNothingController) to isolate the
    PA metabolism effect from Loop dosing decisions -- Loop is deliberately
    never given physical activity directly (it only sees the resulting
    glucose), so this reproduces the physiological defect without depending
    on the Swift Loop dylib being available in the test environment.
    """

    TLR_000_PA_DIR = os.path.join(
        os.path.dirname(__file__), "..",
        "scenario_configs", "tidepool_risk_v2", "loop_risk_v2_0", "test", "TLR-000-pa")

    def _run_isolated(self, config_path):
        parser = ScenarioParserV2(path_to_json_config=config_path)
        for override in parser.override_configs:
            override["controller"] = None
        sims = parser.get_sims()
        sim = list(sims.values())[0]
        sim.multiprocess = False
        sim.run()
        return sim.get_results_df()

    def test_exercise_vs_no_exercise_glucose_traces_diverge_at_activity_onset(self):
        exercise_path = os.path.join(
            self.TLR_000_PA_DIR, "Simulation-Configuration-TLR-000-pa_a_median_v1.json")
        no_exercise_path = os.path.join(
            self.TLR_000_PA_DIR, "Simulation-Configuration-TLR-000-pa_median_v1.json")

        df_exercise = self._run_isolated(exercise_path)
        df_no_exercise = self._run_isolated(no_exercise_path)

        bg_diff = df_exercise["bg"] - df_no_exercise["bg"]

        # jogging_v1's activity starts at 13:00 and the traces track closely
        # beforehand (settings-only drift, a few mg/dL); the fixed PA effect
        # should pull the exercise trace materially lower by two hours later.
        # Pre-fix (w_hr silently 0.0) this checkpoint sits around -13.6 mg/dL;
        # the threshold below only passes once the exercise effect is real.
        checkpoint = datetime.datetime(2019, 8, 15, 15, 0, 0)
        assert bg_diff[checkpoint] <= -40.0, (
            "Expected the 'with exercise' trace to run materially lower than "
            "'no exercise' two hours after activity onset; got a diff of "
            f"{bg_diff[checkpoint]} mg/dL, consistent with the PA metabolism "
            "effect still being dropped."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
