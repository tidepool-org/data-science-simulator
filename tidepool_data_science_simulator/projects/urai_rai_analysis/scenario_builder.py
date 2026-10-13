"""Build paired URAI / RAI scenario configs and glucose seeds from a patient draw (TRSET-74)."""
import datetime
from typing import Any, Dict, List

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import DATETIME_FORMAT
from tidepool_data_science_simulator.projects.urai_rai_analysis.config import RunConfig
from tidepool_data_science_simulator.projects.urai_rai_analysis.meals import Meal
from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import PatientRecord

BASE_CONFIG_POINTER = "reusable.simulations.base_median_autobolus_v1_2"
GLUCOSE_POINTER_PREFIX = "reusable.glucose.urai_rai"
ACCEPT_RECOMMENDATION = "accept_recommendation"

# Arm name -> insulin model. Both the controller's model and the patient's insulin type flip together.
ARM_INSULIN_MODEL = {"urai": "fiasp", "rai": "rapid_acting_adult"}


def glucose_pointer(run_name: str, patient_id: str) -> str:
    """Pointer string the parser resolves to the patient's glucose seed CSV."""
    return f"{GLUCOSE_POINTER_PREFIX}.{run_name}.{patient_id}"


def glucose_seed_rows(record: PatientRecord, config: RunConfig) -> List[Dict[str, str]]:
    """Rows (datetime, value, units) of the glucose seed.

    Selection rule: the first ``glucose_seed_points`` values of the patient's ``actual_blood_glucose`` row, in
    column order, assigned in order to a 5-minute grid starting at ``glucose_seed_start``. Values keep their source
    text (no rounding or reformatting).
    """
    start = datetime.datetime.strptime(config.glucose_seed_start, DATETIME_FORMAT)
    step = datetime.timedelta(minutes=config.glucose_interval_min)
    return [
        {"datetime": (start + i * step).strftime(DATETIME_FORMAT), "value": token, "units": "mg/dL"}
        for i, token in enumerate(record.glucose_tokens)
    ]


def _entries(meals) -> List[Dict[str, Any]]:
    return [
        {"start_time": m.time.strftime(DATETIME_FORMAT), "value": m.carbs_g, "duration": m.absorption_min,
         "type": "carb"}
        for m in meals
    ]


def controller_settings(record: PatientRecord, config: RunConfig, arm: str) -> Dict[str, Any]:
    """Controller settings override for an arm."""
    return {
        "suspend_threshold": record.suspend_threshold,
        "partial_application_factor": config.partial_application_factor,
        "max_bolus": config.max_bolus,
        "model": ARM_INSULIN_MODEL[arm],
    }


def build_scenario(record: PatientRecord, config: RunConfig, arm: str, cohort_name: str, meals: List[Meal],
                   announced: bool) -> Dict[str, Any]:
    """Build one scenario config (base pointer plus a single override) for an arm.

    Every key overridden already exists in the base config, so no override adds a new key.
    """
    insulin_model = ARM_INSULIN_MODEL[arm]
    sim_id = f"urai_rai_{cohort_name}_{record.patient_id}_{arm}"
    carb_entries = _entries(meals)
    pointer = glucose_pointer(config.run_name, record.patient_id)

    def schedule(s):
        return {"start_times": list(s.start_times), "values": list(s.values)}

    def metabolism():
        return {
            "carb_insulin_ratio": schedule(record.carb_ratio),
            "basal_rate": schedule(record.basal),
            "insulin_sensitivity_factor": schedule(record.isf),
        }

    pump = {
        "target_range": {
            "start_times": list(record.target_range.start_times),
            "lower_values": list(record.target_range.lower_values),
            "upper_values": list(record.target_range.upper_values),
        },
        "metabolism_settings": metabolism(),
        "carb_entries": carb_entries if announced else [],
    }
    patient_model = {
        "glucose_history": pointer,
        "metabolism_settings": {**metabolism(), "patient_insulin_type": insulin_model},
        "carb_entries": carb_entries,
        "bolus_entries": (
            [{"time": c["start_time"], "value": ACCEPT_RECOMMENDATION} for c in carb_entries] if announced else []
        ),
    }
    return {
        "metadata": {
            "risk-id": f"TLR-URAI-RAI_{cohort_name}_{record.patient_id}",
            "simulation_id": sim_id,
            "risk_description": "URAI vs RAI autobolus paired comparison",
            "config_format_version": "v1.0",
        },
        "base_config": BASE_CONFIG_POINTER,
        "override_config": [{
            "sim_id": sim_id,
            "time_to_calculate_at": config.run_start,
            "duration_hours": config.duration_hours,
            "patient": {"sensor": {"glucose_history": pointer}, "pump": pump, "patient_model": patient_model},
            "controller": {"settings": controller_settings(record, config, arm)},
        }],
    }
