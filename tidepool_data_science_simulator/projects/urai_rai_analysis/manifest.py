"""Per-scenario manifest for a URAI-RAI run (TRSET-74). The join key for later focused outputs."""
import json
from typing import Any, Dict, List

import pandas as pd

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import DATETIME_FORMAT

MANIFEST_COLUMNS = [
    "file_name", "patient_id", "cohort", "also_in_cohorts", "pair_id", "arm", "insulin_model", "meals_announced",
    "age_band", "age", "isf", "derived_seed", "meal_schedule", "has_seconds_pair", "controller_settings",
    "shortfall_note", "default_suspend", "max_bolus_mode", "tdd_estimate_u", "max_bolus_u", "max_bolus_clamped",
    "meal80_fits",
]
REPLACEMENT_COLUMNS = MANIFEST_COLUMNS + ["replaces_pair_id", "replacement_reason"]
ELIGIBLE_POOL_COLUMNS = [
    "patient_id", "age", "suspend_threshold", "basal_rate", "carb_ratio", "isf", "target_range_minimum",
    "target_range_maximum", "starting_glucose", "default_suspend", "tdd_estimate_u", "max_bolus_u",
    "max_bolus_clamped", "meal80_fits",
]
EXCLUSION_COLUMNS = ["patient_id", "rule", "detail"]


def meal_schedule_json(meals) -> str:
    """Compact JSON of the meal schedule, including absorption times."""
    return json.dumps([
        {"time": m.time.strftime(DATETIME_FORMAT), "carbs_g": m.carbs_g, "absorption_min": m.absorption_min}
        for m in meals
    ])


def manifest_frame(rows: List[Dict[str, Any]], columns: List[str] = MANIFEST_COLUMNS) -> pd.DataFrame:
    """Manifest as a DataFrame with the fixed column order."""
    return pd.DataFrame(rows, columns=columns)


def write_csv(rows: List[Dict[str, Any]], path: str, columns: List[str]) -> pd.DataFrame:
    """Write rows as a CSV with a fixed column order (deterministic content) and return the frame."""
    frame = manifest_frame(rows, columns)
    frame.to_csv(path, index=False)
    return frame


def write_manifest(rows: List[Dict[str, Any]], path: str, columns: List[str] = MANIFEST_COLUMNS) -> pd.DataFrame:
    """Write the manifest CSV (deterministic content) and return it."""
    return write_csv(rows, path, columns)
