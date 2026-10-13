"""Per-scenario manifest for a URAI-RAI run (TRSET-74). The join key for later focused outputs."""
import json
from typing import Any, Dict, List

import pandas as pd

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import DATETIME_FORMAT

MANIFEST_COLUMNS = [
    "file_name", "patient_id", "cohort", "also_in_cohorts", "pair_id", "arm", "insulin_model", "meals_announced",
    "age_band", "age", "isf", "derived_seed", "meal_schedule", "has_seconds_pair", "controller_settings",
    "shortfall_note",
]


def meal_schedule_json(meals) -> str:
    """Compact JSON of the meal schedule, including absorption times."""
    return json.dumps([
        {"time": m.time.strftime(DATETIME_FORMAT), "carbs_g": m.carbs_g, "absorption_min": m.absorption_min}
        for m in meals
    ])


def manifest_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Manifest as a DataFrame with the fixed column order."""
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifest(rows: List[Dict[str, Any]], path: str) -> pd.DataFrame:
    """Write the manifest CSV (deterministic content) and return it."""
    frame = manifest_frame(rows)
    frame.to_csv(path, index=False)
    return frame
