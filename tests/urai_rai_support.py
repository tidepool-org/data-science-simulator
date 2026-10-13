"""Shared helpers for the TRSET-74 URAI-RAI generator tests (synthetic patients, tmp scenario root, parser access)."""
import contextlib
import csv
import io
import os
from typing import Dict, List, Optional, Sequence

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import POINTER_OBJ_DIR, ScenarioParserV2

REPO_SCENARIO_ROOT = os.path.normpath(POINTER_OBJ_DIR)
N_COLS = 150  # numbered value columns written per synthetic row


def make_scenario_root(tmp_path) -> str:
    """A scenario root whose reusable bases are symlinked from the repo; generated output lands in tmp_path."""
    root = tmp_path / "scenario_root"
    glucose = root / "reusable" / "glucose"
    glucose.mkdir(parents=True)
    for sub in ("simulations", "loop_settings", "metabolism_settings"):
        os.symlink(os.path.join(REPO_SCENARIO_ROOT, "reusable", sub), root / "reusable" / sub)
    os.symlink(os.path.join(REPO_SCENARIO_ROOT, "reusable", "glucose", "flat_110_12hr.csv"),
               glucose / "flat_110_12hr.csv")
    return str(root)


def _row(name: str, tokens: Sequence[str], layout: str, scalar: bool) -> List[str]:
    """One raw CSV row. 'observed' = scalars in settings, series from column 0; 'shifted' = the opposite."""
    tokens = [str(t) for t in tokens]
    cells = [""] * (N_COLS + 1)  # settings + numbered columns
    in_settings = (layout == "observed") if scalar else (layout == "shifted")
    start = 0 if in_settings else 1
    cells[start:start + len(tokens)] = tokens
    return [name] + cells


def write_raw_patient(path: str, age: float, isf: Sequence[float] = (60.0,), layout: str = "observed",
                      basal=((("00:00:00",), (0.9,))), carb_ratio=((("00:00:00",), (12.0,))),
                      isf_times: Sequence[str] = ("00:00:00",), target=(("00:00:00",), (100.0,), (120.0,)),
                      suspend_threshold: float = 70.0, glucose: Optional[Sequence[str]] = None,
                      extra_rows: Optional[Dict[str, Sequence[str]]] = None) -> None:
    """Write a synthetic raw-format patient CSV (no real data). ``glucose`` are source value strings."""
    glucose = list(glucose) if glucose is not None else [f"{100 + i * 0.137:.3f}" for i in range(N_COLS)]
    rows = [
        _row("age", [age], layout, True),
        _row("suspend_threshold", [suspend_threshold], layout, True),
        _row("model", ["[360.0, 65]"], layout, True),
        _row("basal_rate_start_times", basal[0], layout, False),
        _row("basal_rate_values", basal[1], layout, False),
        _row("actual_basal_rates", basal[1], layout, False),
        _row("carb_ratio_start_times", carb_ratio[0], layout, False),
        _row("carb_ratio_values", carb_ratio[1], layout, False),
        _row("actual_carb_ratios", carb_ratio[1], layout, False),
        _row("sensitivity_ratio_start_times", isf_times, layout, False),
        _row("sensitivity_ratio_values", isf, layout, False),
        _row("actual_sensitivity_ratios", isf, layout, False),
        _row("target_range_start_times", target[0], layout, False),
        _row("target_range_minimum_values", target[1], layout, False),
        _row("target_range_maximum_values", target[2], layout, False),
        _row("actual_blood_glucose", glucose, layout, False),
    ]
    for name, tokens in (extra_rows or {}).items():
        rows.append(_row(name, tokens, layout, False))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting_name", "settings"] + [str(i) for i in range(N_COLS)])
        writer.writerows(rows)


def load_sim(scenario_path: str, scenario_root: str):
    """Load the single simulation in a generated scenario file through ScenarioParserV2 (overrides applied)."""
    parser = ScenarioParserV2(path_to_json_config=scenario_path, pointer_object_dir=scenario_root + "/")
    with contextlib.redirect_stdout(io.StringIO()):
        sims = parser.get_sims()
    assert len(sims) == 1
    return next(iter(sims.values()))


def schedule_entries(schedule) -> List[tuple]:
    """[(start_time, value)] of a parsed SettingSchedule24Hr, in schedule order; value is the raw number or range."""
    entries = []
    for (start, _end), measure in schedule.schedule.items():
        value = (measure.min_value, measure.max_value) if hasattr(measure, "min_value") else measure.value
        entries.append((start, value))
    return entries


def parsed_settings(sim) -> Dict[str, Dict[str, list]]:
    """Parsed basal / CIR / ISF / target schedules for 'pump' and 'patient_model'."""
    vp = sim.virtual_patient
    pump, patient = vp.pump.pump_config, vp.patient_config
    out = {
        "pump": {
            "basal": schedule_entries(pump.basal_schedule),
            "carb_ratio": schedule_entries(pump.carb_ratio_schedule),
            "isf": schedule_entries(pump.insulin_sensitivity_schedule),
            "target": schedule_entries(pump.target_range_schedule),
        },
        "patient_model": {
            "basal": schedule_entries(patient.basal_schedule),
            "carb_ratio": schedule_entries(patient.carb_ratio_schedule),
            "isf": schedule_entries(patient.insulin_sensitivity_schedule),
        },
    }
    return out
