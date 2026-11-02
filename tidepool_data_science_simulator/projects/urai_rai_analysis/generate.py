"""Generate paired URAI / RAI autobolus scenarios for the A/B study (TRSET-74, TRSET-77).

Usage:
    URAI_RAI_PATIENT_POOL=/path/to/raw_files python -m \
        tidepool_data_science_simulator.projects.urai_rai_analysis.generate --config run_config.json

    # replace failed pairs without changing any other draw (see ``replacement.py``)
    python -m tidepool_data_science_simulator.projects.urai_rai_analysis.generate \
        --replace <original run folder> --replacements pairs.csv --run-name <new_run_name>
"""
import argparse
import csv
import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import DATETIME_FORMAT, POINTER_OBJ_DIR
from tidepool_data_science_simulator.projects.urai_rai_analysis.cohorts import age_band, derive_seed, draw_cohort
from tidepool_data_science_simulator.projects.urai_rai_analysis.config import (
    RunConfig, load_run_config, parse_run_config, resolve_pool_dir,
)
from tidepool_data_science_simulator.projects.urai_rai_analysis.eligibility import (
    EligiblePatient, PoolEvaluation, check_pump_matches_model, eligible_table, evaluate_pool,
)
from tidepool_data_science_simulator.projects.urai_rai_analysis.manifest import (
    ELIGIBLE_POOL_COLUMNS, EXCLUSION_COLUMNS, MANIFEST_COLUMNS, meal_schedule_json, write_csv,
)
from tidepool_data_science_simulator.projects.urai_rai_analysis.max_bolus import controller_max_bolus
from tidepool_data_science_simulator.projects.urai_rai_analysis.meals import MealSchedule, generate_meals
from tidepool_data_science_simulator.projects.urai_rai_analysis.scenario_builder import (
    ARM_INSULIN_MODEL, build_scenario, controller_settings, glucose_seed_rows,
)

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_ROOT = os.path.normpath(POINTER_OBJ_DIR)
SCENARIO_SUBDIR = os.path.join("loop_risk_v2_0", "loop_risk_v2_urai_rai")
GLUCOSE_SUBDIR = os.path.join("reusable", "glucose", "urai_rai")
MANIFEST_NAME = "manifest.csv"
EXCLUSIONS_NAME = "exclusions.csv"
ELIGIBLE_POOL_NAME = "eligible_pool.csv"
CONFIG_COPY_NAME = "run_config.json"
PATIENT_DIR_PREFIX = "TLR-000_"  # the risk runner scans ``TLR-*`` folders; one folder per patient (TRSET-77 AC 8)


def patient_dir_name(patient_id: str) -> str:
    """Name of the folder holding all of one patient's scenario files."""
    return f"{PATIENT_DIR_PREFIX}{patient_id}"


@dataclass(frozen=True)
class RunResult:
    """Where a run's outputs went.

    ``run_dir`` is the scenario root the risk runner is pointed at. It holds one ``TLR-000_<patient>`` folder per
    patient (the scenario JSONs) and the run-level records: manifest, ``exclusions.csv``, ``eligible_pool.csv`` and
    ``run_config.json``.
    """
    run_dir: str
    glucose_dir: str
    manifest_path: str
    manifest: pd.DataFrame
    exclusions: pd.DataFrame
    eligible_pool: pd.DataFrame

    def scenario_path(self, patient_id: str, file_name: str) -> str:
        """Path of one scenario file."""
        return os.path.join(self.run_dir, patient_dir_name(patient_id), file_name)

    def patient_dirs(self) -> List[str]:
        """The ``TLR-000_<patient>`` folders of the run, sorted."""
        return sorted(os.path.join(self.run_dir, d) for d in os.listdir(self.run_dir) if d.startswith(PATIENT_DIR_PREFIX))


@dataclass(frozen=True)
class PlannedScenario:
    """One scenario file and its manifest row, built in memory before anything is written."""
    patient_id: str
    file_name: str
    scenario: Dict[str, Any]
    row: Dict[str, Any]


@dataclass(frozen=True)
class PoolInputs:
    """Everything the pool contributes, ready for sampling: the eligibility pass and the eligible-patient table."""
    evaluation: PoolEvaluation
    table: pd.DataFrame
    patients: Dict[str, EligiblePatient] = field(compare=False)


def _prepare_dir(path: str) -> None:
    if os.path.isdir(path) and os.listdir(path):
        raise FileExistsError(f"Output directory is not empty: {path}. Remove it or choose another run_name.")
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _write_glucose_seed(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime", "value", "units"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_pool(cfg: RunConfig, pool_path: str) -> PoolInputs:
    """Read the pool, apply E1-E6 and check the eligible patients' pump and patient-model settings agree.

    Raises
    ------
    ValueError
        If an eligible patient's pump and patient-model settings differ (S3, S20), or a file cannot be decided.
    """
    evaluation = evaluate_pool(pool_path, cfg)
    for patient in evaluation.eligible:
        check_pump_matches_model(patient)
    return PoolInputs(evaluation, eligible_table(evaluation),
                      {p.record.patient_id: p for p in evaluation.eligible})


def build_planned(cfg: RunConfig, cohort_name: str, announced: bool, patient: EligiblePatient, meal_seed: int,
                  schedule: MealSchedule, shortfall_note: str, also_in_cohorts: str,
                  extra: Optional[Dict[str, Any]] = None) -> List[PlannedScenario]:
    """Both arms' scenario files and manifest rows for one patient in one cohort."""
    record, flags = patient.record, patient.flags
    max_bolus = controller_max_bolus(flags, cfg.max_bolus)
    planned = []
    for arm in ARM_INSULIN_MODEL:
        scenario = build_scenario(record, cfg, arm, cohort_name, list(schedule.meals), announced, max_bolus)
        sim_id = scenario["override_config"][0]["sim_id"]
        row = {
            "file_name": f"Simulation-Configuration-TLR-{sim_id}.json", "patient_id": record.patient_id,
            "cohort": cohort_name, "also_in_cohorts": also_in_cohorts, "pair_id": f"{cohort_name}_{record.patient_id}",
            "arm": arm, "insulin_model": ARM_INSULIN_MODEL[arm], "meals_announced": announced,
            "age_band": age_band(record.age), "age": record.age, "isf": record.isf_for_cohorts,
            "derived_seed": meal_seed, "meal_schedule": meal_schedule_json(schedule.meals),
            "has_seconds_pair": schedule.has_seconds_pair,
            "controller_settings": json.dumps(controller_settings(record, cfg, arm, max_bolus)),
            "shortfall_note": shortfall_note, "default_suspend": patient.default_suspend,
            "max_bolus_mode": "static" if cfg.max_bolus.static_u is not None else "per_patient",
            "tdd_estimate_u": flags.tdd_estimate_u, "max_bolus_u": flags.max_bolus_u,
            "max_bolus_clamped": flags.max_bolus_clamped, "meal80_fits": flags.meal80_fits,
            **(extra or {}),
        }
        planned.append(PlannedScenario(record.patient_id, row["file_name"], scenario, row))
    return planned


def _eligible_pool_rows(evaluation: PoolEvaluation) -> List[Dict[str, Any]]:
    rows = []
    for p in evaluation.eligible:
        r = p.record
        rows.append({
            "patient_id": r.patient_id, "age": r.age, "suspend_threshold": r.suspend_threshold,
            "basal_rate": json.dumps(list(r.basal.values)), "carb_ratio": json.dumps(list(r.carb_ratio.values)),
            "isf": json.dumps(list(r.isf.values)),
            "target_range_minimum": json.dumps(list(r.target_range.lower_values)),
            "target_range_maximum": json.dumps(list(r.target_range.upper_values)),
            "starting_glucose": r.glucose_tokens[0], "default_suspend": p.default_suspend,
            "tdd_estimate_u": p.flags.tdd_estimate_u, "max_bolus_u": p.flags.max_bolus_u,
            "max_bolus_clamped": p.flags.max_bolus_clamped, "meal80_fits": p.flags.meal80_fits,
        })
    return rows


def write_run(cfg: RunConfig, root: str, inputs: PoolInputs, planned: List[PlannedScenario],
              manifest_columns: List[str] = MANIFEST_COLUMNS) -> RunResult:
    """Write a run: scenarios, glucose seeds, manifest, exclusions, eligible pool and the config copy.

    Nothing is written until every check and every scenario build has finished; the caller passes the finished plan.

    Raises
    ------
    FileExistsError
        If an output directory is not empty.
    """
    run_dir = os.path.join(root, SCENARIO_SUBDIR, cfg.run_name)
    glucose_dir = os.path.join(root, GLUCOSE_SUBDIR, cfg.run_name)
    _prepare_dir(run_dir)
    _prepare_dir(glucose_dir)

    seeds_written = set()
    for item in planned:
        patient_dir = os.path.join(run_dir, patient_dir_name(item.patient_id))
        os.makedirs(patient_dir, exist_ok=True)
        _write_json(os.path.join(patient_dir, item.file_name), item.scenario)
        if item.patient_id not in seeds_written:
            record = inputs.patients[item.patient_id].record
            _write_glucose_seed(os.path.join(glucose_dir, f"{item.patient_id}.csv"), glucose_seed_rows(record, cfg))
            seeds_written.add(item.patient_id)

    _write_json(os.path.join(run_dir, CONFIG_COPY_NAME), cfg.raw)
    manifest_path = os.path.join(run_dir, MANIFEST_NAME)
    manifest = write_csv([item.row for item in planned], manifest_path, manifest_columns)
    exclusions = write_csv(
        [{"patient_id": e.patient_id, "rule": e.rule, "detail": e.detail} for e in inputs.evaluation.exclusions],
        os.path.join(run_dir, EXCLUSIONS_NAME), EXCLUSION_COLUMNS)
    eligible_pool = write_csv(_eligible_pool_rows(inputs.evaluation), os.path.join(run_dir, ELIGIBLE_POOL_NAME),
                              ELIGIBLE_POOL_COLUMNS)
    logger.info("Wrote %d scenarios for %d patients to %s", len(planned), len(seeds_written), run_dir)
    return RunResult(run_dir, glucose_dir, manifest_path, manifest, exclusions, eligible_pool)


def generate_run(config: Union[RunConfig, Dict[str, Any]], pool_dir: Optional[str] = None,
                 scenario_root: Optional[str] = None) -> RunResult:
    """Generate every scenario of a run, its manifest and its eligibility records.

    Order of work: eligibility (E1-E6) on the whole pool, the pump/patient-model check on eligible patients, cohort
    sampling, then meal generation. Nothing is written until all of that has finished.

    Parameters
    ----------
    config: RunConfig or raw config dict
    pool_dir: patient pool directory; defaults to the config's environment variable
    scenario_root: root holding ``loop_risk_v2_0`` and ``reusable``; defaults to the repo's
        ``scenario_configs/tidepool_risk_v2``

    Raises
    ------
    ValueError, FileNotFoundError, FileExistsError
        On a missing pool or seed, an undecidable pool file, a settings mismatch, a meal-rule failure, a cohort
        shortfall that is not accepted, or a non-empty output directory.
    """
    cfg = config if isinstance(config, RunConfig) else parse_run_config(config)
    inputs = prepare_pool(cfg, resolve_pool_dir(cfg, pool_dir))
    draws = [draw_cohort(inputs.table, spec, cfg) for spec in cfg.cohorts]
    cohorts_by_patient: Dict[str, List[str]] = {}
    for draw in draws:
        for pid in draw.patient_ids:
            cohorts_by_patient.setdefault(pid, []).append(draw.spec.name)

    run_start = datetime.datetime.strptime(cfg.run_start, DATETIME_FORMAT)
    planned: List[PlannedScenario] = []
    for draw in draws:
        for pid in draw.patient_ids:
            seed = derive_seed(cfg.master_seed, "draw", draw.spec.name, pid)
            schedule = generate_meals(seed, cfg.meals, run_start, cfg.duration_hours)
            planned += build_planned(
                cfg, draw.spec.name, draw.spec.announced, inputs.patients[pid], seed, schedule,
                draw.shortfall_note or "", ";".join(c for c in cohorts_by_patient[pid] if c != draw.spec.name))
    return write_run(cfg, scenario_root or DEFAULT_SCENARIO_ROOT, inputs, planned)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate URAI-RAI paired autobolus scenarios")
    parser.add_argument("--config", help="Path to the run config JSON (a new run)")
    parser.add_argument("--pool-dir", help="Patient pool directory (default: the config's environment variable)")
    parser.add_argument("--scenario-root", help="Override the scenario_configs/tidepool_risk_v2 output root")
    parser.add_argument("--replace", metavar="RUN_DIR", help="Original run folder to replace pairs from")
    parser.add_argument("--replacements", help="CSV with columns pair_id,reason (with --replace)")
    parser.add_argument("--run-name", help="Name of the replacement run (with --replace)")
    parser.add_argument("--prior-run", action="append", default=[], metavar="RUN_DIR",
                        help="Earlier replacement run folder whose replacements must not be reused (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if bool(args.config) == bool(args.replace):
        parser.error("give either --config (a new run) or --replace (a replacement run)")
    if args.replace:
        from tidepool_data_science_simulator.projects.urai_rai_analysis.replacement import (
            generate_replacement_run, read_replacements)
        if not (args.replacements and args.run_name):
            parser.error("--replace needs --replacements and --run-name")
        result = generate_replacement_run(args.replace, read_replacements(args.replacements), args.run_name,
                                          args.pool_dir, args.scenario_root, args.prior_run)
    else:
        result = generate_run(load_run_config(args.config), args.pool_dir, args.scenario_root)
    print(f"{len(result.manifest)} scenarios written to {result.run_dir}")


if __name__ == "__main__":
    main()
