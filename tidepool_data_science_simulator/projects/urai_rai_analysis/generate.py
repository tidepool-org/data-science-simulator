"""Generate paired URAI / RAI autobolus scenarios for the A/B study (TRSET-74).

Usage:
    URAI_RAI_PATIENT_POOL=/path/to/raw_files python -m \
        tidepool_data_science_simulator.projects.urai_rai_analysis.generate --config run_config.json
"""
import argparse
import csv
import datetime
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import DATETIME_FORMAT, POINTER_OBJ_DIR
from tidepool_data_science_simulator.projects.urai_rai_analysis.cohorts import age_band, derive_seed, draw_cohort
from tidepool_data_science_simulator.projects.urai_rai_analysis.config import (
    RunConfig, load_run_config, parse_run_config, resolve_pool_dir,
)
from tidepool_data_science_simulator.projects.urai_rai_analysis.manifest import meal_schedule_json, write_manifest
from tidepool_data_science_simulator.projects.urai_rai_analysis.meals import generate_meals
from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import parse_patient, scan_pool
from tidepool_data_science_simulator.projects.urai_rai_analysis.scenario_builder import (
    ARM_INSULIN_MODEL, build_scenario, controller_settings, glucose_seed_rows,
)

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO_ROOT = os.path.normpath(POINTER_OBJ_DIR)
SCENARIO_SUBDIR = os.path.join("loop_risk_v2_0", "loop_risk_v2_urai_rai")
GLUCOSE_SUBDIR = os.path.join("reusable", "glucose", "urai_rai")
MANIFEST_NAME = "manifest.csv"
CONFIG_COPY_NAME = "run_config.json"


@dataclass(frozen=True)
class RunResult:
    """Where a run's outputs went."""
    scenario_dir: str
    glucose_dir: str
    manifest_path: str
    manifest: pd.DataFrame


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


def generate_run(config: Union[RunConfig, Dict[str, Any]], pool_dir: Optional[str] = None,
                 scenario_root: Optional[str] = None) -> RunResult:
    """Generate every scenario of a run and its manifest.

    Parameters
    ----------
    config: RunConfig or raw config dict
    pool_dir: patient pool directory; defaults to the config's environment variable
    scenario_root: root holding ``loop_risk_v2_0`` and ``reusable``; defaults to the repo's
        ``scenario_configs/tidepool_risk_v2``

    Raises
    ------
    ValueError, FileNotFoundError, FileExistsError
        On a missing pool or seed, a cohort shortfall that is not accepted, or a non-empty output directory.
    """
    cfg = config if isinstance(config, RunConfig) else parse_run_config(config)
    pool_path = resolve_pool_dir(cfg, pool_dir)
    root = scenario_root or DEFAULT_SCENARIO_ROOT
    scenario_dir = os.path.join(root, SCENARIO_SUBDIR, cfg.run_name)
    glucose_dir = os.path.join(root, GLUCOSE_SUBDIR, cfg.run_name)

    table = scan_pool(pool_path)
    draws = [draw_cohort(table, spec, cfg) for spec in cfg.cohorts]  # raises before anything is written
    cohorts_by_patient: Dict[str, List[str]] = {}
    for draw in draws:
        for pid in draw.patient_ids:
            cohorts_by_patient.setdefault(pid, []).append(draw.spec.name)

    _prepare_dir(scenario_dir)
    _prepare_dir(glucose_dir)
    run_start = datetime.datetime.strptime(cfg.run_start, DATETIME_FORMAT)
    paths = dict(zip(table["patient_id"], table["path"]))

    rows: List[Dict[str, Any]] = []
    seeds_written = set()
    for draw in draws:
        for pid in draw.patient_ids:
            record = parse_patient(paths[pid], cfg.glucose_seed_points)
            if pid not in seeds_written:
                _write_glucose_seed(os.path.join(glucose_dir, f"{pid}.csv"), glucose_seed_rows(record, cfg))
                seeds_written.add(pid)
            seed = derive_seed(cfg.master_seed, "draw", draw.spec.name, pid)
            schedule = generate_meals(seed, cfg.meals, run_start, cfg.duration_hours)
            for arm in ARM_INSULIN_MODEL:
                scenario = build_scenario(record, cfg, arm, draw.spec.name, list(schedule.meals), draw.spec.announced)
                sim_id = scenario["override_config"][0]["sim_id"]
                file_name = f"Simulation-Configuration-TLR-{sim_id}.json"
                _write_json(os.path.join(scenario_dir, file_name), scenario)
                rows.append({
                    "file_name": file_name, "patient_id": pid, "cohort": draw.spec.name,
                    "also_in_cohorts": ";".join(c for c in cohorts_by_patient[pid] if c != draw.spec.name),
                    "pair_id": f"{draw.spec.name}_{pid}", "arm": arm, "insulin_model": ARM_INSULIN_MODEL[arm],
                    "meals_announced": draw.spec.announced, "age_band": age_band(record.age), "age": record.age,
                    "isf": record.isf_for_cohorts, "derived_seed": seed,
                    "meal_schedule": meal_schedule_json(schedule.meals),
                    "has_seconds_pair": schedule.has_seconds_pair,
                    "controller_settings": json.dumps(controller_settings(record, cfg, arm)),
                    "shortfall_note": draw.shortfall_note or "",
                })

    _write_json(os.path.join(scenario_dir, CONFIG_COPY_NAME), cfg.raw)
    manifest_path = os.path.join(scenario_dir, MANIFEST_NAME)
    manifest = write_manifest(rows, manifest_path)
    logger.info("Wrote %d scenarios to %s", len(rows), scenario_dir)
    return RunResult(scenario_dir, glucose_dir, manifest_path, manifest)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate URAI-RAI paired autobolus scenarios")
    parser.add_argument("--config", required=True, help="Path to the run config JSON")
    parser.add_argument("--pool-dir", help="Patient pool directory (default: the config's environment variable)")
    parser.add_argument("--scenario-root", help="Override the scenario_configs/tidepool_risk_v2 output root")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = generate_run(load_run_config(args.config), args.pool_dir, args.scenario_root)
    print(f"{len(result.manifest)} scenarios written to {result.scenario_dir}")


if __name__ == "__main__":
    main()
