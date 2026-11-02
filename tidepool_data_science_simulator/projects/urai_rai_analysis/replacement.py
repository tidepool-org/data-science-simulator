"""Replacement runs for the URAI-RAI generator (TRSET-77 AC 10).

A replacement run swaps failed pairs for the next patients in each cohort's seeded order. It is a new run folder with its
own manifest; the original run, its draws and its files are only read, never written.
"""
import csv
import dataclasses
import datetime
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import DATETIME_FORMAT
from tidepool_data_science_simulator.projects.urai_rai_analysis.cohorts import cohort_order, derive_seed
from tidepool_data_science_simulator.projects.urai_rai_analysis.config import RunConfig, load_run_config, resolve_pool_dir
from tidepool_data_science_simulator.projects.urai_rai_analysis.generate import (
    CONFIG_COPY_NAME, DEFAULT_SCENARIO_ROOT, MANIFEST_NAME, PlannedScenario, RunResult, build_planned, prepare_pool,
    write_run,
)
from tidepool_data_science_simulator.projects.urai_rai_analysis.manifest import REPLACEMENT_COLUMNS
from tidepool_data_science_simulator.projects.urai_rai_analysis.meals import generate_meals


def read_replacements(path: str) -> List[Tuple[str, str]]:
    """(pair_id, reason) rows of a replacements CSV with columns ``pair_id,reason``.

    Raises
    ------
    ValueError
        If a column is missing, or a row has no pair ID or no reason.
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or {"pair_id", "reason"} - set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns pair_id,reason")
        rows = [(r["pair_id"].strip(), r["reason"].strip()) for r in reader]
    if not rows or any(not pair or not reason for pair, reason in rows):
        raise ValueError(f"{path}: every row needs a pair_id and a reason")
    return rows


def _manifest(run_dir: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(run_dir, MANIFEST_NAME), dtype=str, keep_default_na=False)


def generate_replacement_run(original_run_dir: str, replacements: Sequence[Tuple[str, str]], run_name: str,
                             pool_dir: Optional[str] = None, scenario_root: Optional[str] = None,
                             prior_run_dirs: Sequence[str] = ()) -> RunResult:
    """Write a new run holding a replacement pair for each (pair_id, reason).

    The replacement is the next patient in the cohort's seeded order that is not in the original cohort, not a
    replacement in any ``prior_run_dirs`` run, and not already used in this run. It gets its own derived meal seed.

    Raises
    ------
    ValueError
        If a pair is unknown, the pool or config no longer reproduces the original draw, or a cohort has no eligible
        patient left to use.
    FileExistsError
        If the new run's output directory is not empty.
    """
    original_cfg = load_run_config(os.path.join(original_run_dir, CONFIG_COPY_NAME))
    cfg = dataclasses.replace(original_cfg, run_name=run_name, raw={**original_cfg.raw, "run_name": run_name})
    original, priors = _manifest(original_run_dir), [_manifest(d) for d in prior_run_dirs]

    known_pairs = {r["pair_id"]: (r["cohort"], r["patient_id"]) for m in [original, *priors] for _, r in m.iterrows()}
    in_cohort: Dict[str, List[str]] = {}
    for cohort, patient in dict.fromkeys(zip(original["cohort"], original["patient_id"])):
        in_cohort.setdefault(cohort, []).append(patient)
    taken: Dict[str, Set[str]] = {c: set(ids) for c, ids in in_cohort.items()}
    for m in priors:
        for cohort, patient in zip(m["cohort"], m["patient_id"]):
            taken.setdefault(cohort, set()).add(patient)

    inputs = prepare_pool(cfg, resolve_pool_dir(cfg, pool_dir))
    specs = {spec.name: spec for spec in cfg.cohorts}
    orders: Dict[str, List[str]] = {}
    run_start = datetime.datetime.strptime(cfg.run_start, DATETIME_FORMAT)
    chosen: List[Tuple[str, str, str, str]] = []  # cohort, replaced pair, replacement patient, reason
    for pair_id, reason in replacements:
        if pair_id not in known_pairs:
            raise ValueError(f"Unknown pair {pair_id!r}: it is in neither the original run nor a prior replacement run")
        cohort = known_pairs[pair_id][0]
        if cohort not in orders:
            orders[cohort] = cohort_order(inputs.table, specs[cohort], cfg.master_seed)
            drawn = in_cohort[cohort]
            if orders[cohort][:len(drawn)] != drawn:
                raise ValueError(f"Cohort {cohort}: the pool or config no longer reproduces the original draw")
        candidate = next((p for p in orders[cohort] if p not in taken[cohort]), None)
        if candidate is None:
            raise ValueError(f"Cohort {cohort}: no eligible patient left to replace {pair_id}")
        taken[cohort].add(candidate)
        chosen.append((cohort, pair_id, candidate, reason))

    cohorts_by_patient: Dict[str, List[str]] = {}
    for cohort, _, patient, _ in chosen:
        cohorts_by_patient.setdefault(patient, []).append(cohort)
    planned: List[PlannedScenario] = []
    for cohort, pair_id, patient, reason in chosen:
        seed = derive_seed(cfg.master_seed, "replacement", cohort, patient, pair_id)
        schedule = generate_meals(seed, cfg.meals, run_start, cfg.duration_hours)
        planned += build_planned(
            cfg, cohort, specs[cohort].announced, inputs.patients[patient], seed, schedule, "",
            ";".join(c for c in cohorts_by_patient[patient] if c != cohort),
            extra={"replaces_pair_id": pair_id, "replacement_reason": reason})
    return write_run(cfg, scenario_root or DEFAULT_SCENARIO_ROOT, inputs, planned, REPLACEMENT_COLUMNS)
