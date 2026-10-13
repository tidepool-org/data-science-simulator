"""Run configuration for the URAI-RAI scenario generator (TRSET-74)."""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")  # no "." -- it is the pointer separator in glucose pointers
DEFAULT_POOL_ENV_VAR = "URAI_RAI_PATIENT_POOL"
FILTER_KEYS = ("age_min", "age_max", "isf_min", "isf_max")


@dataclass(frozen=True)
class CohortSpec:
    """A named patient filter plus a group size.

    Filter bounds: ``age_min`` and ``isf_min`` are inclusive lower bounds, ``age_max`` is an exclusive upper
    bound and ``isf_max`` is an inclusive upper bound.
    """
    name: str
    n: int
    filter: Dict[str, float]
    announced: bool = True


@dataclass(frozen=True)
class MealConfig:
    """Meal generation rules. Times are minutes on a 5-minute grid; clock times are ``HH:MM`` strings."""
    meals_per_day: Tuple[int, int] = (2, 4)
    window_start: str = "05:00"
    window_end: str = "22:00"
    carbs_g: Tuple[int, int] = (20, 80)
    min_spacing_min: int = 60
    seconds_pair_prob: float = 0.25
    seconds_pair_gap_min: Tuple[int, int] = (5, 55)
    absorption_times_min: Tuple[int, ...] = (30, 180, 300)
    grid_min: int = 5


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to generate one URAI-RAI run."""
    run_name: str
    master_seed: int
    cohorts: Tuple[CohortSpec, ...]
    patient_pool_env_var: str = DEFAULT_POOL_ENV_VAR
    min_age: float = 6
    accept_smaller_group: bool = False
    partial_application_factor: float = 0.4
    max_bolus: float = 20
    meals: MealConfig = field(default_factory=MealConfig)
    run_start: str = "08/15/2019 12:00:00"
    duration_hours: float = 24.0
    glucose_seed_start: str = "08/15/2019 00:40:00"
    glucose_seed_points: int = 137
    glucose_interval_min: int = 5
    raw: Dict[str, Any] = field(default_factory=dict, compare=False)


def _tuple(value: Any, length: Optional[int] = None) -> tuple:
    result = tuple(value)
    if length is not None and len(result) != length:
        raise ValueError(f"Expected {length} values, got {value!r}")
    return result


def parse_run_config(config: Dict[str, Any]) -> RunConfig:
    """Validate a raw config dict and return a RunConfig.

    Raises
    ------
    ValueError
        If the master seed or run name is missing/invalid, a cohort filter is unknown, or a key is unrecognised.
    """
    cfg = dict(config)
    for required in ("run_name", "master_seed", "cohorts"):
        if cfg.get(required) is None:
            raise ValueError(f"Required run config value '{required}' is missing")
    if isinstance(cfg["master_seed"], bool) or not isinstance(cfg["master_seed"], int):
        raise ValueError("master_seed must be an integer")
    run_name = cfg.pop("run_name")
    if not RUN_NAME_PATTERN.match(str(run_name)):
        raise ValueError(f"run_name {run_name!r} must match {RUN_NAME_PATTERN.pattern}")

    cohorts: List[CohortSpec] = []
    default_n = cfg.pop("default_n", 30)
    for c in cfg.pop("cohorts"):
        unknown = set(c.get("filter", {})) - set(FILTER_KEYS)
        if unknown:
            raise ValueError(f"Cohort {c.get('name')!r} has unknown filter keys {sorted(unknown)}")
        cohorts.append(CohortSpec(
            name=c["name"], n=int(c.get("n", default_n)), filter=dict(c.get("filter", {})),
            announced=bool(c.get("announced", True)),
        ))
    names = [c.name for c in cohorts]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate cohort names in {names}")

    meal_raw = dict(cfg.pop("meals", {}))
    for key in ("meals_per_day", "carbs_g", "seconds_pair_gap_min"):
        if key in meal_raw:
            meal_raw[key] = _tuple(meal_raw[key], 2)
    if "absorption_times_min" in meal_raw:
        meal_raw["absorption_times_min"] = _tuple(meal_raw["absorption_times_min"])
    meals = MealConfig(**meal_raw)

    controller = cfg.pop("controller", {})
    glucose = cfg.pop("glucose_seed", {})
    run = cfg.pop("run", {})
    kwargs: Dict[str, Any] = dict(
        run_name=run_name, master_seed=cfg.pop("master_seed"), cohorts=tuple(cohorts), meals=meals, raw=dict(config),
    )
    for key in ("patient_pool_env_var", "min_age", "accept_smaller_group"):
        if key in cfg:
            kwargs[key] = cfg.pop(key)
    for key in ("partial_application_factor", "max_bolus"):
        if key in controller:
            kwargs[key] = controller[key]
    if "start" in run:
        kwargs["run_start"] = run["start"]
    if "duration_hours" in run:
        kwargs["duration_hours"] = float(run["duration_hours"])
    for key, target in (("start", "glucose_seed_start"), ("points", "glucose_seed_points"),
                        ("interval_min", "glucose_interval_min")):
        if key in glucose:
            kwargs[target] = glucose[key]
    if cfg:
        raise ValueError(f"Unrecognised run config keys: {sorted(cfg)}")
    return RunConfig(**kwargs)


def load_run_config(path: str) -> RunConfig:
    """Load and validate a run config JSON file."""
    with open(path) as f:
        return parse_run_config(json.load(f))


def resolve_pool_dir(config: RunConfig, pool_dir: Optional[str] = None) -> str:
    """Return the patient pool directory from the explicit argument or the configured environment variable.

    Raises
    ------
    ValueError
        If neither is set.
    """
    resolved = pool_dir or os.environ.get(config.patient_pool_env_var)
    if not resolved:
        raise ValueError(
            f"Patient pool directory not given: set environment variable {config.patient_pool_env_var} or pass pool_dir"
        )
    return resolved
