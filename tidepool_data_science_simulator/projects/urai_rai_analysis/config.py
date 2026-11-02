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
class Guardrails:
    """Inclusive ``(low, high)`` ranges used by eligibility rules E3 and E5 (TRSET-77 AC 6).

    ``suspend_threshold`` and ``target`` also carry the E6 relationships (suspend <= target minimum <= target maximum).
    """
    basal_rate: Tuple[float, float]
    carb_ratio: Tuple[float, float]
    isf: Tuple[float, float]
    target: Tuple[float, float]
    suspend_threshold: Tuple[float, float]
    glucose_seed: Tuple[float, float]


@dataclass(frozen=True)
class MaxBolusConfig:
    """Per-patient max bolus = min(fraction_of_tdd * TDD, ceiling_u) rounded down to ``increment_u``.

    TDD = ``tdd_basal_multiplier`` * daily scheduled basal. ``static_u`` replaces every patient's controller max bolus
    (the comparator run); the settings flags always describe the per-patient computation.
    """
    fraction_of_tdd: float
    tdd_basal_multiplier: float
    ceiling_u: float
    increment_u: float
    static_u: Optional[float] = None


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to generate one URAI-RAI run."""
    run_name: str
    master_seed: int
    cohorts: Tuple[CohortSpec, ...]
    guardrails: Guardrails
    max_bolus: MaxBolusConfig
    default_suspend_threshold: Optional[float] = None
    patient_pool_env_var: str = DEFAULT_POOL_ENV_VAR
    min_age: float = 6
    accept_smaller_group: bool = False
    partial_application_factor: float = 0.4
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


GUARDRAIL_KEYS = ("basal_rate", "carb_ratio", "isf", "target", "suspend_threshold", "glucose_seed")
MAX_BOLUS_KEYS = ("fraction_of_tdd", "tdd_basal_multiplier", "ceiling_u", "increment_u", "static_u")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return value


def _parse_guardrails(raw: Any) -> Guardrails:
    if not isinstance(raw, dict) or set(raw) != set(GUARDRAIL_KEYS):
        raise ValueError(f"guardrails must define exactly {list(GUARDRAIL_KEYS)}")
    ranges = {}
    for key in GUARDRAIL_KEYS:
        low, high = _tuple(raw[key], 2)
        if _number(low, f"guardrails.{key}") > _number(high, f"guardrails.{key}"):
            raise ValueError(f"guardrails.{key} low {low} exceeds high {high}")
        ranges[key] = (low, high)
    return Guardrails(**ranges)


def _parse_max_bolus(raw: Any) -> MaxBolusConfig:
    required = set(MAX_BOLUS_KEYS) - {"static_u"}
    if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - set(MAX_BOLUS_KEYS):
        raise ValueError(f"max_bolus must define {sorted(required)} and may define static_u")
    values = {key: _number(raw[key], f"max_bolus.{key}") for key in required}
    if any(v <= 0 for v in values.values()):
        raise ValueError("max_bolus fraction_of_tdd, tdd_basal_multiplier, ceiling_u and increment_u must be positive")
    static = raw.get("static_u")
    if static is not None and _number(static, "max_bolus.static_u") <= 0:
        raise ValueError("max_bolus.static_u must be positive or null")
    return MaxBolusConfig(static_u=static, **values)


def parse_run_config(config: Dict[str, Any]) -> RunConfig:
    """Validate a raw config dict and return a RunConfig.

    Raises
    ------
    ValueError
        If the master seed or run name is missing/invalid, a cohort filter is unknown, or a key is unrecognised.
    """
    cfg = dict(config)
    for required in ("run_name", "master_seed", "cohorts", "guardrails", "max_bolus"):
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
    if "max_bolus" in controller:
        raise ValueError("controller.max_bolus is replaced by the top-level max_bolus block")
    glucose = cfg.pop("glucose_seed", {})
    run = cfg.pop("run", {})
    kwargs: Dict[str, Any] = dict(
        run_name=run_name, master_seed=cfg.pop("master_seed"), cohorts=tuple(cohorts), meals=meals, raw=dict(config),
        guardrails=_parse_guardrails(cfg.pop("guardrails")), max_bolus=_parse_max_bolus(cfg.pop("max_bolus")),
    )
    for key in ("patient_pool_env_var", "min_age", "accept_smaller_group"):
        if key in cfg:
            kwargs[key] = cfg.pop(key)
    default_suspend = cfg.pop("default_suspend_threshold", None)
    if default_suspend is not None:
        kwargs["default_suspend_threshold"] = _number(default_suspend, "default_suspend_threshold")
    for key in ("partial_application_factor",):
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
