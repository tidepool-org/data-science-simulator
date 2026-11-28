"""
Read-only resolution of baseline scenario configs through ScenarioParserV2.

Resolves pointers and applies overrides exactly as ``ScenarioParserV2.get_sims`` does, but stops before
``build_sim_from_config``: nothing is built or simulated. The parser is used, never modified.
"""
import contextlib
import copy
import io
import json
from pathlib import Path

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

UNSET = "<unset>"
NO_CONTROLLER = "<no controller>"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = REPO_ROOT / "scenario_configs" / "tidepool_risk_v2"
DEFAULT_BASELINE = SCENARIO_ROOT / "loop_risk_v2_0" / "loop_risk_v2_2_0_full"
EXCLUDE_LIST_PATH = Path(__file__).resolve().parent / "exclude_list.json"


def load_exclude_list(path=EXCLUDE_LIST_PATH):
    with open(path) as f:
        return list(json.load(f)["exclude"])


def tlr_dirs(baseline_dir, exclude):
    """Sorted TLR subdirectories of the baseline, minus the exclude list."""
    baseline_dir = Path(baseline_dir)
    dirs = [d for d in baseline_dir.iterdir() if d.is_dir()]
    unknown = sorted(set(exclude) - {d.name for d in dirs})
    if unknown:  # a typo here would silently leave a TLR in scope
        raise ValueError(f"exclude list names directories that do not exist in {baseline_dir.name}: {unknown}")
    return sorted(d for d in dirs if d.name not in set(exclude))


def _dig(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def extract_fields(resolved):
    """The three insulin fields of one resolved sim config (patient side only; pump side is ignored)."""
    ms = _dig(resolved, "patient", "patient_model", "metabolism_settings")
    ms = ms if isinstance(ms, dict) else {}
    controller = resolved.get("controller")
    if controller is None:
        ctrl_model = NO_CONTROLLER
    else:
        ctrl_model = (_dig(controller, "settings", "model")) or UNSET
    return {
        "patient_insulin_type": ms.get("patient_insulin_type", UNSET),
        "insulin_timing": ms.get("insulin_timing", UNSET),
        "controller_model": ctrl_model,
    }


def override_flags(raw_override):
    """Resolution traps visible in the raw (pre-resolution) override."""
    return {
        "override_sets_patient_metabolism": _dig(raw_override, "patient", "patient_model", "metabolism_settings") is not None,
        "override_sets_controller_settings": _dig(raw_override, "controller", "settings") is not None,
    }


def resolve_config_file(config_path, pointer_object_dir=None):
    """
    Resolve every sim in one scenario config file without building or simulating.

    Returns a list of dicts: sim_id, the three insulin fields, override flags, and the resolved config.
    Raises on any resolution failure (the caller reports it; failures are never swallowed here).
    """
    kwargs = {} if pointer_object_dir is None else {"pointer_object_dir": str(pointer_object_dir)}
    # The parser prints extensive override diagnostics; silence them.
    with contextlib.redirect_stdout(io.StringIO()):
        parser = ScenarioParserV2(str(config_path), **kwargs)
        out = []
        for override in parser.override_configs:
            resolved = copy.deepcopy(parser.base_sim_config)
            raw = copy.deepcopy(override)
            parser.apply_config_override(resolved, copy.deepcopy(override))
            rec = {"sim_id": resolved.get("sim_id"), **extract_fields(resolved), **override_flags(raw), "resolved": resolved}
            out.append(rec)
    return out


def resolve_baseline(baseline_dir=DEFAULT_BASELINE, exclude=None, pointer_object_dir=None):
    """
    Resolve every sim of every in-scope TLR directory.

    Returns (sims, errors). Each sim is a dict with tlr, file, sim_id and the fields above (no resolved config).
    Each error is {tlr, file, error}.
    """
    exclude = load_exclude_list() if exclude is None else exclude
    sims, errors = [], []
    for tlr in tlr_dirs(baseline_dir, exclude):
        for cfg in sorted(tlr.glob("*.json")):
            try:
                recs = resolve_config_file(cfg, pointer_object_dir)
            except Exception as e:  # report, never skip silently
                errors.append({"tlr": tlr.name, "file": cfg.name, "error": f"{type(e).__name__}: {e}"})
                continue
            for r in recs:
                r.pop("resolved")
                sims.append({"tlr": tlr.name, "file": cfg.name, **r})
    return sims, errors
