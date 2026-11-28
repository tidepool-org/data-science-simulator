"""Verification by difference: a generated sim must equal its baseline sim except for the three insulin fields."""
import copy

from .resolve import NO_CONTROLLER, UNSET
from .translation import CONTROLLER_UNCHANGED

HARD = "hard"


def target_for(row, arm, fields):
    """(patient_insulin_type, insulin_timing, controller_model) a sim must have in ``arm``."""
    a = row["arms"][arm]
    model = fields["controller_model"] if a["controller_model"] == CONTROLLER_UNCHANGED else a["controller_model"]
    return a["patient_insulin_type"], a["insulin_timing"], model


def _strip(resolved):
    r = copy.deepcopy(resolved)
    ms = (r.get("patient") or {}).get("patient_model", {}).get("metabolism_settings")
    if isinstance(ms, dict):
        ms.pop("patient_insulin_type", None)
        ms.pop("insulin_timing", None)
    ctrl = r.get("controller")
    if isinstance(ctrl, dict) and isinstance(ctrl.get("settings"), dict):
        ctrl["settings"].pop("model", None)
    return r


def first_difference(a, b, path=""):
    if type(a) != type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k}: present in only one side"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: list length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != a and b != b:       # both NaN (e.g. missing glucose history entries): the same value, though nan != nan
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def timing_ok(expected, got, got_type):
    """None means 'n/a'. A non-v1 patient ignores insulin_timing 'main' (it is also the default), so absent and
    'main' are the same behaviour for those types (SimpleMetabolismModelV1 rejects any other value)."""
    if expected is not None:
        return got == expected
    return got == UNSET or (got == "main" and not str(got_type).endswith("_v1"))


def compare(base_rec, gen_rec, target):
    """
    Returns a list of problems. Each is {"kind": "hard"|"patient"|"model", ...}. "patient" and "model" problems can
    be corrected with a minimal override; "hard" ones fail the run.
    """
    problems = []
    if base_rec["sim_id"] != gen_rec["sim_id"]:
        return [{"kind": HARD, "detail": f"sim_id {base_rec['sim_id']!r} != {gen_rec['sim_id']!r}"}]
    d = first_difference(_strip(base_rec["resolved"]), _strip(gen_rec["resolved"]))
    if d:
        problems.append({"kind": HARD, "detail": f"resolved sim differs outside the three insulin fields at {d}"})
    ptype, timing, model = target
    if gen_rec["patient_insulin_type"] != ptype or not timing_ok(timing, gen_rec["insulin_timing"], gen_rec["patient_insulin_type"]):
        problems.append({"kind": "patient", "expected": (ptype, timing),
                         "got": (gen_rec["patient_insulin_type"], gen_rec["insulin_timing"])})
    if gen_rec["controller_model"] != model:
        problems.append({"kind": "model", "expected": model, "got": gen_rec["controller_model"]})
    return problems
