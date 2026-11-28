"""
Per-arm translation table, keyed on the baseline (patient_insulin_type, controller_model) pair.

Only plain rapid_acting_adult rows are pre-filled as NOMINAL. Every other pair is written as NEEDS_REVIEW with
empty arm cells, so Phase 2 fails on it until the table is filled by hand.
"""
import hashlib
import json
from pathlib import Path

from .resolve import NO_CONTROLLER, UNSET

TABLE_PATH = Path(__file__).resolve().parent / "translation_table.json"
CONTROLLER_UNCHANGED = "<unchanged>"
BASELINE_PLAIN_PATIENT = "rapid_acting_adult"
BASELINE_PLAIN_CONTROLLER = "rapid_acting_adult"

# arm directory -> (patient_insulin_type, insulin_timing, controller_model or CONTROLLER_UNCHANGED)
ARMS = {
    "loop_risk_v2_rai_main": ("rapid_acting_v1", "main", CONTROLLER_UNCHANGED),
    "loop_risk_v2_urai_main": ("ultra_rapid_v1", "main", "fiasp"),
    "loop_risk_v2_urai_old": ("fiasp", None, "fiasp"),  # A2: old URAI patient model, no insulin_timing
}
# Rev 1 (2026-11-25): faster/slower arms are out of scope; reinstating them is a new ticket (re-add them to ARMS).


def pair_key(patient_insulin_type, controller_model):
    return f"{patient_insulin_type} | {controller_model}"


def nominal_arms(controller_model):
    """Nominal per-arm outputs. No-controller sims keep no controller in every arm."""
    arms = {}
    for name, (ptype, timing, ctrl) in ARMS.items():
        if controller_model == NO_CONTROLLER:
            ctrl = CONTROLLER_UNCHANGED  # nothing to set: the sim has no controller
        arms[name] = {"patient_insulin_type": ptype, "insulin_timing": timing, "controller_model": ctrl}
    return arms


def is_nominal_pair(patient_insulin_type, controller_model):
    return patient_insulin_type == BASELINE_PLAIN_PATIENT and controller_model in (BASELINE_PLAIN_CONTROLLER, NO_CONTROLLER)


def build_table(pairs, existing=None):
    """
    Build the table for the given (patient_insulin_type, controller_model) pairs.

    Rows already in ``existing`` are kept untouched, so hand-filled deviation rows are never overwritten.
    """
    existing_rows = {r["key"]: r for r in (existing or {}).get("rows", [])}
    rows = []
    for ptype, ctrl in sorted(set(pairs)):
        key = pair_key(ptype, ctrl)
        if key in existing_rows:
            rows.append(existing_rows[key])
        elif is_nominal_pair(ptype, ctrl):
            rows.append({"key": key, "baseline": {"patient_insulin_type": ptype, "controller_model": ctrl},
                         "status": "NOMINAL", "arms": nominal_arms(ctrl)})
        else:
            rows.append({"key": key, "baseline": {"patient_insulin_type": ptype, "controller_model": ctrl},
                         "status": "NEEDS_REVIEW", "arms": {}})
    return {
        "_comment": "TRSET-84 translation table. NOMINAL rows are pre-filled; NEEDS_REVIEW rows are filled by hand before Phase 2.",
        "controller_unchanged_sentinel": CONTROLLER_UNCHANGED,
        "arm_order": list(ARMS),
        "rows": rows,
    }


def load_table(path=TABLE_PATH):
    with open(path) as f:
        return json.load(f)


def save_table(table, path=TABLE_PATH):
    with open(path, "w") as f:
        json.dump(table, f, indent=2, sort_keys=False)
        f.write("\n")


def table_hash(table):
    return hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()


def unresolved_rows(table):
    """Rows Phase 2 cannot use: NEEDS_REVIEW, or any arm cell missing/empty."""
    bad = []
    for r in table["rows"]:
        arms = r.get("arms", {})
        if r.get("status") == "NEEDS_REVIEW" or set(arms) != set(ARMS):
            bad.append(r["key"])
    return bad
