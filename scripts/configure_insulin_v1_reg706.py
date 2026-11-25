"""
configure_insulin_v1_reg706.py  (TRSET-81, supports REG-706)

Builds the REG-706 comparison arms from the loop_risk_v2_2_0_full baseline (read-only). Each arm is a copy of the
baseline configs for the REG-706 test set with only the patient insulin settings (and, for URAI arms, the Loop
controller model) changed:

    A1_old_RAI          baseline, as-is (plain copy)
    A2_old_URAI         patient fiasp,          controller model fiasp
    B1_new_RAI_main     patient rapid_acting_v1, insulin_timing main,   controller baseline
    B2_new_RAI_faster   patient rapid_acting_v1, insulin_timing faster, controller baseline
    B3_new_RAI_slower   patient rapid_acting_v1, insulin_timing slower, controller baseline
    B4_new_URAI_main    patient ultra_rapid_v1,  insulin_timing main,   controller model fiasp
    B5_new_URAI_faster  ...faster
    B6_new_URAI_slower  ...slower

The scenario parser only overrides keys that exist in the base, and the baseline profiles do not carry
``insulin_timing`` (and some lack ``patient_insulin_type``). So each generated file inlines the base config's patient
metabolism settings with explicit defaults (``patient_insulin_type`` keeps its existing value, else
``rapid_acting_adult``; ``insulin_timing`` ``main``), which is behaviourally identical to the baseline.

Usage:
    python scripts/configure_insulin_v1_reg706.py            # all arms, full test set
    python scripts/configure_insulin_v1_reg706.py --tlr TLR-843   # one TLR group, for a quick check
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from configure_insulin_model_compare import apply_controller_fiasp, resolve_reusable_ref  # noqa: E402

_SIM_ROOT = _SCRIPT_DIR.parent
_SCENARIO_ROOT = _SIM_ROOT / "scenario_configs" / "tidepool_risk_v2"
_REUSABLE_ROOT = _SCENARIO_ROOT / "reusable"
SOURCE_DIR = _SCENARIO_ROOT / "loop_risk_v2_0" / "loop_risk_v2_2_0_full"
DEST_BASE = _SCENARIO_ROOT / "loop_risk_v2_0" / "loop_risk_v2_insulin_v1_reg706"

HYPO_TLRS = ["TLR-843", "TLR-959", "TLR-861", "TLR-844", "TLR-899", "TLR-553"]
DKA_TLRS = ["TLR-613", "TLR-1122", "TLR-552"]
TEST_SET_TLRS = HYPO_TLRS + DKA_TLRS

# arm name -> (patient_insulin_type or None for baseline, insulin_timing or None, controller model fiasp?)
ARMS = {
    "A1_old_RAI": (None, None, False),
    "A2_old_URAI": ("fiasp", None, True),
    "B1_new_RAI_main": ("rapid_acting_v1", "main", False),
    "B2_new_RAI_faster": ("rapid_acting_v1", "faster", False),
    "B3_new_RAI_slower": ("rapid_acting_v1", "slower", False),
    "B4_new_URAI_main": ("ultra_rapid_v1", "main", True),
    "B5_new_URAI_faster": ("ultra_rapid_v1", "faster", True),
    "B6_new_URAI_slower": ("ultra_rapid_v1", "slower", True),
}

DEFAULT_TYPE = "rapid_acting_adult"


def matched_dirs(source_dir, tlr_ids):
    """Every directory named for a TLR id, including its sub-scenario directories (TLR-844_170_corr, ...)."""
    out = []
    for tlr in tlr_ids:
        out += sorted(d for d in source_dir.iterdir() if d.is_dir() and (d.name == tlr or d.name.startswith(tlr + "_")))
    return out


def _inline_ms(ms):
    """Resolve a metabolism_settings pointer string to a dict; dicts are copied."""
    if isinstance(ms, str):
        return resolve_reusable_ref(ms, _REUSABLE_ROOT)
    return copy.deepcopy(ms)


def inline_base(base_config):
    """Resolve the base config pointer and give the patient's metabolism settings explicit type/timing defaults."""
    if isinstance(base_config, str):
        # same file lookup as ScenarioParserV2.load_pointer for simulations
        base = resolve_reusable_ref(base_config, _REUSABLE_ROOT)
    else:
        base = copy.deepcopy(base_config)
    pm = base["patient"]["patient_model"]
    ms = _inline_ms(pm["metabolism_settings"])
    ms.setdefault("patient_insulin_type", DEFAULT_TYPE)
    ms.setdefault("insulin_timing", "main")
    pm["metabolism_settings"] = ms
    return base


def apply_patient_v1(override_item, patient_type, timing):
    """Set the patient insulin settings in an override item (only these keys change)."""
    pm = override_item.setdefault("patient", {}).setdefault("patient_model", {})
    ms = pm.get("metabolism_settings")
    if ms is None:
        # base is inlined with both keys, so a partial dict overrides them
        pm["metabolism_settings"] = {"patient_insulin_type": patient_type, "insulin_timing": timing}
        return
    ms = _inline_ms(ms)
    ms["patient_insulin_type"] = patient_type
    ms["insulin_timing"] = timing
    pm["metabolism_settings"] = ms


def transform_config(config, arm):
    patient_type, timing, controller_fiasp = ARMS[arm]
    if patient_type is None:
        return config
    config["base_config"] = inline_base(config["base_config"])
    if patient_type == "fiasp":
        # old URAI patient: type only (no insulin_timing key needed)
        for item in config["override_config"]:
            pm = item.setdefault("patient", {}).setdefault("patient_model", {})
            ms = pm.get("metabolism_settings")
            if ms is None:
                pm["metabolism_settings"] = {"patient_insulin_type": "fiasp"}
            else:
                ms = _inline_ms(ms)
                ms["patient_insulin_type"] = "fiasp"
                pm["metabolism_settings"] = ms
    else:
        for item in config["override_config"]:
            apply_patient_v1(item, patient_type, timing)
    if controller_fiasp:
        for item in config["override_config"]:
            apply_controller_fiasp(item, _REUSABLE_ROOT)
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tlr", action="append", help="limit to this TLR group (repeatable)")
    ap.add_argument("--dest", default=str(DEST_BASE))
    args = ap.parse_args()
    tlrs = args.tlr or TEST_SET_TLRS
    dest_base = Path(args.dest)
    dirs = matched_dirs(SOURCE_DIR, tlrs)
    print(f"{len(dirs)} source directories, {len(ARMS)} arms -> {dest_base}")
    for arm in ARMS:
        for d in dirs:
            out = dest_base / arm / d.name
            out.mkdir(parents=True, exist_ok=True)
            for f in sorted(d.glob("*.json")):
                if ARMS[arm][0] is None:
                    shutil.copyfile(f, out / f.name)
                else:
                    cfg = transform_config(json.load(open(f)), arm)
                    with open(out / f.name, "w") as fh:
                        json.dump(cfg, fh, indent=2)


if __name__ == "__main__":
    main()
