"""
configure_insulin_model_compare.py

Copies a subset of TLR scenario directories from loop_risk_v2_2_0_full into three
insulin model comparison directories and modifies the JSON configs to create
controlled insulin model mismatches:

  ab_URAI_ptModel  – patient physiology uses Fiasp; Loop controller uses rapid_acting_adult
  ab_URAI_pump     – patient physiology uses rapid_acting_adult; Loop controller uses Fiasp
  URAI             – both patient physiology and Loop controller use Fiasp

Usage (single TLR for testing):
    python configure_insulin_model_compare.py

Usage (full run, all in-scope TLRs):
    python configure_insulin_model_compare.py --all
"""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_SIM_ROOT = _SCRIPT_DIR.parent  # data-science-simulator/
_SCENARIO_ROOT = _SIM_ROOT / "scenario_configs" / "tidepool_risk_v2"
_REUSABLE_ROOT = _SCENARIO_ROOT / "reusable"
_LOOP_RISK_ROOT = _SCENARIO_ROOT / "loop_risk_v2_0"

SOURCE_DIR = _LOOP_RISK_ROOT / "loop_risk_v2_2_0_full"
DEST_BASE = _LOOP_RISK_ROOT / "insulin_model_compare"

DEST_DIRS = {
    "ab_URAI_ptModel": DEST_BASE / "ab_URAI_ptModel",
    "ab_URAI_pump":    DEST_BASE / "ab_URAI_pump",
    "URAI":            DEST_BASE / "URAI",
}

# ---------------------------------------------------------------------------
# In-scope TLR base IDs (from TLRs_in_scope.rtf).
# The script expands these to all matching subdirectories at runtime.
# ---------------------------------------------------------------------------
IN_SCOPE_TLRS = [
    "TLR-1011", "TLR-1023", "TLR-1032", "TLR-1034", "TLR-1053", "TLR-1062",
    "TLR-1065", "TLR-1066", "TLR-1078", "TLR-1116", "TLR-1117", "TLR-1118",
    "TLR-1120", "TLR-1121", "TLR-1130", "TLR-1131", "TLR-1136", "TLR-1142",
    "TLR-1143", "TLR-1145", "TLR-1147", "TLR-549",  "TLR-552",  "TLR-553",
    "TLR-554",  "TLR-555",  "TLR-556",  "TLR-558",  "TLR-561",  "TLR-562",
    "TLR-564",  "TLR-566",  "TLR-568",  "TLR-576",  "TLR-577",  "TLR-578",
    "TLR-579",  "TLR-586",  "TLR-587",  "TLR-590",  "TLR-596",  "TLR-604",
    "TLR-605",  "TLR-606",  "TLR-607",  "TLR-613",  "TLR-615",  "TLR-616",
    "TLR-627",  "TLR-629",  "TLR-660",  "TLR-664",  "TLR-667",  "TLR-668",
    "TLR-675",  "TLR-676",  "TLR-682",  "TLR-684",  "TLR-687",  "TLR-688",
    "TLR-689",  "TLR-690",  "TLR-696",  "TLR-697",  "TLR-703",  "TLR-704",
    "TLR-710",  "TLR-716",  "TLR-723",  "TLR-725",  "TLR-726",  "TLR-727",
    "TLR-731",  "TLR-736",  "TLR-739",  "TLR-742",  "TLR-788",  "TLR-789",
    "TLR-790",  "TLR-792",  "TLR-793",  "TLR-822",  "TLR-826",  "TLR-843",
    "TLR-845",  "TLR-846",  "TLR-847",  "TLR-861",  "TLR-899",  "TLR-901",
    "TLR-911",  "TLR-912",  "TLR-950",  "TLR-969",
]

# Single TLR used when running in test mode (--all not specified)
TEST_TLR = "TLR-549"

# Profiles that have Fiasp equivalents in
#   scenario_configs/tidepool_risk_v2/reusable/metabolism_settings/profiles/
PROFILE_FIASP_MAP = {
    "adolescent_v1": "adolescent_fiasp_v1",
    "median_v1":     "median_fiasp_v1",
    "resistant_v1":  "resistant_fiasp_v1",
    "sensitive_v1":  "sensitive_fiasp_v1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_reusable_ref(ref_str: str, reusable_root: Path) -> dict:
    """
    Convert a dot-notation pointer (e.g. ``reusable.mitigations.guardrails.controller_settings_adolescent_wmax``)
    to a JSON dict by locating and loading the corresponding file under *reusable_root*.

    Mirrors the subdirectory search behaviour of ScenarioParserV2.load_pointer:
    - For ``metabolism_settings`` paths, also searches profiles/, suspensions/,
      presets/, versions/, and types/ subdirectories.
    - For ``simulations`` paths, also searches base/, suspend/, and specialized/.
    """
    segments = ref_str.split(".")
    # segments[0] == "reusable" → strip it; remainder are the path parts
    rel_parts = segments[1:]
    filename = rel_parts[-1] + ".json"
    dir_parts = rel_parts[:-1]
    primary_dir = reusable_root.joinpath(*dir_parts)

    search_dirs = [primary_dir]
    if "metabolism_settings" in dir_parts:
        for sub in ("profiles", "suspensions", "presets", "versions", "types"):
            search_dirs.append(primary_dir / sub)
    elif "simulations" in dir_parts:
        for sub in ("base", "suspend", "specialized"):
            search_dirs.append(primary_dir / sub)

    for candidate_dir in search_dirs:
        candidate = candidate_dir / filename
        if candidate.is_file():
            with open(candidate) as f:
                return json.load(f)

    searched = [str(d / filename) for d in search_dirs]
    raise FileNotFoundError(
        f"Could not resolve reusable ref '{ref_str}'. Searched:\n  " + "\n  ".join(searched)
    )


def swap_profile_ref_to_fiasp(ref_str: str) -> str | None:
    """
    If *ref_str* ends with a known profile base name (e.g. ``adolescent_v1``),
    return the same ref string with the trailing segment replaced by the Fiasp
    equivalent. Returns ``None`` if the profile is not in PROFILE_FIASP_MAP.
    """
    last_segment = ref_str.rsplit(".", 1)[-1]
    if last_segment in PROFILE_FIASP_MAP:
        prefix = ref_str.rsplit(".", 1)[0]
        return f"{prefix}.{PROFILE_FIASP_MAP[last_segment]}"
    return None


# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------

def apply_patient_fiasp(override_item: dict, reusable_root: Path) -> str:
    """
    Modify *override_item* in place so that ``patient.patient_model.metabolism_settings``
    reflects Fiasp insulin type.

    Returns a short human-readable description of the change made.
    """
    # Ensure the nested path exists
    patient = override_item.setdefault("patient", {})
    patient_model = patient.setdefault("patient_model", {})
    ms = patient_model.get("metabolism_settings")

    if ms is None:
        patient_model["metabolism_settings"] = {"patient_insulin_type": "fiasp"}
        return "added metabolism_settings: {patient_insulin_type: fiasp}"

    if isinstance(ms, str):
        new_ref = swap_profile_ref_to_fiasp(ms)
        if new_ref is not None:
            patient_model["metabolism_settings"] = new_ref
            return f"swapped profile ref: {ms} → {new_ref}"
        else:
            # Non-standard profile ref: resolve and inline with fiasp type injected
            resolved = resolve_reusable_ref(ms, reusable_root)
            resolved["patient_insulin_type"] = "fiasp"
            patient_model["metabolism_settings"] = resolved
            return f"resolved and inlined '{ms}' with patient_insulin_type: fiasp"

    if isinstance(ms, dict):
        old = ms.get("patient_insulin_type", "<absent>")
        ms["patient_insulin_type"] = "fiasp"
        return f"updated patient_insulin_type: {old} → fiasp"

    return "no change (unrecognised metabolism_settings type)"


def apply_controller_fiasp(override_item: dict, reusable_root: Path) -> str:
    """
    Modify *override_item* in place so that ``controller.settings.model`` is ``"fiasp"``.

    - ``controller`` absent  → add ``{"settings": {"model": "fiasp"}}``
    - ``controller`` is null → skip (no-Loop simulation)
    - ``settings`` is a string ref → resolve the file, add ``model: fiasp``, inline it
    - ``settings`` is an inline dict → add/overwrite ``model: fiasp``

    Returns a short human-readable description of the change made.
    """
    if "controller" not in override_item:
        override_item["controller"] = {"settings": {"model": "fiasp"}}
        return "added controller: {settings: {model: fiasp}}"

    controller = override_item["controller"]

    if controller is None:
        return "skipped (controller is null)"

    settings = controller.get("settings")

    if settings is None:
        controller["settings"] = {"model": "fiasp"}
        return "added settings: {model: fiasp}"

    if isinstance(settings, str):
        resolved = resolve_reusable_ref(settings, reusable_root)
        resolved["model"] = "fiasp"
        controller["settings"] = resolved
        return f"resolved '{settings}' and set model: fiasp"

    if isinstance(settings, dict):
        old = settings.get("model", "<absent>")
        settings["model"] = "fiasp"
        return f"updated model: {old} → fiasp"

    return "no change (unrecognised settings type)"


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_scenario_file(
    src_path: Path,
    dest_path: Path,
    mode: str,
    reusable_root: Path,
) -> list[dict]:
    """
    Load *src_path*, apply transformations based on *mode*, write to *dest_path*.

    *mode* is one of:
        ``"patient"``    – apply_patient_fiasp only
        ``"controller"`` – apply_controller_fiasp only
        ``"both"``       – both

    Returns a list of per-override-item change records:
        [{sim_id, patient_change, controller_change}, ...]
    """
    with open(src_path) as f:
        config = json.load(f)

    change_log = []

    for override_item in config.get("override_config", []):
        sim_id = override_item.get("sim_id", "<unknown>")
        record = {"sim_id": sim_id, "patient_change": None, "controller_change": None}

        if mode in ("patient", "both"):
            record["patient_change"] = apply_patient_fiasp(override_item, reusable_root)

        if mode in ("controller", "both"):
            record["controller_change"] = apply_controller_fiasp(override_item, reusable_root)

        change_log.append(record)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w") as f:
        json.dump(config, f, indent=2)

    return change_log


# ---------------------------------------------------------------------------
# Directory-level processing
# ---------------------------------------------------------------------------

def get_matched_dirs(source_dir: Path, tlr_ids: list[str]) -> list[Path]:
    """
    Return all subdirectories of *source_dir* whose name exactly matches a TLR ID
    or starts with ``<TLR-ID>_``.
    """
    matched = []
    missing = []
    for tlr_id in tlr_ids:
        hits = [
            d for d in source_dir.iterdir()
            if d.is_dir() and (d.name == tlr_id or d.name.startswith(tlr_id + "_"))
        ]
        if hits:
            matched.extend(sorted(hits))
        else:
            missing.append(tlr_id)

    if missing:
        print(f"\n⚠  {len(missing)} TLR ID(s) not found in source directory (expected):")
        for m in missing:
            print(f"   {m}")

    return matched


def process_tlr_directory(
    src_tlr_dir: Path,
    reusable_root: Path,
) -> dict:
    """
    Copy *src_tlr_dir* to each of the three destination directories and apply
    the appropriate modifications.  Returns a summary dict.
    """
    summary = {
        "tlr_dir": src_tlr_dir.name,
        "files_processed": 0,
        "targets": {},
    }

    mode_map = {
        "ab_URAI_ptModel": "patient",
        "ab_URAI_pump":    "controller",
        "URAI":            "both",
    }

    json_files = sorted(src_tlr_dir.glob("*.json"))
    if not json_files:
        print(f"  ⚠  No JSON files found in {src_tlr_dir.name}")
        return summary

    for target_name, mode in mode_map.items():
        dest_tlr_dir = DEST_DIRS[target_name] / src_tlr_dir.name
        dest_tlr_dir.mkdir(parents=True, exist_ok=True)
        target_changes = []

        for json_file in json_files:
            dest_file = dest_tlr_dir / json_file.name
            changes = process_scenario_file(json_file, dest_file, mode, reusable_root)
            target_changes.append({"file": json_file.name, "changes": changes})

        summary["targets"][target_name] = target_changes

    summary["files_processed"] = len(json_files)
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(summaries: list[dict]) -> None:
    total_dirs = len(summaries)
    total_files = sum(s["files_processed"] for s in summaries)
    print("\n" + "=" * 70)
    print("CONFIGURE INSULIN MODEL COMPARE — SUMMARY")
    print("=" * 70)
    print(f"  TLR directories processed : {total_dirs}")
    print(f"  JSON files modified        : {total_files * 3}  ({total_files} × 3 targets)")

    for s in summaries:
        print(f"\n  {s['tlr_dir']}  ({s['files_processed']} file(s))")
        for target_name, file_changes in s["targets"].items():
            print(f"    [{target_name}]")
            for fc in file_changes:
                print(f"      {fc['file']}")
                for chg in fc["changes"]:
                    sim_id = chg["sim_id"]
                    if chg["patient_change"]:
                        print(f"        patient    [{sim_id}]: {chg['patient_change']}")
                    if chg["controller_change"]:
                        print(f"        controller [{sim_id}]: {chg['controller_change']}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all in-scope TLRs. Without this flag only TLR-549 is processed.",
    )
    args = parser.parse_args()

    tlr_ids = IN_SCOPE_TLRS if args.all else [TEST_TLR]
    mode_label = "FULL RUN" if args.all else f"TEST MODE (single TLR: {TEST_TLR})"

    print(f"\nConfigure Insulin Model Compare — {mode_label}")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Targets: {', '.join(DEST_DIRS.keys())}")

    matched_dirs = get_matched_dirs(SOURCE_DIR, tlr_ids)
    if not matched_dirs:
        print("No matching source directories found. Exiting.")
        return

    print(f"\nProcessing {len(matched_dirs)} director(ies)...\n")

    summaries = []
    for tlr_dir in matched_dirs:
        print(f"  → {tlr_dir.name}")
        summary = process_tlr_directory(tlr_dir, _REUSABLE_ROOT)
        summaries.append(summary)

    print_summary(summaries)


if __name__ == "__main__":
    main()
