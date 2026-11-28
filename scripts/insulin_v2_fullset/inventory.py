"""
Phase 1 read-only inventory for TRSET-84.

Resolves every in-scope baseline sim through ScenarioParserV2 (no build, no simulation), groups them by
(patient_insulin_type, controller_model), and writes a report plus the translation table with nominal rows.
It never writes under the baseline or reusable/.

Usage:
    python -m scripts.insulin_v2_fullset.inventory [--baseline DIR] [--out-dir DIR] [--write-table]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .resolve import DEFAULT_BASELINE, load_exclude_list, resolve_baseline
from .translation import TABLE_PATH, build_table, load_table, pair_key, save_table

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output"


def group_pairs(sims):
    groups = defaultdict(lambda: {"sim_ids": [], "tlrs": set(), "timings": set(), "metabolism_override_sims": 0,
                                  "controller_settings_override_sims": 0})
    for s in sims:
        g = groups[(s["patient_insulin_type"], s["controller_model"])]
        g["sim_ids"].append(f'{s["tlr"]}/{s["sim_id"]}')
        g["tlrs"].add(s["tlr"])
        g["timings"].add(s["insulin_timing"])
        g["metabolism_override_sims"] += s["override_sets_patient_metabolism"]
        g["controller_settings_override_sims"] += s["override_sets_controller_settings"]
    return groups


def build_report(sims, errors, exclude):
    groups = group_pairs(sims)
    pairs = []
    for (ptype, ctrl), g in sorted(groups.items()):
        pairs.append({
            "key": pair_key(ptype, ctrl), "patient_insulin_type": ptype, "controller_model": ctrl,
            "n_sims": len(g["sim_ids"]), "n_tlrs": len(g["tlrs"]), "tlrs": sorted(g["tlrs"]),
            "insulin_timing_values": sorted(g["timings"]),
            "sims_with_metabolism_override": g["metabolism_override_sims"],
            "sims_with_controller_settings_override": g["controller_settings_override_sims"],
            "sim_ids": sorted(g["sim_ids"]),
        })
    return {"excluded": exclude, "n_sims": len(sims), "n_pairs": len(pairs), "pairs": pairs, "errors": errors}


def format_report(report):
    lines = [f"Excluded TLR dirs: {report['excluded']}",
             f"Sims resolved: {report['n_sims']}   Distinct pairs: {report['n_pairs']}   Resolution errors: {len(report['errors'])}", ""]
    for p in report["pairs"]:
        lines += [f"PAIR  patient={p['patient_insulin_type']}  controller={p['controller_model']}",
                  f"  sims={p['n_sims']}  TLR dirs={p['n_tlrs']}  insulin_timing={p['insulin_timing_values']}",
                  f"  sims overriding patient metabolism: {p['sims_with_metabolism_override']}   "
                  f"sims overriding controller settings: {p['sims_with_controller_settings_override']}",
                  f"  TLRs: {', '.join(p['tlrs'])}", ""]
    if report["errors"]:
        lines.append("RESOLUTION ERRORS")
        lines += [f"  {e['tlr']}/{e['file']}: {e['error']}" for e in report["errors"]]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--write-table", action="store_true",
                    help="write/update the translation table (existing rows are kept untouched)")
    args = ap.parse_args(argv)

    exclude = load_exclude_list()
    sims, errors = resolve_baseline(args.baseline, exclude)
    report = build_report(sims, errors, exclude)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "inventory_report.txt").write_text(format_report(report) + "\n")
    print(format_report(report))

    if args.write_table:
        existing = load_table() if TABLE_PATH.exists() else None
        save_table(build_table([(p["patient_insulin_type"], p["controller_model"]) for p in report["pairs"]], existing))
        print(f"\nTranslation table written: {TABLE_PATH}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
