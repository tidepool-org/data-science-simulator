#!/usr/bin/env python3
"""
standardize_sim_ids.py

Standardizes sim_id values in JSON simulation configuration files to comply
with the Loop risk v2 naming convention:

  1. Pre-loop:  "pre-Loop_NoMitigations_<t1|t2>_<profile>"
  2. No-loop:   "pre-noLoop_<t1|t2>_<profile>"
  3. Post-loop: "post-Loop-WithMitigations_<t1|t2>_<profile>"

Usage:
  # Dry-run (default — no files modified, produces CSVs showing proposed changes):
  python standardize_sim_ids.py --dir <path>

  # Apply changes:
  python standardize_sim_ids.py --dir <path> --apply

Output files (written to <dir>):
  sim_id_auto_fixes.csv        — all auto-fixable changes (proposed or applied)
  sim_id_flagged_for_review.csv — entries that could not be auto-resolved
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Compliance pattern — anything matching this is already correct
# ---------------------------------------------------------------------------
COMPLIANT_RE = re.compile(
    r"^(pre-Loop_NoMitigations|pre-noLoop|post-Loop-WithMitigations)_(t1|t2)_\S+$"
)

# ---------------------------------------------------------------------------
# Human-readable descriptions for each flag reason
# ---------------------------------------------------------------------------
FLAG_REASONS = {
    "missing_t1_t2": (
        "Cannot determine t1/t2 — component is absent from sim_id; "
        "manual inspection of file required"
    ),
    "missing_t_and_profile": (
        "sim_id has neither t1/t2 nor a profile component"
    ),
    "different_scheme": (
        "Completely different naming scheme — not a Loop risk scenario "
        "(e.g. scenario_icgm_*)"
    ),
    "unrecognized": (
        "Does not match any known pattern after rule application"
    ),
}


# ---------------------------------------------------------------------------
# Rewrite rules
# ---------------------------------------------------------------------------

def apply_rules(sim_id: str) -> str:
    """
    Apply ordered rewrite rules to a sim_id.
    Returns the corrected string (unchanged if already compliant or unrecognized).
    Rules are grouped by scenario family and applied from most-specific to least.
    """

    # ── POST-LOOP ────────────────────────────────────────────────────────────
    if sim_id.startswith("post"):

        # Trailing comma inside the string value
        sim_id = sim_id.rstrip(",")

        # Typo: resistnat → resistant
        sim_id = sim_id.replace("resistnat", "resistant")

        # Misplaced t2 token:
        #   post-Loop-t2_WithMitigations_t2_<p> → post-Loop-WithMitigations_t2_<p>
        sim_id = re.sub(
            r"^post-Loop-t2_WithMitigations_(t2_.+)$",
            r"post-Loop-WithMitigations_\1",
            sim_id,
        )

        # All-underscores prefix:  post_Loop_WithMitigations_ → post-Loop-WithMitigations_
        sim_id = re.sub(
            r"^post_Loop_WithMitigations_",
            "post-Loop-WithMitigations_",
            sim_id,
        )

        # Missing hyphen:  post-LoopWithMitigations_ → post-Loop-WithMitigations_
        sim_id = re.sub(
            r"^post-LoopWithMitigations_",
            "post-Loop-WithMitigations_",
            sim_id,
        )

        # Underscore separator (any case of 'w'):
        #   post-Loop_[Ww]ithMitigations_ → post-Loop-WithMitigations_
        sim_id = re.sub(
            r"^post-Loop_[Ww]ithMitigations_",
            "post-Loop-WithMitigations_",
            sim_id,
        )

        # Lowercase 'w':  post-Loop-withMitigations_ → post-Loop-WithMitigations_
        sim_id = re.sub(
            r"^post-Loop-withMitigations_",
            "post-Loop-WithMitigations_",
            sim_id,
        )

        # Uppercase T1:  _T1_ → _t1_
        sim_id = re.sub(r"_(T1)_", "_t1_", sim_id)

        # Redundant profile prefix before t1/t2 (applied last, after structural fixes):
        #   post-Loop-WithMitigations_adolescent_t1_adolescent → _t1_adolescent
        #   Matches: <correct-prefix>_<profile>_t{1|2}_<same-profile>
        sim_id = re.sub(
            r"^(post-Loop-WithMitigations)"
            r"_(adolescent|median|resistant|sensitive)"
            r"_(t[12]_(?:adolescent|median|resistant|sensitive))$",
            r"\1_\3",
            sim_id,
        )

    # ── PRE-LOOP  (Loop / NoMitigations family) ──────────────────────────────
    elif sim_id.lower().startswith("pre-loop"):

        # Duplicated suffix:
        #   _t1_<profile>_t1_<profile> → _t1_<profile>
        #   _t1_<profile>_t2_<profile> → _t1_<profile>  (picks first)
        sim_id = re.sub(r"_(t[12]_\w+)_t[12]_\w+$", r"_\1", sim_id)

        # Redundant profile prefix:
        #   pre-Loop_NoMitigations_median_t1_median → pre-Loop_NoMitigations_t1_median
        sim_id = re.sub(
            r"^(pre-Loop_NoMitigations)"
            r"_(adolescent|median|resistant|sensitive)"
            r"_(t[12]_(?:adolescent|median|resistant|sensitive))$",
            r"\1_\3",
            sim_id,
        )

        # Double-s typo:  pre-LoopNoMitigationss_ → pre-Loop_NoMitigations_
        sim_id = re.sub(
            r"^pre-LoopNoMitigationss_",
            "pre-Loop_NoMitigations_",
            sim_id,
        )

        # Missing underscore:  pre-LoopNoMitigations_ → pre-Loop_NoMitigations_
        sim_id = re.sub(
            r"^pre-LoopNoMitigations_",
            "pre-Loop_NoMitigations_",
            sim_id,
        )

        # Hyphen or underscore separator, any case of 'n':
        #   pre-Loop[-_][Nn]oMitigations_ → pre-Loop_NoMitigations_
        sim_id = re.sub(
            r"^pre-Loop[-_][Nn]oMitigations_",
            "pre-Loop_NoMitigations_",
            sim_id,
        )

        # Wrong scenario type — pre-loop labelled WithMitigations should be post-loop:
        #   pre-Loop-WithMitigations_<t>_<profile> → post-Loop-WithMitigations_<t>_<profile>
        sim_id = re.sub(
            r"^pre-Loop-WithMitigations_",
            "post-Loop-WithMitigations_",
            sim_id,
        )

        # Uppercase T1
        sim_id = re.sub(r"_(T1)_", "_t1_", sim_id)

    # ── NO-LOOP  (noLoop / NoLoop family) ────────────────────────────────────
    elif sim_id.lower().startswith("pre-noloop") or sim_id.startswith("pre-NoLoop"):

        # Redundant profile prefix:
        #   pre-noLoop_resistant_t1_resistant → pre-noLoop_t1_resistant
        #   Works for any capitalisation of 'no' and 'loop'
        sim_id = re.sub(
            r"^pre-[Nn]o[Ll]oop"
            r"_(adolescent|median|resistant|sensitive)"
            r"_(t[12]_(?:adolescent|median|resistant|sensitive))$",
            r"pre-noLoop_\2",
            sim_id,
        )

        # Capital N:  pre-NoLoop_ → pre-noLoop_
        sim_id = re.sub(r"^pre-NoLoop_", "pre-noLoop_", sim_id)

        # Uppercase T1:  pre-noLoop_T1_ → pre-noLoop_t1_
        sim_id = re.sub(r"^pre-noLoop_T1_", "pre-noLoop_t1_", sim_id)

        # Stray whitespace between t-token and profile:  _t1_ adolescent → _t1_adolescent
        sim_id = re.sub(r"_(t[12])_\s+(\S)", r"_\1_\2", sim_id)

    return sim_id


# ---------------------------------------------------------------------------
# Post-rule classification for non-compliant sim_ids
# ---------------------------------------------------------------------------

def classify_flagged(sim_id: str) -> str:
    """Return a flag-reason code for a sim_id that is still non-compliant."""
    if sim_id.startswith("scenario_icgm"):
        return "different_scheme"

    # Bare prefix with no t1/t2 or profile — includes separator variants that
    # were not auto-fixable because the t1/t2 + profile parts are missing
    # e.g. "post-Loop_WithMitigations", "pre-NoLoop"
    if re.match(
        r"^(pre-Loop_NoMitigations|pre-noLoop|pre-NoLoop"
        r"|post-Loop-WithMitigations|post-Loop_WithMitigations)$",
        sim_id,
    ):
        return "missing_t_and_profile"

    # Correct (or near-correct) prefix present but t1/t2 component is absent
    # e.g. "pre-Loop_NoMitigations_adolescent", "pre-noLoop_Median"
    if re.match(
        r"^(pre-Loop_NoMitigations|pre-noLoop|post-Loop-WithMitigations)"
        r"_[^t]\S*$",
        sim_id,
    ):
        return "missing_t1_t2"

    return "unrecognized"


# ---------------------------------------------------------------------------
# JSON traversal
# ---------------------------------------------------------------------------

def extract_sim_ids(data) -> list[str]:
    """Recursively collect all sim_id string values from a parsed JSON structure."""
    found = []
    if isinstance(data, dict):
        if "sim_id" in data and isinstance(data["sim_id"], str):
            found.append(data["sim_id"])
        for v in data.values():
            found.extend(extract_sim_ids(v))
    elif isinstance(data, list):
        for item in data:
            found.extend(extract_sim_ids(item))
    return found


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(path: Path, apply: bool) -> dict:
    """
    Parse one JSON file, evaluate every sim_id, and optionally apply fixes.

    Returns a dict with:
      path     — Path object
      entries  — list of (original, corrected, status, reason)
                 status ∈ {'compliant', 'auto_fixed', 'flagged'}
      changed  — True if the file was written (only possible when apply=True)
      error    — exception message string, or None
    """
    result = {"path": path, "entries": [], "changed": False, "error": None}

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        result["error"] = str(exc)
        return result

    # Deduplicate within the file (same id may appear in multiple scenarios)
    sim_ids = list(dict.fromkeys(extract_sim_ids(data)))
    if not sim_ids:
        return result

    new_raw = raw
    for orig in sim_ids:
        corrected = apply_rules(orig)

        if COMPLIANT_RE.match(corrected):
            status = "compliant" if corrected == orig else "auto_fixed"
            reason = None
        else:
            status = "flagged"
            reason = classify_flagged(corrected)

        result["entries"].append((orig, corrected, status, reason))

        if status == "auto_fixed" and apply:
            # Targeted in-place replacement preserves all original whitespace /
            # indentation; handles both `"sim_id": "..."` and `"sim_id":"..."`.
            new_raw = re.sub(
                r'("sim_id"\s*:\s*")' + re.escape(orig) + r'"',
                r"\g<1>" + corrected + '"',
                new_raw,
            )

    if apply and new_raw != raw:
        path.write_text(new_raw, encoding="utf-8")
        result["changed"] = True

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        required=True,
        metavar="PATH",
        help="Root directory to scan recursively for .json files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write corrected sim_ids back to files (default: dry-run only)",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(root.rglob("*.json"))
    mode_label = "APPLY" if args.apply else "DRY-RUN (no files modified)"
    print(f"Scanning {len(json_files):,} JSON files under {root}")
    print(f"Mode    : {mode_label}")
    print()

    counts = Counter()
    fixed_rows: list[dict] = []
    flagged_rows: list[dict] = []
    files_changed = 0

    for path in json_files:
        result = process_file(path, apply=args.apply)
        rel = path.relative_to(root)

        if result["error"]:
            counts["errors"] += 1
            print(f"  PARSE ERROR  {rel}: {result['error']}", file=sys.stderr)
            continue

        if result["changed"]:
            files_changed += 1

        for orig, corrected, status, reason in result["entries"]:
            counts[status] += 1

            if status == "auto_fixed":
                fixed_rows.append(
                    {
                        "file": str(rel),
                        "original": orig,
                        "corrected": corrected,
                    }
                )
            elif status == "flagged":
                flagged_rows.append(
                    {
                        "file": str(rel),
                        "original_sim_id": orig,
                        "sim_id_after_rules": corrected,
                        "reason": reason,
                        "description": FLAG_REASONS.get(reason, ""),
                    }
                )

    # ── Summary ───────────────────────────────────────────────────────────────
    total = counts["compliant"] + counts["auto_fixed"] + counts["flagged"]
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Total sim_id entries found   : {total:,}")
    print(f"  Already compliant            : {counts['compliant']:,}")
    print(f"  Auto-{'applied' if args.apply else 'fixable'}                  : {counts['auto_fixed']:,}")
    print(f"  Flagged for manual review    : {counts['flagged']:,}")
    if counts["errors"]:
        print(f"  Parse errors                 : {counts['errors']:,}")
    if args.apply:
        print(f"  Files modified               : {files_changed:,}")
    print()

    # ── Write CSV reports ─────────────────────────────────────────────────────
    if fixed_rows:
        fixed_path = root / "sim_id_auto_fixes.csv"
        with open(fixed_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "original", "corrected"])
            writer.writeheader()
            writer.writerows(fixed_rows)
        verb = "Applied" if args.apply else "Would apply"
        print(f"{verb} {len(fixed_rows):,} auto-fix(es).  Details → {fixed_path.name}")

    if flagged_rows:
        flagged_path = root / "sim_id_flagged_for_review.csv"
        with open(flagged_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "original_sim_id",
                    "sim_id_after_rules",
                    "reason",
                    "description",
                ],
            )
            writer.writeheader()
            writer.writerows(flagged_rows)
        print(
            f"Flagged {len(flagged_rows):,} sim_id(s) for manual review.  "
            f"Details → {flagged_path.name}"
        )

        print()
        print("Flagged reasons breakdown:")
        reason_counts = Counter(r["reason"] for r in flagged_rows)
        for reason, count in reason_counts.most_common():
            print(f"  {reason:<32} {count:>5}  — {FLAG_REASONS.get(reason, '')}")

    if not args.apply and (fixed_rows or flagged_rows):
        print()
        print("Re-run with --apply to write auto-fixes to disk.")


if __name__ == "__main__":
    main()
