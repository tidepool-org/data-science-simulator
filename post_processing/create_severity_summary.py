#!/usr/bin/env python3
"""
Create RTF shell documents for Tidepool Risk Severity Evaluation results.

Phase 2 refactor: all severity COMPUTATION now lives in severity_model.py, which
produces a structured SeverityAssessment object. This module is now the RTF
RENDERER — a thin consumer of that object. The GUI is a separate consumer of the
same object.

The RTF output is intended to be byte-for-byte identical to the pre-refactor
output; render_rtf() and the two render_* helpers reproduce the original template
and message strings exactly, driven by the structured object instead of computing
inline.

Backwards compatibility: round_half_up and calculate_integer_averages are
re-exported from severity_model so the existing test_create_severity_summary.py
imports continue to resolve unchanged.

Failure contract: process_results_directory raises SeveritySummaryError when a
results directory cannot be summarized at all, and returns a SummaryResult
naming what it wrote and what it skipped. It previously printed and returned
None on both, so neither the CLI nor the GUI export could tell a no-op run from
a successful one.
"""

import os
import json
from dataclasses import dataclass, field

from severity_model import (
    build_assessment_result,
    STAGE_ORDER,
    STAGE_DISPLAY,
    # re-exported for backwards-compat with existing tests:
    build_assessment,                  # noqa: F401  (re-export)
    # re-exported for backwards-compat with existing tests:
    round_half_up,                     # noqa: F401  (re-export)
    calculate_integer_averages,        # noqa: F401  (re-export)
    calculate_stage_averages,          # noqa: F401  (re-export)
    calculate_hyperglycemia_score,     # noqa: F401  (re-export)
    determine_harm_and_severity,       # noqa: F401  (re-export)
    identify_severity_4_hypoglycemia,  # noqa: F401  (re-export)
    extract_metric_data,               # noqa: F401  (re-export)
    get_profile_metrics,               # noqa: F401  (re-export)
)


# =============================================================================
# RTF rendering (consumes the structured findings; formatting only)
# =============================================================================

# Results-table column layout. Eight columns of even 1275-twip width, summing to
# the same 10200-twip total the six-column table used, so page fit is unchanged.
# Defined ONCE and emitted on every row (header + the three stage rows): the RTF
# spec requires the stops to be repeated per row, and keeping the literal in one
# place is what stops the four copies drifting apart when a column is added.
# Column order (must match the cells emitted in render_rtf):
#   Evaluation stage | Harm | Severity | TIR % | TBR % | LBGI | DKAI | TAR %
TABLE_CELL_STOPS = "".join(
    r"\cellx{}".format(1275 * column) for column in range(1, 9)
)


def render_catastrophic_identifier(catastrophic_findings):
    """RTF for the Critical/Catastrophic Identifier section.

    Reproduces the original generate_catastrophic_identifier_section() output
    exactly, driven by structured CatastrophicFinding objects.
    """
    if not catastrophic_findings:
        return "Critical/Catastrophic level determination not relevant to this scenario."

    rtf_lines = []
    for finding in sorted(catastrophic_findings, key=lambda f: f.sim_id):
        sim_id = finding.sim_id
        condition = finding.condition
        if condition == 'zero_or_negative':
            message = f"Glucose trace for {sim_id} includes values \\u8804? 0 mg/dL."
        elif condition == 'extended_low':
            message = f"Glucose trace for {sim_id} includes values \\u8804? 40 mg/dL for 4 or more hours."
        else:  # 'none'
            message = f"Glucose trace for {sim_id} includes no values \\u8804? 0 mg/dL or \\u8804? 40 mg/dL for \\u8805? 4 hours."
        rtf_lines.append(f"\\bullet  {message}")
        rtf_lines.append("\\par")
    return "\n".join(rtf_lines)


def render_outlier_results(outlier_findings, status):
    """Text for the Outlier Results section.

    Reproduces the original generate_outlier_results_section() output exactly,
    driven by structured OutlierFinding objects + a status string.
    """
    if status == 'malformed_data':
        # NOT "Data not available": the data was there and could not be read.
        # Deliberately does not name the file -- the path stays in the console
        # diagnostic; a regulatory document should not carry local filesystem paths.
        return ("Outlier analysis not performed: profile data is present but could "
                "not be read. Check data configuration.")
    if status == 'no_data':
        return "Data not available for outlier analysis."
    if status == 'single_profile':
        return "Only one profile present, so outliers are not relevant."

    if not outlier_findings:
        return "No outlier profiles exist. All results are within 1 severity level of one another."

    messages = []
    for f in outlier_findings:
        stage_name = STAGE_DISPLAY[f.stage]
        if f.harm_type == 'Hypoglycemia':
            messages.append(
                f"Outlier profile exists. {f.profile} has a Hypoglycemia score of {int(f.value)} at {stage_name}, "
                f"while other profiles have a Hypoglycemia score of {int(f.comparison_median)}."
            )
        elif f.harm_type == 'DKA':
            messages.append(
                f"Outlier profile exists. {f.profile} has a DKA score of {int(f.value)} at {stage_name}, "
                f"while other profiles have a DKA score of {int(f.comparison_median)}."
            )
        elif f.harm_type == 'Hyperglycemia':
            messages.append(
                f"Outlier profile exists. {f.profile} has a Hyperglycemia percent_cgm_gt_180 of 0.0 at {stage_name}, "
                f"while other profiles have a Hyperglycemia percent_cgm_gt_180 of {f.comparison_median:.1f}."
            )
    return " ".join(messages)


def render_profile_count(profile_count, usable_profile_count=None):
    """The 'N virtual patient profiles aggregated for this summary.' line.

    profile_count is N (files present); usable_profile_count is M (files that could
    contribute a value). M == N -- the normal clean-data case -- renders the
    original sentence byte-for-byte. M < N surfaces the discrepancy, because
    extract_metric_data drops a malformed file and averages the remainder, so the
    unqualified count could name more profiles than contributed a single value.

    usable_profile_count None means "not measured" and renders as M == N, so an
    assessment built by an older positional constructor is unaffected.
    """
    if usable_profile_count is None or usable_profile_count >= profile_count:
        return f"{profile_count} virtual patient profiles aggregated for this summary."
    dropped = profile_count - usable_profile_count
    noun = "file" if dropped == 1 else "files"
    return (
        f"{usable_profile_count} of {profile_count} virtual patient profiles "
        f"aggregated for this summary. "
        f"{dropped} summary results {noun} could not be read."
    )


def render_rtf(assessment):
    """Render a full RTF document string from a SeverityAssessment.

    Reproduces the original create_rtf_shell() template exactly. All inputs come
    from the structured object — including outlier_status — so no directory
    re-reading or branch re-derivation is needed.
    """
    formatted_timestamp = assessment.timestamp.replace('T', ' ').split('.')[0]

    stages = assessment.stages
    hs = {stage: (stages[stage].harm_type, stages[stage].severity) for stage in STAGE_ORDER}
    tir = {stage: stages[stage].tir for stage in STAGE_ORDER}
    tbr = {stage: stages[stage].tbr for stage in STAGE_ORDER}
    tar = {stage: stages[stage].tar for stage in STAGE_ORDER}
    # Raw averaged LBGI / DKA-index values. Already display-ready strings from
    # severity_model (truncated to 2dp, or 'NA') -- no formatting applied here.
    lbgi = {stage: stages[stage].lbgi_value_avg for stage in STAGE_ORDER}
    dkai = {stage: stages[stage].dka_index_value_avg for stage in STAGE_ORDER}

    catastrophic_content = render_catastrophic_identifier(assessment.catastrophic_findings)
    outlier_content = render_outlier_results(assessment.outlier_findings, assessment.outlier_status)
    profile_count_content = render_profile_count(
        assessment.profile_count, assessment.usable_profile_count
    )

    rtf_content = r"""{\rtf1\ansi\deff0
{\fonttbl{\f0 Arial;}}
\f0\fs24

{\b\fs28 Risk severity summary for """ + assessment.subdirectory_name + r"""}
\par\par

{\b Date and time of simulation run:} """ + formatted_timestamp + r"""
\par\par

Auto-generated output from Tidepool Risk Severity Evaluation Simulator Tool
\par\par

{\b Table of results}
\par\par

\trowd
""" + TABLE_CELL_STOPS + r"""
\pard\intbl {\b Evaluation stage}\cell
\pard\intbl {\b Harm}\cell
\pard\intbl {\b Severity}\cell
\pard\intbl {\b TIR % (70 - 180 mg/dL)}\cell
\pard\intbl {\b TBR % (<54 mg/dL)}\cell
\pard\intbl {\b LBGI}\cell
\pard\intbl {\b DKAI}\cell
\pard\intbl {\b TAR % (>180 mg/dL)}\cell
\row

\trowd
""" + TABLE_CELL_STOPS + r"""
\pard\intbl Pre-mitigation\cell
\pard\intbl """ + hs['pre'][0] + r"""\cell
\pard\intbl """ + hs['pre'][1] + r"""\cell
\pard\intbl """ + tir['pre'] + r"""\cell
\pard\intbl """ + tbr['pre'] + r"""\cell
\pard\intbl """ + lbgi['pre'] + r"""\cell
\pard\intbl """ + dkai['pre'] + r"""\cell
\pard\intbl """ + tar['pre'] + r"""\cell
\row

\trowd
""" + TABLE_CELL_STOPS + r"""
\pard\intbl No Loop\cell
\pard\intbl """ + hs['no_loop'][0] + r"""\cell
\pard\intbl """ + hs['no_loop'][1] + r"""\cell
\pard\intbl """ + tir['no_loop'] + r"""\cell
\pard\intbl """ + tbr['no_loop'] + r"""\cell
\pard\intbl """ + lbgi['no_loop'] + r"""\cell
\pard\intbl """ + dkai['no_loop'] + r"""\cell
\pard\intbl """ + tar['no_loop'] + r"""\cell
\row

\trowd
""" + TABLE_CELL_STOPS + r"""
\pard\intbl Post-mitigation\cell
\pard\intbl """ + hs['post'][0] + r"""\cell
\pard\intbl """ + hs['post'][1] + r"""\cell
\pard\intbl """ + tir['post'] + r"""\cell
\pard\intbl """ + tbr['post'] + r"""\cell
\pard\intbl """ + lbgi['post'] + r"""\cell
\pard\intbl """ + dkai['post'] + r"""\cell
\pard\intbl """ + tar['post'] + r"""\cell
\row

\pard
\par\par

""" + profile_count_content + r"""
\par\par

{\b Critical/Catastrophic Identifier}
\par\par

""" + catastrophic_content + r"""
\par\par

{\b Outlier Results}
\par\par

""" + outlier_content + r"""

\pard
}"""
    return rtf_content


# =============================================================================
# Directory processing (orchestration + file writing)
# =============================================================================

METADATA_FILENAME = 'metadata.json'
TLR_DIR_PREFIX = 'TLR-'
# The keys a run's metadata.json may carry its run timestamp under. Read in this
# order; absent both, the run cannot be dated and is an error rather than a
# summary stamped with a placeholder.
_TIMESTAMP_KEYS = ('timestamp', 'run_timestamp')


class SeveritySummaryError(Exception):
    """A results directory cannot be summarized at all.

    Raised instead of printing and returning, so a caller (the CLI, the GUI
    export) can tell an unusable directory from a successful run. A single TLR
    directory with no usable data is NOT this -- that is a partial outcome,
    reported in SummaryResult.skipped.
    """


@dataclass
class SummaryResult:
    """What process_results_directory actually did.

    written: the RTF paths it wrote, one per TLR directory that produced an
    assessment. skipped: ``(tlr_dir, reason)`` for each directory that did not,
    so a caller can report them rather than silently shipping fewer summaries
    than there were directories.

    The reason distinguishes an empty directory from malformed data (it used to be
    one shared string for both). A caller that needs to BRANCH on which, rather
    than report it, should call ``severity_model.build_assessment_result`` and read
    its typed ``status``; the tuple shape is left alone so existing consumers of
    ``skipped`` keep working.
    """
    written: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _read_run_timestamp(results_dir):
    """The run timestamp from results_dir/metadata.json.

    Raises SeveritySummaryError if the file is missing, unreadable, or carries no
    recognized timestamp key -- every summary is dated from this, so a placeholder
    would silently put "Unknown" on a regulatory document.
    """
    metadata_path = os.path.join(results_dir, METADATA_FILENAME)
    if not os.path.exists(metadata_path):
        raise SeveritySummaryError(
            f"{METADATA_FILENAME} not found in {results_dir}: the run cannot be dated. "
            "A GUI run writes it automatically; for a CLI run, check that this is a "
            "run directory produced by loop_risk_v2_0."
        )
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise SeveritySummaryError(f"Could not read {metadata_path}: {exc}") from exc

    for key in _TIMESTAMP_KEYS:
        if metadata.get(key):
            return metadata[key]
    raise SeveritySummaryError(
        f"{metadata_path} carries none of {list(_TIMESTAMP_KEYS)}, so the run cannot be dated."
    )


def _find_tlr_dirs(results_dir):
    """The TLR-* subdirectories of results_dir, sorted.

    Directories only -- a file named e.g. TLR-notes.txt is not a run directory and
    must not be handed to build_assessment. Sorted so processing order (and so the
    reported order) is the same on every machine.
    """
    return sorted(
        os.path.join(results_dir, name)
        for name in os.listdir(results_dir)
        if name.startswith(TLR_DIR_PREFIX) and os.path.isdir(os.path.join(results_dir, name))
    )


def process_results_directory(results_dir):
    """Build each TLR directory's assessment and write its RTF; return a SummaryResult.

    Raises SeveritySummaryError if the directory cannot be summarized at all --
    unreadable/undated metadata.json, or no TLR-* subdirectory. A single TLR
    directory that yields no assessment is recorded in the result's ``skipped``
    list instead, since a run legitimately can contain one -- with a reason naming
    whether it was empty or malformed.
    """
    timestamp = _read_run_timestamp(results_dir)

    tlr_dirs = _find_tlr_dirs(results_dir)
    if not tlr_dirs:
        raise SeveritySummaryError(
            f"No {TLR_DIR_PREFIX}* subdirectories found in {results_dir}: nothing to summarize."
        )

    print(f"Found {len(tlr_dirs)} TLR subdirectories")
    result = SummaryResult()

    for tlr_dir in tlr_dirs:
        print(f"\nProcessing: {tlr_dir}")

        outcome = build_assessment_result(tlr_dir, timestamp)
        assessment = outcome.assessment
        if assessment is None:
            # 'empty' (nothing ran here) and 'malformed' (the data is broken) both
            # used to report the same "no usable summary results data".
            reason = f"{outcome.status}: {outcome.detail}"
            print(f"  Skipped: {reason}")
            result.skipped.append((tlr_dir, reason))
            continue

        print(f"  Simulation ID: {assessment.simulation_id}")
        print(f"  Profile count: {assessment.usable_profile_count} usable of "
              f"{assessment.profile_count} present")

        for stage, label in [('pre', 'Pre'), ('no_loop', 'No Loop'), ('post', 'Post')]:
            sr = assessment.stages[stage]
            print(f"  [{label}] harm={sr.harm_type} severity={sr.severity} "
                  f"TIR={sr.tir} TBR={sr.tbr} TAR={sr.tar} "
                  f"(LBGI={sr.lbgi_score_avg}, DKA={sr.dka_score_avg}, "
                  f"hyper={sr.hyperglycemia_score}, n={sr.n_sims})")

        for finding in assessment.catastrophic_findings:
            print(f"    CATASTROPHIC {finding.sim_id}: severity {finding.updated_severity} ({finding.condition})")

        rtf = render_rtf(assessment)
        output_path = os.path.join(tlr_dir, f"risk_summary_{assessment.simulation_id}.rtf")
        with open(output_path, 'w') as f:
            f.write(rtf)
        result.written.append(output_path)
        print(f"Created shell document: {output_path}")

    return result


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Create RTF documents for risk severity evaluation results'
    )
    parser.add_argument('results_dir', help='Path to results directory')
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Directory not found: {args.results_dir}")
        return 1

    try:
        result = process_results_directory(args.results_dir)
    except SeveritySummaryError as exc:
        print(f"Error: {exc}")
        return 1

    for tlr_dir, reason in result.skipped:
        print(f"Skipped {tlr_dir}: {reason}")
    print(f"\nWrote {len(result.written)} summary document(s); "
          f"skipped {len(result.skipped)} directory(ies).")
    # An invocation that produced no document is not a success, whatever the
    # per-directory reasons were -- callers (and CI) read this exit code.
    if not result.written:
        print("Error: no summary documents were written.")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
