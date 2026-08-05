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
"""

import os
import json
import glob

from severity_model import (
    build_assessment,
    STAGE_ORDER,
    STAGE_DISPLAY,
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

""" + str(assessment.profile_count) + r""" virtual patient profiles aggregated for this summary.
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

def process_results_directory(results_dir):
    """Process a results directory: build each TLR's assessment and write its RTF."""
    metadata_path = os.path.join(results_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {results_dir}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    timestamp = metadata.get('timestamp', metadata.get('run_timestamp', 'Unknown'))

    tlr_dirs = glob.glob(os.path.join(results_dir, 'TLR-*'))
    if not tlr_dirs:
        print(f"No TLR-* subdirectories found in {results_dir}")
        return

    print(f"Found {len(tlr_dirs)} TLR subdirectories")

    for tlr_dir in tlr_dirs:
        print(f"\nProcessing: {tlr_dir}")

        assessment = build_assessment(tlr_dir, timestamp)
        if assessment is None:
            continue

        print(f"  Simulation ID: {assessment.simulation_id}")
        print(f"  Profile count: {assessment.profile_count}")

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
        print(f"Created shell document: {output_path}")


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

    process_results_directory(args.results_dir)
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
