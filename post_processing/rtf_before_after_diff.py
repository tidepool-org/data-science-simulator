#!/usr/bin/env python3
"""
rtf_before_after_diff.py -- TRSET-82 before/after validation on real results.

Companion to rtf_regression_diff.py. That gate renders committed synthetic fixtures
against stored goldens; this one renders every TLR directory of a REAL results run
twice -- with the code at a baseline git ref (default origin/main) and with the
working tree -- and classifies each difference.

An RTF difference is ALLOWED only if it is one of:
    harm/severity  a Harm or Severity cell of the results table changed
    note           an added "Harm not constrained by delivery risk type" note
    outlier        a changed or removed Outlier Results sentence
    catastrophic   a Critical/Catastrophic Identifier section changed to "not relevant"
Anything else (any other table cell, the count line, a changed catastrophic section
that is not "not relevant", a file that exists on one side only) is a FAILURE and the
exit code is 1. The run-timestamp line is neutralised (the only non-deterministic field).

Usage:
    python post_processing/rtf_before_after_diff.py RISK_RUN_DIR [--baseline-ref REF] [--markdown]

RISK_RUN_DIR is a Risk_Run_* directory containing TLR-* subdirectories and a
metadata.json. The originals are never written: both renders work on temporary copies.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOT_RELEVANT = "Critical/Catastrophic level determination not relevant to this scenario."
NOTE_PREFIX = "Harm not constrained by delivery risk type: "
STAGE_LABELS = ('Pre-mitigation', 'No Loop', 'Post-mitigation')
CATASTROPHIC_MARK = r"{\b Critical/Catastrophic Identifier}"
OUTLIER_MARK = r"{\b Outlier Results}"

_RENDER_SNIPPET = (
    "import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]);"
    "from create_severity_summary import process_results_directory;"
    "r = process_results_directory(sys.argv[3]);"
    "print('WRITTEN', len(r.written), 'SKIPPED', len(r.skipped))"
)


def copy_run(run_dir, dest):
    shutil.copytree(
        run_dir, dest,
        ignore=shutil.ignore_patterns('*.png', 'loop_algo_io', '*.rtf', '*.pdf'),
    )


def render_with(code_dir, run_dir, python):
    """Run process_results_directory from code_dir/post_processing on run_dir."""
    proc = subprocess.run(
        [python, '-I', '-c', _RENDER_SNIPPET,
         os.path.join(code_dir, 'post_processing'), code_dir, run_dir],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"render failed for {code_dir}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def export_ref(ref, dest):
    """Extract post_processing/ and the package at git ref into dest."""
    archive = subprocess.run(
        ['git', '-C', REPO_ROOT, 'archive', ref, 'post_processing'],
        capture_output=True, check=True,
    )
    subprocess.run(['tar', '-x', '-C', dest], input=archive.stdout, check=True)


def _neutralise(rtf):
    return re.sub(r'\{\\b Date and time of simulation run:\} [^\n]*', 'TS', rtf)


def _split(rtf):
    head, rest = rtf.split(CATASTROPHIC_MARK, 1)
    catastrophic, outlier = rest.split(OUTLIER_MARK, 1)
    return head, catastrophic, outlier


def _mask_harm_cells(head):
    """Replace each stage row's Harm and Severity cells with a placeholder; return
    (masked_text, [(harm, severity) per stage])."""
    lines = head.split('\n')
    cells, out, i = [], [], 0
    while i < len(lines):
        out.append(lines[i])
        m = re.match(r'^\\pard\\intbl (.*)\\cell$', lines[i])
        if m and m.group(1) in STAGE_LABELS and i + 2 < len(lines):
            harm = re.match(r'^\\pard\\intbl (.*)\\cell$', lines[i + 1]).group(1)
            sev = re.match(r'^\\pard\\intbl (.*)\\cell$', lines[i + 2]).group(1)
            cells.append((harm, sev))
            out.extend([r'\pard\intbl <HARM>\cell', r'\pard\intbl <SEV>\cell'])
            i += 3
            continue
        i += 1
    return '\n'.join(out), cells


def _strip_note(head):
    return re.sub(r'\n\\par\n' + re.escape(NOTE_PREFIX) + r'[^\n]*', '', head)


def classify(before, after):
    """(allowed_kinds: set, failures: list[str]) for one TLR's RTF pair."""
    before, after = _neutralise(before), _neutralise(after)
    if before == after:
        return set(), []
    kinds, failures = set(), []
    b_head, b_cat, b_out = _split(before)
    a_head, a_cat, a_out = _split(after)

    a_head_no_note = _strip_note(a_head)
    if a_head_no_note != a_head:
        kinds.add('note')
    if NOTE_PREFIX in b_head:
        failures.append("baseline already carried a delivery-risk note")
    b_masked, b_cells = _mask_harm_cells(b_head)
    a_masked, a_cells = _mask_harm_cells(a_head_no_note)
    if b_masked != a_masked:
        failures.append("table or count line differs outside the Harm/Severity cells")
    if b_cells != a_cells:
        kinds.add('harm/severity')

    if b_cat != a_cat:
        if a_cat.replace(r'\par', '').strip() == NOT_RELEVANT:
            kinds.add('catastrophic')
        else:
            failures.append("Catastrophic Identifier section changed to something "
                            "other than 'not relevant'")
    if b_out != a_out:
        kinds.add('outlier')
    return kinds, failures


def compare_runs(before_run, after_run):
    """{tlr: (kinds, failures, before_cells, after_cells)} over every TLR directory."""
    results = {}
    names = sorted({os.path.basename(d) for d in
                    glob.glob(os.path.join(before_run, 'TLR-*')) +
                    glob.glob(os.path.join(after_run, 'TLR-*'))})
    for name in names:
        b = glob.glob(os.path.join(before_run, name, 'risk_summary_*.rtf'))
        a = glob.glob(os.path.join(after_run, name, 'risk_summary_*.rtf'))
        if bool(b) != bool(a):
            side = 'baseline only' if b else 'new code only'
            results[name] = (set(), [f"RTF written by {side}"], None, None)
            continue
        if not b:
            results[name] = (set(), [], None, None)  # neither side produced one
            continue
        before = open(b[0], encoding='utf-8', newline='').read()
        after = open(a[0], encoding='utf-8', newline='').read()
        kinds, failures = classify(before, after)
        results[name] = (kinds, failures,
                         _mask_harm_cells(_split(_neutralise(before))[0])[1],
                         _mask_harm_cells(_split(_neutralise(after))[0])[1])
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--baseline-ref', default='origin/main')
    ap.add_argument('--markdown', action='store_true', help='print a markdown table')
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        base_code = os.path.join(tmp, 'baseline_code')
        os.makedirs(base_code)
        export_ref(args.baseline_ref, base_code)
        before_run, after_run = os.path.join(tmp, 'before'), os.path.join(tmp, 'after')
        copy_run(args.run_dir, before_run)
        copy_run(args.run_dir, after_run)
        render_with(base_code, before_run, sys.executable)
        render_with(REPO_ROOT, after_run, sys.executable)
        results = compare_runs(before_run, after_run)

    failed = False
    changed = []
    for name, (kinds, failures, bc, ac) in results.items():
        if failures:
            failed = True
            print(f"FAIL {name}: " + "; ".join(failures))
        if kinds:
            changed.append((name, kinds, bc, ac))
    unchanged = len(results) - len(changed) - sum(1 for r in results.values() if r[1])
    if args.markdown:
        print("| TLR | change kinds | before (pre / no Loop / post) | after |")
        print("|---|---|---|---|")
    for name, kinds, bc, ac in changed:
        fmt = lambda cells: ' / '.join(f"{h} {s}" for h, s in cells) if cells else '-'
        row = (name, ', '.join(sorted(kinds)), fmt(bc), fmt(ac))
        print("| " + " | ".join(row) + " |" if args.markdown else
              f"CHANGED {row[0]} [{row[1]}]  {row[2]}  ->  {row[3]}")
    print(f"\n{len(results)} TLR directories: {len(changed)} changed (allowed), "
          f"{unchanged} byte-identical, "
          f"{sum(1 for r in results.values() if r[1])} failing.")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
