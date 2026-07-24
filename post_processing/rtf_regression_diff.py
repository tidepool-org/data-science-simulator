#!/usr/bin/env python3
"""
rtf_regression_diff.py — RTF regression gate (stored-golden).

Proves the RTF renderer still produces the agreed table layout by rendering a
small, committed synthetic fixture through the CURRENT code and byte-diffing the
result against a committed golden RTF.

History: this used to byte-diff the pre-refactor monolith
(create_severity_summary_ORIGINAL.py) against the refactored code over a real
Risk_Run_* directory on Shawn's Mac. That code-vs-code design could not survive
an intentional output change (the LBGI/DKAI columns) and required private data.
It is now a self-contained, reproducible golden-file gate: the fixture and golden
live under tests/fixtures/rtf_regression/, so it runs in CI and as a pytest with
no external data.

Regenerating the golden (only when a table change is intentional):
    python post_processing/rtf_regression_diff.py --regenerate

Checking (default; exit 0 = matches golden, exit 1 = differs):
    python post_processing/rtf_regression_diff.py
"""

import argparse
import difflib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make both the renderer module (post_processing/) and the package (repo root)
# importable whether this runs as a script or under pytest.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from create_severity_summary import build_assessment, render_rtf  # noqa: E402

FIXTURE_ROOT = os.path.join(REPO_ROOT, 'tests', 'fixtures', 'rtf_regression')
FIXTURE_TLR_DIR = os.path.join(
    FIXTURE_ROOT, 'Risk_Run_2026-01-01T00_00_00.000000', 'TLR-999'
)
# Timestamp is read from the fixture's metadata.json in the real pipeline; the
# gate pins it directly so the rendered header is deterministic.
FIXTURE_TIMESTAMP = '2026-01-01T00:00:00.000000'
GOLDEN_PATH = os.path.join(FIXTURE_ROOT, 'golden', 'expected_risk_summary_TLR-999.rtf')


def render_fixture():
    """Render the committed fixture through the current renderer; return RTF text."""
    assessment = build_assessment(FIXTURE_TLR_DIR, FIXTURE_TIMESTAMP)
    if assessment is None:
        raise RuntimeError(f"Fixture produced no assessment: {FIXTURE_TLR_DIR}")
    return render_rtf(assessment)


def regenerate():
    """Overwrite the golden with freshly rendered output (intentional changes only)."""
    rtf = render_fixture()
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, 'w') as f:
        f.write(rtf)
    print(f"Regenerated golden: {os.path.relpath(GOLDEN_PATH, REPO_ROOT)}")


def diff_against_golden():
    """Return (matches: bool, diff_lines: list[str]) for current output vs golden."""
    actual = render_fixture()
    if not os.path.exists(GOLDEN_PATH):
        return (False, [f"Golden file missing: {GOLDEN_PATH}. Run with --regenerate."])
    with open(GOLDEN_PATH) as f:
        expected = f.read()
    if actual == expected:
        return (True, [])
    diff = list(difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile='golden', tofile='current',
    ))
    return (False, diff)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--regenerate', action='store_true',
                    help='Overwrite the golden RTF with current output (intentional table changes only).')
    args = ap.parse_args()

    if args.regenerate:
        regenerate()
        return 0

    matches, diff = diff_against_golden()
    if matches:
        print("RESULT: PASS — rendered fixture is byte-identical to the golden RTF.")
        return 0
    print("RESULT: FAIL — rendered fixture differs from the golden RTF:")
    for line in diff[:60]:
        print("  " + line.rstrip("\n"))
    print("\nIf this change is intentional, regenerate with --regenerate.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
