#!/usr/bin/env python3
"""
rtf_regression_diff.py -- byte-level RTF regression gate (stored-golden).

Renders committed synthetic fixtures through the CURRENT renderer and compares
the result, byte for byte, against committed golden RTF files. Any difference --
a word of static text, a \\cellx tab stop, a control word, whitespace, a line
ending -- fails the check and the diff names where.

Fixtures (tests/fixtures/rtf_regression/Risk_Run_2026-01-01T00_00_00.000000/):
    TLR-999  populated table, all three stages, two profiles.
    TLR-998  one profile, post-mitigation stage absent, lbgi/dka_index columns
             absent: exercises "NA" cells and a truncated stage.
Goldens live under tests/fixtures/rtf_regression/golden/.

NON-DETERMINISTIC FIELDS NEUTRALISED (every one is named here; nothing else is):
    1. Run timestamp -- the "Date and time of simulation run" line. The real
       pipeline reads it from metadata.json; this gate pins FIXTURE_TIMESTAMP.
    No other field varies between renders (TestRtfRegression proves two renders
    are identical and that the timestamp is the only run-varying input).

Checking (also runs in the normal pytest suite via test_rtf_regression.py):
    python post_processing/rtf_regression_diff.py        # exit 0 = match, 1 = differs

Regenerating the goldens -- ONLY for an intentional, reviewed RTF change:
    python post_processing/rtf_regression_diff.py --regenerate
READ THE RESULTING `git diff` OF THE GOLDENS BEFORE COMMITTING. Regenerating
without reading the diff defeats this check; it must never be the automatic
response to a failure.

Note: TestRtfOutputUnchanged in test_create_severity_summary.py is NOT this
check -- it compares the renderer to itself and cannot detect a text change.
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
FIXTURE_RUN_DIR = os.path.join(FIXTURE_ROOT, 'Risk_Run_2026-01-01T00_00_00.000000')
GOLDEN_DIR = os.path.join(FIXTURE_ROOT, 'golden')

# Neutralised non-deterministic field #1: the run timestamp.
FIXTURE_TIMESTAMP = '2026-01-01T00:00:00.000000'

FIXTURE_NAMES = ('TLR-999', 'TLR-998')


def fixture_dir(name):
    return os.path.join(FIXTURE_RUN_DIR, name)


def golden_path(name):
    return os.path.join(GOLDEN_DIR, f'expected_risk_summary_{name}.rtf')


def render_fixture(name, timestamp=FIXTURE_TIMESTAMP):
    """Render one committed fixture through the current renderer; return RTF text."""
    assessment = build_assessment(fixture_dir(name), timestamp)
    if assessment is None:
        raise RuntimeError(f"Fixture produced no assessment: {fixture_dir(name)}")
    return render_rtf(assessment)


def _read_bytes_as_text(path):
    # newline='' + explicit encoding: no newline translation, so \r\n vs \n
    # differences are visible rather than silently normalised.
    with open(path, encoding='utf-8', newline='') as f:
        return f.read()


def regenerate():
    """Overwrite the goldens with freshly rendered output (intentional changes only)."""
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    for name in FIXTURE_NAMES:
        with open(golden_path(name), 'w', encoding='utf-8', newline='') as f:
            f.write(render_fixture(name))
        print(f"Regenerated golden: {os.path.relpath(golden_path(name), REPO_ROOT)}")
    print("Now READ `git diff` on the goldens before committing.")


def diff_against_golden(name, actual=None):
    """Return (matches: bool, diff_lines: list[str]) for current output vs golden."""
    if actual is None:
        actual = render_fixture(name)
    path = golden_path(name)
    if not os.path.exists(path):
        return (False, [f"Golden file missing: {path}. Run with --regenerate."])
    expected = _read_bytes_as_text(path)
    if actual == expected:
        return (True, [])
    diff = list(difflib.unified_diff(
        # repr-free but line-ending-preserving; splitlines(keepends) keeps \r\n.
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile=f'golden/{name}', tofile=f'current/{name}',
    ))
    if not diff:  # differs, but only in something splitlines treats as equal
        diff = [f"Byte-level difference not visible in line diff (lengths "
                f"{len(expected)} vs {len(actual)}); compare with repr()."]
    return (False, diff)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--regenerate', action='store_true',
                    help='Overwrite the golden RTFs with current output '
                         '(intentional changes only; read the git diff afterwards).')
    args = ap.parse_args()

    if args.regenerate:
        regenerate()
        return 0

    failed = False
    for name in FIXTURE_NAMES:
        matches, diff = diff_against_golden(name)
        if matches:
            print(f"RESULT: PASS -- {name} is byte-identical to its golden RTF.")
            continue
        failed = True
        print(f"RESULT: FAIL -- {name} differs from its golden RTF:")
        for line in diff[:60]:
            print("  " + line.rstrip("\r\n"))
    if failed:
        print("\nIf this change is intentional, regenerate with --regenerate "
              "and READ the golden diff before committing.")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
