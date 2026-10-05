"""
Byte-level RTF regression gate (TRSET-42), run in the normal pytest suite.

NOT the same as TestRtfOutputUnchanged (test_create_severity_summary.py), which
compares the renderer to itself and cannot detect a text change. These tests
compare against COMMITTED golden bytes. See rtf_regression_diff.py for the
regeneration command and the list of neutralised non-deterministic fields.
"""

import os
import shutil
import subprocess
import sys

import pytest

import create_severity_summary
import rtf_regression_diff as gate

ALL_FIXTURES = list(gate.FIXTURE_NAMES)


def _first_diff_line(diff):
    return "\n".join(diff)


# IT-1 -- clean render matches the committed golden, byte for byte.
@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_clean_render_matches_golden(name):
    matches, diff = gate.diff_against_golden(name)
    assert matches, _first_diff_line(diff)


# The fixtures must keep exercising what AC 2 requires.
def test_fixtures_cover_populated_table_na_cell_and_absent_stage():
    populated = gate.render_fixture("TLR-999")
    sparse = gate.render_fixture("TLR-998")
    assert "NA\\cell" not in populated
    assert "NA\\cell" in sparse
    # TLR-998 has no post-Loop rows: that stage's TIR/TBR/TAR cells are NA.
    post = sparse.split("Post-mitigation\\cell", 1)[1]
    assert post.count("NA\\cell") >= 3


# IT-2 (AC 5, load-bearing) -- a one-word change to static RTF text fails, and
# the diff names the changed region.
@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_one_word_static_text_mutation_is_detected(name):
    actual = gate.render_fixture(name)
    assert "Table of results" in actual
    mutated = actual.replace("Table of results", "Table of Results", 1)
    assert mutated != actual
    matches, diff = gate.diff_against_golden(name, actual=mutated)
    assert not matches
    joined = _first_diff_line(diff)
    assert "Table of results" in joined and "Table of Results" in joined


def test_one_word_mutation_in_the_real_renderer_is_detected(monkeypatch):
    # Same, but through the renderer's own code path rather than post-hoc string
    # surgery: patch the real static line the way a source edit would.
    real = create_severity_summary.render_rtf
    monkeypatch.setattr(
        gate, "render_rtf",
        lambda a: real(a).replace("Auto-generated output", "Auto-generated outputs", 1),
    )
    matches, diff = gate.diff_against_golden("TLR-999")
    assert not matches
    assert "Auto-generated outputs" in _first_diff_line(diff)


# IT-3 -- a numeric cell value change fails (through the real build/render path).
def test_numeric_cell_mutation_is_detected(tmp_path, monkeypatch):
    run_copy = tmp_path / "run"
    shutil.copytree(gate.FIXTURE_RUN_DIR, run_copy)
    csv_path = next(
        p for p in (run_copy / "TLR-999").iterdir() if "Median" in p.name
    )
    text = csv_path.read_text()
    # Pre-Loop median percent_cgm_gt_180: 79.0 -> 78.0 (feeds the TAR cell).
    assert ",79.0,66.787,20.0" in text
    csv_path.write_text(text.replace(",79.0,66.787,20.0", ",78.0,66.787,20.0"))
    monkeypatch.setattr(gate, "FIXTURE_RUN_DIR", str(run_copy))
    matches, diff = gate.diff_against_golden("TLR-999")
    assert not matches
    assert "79.0" in _first_diff_line(diff) or "78.0" in _first_diff_line(diff)


# IT-4 -- a \cellx tab stop change fails. Structural (parsed-cell) tests would
# not see this.
def test_cellx_tab_stop_mutation_is_detected(monkeypatch):
    stops = create_severity_summary.TABLE_CELL_STOPS
    assert "\\cellx1275" in stops
    monkeypatch.setattr(
        create_severity_summary, "TABLE_CELL_STOPS",
        stops.replace("\\cellx1275", "\\cellx1276", 1),
    )
    matches, diff = gate.diff_against_golden("TLR-999")
    assert not matches
    assert "cellx1276" in _first_diff_line(diff)


def test_line_ending_change_is_detected():
    # \r\n vs \n must not be normalised away by the golden read.
    actual = gate.render_fixture("TLR-999")
    matches, _ = gate.diff_against_golden("TLR-999", actual=actual.replace("\n", "\r\n"))
    assert not matches


# IT-5 -- two consecutive clean renders are identical, and the run timestamp is
# the ONLY run-varying input (the one neutralised field).
@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_consecutive_renders_are_identical(name):
    assert gate.render_fixture(name) == gate.render_fixture(name)


def test_timestamp_is_the_only_run_varying_input():
    a = gate.render_fixture("TLR-999", timestamp="2026-01-01T00:00:00.000000")
    b = gate.render_fixture("TLR-999", timestamp="2031-12-31T23:59:59.000000")
    assert a != b
    diff_lines = [
        (x, y) for x, y in zip(a.splitlines(), b.splitlines()) if x != y
    ]
    assert len(a.splitlines()) == len(b.splitlines())
    assert len(diff_lines) == 1
    assert "Date and time of simulation run" in diff_lines[0][0]


# IT-6 -- the documented regeneration command produces goldens that then pass.
# Run against a temp copy of the goldens so the test never rewrites committed files.
def test_regeneration_command_produces_passing_goldens(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "GOLDEN_DIR", str(tmp_path))
    monkeypatch.setattr(gate, "REPO_ROOT", gate.REPO_ROOT)
    gate.regenerate()
    for name in ALL_FIXTURES:
        assert os.path.exists(gate.golden_path(name))
        matches, diff = gate.diff_against_golden(name)
        assert matches, _first_diff_line(diff)
        # Regenerated output equals the committed golden: today's tree is frozen.
    for name in ALL_FIXTURES:
        committed = os.path.join(
            os.path.dirname(gate.__file__), "..", "tests", "fixtures",
            "rtf_regression", "golden", f"expected_risk_summary_{name}.rtf",
        )
        with open(committed, "rb") as c, open(gate.golden_path(name), "rb") as r:
            assert c.read() == r.read()


def test_script_exit_code_reflects_match():
    script = os.path.join(os.path.dirname(gate.__file__), "rtf_regression_diff.py")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout
    assert "PASS" in proc.stdout
