"""
Unit tests for create_severity_summary.py

Covers:
  - round_half_up() and calculate_integer_averages() -- conservative
    (round-half-up) rounding behavior for risk severity scores.
  - render_rtf()'s results table -- column order/count and the LBGI/DKAI
    columns, driven by a real build_assessment() over temp summary CSVs.
  - process_results_directory()'s failure contract -- what raises, what is
    reported as a skipped directory, and what it returns.
"""

import pytest
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from create_severity_summary import (
    METADATA_FILENAME,
    SeveritySummaryError,
    SummaryResult,
    TABLE_CELL_STOPS,
    calculate_integer_averages,
    process_results_directory,
    render_outlier_results,
    render_profile_count,
    render_rtf,
    round_half_up,
)
from severity_model import build_assessment, OutlierFinding


class TestRoundHalfUp:
    """Tests for the round_half_up() helper function."""

    # Standard rounding (no ambiguity)
    def test_rounds_down_below_half(self):
        assert round_half_up(2.3) == 2

    def test_rounds_up_above_half(self):
        assert round_half_up(2.7) == 3

    def test_rounds_down_well_below_half(self):
        assert round_half_up(3.1) == 3

    def test_rounds_up_well_above_half(self):
        assert round_half_up(3.9) == 4

    # Exact half values — where banker's rounding differs
    def test_half_2_5_rounds_up(self):
        """2.5 should round to 3, not 2 (banker's rounding would give 2)."""
        assert round_half_up(2.5) == 3

    def test_half_3_5_rounds_up(self):
        """3.5 should round to 4 (banker's rounding also gives 4, but for different reasons)."""
        assert round_half_up(3.5) == 4

    def test_half_4_5_rounds_up(self):
        """4.5 should round to 5, not 4 (banker's rounding would give 4)."""
        assert round_half_up(4.5) == 5

    def test_half_1_5_rounds_up(self):
        """1.5 should round to 2."""
        assert round_half_up(1.5) == 2

    # Integer and zero inputs
    def test_zero(self):
        assert round_half_up(0) == 0

    def test_integer_value(self):
        assert round_half_up(3.0) == 3

    def test_large_integer(self):
        assert round_half_up(100.0) == 100

    # Return type
    def test_returns_int(self):
        assert isinstance(round_half_up(2.5), int)

    def test_returns_int_for_whole_number(self):
        assert isinstance(round_half_up(3.0), int)


class TestCalculateIntegerAverages:
    """Tests for calculate_integer_averages() using round_half_up."""

    def test_basic_averaging(self):
        metric_data = {'pre': [1, 2, 3], 'no_loop': [2, 2, 2], 'post': [3, 4, 5]}
        result = calculate_integer_averages(metric_data)
        assert result['pre'] == 2       # avg = 2.0
        assert result['no_loop'] == 2   # avg = 2.0
        assert result['post'] == 4      # avg = 4.0

    def test_half_value_rounds_up(self):
        """Average of [2, 3] = 2.5, should round to 3 (not 2 via banker's rounding)."""
        metric_data = {'pre': [2, 3], 'no_loop': [2, 3], 'post': [2, 3]}
        result = calculate_integer_averages(metric_data)
        assert result['pre'] == 3
        assert result['no_loop'] == 3
        assert result['post'] == 3

    def test_half_value_4_5_rounds_up(self):
        """Average of [4, 5] = 4.5, should round to 5 (not 4 via banker's rounding)."""
        metric_data = {'pre': [4, 5], 'no_loop': [4, 5], 'post': [4, 5]}
        result = calculate_integer_averages(metric_data)
        assert result['pre'] == 5
        assert result['no_loop'] == 5
        assert result['post'] == 5

    def test_empty_stage_returns_zero(self):
        metric_data = {'pre': [], 'no_loop': [2, 3], 'post': [4]}
        result = calculate_integer_averages(metric_data)
        assert result['pre'] == 0

    def test_all_empty_stages_return_zero(self):
        metric_data = {'pre': [], 'no_loop': [], 'post': []}
        result = calculate_integer_averages(metric_data)
        assert result == {'pre': 0, 'no_loop': 0, 'post': 0}

    def test_single_value_per_stage(self):
        metric_data = {'pre': [4], 'no_loop': [2], 'post': [5]}
        result = calculate_integer_averages(metric_data)
        assert result['pre'] == 4
        assert result['no_loop'] == 2
        assert result['post'] == 5

    def test_all_stages_present(self):
        """Verify all three expected keys are returned."""
        metric_data = {'pre': [1], 'no_loop': [2], 'post': [3]}
        result = calculate_integer_averages(metric_data)
        assert set(result.keys()) == {'pre', 'no_loop', 'post'}

    def test_returns_integers(self):
        metric_data = {'pre': [2, 3], 'no_loop': [1], 'post': [4, 5]}
        result = calculate_integer_averages(metric_data)
        for stage in ['pre', 'no_loop', 'post']:
            assert isinstance(result[stage], int), f"{stage} should be int"


# =============================================================================
# render_rtf() results table
# =============================================================================

# Expected column order after adding LBGI/DKAI: LBGI immediately after TBR,
# DKAI immediately before TAR.
EXPECTED_HEADERS = [
    "Evaluation stage", "Harm", "Severity", "TIR % (70 - 180 mg/dL)",
    "TBR % (<54 mg/dL)", "LBGI", "DKAI", "TAR % (>180 mg/dL)",
]

_SUMMARY_COLUMNS = [
    "sim_id", "percent_values_ge_70_le_180", "percent_cgm_lt_54",
    "percent_cgm_gt_180", "lbgi_risk_score", "dka_risk_score", "lbgi", "dka_index",
]
# Raw lbgi/dka_index values chosen so each stage exercises a different
# truncation/formatting branch, and so LBGI vs DKAI cells can't be confused:
#   pre     lbgi (2.0+3.339)/2 = 2.6695 -> '2.66'  ; dka (20+22)/2   = 21    -> '21'
#   no_loop lbgi (4.0+5.0)/2   = 4.5    -> '4.5'   ; dka (30+28)/2   = 29    -> '29'
#   post    lbgi (1.0+1.0)/2   = 1      -> '1'     ; dka (10+12.5)/2 = 11.25 -> '11.25'
_PROFILE_A_ROWS = [
    ("pre-Loop_NoMitigations_t1_median",    78.0, 4.5, 17.5, 3, 1, 2.0, 20.0),
    ("pre-noLoop_t1_median",                69.0, 3.5, 27.5, 3, 2, 4.0, 30.0),
    ("post-Loop_WithMitigations_t1_median", 94.5, 0.0,  5.5, 1, 0, 1.0, 10.0),
]
_PROFILE_B_ROWS = [
    ("pre-Loop_NoMitigations_t1_median",    80.0, 4.0, 16.0, 3, 1, 3.339, 22.0),
    ("pre-noLoop_t1_median",                70.0, 3.0, 26.0, 3, 2, 5.0, 28.0),
    ("post-Loop_WithMitigations_t1_median", 95.0, 0.0,  5.0, 1, 0, 1.0, 12.5),
]


def _write_summary_csv(directory, profile, rows, columns=_SUMMARY_COLUMNS):
    path = os.path.join(
        directory,
        f"summary_results_Simulation-Configuration-TLR-999-test_{profile}_profile.csv",
    )
    with open(path, "w") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join(str(v) for v in row[: len(columns)]) + "\n")
    return path


def _table_rows(rtf):
    """Parse the results table into a list of rows, each a list of cell texts.

    Asserting on parsed cell ORDER (rather than substring presence) is what makes
    a mis-placed column actually fail: '21' appears somewhere in the document
    either way, but only the correct layout puts it in the DKAI position.
    """
    rows = []
    for block in re.findall(r"\\trowd(.*?)\\row", rtf, re.DOTALL):
        cells = re.findall(r"\\pard\\intbl\s*(.*?)\\cell", block, re.DOTALL)
        # Strip the bold wrapper from header cells: '{\b Harm}' -> 'Harm'.
        rows.append([
            re.sub(r"^\{\\b\s*(.*)\}$", r"\1", cell.strip()).strip()
            for cell in cells
        ])
    return rows


@pytest.fixture
def rendered_rtf(tmp_path):
    """A real assessment (temp summary CSVs -> build_assessment) rendered to RTF.

    Deliberately not a hand-built StageResult: this exercises the real
    CSV -> compute -> render path, so a contract change between severity_model
    and the renderer surfaces here.
    """
    tlr = str(tmp_path)
    _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
    _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)
    assessment = build_assessment(tlr, "2026-07-30T00:00:00")
    assert assessment is not None, "fixture failed to build an assessment"
    return render_rtf(assessment)


class TestRenderRtfTableLayout:
    """Column layout of the results table (8 columns incl. LBGI/DKAI)."""

    def test_table_has_header_plus_three_stage_rows(self, rendered_rtf):
        assert len(_table_rows(rendered_rtf)) == 4

    def test_header_column_order(self, rendered_rtf):
        assert _table_rows(rendered_rtf)[0] == EXPECTED_HEADERS

    def test_lbgi_follows_tbr_and_dkai_precedes_tar(self, rendered_rtf):
        headers = _table_rows(rendered_rtf)[0]
        assert headers.index("LBGI") == headers.index("TBR % (<54 mg/dL)") + 1
        assert headers.index("DKAI") == headers.index("TAR % (>180 mg/dL)") - 1

    def test_every_row_has_one_cell_per_stop(self, rendered_rtf):
        expected = TABLE_CELL_STOPS.count(r"\cellx")
        assert expected == len(EXPECTED_HEADERS)
        for index, row in enumerate(_table_rows(rendered_rtf)):
            assert len(row) == expected, f"row {index} has {len(row)} cells"

    def test_stops_are_emitted_once_per_row(self, rendered_rtf):
        assert rendered_rtf.count(TABLE_CELL_STOPS) == 4

    def test_total_table_width_unchanged(self):
        """Eight columns still span the original 10200 twips (page fit)."""
        stops = [int(v) for v in re.findall(r"\\cellx(\d+)", TABLE_CELL_STOPS)]
        assert stops[-1] == 10200
        assert stops == sorted(stops), "stops must be strictly increasing"
        assert len(set(stops)) == len(stops), "stops must not repeat"


class TestRenderRtfTableValues:
    """LBGI/DKAI cells carry the truncated raw values, in the right positions."""

    @pytest.mark.parametrize("stage_label,lbgi,dkai", [
        ("Pre-mitigation", "2.66", "21"),
        ("No Loop", "4.5", "29"),
        ("Post-mitigation", "1", "11.25"),
    ])
    def test_stage_row_lbgi_dkai_cells(self, rendered_rtf, stage_label, lbgi, dkai):
        rows = _table_rows(rendered_rtf)
        row = next(r for r in rows if r[0] == stage_label)
        lbgi_index = rows[0].index("LBGI")
        dkai_index = rows[0].index("DKAI")
        assert row[lbgi_index] == lbgi
        assert row[dkai_index] == dkai

    def test_existing_columns_still_populated(self, rendered_rtf):
        """Pre-existing columns are unchanged by the insertion."""
        rows = _table_rows(rendered_rtf)
        headers, pre = rows[0], next(r for r in rows if r[0] == "Pre-mitigation")
        assert pre[headers.index("Harm")] == "Hypoglycemia"
        assert pre[headers.index("Severity")] == "3"
        assert pre[headers.index("TIR % (70 - 180 mg/dL)")] == "79.0"
        assert pre[headers.index("TBR % (<54 mg/dL)")] == "4.2"
        assert pre[headers.index("TAR % (>180 mg/dL)")] == "16.8"

    def test_na_rendered_when_raw_columns_absent(self, tmp_path):
        """A stage with no lbgi/dka_index data renders 'NA', not a blank cell."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS, columns=_SUMMARY_COLUMNS[:-2])
        assessment = build_assessment(tlr, "2026-07-30T00:00:00")
        rows = _table_rows(render_rtf(assessment))
        lbgi_index, dkai_index = rows[0].index("LBGI"), rows[0].index("DKAI")
        for row in rows[1:]:
            assert row[lbgi_index] == "NA"
            assert row[dkai_index] == "NA"


# =============================================================================
# process_results_directory() failure contract
# =============================================================================

# Every path through this function used to print a diagnostic and return None,
# so a caller could not distinguish "wrote three summaries" from "wrote nothing".
# These tests pin the replacement: raise when the directory is unusable, report
# a per-directory skip, and return what was actually done.

import json  # noqa: E402

from create_severity_summary import main  # noqa: E402

_RUN_TIMESTAMP = "2026-08-06T09:15:00.123456"


def _write_metadata(run_dir, payload={"timestamp": _RUN_TIMESTAMP}):
    path = os.path.join(run_dir, METADATA_FILENAME)
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return path


def _usable_tlr_dir(run_dir, name="TLR-999-test"):
    """A TLR directory with the real summary CSVs build_assessment needs."""
    tlr_dir = os.path.join(run_dir, name)
    os.makedirs(tlr_dir)
    _write_summary_csv(tlr_dir, "median", _PROFILE_A_ROWS)
    _write_summary_csv(tlr_dir, "adolescent", _PROFILE_B_ROWS)
    return tlr_dir


class TestProcessResultsDirectoryRaises:
    """A results directory that cannot be summarized at all is an error."""

    def test_missing_metadata_raises(self, tmp_path):
        _usable_tlr_dir(str(tmp_path))

        with pytest.raises(SeveritySummaryError, match=METADATA_FILENAME):
            process_results_directory(str(tmp_path))

    def test_unparseable_metadata_raises_rather_than_leaking_a_json_error(self, tmp_path):
        with open(os.path.join(str(tmp_path), METADATA_FILENAME), "w") as fh:
            fh.write("{not json")
        _usable_tlr_dir(str(tmp_path))

        with pytest.raises(SeveritySummaryError):
            process_results_directory(str(tmp_path))

    @pytest.mark.parametrize("payload", [{}, {"other_key": "x"}, {"timestamp": ""}])
    def test_metadata_without_a_usable_timestamp_raises(self, tmp_path, payload):
        """The old code fell back to the literal 'Unknown', which then rendered
        into the summary's 'Date and time of simulation run' field."""
        _write_metadata(str(tmp_path), payload)
        _usable_tlr_dir(str(tmp_path))

        with pytest.raises(SeveritySummaryError, match="cannot be dated"):
            process_results_directory(str(tmp_path))

    def test_run_timestamp_key_is_still_accepted(self, tmp_path):
        """The alternate key the original code read is still honored."""
        _write_metadata(str(tmp_path), {"run_timestamp": _RUN_TIMESTAMP})
        _usable_tlr_dir(str(tmp_path))

        result = process_results_directory(str(tmp_path))

        assert len(result.written) == 1

    def test_no_tlr_directories_raises(self, tmp_path):
        _write_metadata(str(tmp_path))

        with pytest.raises(SeveritySummaryError, match="TLR-"):
            process_results_directory(str(tmp_path))

    def test_a_file_named_like_a_tlr_dir_is_not_a_run_directory(self, tmp_path):
        """glob matched files too, so a stray TLR-*.txt was handed to
        build_assessment as though it were a run directory."""
        _write_metadata(str(tmp_path))
        with open(os.path.join(str(tmp_path), "TLR-notes.txt"), "w") as fh:
            fh.write("scratch notes\n")

        with pytest.raises(SeveritySummaryError, match="TLR-"):
            process_results_directory(str(tmp_path))


class TestProcessResultsDirectoryResult:
    """What it returns, and what it reports rather than swallowing."""

    def test_writes_one_rtf_per_usable_directory_and_returns_their_paths(self, tmp_path):
        _write_metadata(str(tmp_path))
        tlr_dir = _usable_tlr_dir(str(tmp_path))

        result = process_results_directory(str(tmp_path))

        assert isinstance(result, SummaryResult)
        assert result.skipped == []
        assert len(result.written) == 1
        written = result.written[0]
        assert os.path.dirname(written) == tlr_dir
        assert os.path.basename(written).startswith("risk_summary_")
        assert os.path.isfile(written), "returned a path it did not write"

    def test_a_directory_with_no_usable_data_is_reported_not_swallowed(self, tmp_path):
        """This is a legitimate partial outcome -- a real run can contain one --
        so it is reported in `skipped`, not raised."""
        _write_metadata(str(tmp_path))
        empty_dir = os.path.join(str(tmp_path), "TLR-000-empty")
        os.makedirs(empty_dir)

        result = process_results_directory(str(tmp_path))

        assert result.written == []
        assert len(result.skipped) == 1
        skipped_dir, reason = result.skipped[0]
        assert skipped_dir == empty_dir
        assert reason

    def test_a_mixed_run_reports_both_halves(self, tmp_path):
        _write_metadata(str(tmp_path))
        usable = _usable_tlr_dir(str(tmp_path))
        empty = os.path.join(str(tmp_path), "TLR-000-empty")
        os.makedirs(empty)

        result = process_results_directory(str(tmp_path))

        assert [os.path.dirname(p) for p in result.written] == [usable]
        assert [d for d, _ in result.skipped] == [empty]

    def test_directories_are_processed_in_sorted_order(self, tmp_path):
        """Order came from glob (arbitrary), so the same run reported differently
        on different machines."""
        _write_metadata(str(tmp_path))
        for name in ("TLR-300-test", "TLR-100-test", "TLR-200-test"):
            _usable_tlr_dir(str(tmp_path), name)

        result = process_results_directory(str(tmp_path))

        assert [os.path.basename(os.path.dirname(p)) for p in result.written] == [
            "TLR-100-test", "TLR-200-test", "TLR-300-test",
        ]


class TestCliExitCodes:
    """The CLI printed 'Done!' and exited 0 no matter what happened."""

    def test_success_exits_zero(self, tmp_path, monkeypatch):
        _write_metadata(str(tmp_path))
        _usable_tlr_dir(str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["create_severity_summary.py", str(tmp_path)])

        assert main() == 0

    def test_unusable_directory_exits_nonzero(self, tmp_path, monkeypatch):
        _usable_tlr_dir(str(tmp_path))  # no metadata.json
        monkeypatch.setattr(sys, "argv", ["create_severity_summary.py", str(tmp_path)])

        assert main() == 1

    def test_writing_nothing_exits_nonzero_even_without_an_error(self, tmp_path, monkeypatch):
        """Every directory skipped is not a successful run, whatever the reasons."""
        _write_metadata(str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "TLR-000-empty"))
        monkeypatch.setattr(sys, "argv", ["create_severity_summary.py", str(tmp_path)])

        assert main() == 1

    def test_missing_directory_still_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["create_severity_summary.py", "/does/not/exist/at/all"]
        )

        assert main() == 1


class TestRtfOutputUnchanged:
    """The whole point of the ticket's 'no output change' constraint."""

    def test_written_rtf_is_byte_identical_to_the_renderer_output(self, tmp_path):
        _write_metadata(str(tmp_path))
        tlr_dir = _usable_tlr_dir(str(tmp_path))

        result = process_results_directory(str(tmp_path))

        assessment = build_assessment(tlr_dir, _RUN_TIMESTAMP)
        with open(result.written[0]) as fh:
            assert fh.read() == render_rtf(assessment)


# =============================================================================
# TRSET-28 -- the two rendered strings that change, and only on degraded paths
# =============================================================================

# The verdict-input columns. Dropping them makes a file unusable; the fixture
# below is how a "present but unreadable" profile is simulated.
_MISSING_REQUIRED_COLUMNS = [
    "sim_id", "percent_values_ge_70_le_180", "percent_cgm_lt_54",
]

# The clean-path strings, pinned VERBATIM. TestRtfOutputUnchanged compares the
# written file against the renderer's own output, so it is a self-consistency
# check that would pass even if every string here changed. These are the actual
# byte-identity guard for the two lines TRSET-28 touches.
_CLEAN_PROFILE_COUNT_LINE = "2 virtual patient profiles aggregated for this summary."
_CLEAN_OUTLIER_LINE = (
    "No outlier profiles exist. All results are within 1 severity level of one another."
)
_MALFORMED_OUTLIER_LINE = (
    "Outlier analysis not performed: profile data is present but could not be read. "
    "Check data configuration."
)


def _malformed_tlr_dir(run_dir, name="TLR-500-malformed"):
    """A TLR directory whose only summary CSV cannot contribute a value."""
    tlr_dir = os.path.join(run_dir, name)
    os.makedirs(tlr_dir)
    _write_summary_csv(tlr_dir, "median", _PROFILE_A_ROWS,
                       columns=_MISSING_REQUIRED_COLUMNS)
    return tlr_dir


def _partial_tlr_dir(run_dir, name="TLR-400-partial"):
    """Two usable profiles plus one unusable file: M == 2, N == 3."""
    tlr_dir = _usable_tlr_dir(run_dir, name)
    _write_summary_csv(tlr_dir, "broken", _PROFILE_A_ROWS,
                       columns=_MISSING_REQUIRED_COLUMNS)
    return tlr_dir


class TestRenderOutlierResults:
    """Finding 2: malformed input must not render as absent input."""

    def test_malformed_data_gets_its_own_text(self):
        assert render_outlier_results([], "malformed_data") == _MALFORMED_OUTLIER_LINE

    def test_malformed_text_does_not_claim_the_data_was_unavailable(self):
        """The defect: a corrupt CSV produced a document asserting absence."""
        assert "not available" not in render_outlier_results([], "malformed_data")

    def test_no_data_text_is_unchanged(self):
        assert render_outlier_results([], "no_data") == "Data not available for outlier analysis."

    def test_single_profile_text_is_unchanged(self):
        assert render_outlier_results([], "single_profile") == (
            "Only one profile present, so outliers are not relevant."
        )

    def test_ok_with_no_findings_text_is_unchanged(self):
        assert render_outlier_results([], "ok") == _CLEAN_OUTLIER_LINE


class TestRenderProfileCount:
    """Finding 3: M of N, and byte-identical when M == N."""

    def test_m_equals_n_is_the_original_sentence(self):
        assert render_profile_count(2, 2) == _CLEAN_PROFILE_COUNT_LINE

    def test_unmeasured_count_is_the_original_sentence(self):
        """usable=None means 'not measured' -- an older assessment renders as before."""
        assert render_profile_count(2, None) == _CLEAN_PROFILE_COUNT_LINE
        assert render_profile_count(2) == _CLEAN_PROFILE_COUNT_LINE

    def test_one_dropped_file_is_singular(self):
        assert render_profile_count(3, 2) == (
            "2 of 3 virtual patient profiles aggregated for this summary. "
            "1 summary results file could not be read."
        )

    def test_two_dropped_files_are_plural(self):
        assert render_profile_count(4, 2) == (
            "2 of 4 virtual patient profiles aggregated for this summary. "
            "2 summary results files could not be read."
        )

    def test_a_count_above_n_degrades_to_the_original_sentence(self):
        """Defensive: never render 'more usable than present'."""
        assert render_profile_count(2, 3) == _CLEAN_PROFILE_COUNT_LINE


class TestRenderedCleanPathIsPinned:
    """The clean path (M == N, well-formed data) renders today's exact strings."""

    def test_profile_count_line_is_verbatim(self, rendered_rtf):
        assert _CLEAN_PROFILE_COUNT_LINE in rendered_rtf

    def test_outlier_line_is_verbatim(self, rendered_rtf):
        assert _CLEAN_OUTLIER_LINE in rendered_rtf

    def test_no_degraded_wording_leaks_onto_the_clean_path(self, rendered_rtf):
        assert "of 2 virtual patient profiles" not in rendered_rtf
        assert "could not be read" not in rendered_rtf
        assert "Outlier analysis not performed" not in rendered_rtf


class TestRenderedDegradedPath:
    """A document with a dropped file carries both new strings."""

    @pytest.fixture
    def partial_rtf(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)
        assessment = build_assessment(tlr, "2026-08-06T00:00:00")
        assert assessment is not None, "0 < M < N must still produce a document"
        return render_rtf(assessment)

    def test_the_count_line_names_both_numbers(self, partial_rtf):
        assert (
            "2 of 3 virtual patient profiles aggregated for this summary. "
            "1 summary results file could not be read."
        ) in partial_rtf

    def test_the_outlier_analysis_runs_over_the_readable_profiles(self, partial_rtf):
        """Decision C: the excluded file used to abandon outlier analysis for the
        whole directory, rendering the malformed sentence. The two readable profiles
        are now compared, and the section scopes its claim instead."""
        assert _MALFORMED_OUTLIER_LINE not in partial_rtf
        assert "Data not available for outlier analysis." not in partial_rtf
        assert "No outlier profiles exist." in partial_rtf

    def test_the_outlier_section_scopes_its_claim_to_what_was_analyzed(self, partial_rtf):
        assert "This analysis covered 2 of 3 profiles; 1 could not be read." in partial_rtf

    def test_the_table_still_renders(self, partial_rtf):
        """A partial document is still a complete document."""
        assert len(_table_rows(partial_rtf)) == 4


class TestSkipReasonsDistinguishEmptyFromMalformed:
    """Finding 1 at the reporting boundary: both used to be one shared string."""

    def test_an_empty_directory_reports_empty(self, tmp_path):
        _write_metadata(str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "TLR-000-empty"))

        result = process_results_directory(str(tmp_path))

        assert "empty" in result.skipped[0][1]

    def test_a_malformed_directory_reports_malformed(self, tmp_path):
        _write_metadata(str(tmp_path))
        _malformed_tlr_dir(str(tmp_path))

        result = process_results_directory(str(tmp_path))

        assert result.written == []
        assert "malformed" in result.skipped[0][1]

    def test_the_two_reasons_are_not_the_same_string(self, tmp_path):
        _write_metadata(str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "TLR-000-empty"))
        _malformed_tlr_dir(str(tmp_path))

        reasons = {reason for _, reason in process_results_directory(str(tmp_path)).skipped}

        assert len(reasons) == 2

    def test_a_malformed_directory_writes_no_document(self, tmp_path):
        """It used to write a complete document with every metric NA/0."""
        _write_metadata(str(tmp_path))
        tlr_dir = _malformed_tlr_dir(str(tmp_path))

        process_results_directory(str(tmp_path))

        assert [f for f in os.listdir(tlr_dir) if f.endswith(".rtf")] == []

    def test_one_malformed_directory_does_not_cost_the_others(self, tmp_path):
        """TRSET-27's fatal-vs-partial split: malformed is still per-directory."""
        _write_metadata(str(tmp_path))
        usable = _usable_tlr_dir(str(tmp_path))
        malformed = _malformed_tlr_dir(str(tmp_path))

        result = process_results_directory(str(tmp_path))

        assert [os.path.dirname(p) for p in result.written] == [usable]
        assert [d for d, _ in result.skipped] == [malformed]

    def test_a_partially_usable_directory_is_written_not_skipped(self, tmp_path):
        _write_metadata(str(tmp_path))
        partial = _partial_tlr_dir(str(tmp_path))

        result = process_results_directory(str(tmp_path))

        assert [os.path.dirname(p) for p in result.written] == [partial]
        assert result.skipped == []

    def test_a_malformed_only_run_exits_nonzero(self, tmp_path, monkeypatch):
        _write_metadata(str(tmp_path))
        _malformed_tlr_dir(str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["create_severity_summary.py", str(tmp_path)])

        assert main() == 1


_INCOMPLETE_STAGES_LINE = (
    "Outlier analysis not performed: no profile has results for all "
    "three evaluation stages."
)


class TestRenderOutlierResultsIncompleteStages:
    """TRSET-28 Decision B: the third condition that hid inside 'no_data'."""

    def test_incomplete_stages_gets_its_own_text(self):
        assert render_outlier_results([], "incomplete_stages") == _INCOMPLETE_STAGES_LINE

    def test_it_does_not_claim_the_data_was_unavailable(self):
        assert "not available" not in render_outlier_results([], "incomplete_stages")

    def test_it_is_distinct_from_the_malformed_text(self):
        """Both are "present but unusable for this analysis" -- for different, and
        differently actionable, reasons."""
        assert render_outlier_results([], "incomplete_stages") != render_outlier_results(
            [], "malformed_data"
        )

    def test_the_three_degraded_statuses_render_three_different_sentences(self):
        rendered = {
            render_outlier_results([], status)
            for status in ("no_data", "malformed_data", "incomplete_stages")
        }
        assert len(rendered) == 3

    def test_an_unknown_status_still_falls_through_to_the_findings_text(self):
        """Defensive: a status this renderer does not know must not silently render
        a degraded sentence. With no findings it reports none found."""
        assert render_outlier_results([], "some_future_status") == _CLEAN_OUTLIER_LINE


class TestRenderedIncompleteStagesDocument:
    """The whole path: partial-stage CSVs -> assessment -> rendered document."""

    _OUTLIER_COLUMNS = [
        "sim_id", "percent_values_ge_70_le_180", "percent_cgm_lt_54",
        "percent_cgm_gt_180", "lbgi_risk_score", "dka_risk_score",
    ]

    @pytest.fixture
    def incomplete_rtf(self, tmp_path):
        tlr = str(tmp_path)
        for profile in ("median", "adolescent"):
            rows = [
                (f"{stem}_{profile}", 80.0, 1.0, 20.0, 1, 0)
                for stem in ("pre-Loop_NoMitigations_t1", "pre-noLoop_t1")
            ]  # no post- row -> no profile is stage-complete
            _write_summary_csv(tlr, profile, rows, columns=self._OUTLIER_COLUMNS)
        assessment = build_assessment(tlr, "2026-08-06T00:00:00")
        assert assessment is not None, "readable data must still produce a document"
        return render_rtf(assessment)

    def test_the_outlier_section_names_the_missing_stages(self, incomplete_rtf):
        assert _INCOMPLETE_STAGES_LINE in incomplete_rtf

    def test_it_does_not_say_the_data_was_unavailable(self, incomplete_rtf):
        assert "Data not available for outlier analysis." not in incomplete_rtf

    def test_the_count_line_is_clean_because_both_files_are_usable(self, incomplete_rtf):
        """Thin data is not malformed data: M == N, so finding 3's line is unchanged."""
        assert "2 virtual patient profiles aggregated for this summary." in incomplete_rtf

    def test_the_missing_stage_renders_na_not_a_blank_cell(self, incomplete_rtf):
        rows = _table_rows(incomplete_rtf)
        post_row = next(row for row in rows[1:] if row[0] == "Post-mitigation")
        tir_index = rows[0].index("TIR % (70 - 180 mg/dL)")
        assert post_row[tir_index] == "NA"


class TestOutlierScopeSentence:
    """Decision C: an outlier claim over a subset must say so.

    The count line above the section already reports the drop, but the Outlier
    Results sentences are absolute claims read (and quoted) on their own.
    """

    _CLEAN_FINDING = OutlierFinding("pre", "sensitive", "Hypoglycemia", 4.0, 2.0)

    def test_no_scope_sentence_when_nothing_was_excluded(self):
        assert render_outlier_results([], "ok", 3, 3) == _CLEAN_OUTLIER_LINE

    def test_no_scope_sentence_when_counts_are_unknown(self):
        """Both default to None, so an older caller renders exactly as before."""
        assert render_outlier_results([], "ok") == _CLEAN_OUTLIER_LINE
        assert render_outlier_results([], "ok", None, None) == _CLEAN_OUTLIER_LINE

    def test_the_no_outliers_claim_is_scoped(self):
        assert render_outlier_results([], "ok", 3, 2) == (
            _CLEAN_OUTLIER_LINE
            + " This analysis covered 2 of 3 profiles; 1 could not be read."
        )

    def test_a_findings_claim_is_scoped_too(self):
        rendered = render_outlier_results([self._CLEAN_FINDING], "ok", 4, 2)

        assert rendered.startswith("Outlier profile exists.")
        assert rendered.endswith(
            " This analysis covered 2 of 4 profiles; 2 could not be read."
        )

    def test_single_profile_is_scoped_because_it_asserts_a_count(self):
        """'Only one profile present' would be false when three were present and two
        were excluded."""
        rendered = render_outlier_results([], "single_profile", 3, 1)

        assert rendered == (
            "Only one profile present, so outliers are not relevant."
            " This analysis covered 1 of 3 profiles; 2 could not be read."
        )

    def test_statuses_that_assert_nothing_are_not_scoped(self):
        """These already say the analysis did not happen, and why; appending a scope
        sentence would only repeat the count line."""
        for status in ("no_data", "malformed_data", "incomplete_stages"):
            assert "This analysis covered" not in render_outlier_results([], status, 3, 1)

    def test_a_count_above_the_total_is_not_scoped(self):
        """Defensive: never render 'more analyzed than present'."""
        assert render_outlier_results([], "ok", 2, 3) == _CLEAN_OUTLIER_LINE
