"""
Unit tests for create_severity_summary.py

Covers:
  - round_half_up() and calculate_integer_averages() -- conservative
    (round-half-up) rounding behavior for risk severity scores.
  - render_rtf()'s results table -- column order/count and the LBGI/DKAI
    columns, driven by a real build_assessment() over temp summary CSVs.
"""

import pytest
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from create_severity_summary import (
    TABLE_CELL_STOPS,
    calculate_integer_averages,
    render_rtf,
    round_half_up,
)
from severity_model import build_assessment


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
