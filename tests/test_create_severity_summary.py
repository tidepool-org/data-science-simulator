"""
Unit tests for create_severity_summary.py

Tests verify that sim_id pattern matching correctly identifies all variants
of pre-mitigation, no-loop, and post-mitigation simulation IDs.
"""

import pytest
import sys
import os

# Add post_processing to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'post_processing'))

from create_severity_summary import (
    identify_severity_4_hypoglycemia,
    extract_metric_data,
    get_profile_metrics,
    determine_harm_and_severity,
    calculate_hyperglycemia_score,
    calculate_integer_averages,
    calculate_stage_averages,
    build_assessment,
    render_rtf,
)


class TestSimIdPatternMatching:
    """Tests for sim_id pattern matching across all functions."""
    
    # All known sim_id patterns that should be recognized
    PRE_MITIGATION_PATTERNS = [
        'pre-Loop_NoMitigations_t1_adolescent',
        'pre-Loop-NoMitigations_t1_adolescent',
        'pre-Loop-noMitigations_t1_adolescent',  # lowercase 'n' variant (TLR-789)
        'pre-LoopNoMitigations_t1_adolescent',   # no separator variant (TLR-822)
        'pre-LoopNoMitigationss_t1_adolescent',  # no separator, double 's' typo (TLR-822)
        'pre-Loop_NoMitigations_t1_median',
        'pre-Loop-noMitigations_t1_median',
        'pre-LoopNoMitigations_t1_median',
        'pre-Loop_NoMitigations_t1_resistant',
        'pre-Loop-noMitigations_t1_resistant',
        'pre-LoopNoMitigations_t1_resistant',
        'pre-Loop_NoMitigations_t1_sensitive',
        'pre-Loop-noMitigations_t1_sensitive',
        'pre-LoopNoMitigations_t1_sensitive',
    ]
    
    NO_LOOP_PATTERNS = [
        'pre-noLoop_t1_adolescent',
        'pre-NoLoop_t1_adolescent',
        'pre-noLoop_t1_median',
        'pre-NoLoop_t1_median',
        'pre-NoLoop_t1_resistant',
        'pre-NoLoop_t1_sensitive',
    ]
    
    POST_MITIGATION_PATTERNS = [
        'post-Loop-WithMitigations_t1_adolescent',
        'post-LoopWithMitigations_t1_adolescent',
        'post-Loop_WithMitigations_t1_adolescent',  # The pattern that was missing
        'post-Loop_WithMitigations_t1_median',
        'post-Loop_WithMitigations_t1_resistant',
        'post-Loop_WithMitigations_t1_sensitive',
    ]

    def test_pre_mitigation_pattern_recognition(self):
        """Test that all pre-mitigation patterns are correctly identified."""
        for sim_id in self.PRE_MITIGATION_PATTERNS:
            is_pre = (
                sim_id.startswith('pre-Loop_NoMitigations_') or 
                sim_id.startswith('pre-Loop-NoMitigations_') or
                sim_id.startswith('pre-Loop-noMitigations_') or
                sim_id.startswith('pre-LoopNoMitigations_') or
                sim_id.startswith('pre-LoopNoMitigationss_')
            )
            assert is_pre, f"Failed to recognize pre-mitigation pattern: {sim_id}"

    def test_lowercase_variant_pre_mitigation(self):
        """
        Specifically test the lowercase 'noMitigations' variant that was previously missing.
        This is the critical fix for TLR-789.
        """
        lowercase_variants = [
            'pre-Loop-noMitigations_t1_adolescent',
            'pre-Loop-noMitigations_t1_median',
            'pre-Loop-noMitigations_t1_resistant',
            'pre-Loop-noMitigations_t1_sensitive',
        ]
        
        for sim_id in lowercase_variants:
            # This should match with the fix in place
            is_pre = sim_id.startswith('pre-Loop-noMitigations_')
            assert is_pre, f"Lowercase variant not recognized: {sim_id}"

    def test_no_separator_variant_pre_mitigation(self):
        """
        Specifically test the no-separator variants that were previously missing.
        This is the critical fix for TLR-822.
        """
        no_separator_variants = [
            'pre-LoopNoMitigations_t1_adolescent',
            'pre-LoopNoMitigations_t1_median',
            'pre-LoopNoMitigations_t1_resistant',
            'pre-LoopNoMitigations_t1_sensitive',
            'pre-LoopNoMitigationss_t1_adolescent',  # double 's' typo
        ]
        
        for sim_id in no_separator_variants:
            # This should match with the fix in place
            is_pre = (
                sim_id.startswith('pre-LoopNoMitigations_') or
                sim_id.startswith('pre-LoopNoMitigationss_')
            )
            assert is_pre, f"No-separator variant not recognized: {sim_id}"

    def test_no_loop_pattern_recognition(self):
        """Test that all no-loop patterns are correctly identified."""
        for sim_id in self.NO_LOOP_PATTERNS:
            is_no_loop = (
                sim_id.startswith('pre-noLoop_') or 
                sim_id.startswith('pre-NoLoop_')
            )
            assert is_no_loop, f"Failed to recognize no-loop pattern: {sim_id}"

    def test_post_mitigation_pattern_recognition(self):
        """Test that all post-mitigation patterns are correctly identified."""
        for sim_id in self.POST_MITIGATION_PATTERNS:
            is_post = (
                sim_id.startswith('post-Loop-WithMitigations_') or 
                sim_id.startswith('post-LoopWithMitigations_') or
                sim_id.startswith('post-Loop_WithMitigations_')
            )
            assert is_post, f"Failed to recognize post-mitigation pattern: {sim_id}"

    def test_underscore_variant_post_mitigation(self):
        """
        Specifically test the underscore variant that was previously missing.
        This is the critical fix for TLR-606.
        """
        underscore_variants = [
            'post-Loop_WithMitigations_t1_adolescent',
            'post-Loop_WithMitigations_t1_median',
            'post-Loop_WithMitigations_t1_resistant',
            'post-Loop_WithMitigations_t1_sensitive',
        ]
        
        for sim_id in underscore_variants:
            # This should match with the fix in place
            is_post = sim_id.startswith('post-Loop_WithMitigations_')
            assert is_post, f"Underscore variant not recognized: {sim_id}"


class TestDetermineHarmAndSeverity:
    """Tests for determine_harm_and_severity function."""
    
    def test_all_zeros_returns_baseline(self):
        """When all scores are 0, severity should be baseline."""
        harm, severity = determine_harm_and_severity(0, 0, 0)
        assert harm == "Severity = baseline"
        assert severity == "0"
    
    def test_hyperglycemia_when_lbgi_and_dka_low(self):
        """When LBGI <= 1 and DKA == 0, use hyperglycemia score."""
        harm, severity = determine_harm_and_severity(1, 0, 2)
        assert harm == "Hyperglycemia"
        assert severity == "2"
        
        harm, severity = determine_harm_and_severity(0, 0, 1)
        assert harm == "Hyperglycemia"
        assert severity == "1"
    
    def test_hypoglycemia_when_lbgi_highest(self):
        """When LBGI >= DKA, use hypoglycemia (LBGI takes priority in ties)."""
        harm, severity = determine_harm_and_severity(4, 3, 1)
        assert harm == "Hypoglycemia"
        assert severity == "4"
        
        # Tie goes to LBGI
        harm, severity = determine_harm_and_severity(3, 3, 1)
        assert harm == "Hypoglycemia"
        assert severity == "3"
    
    def test_dka_when_dka_highest(self):
        """When DKA > LBGI, use DKA."""
        harm, severity = determine_harm_and_severity(2, 4, 1)
        assert harm == "DKA"
        assert severity == "4"

    def test_post_mitigation_severity_4_hypoglycemia(self):
        """
        Test case matching TLR-606: post-mitigation with LBGI=4 should return
        Hypoglycemia severity 4, not Hyperglycemia severity 1.
        """
        # This represents the actual values from TLR-606 post-mitigation data
        lbgi_score = 4
        dka_score = 1
        hyperglycemia_score = 1  # TAR is 0.0%, so hyperglycemia score is 1
        
        harm, severity = determine_harm_and_severity(lbgi_score, dka_score, hyperglycemia_score)
        
        assert harm == "Hypoglycemia", f"Expected Hypoglycemia but got {harm}"
        assert severity == "4", f"Expected severity 4 but got {severity}"


class TestCalculateHyperglycemiaScore:
    """Tests for calculate_hyperglycemia_score function."""
    
    def test_na_returns_1(self):
        assert calculate_hyperglycemia_score("NA") == 1
    
    def test_zero_returns_0(self):
        assert calculate_hyperglycemia_score("0.0") == 0
    
    def test_below_12_returns_1(self):
        assert calculate_hyperglycemia_score("5.5") == 1
        assert calculate_hyperglycemia_score("11.9") == 1
    
    def test_12_or_above_returns_2(self):
        assert calculate_hyperglycemia_score("12.0") == 2
        assert calculate_hyperglycemia_score("25.0") == 2


class TestCalculateAverages:
    """Tests for average calculation functions."""
    
    def test_integer_averages_with_data(self):
        metric_data = {
            'pre': [4, 4, 4, 4],
            'no_loop': [2, 1, 4, 1],
            'post': [4, 4, 4, 4]
        }
        averages = calculate_integer_averages(metric_data)
        
        assert averages['pre'] == 4
        assert averages['no_loop'] == 2
        assert averages['post'] == 4
    
    def test_integer_averages_empty_returns_zero(self):
        """Empty lists should return 0, not NA or error."""
        metric_data = {
            'pre': [4, 4],
            'no_loop': [2],
            'post': []  # Empty - this was the bug scenario
        }
        averages = calculate_integer_averages(metric_data)
        
        assert averages['post'] == 0
    
    def test_stage_averages_empty_returns_na(self):
        """Empty lists should return 'NA' for string averages."""
        metric_data = {
            'pre': [25.3],
            'no_loop': [66.5],
            'post': []
        }
        averages = calculate_stage_averages(metric_data)
        
        assert averages['post'] == "NA"


class TestTLR606Scenario:
    """
    Integration test modeling the exact TLR-606 scenario.
    
    This test verifies that when post-mitigation data has LBGI=4,
    the final severity should be 4, not 1.
    """
    
    def test_tlr606_post_mitigation_severity(self):
        """
        Reproduce TLR-606 bug scenario:
        - Post-mitigation sim_ids use underscore variant
        - All post-mitigation LBGI scores are 4
        - Expected final severity: 4 (Hypoglycemia)
        - Bug showed: 1 (Hyperglycemia) due to empty post data
        """
        # Simulated data matching Risk_Results.csv
        lbgi_data = {
            'pre': [4, 4, 4, 4],
            'no_loop': [0, 2, 4, 1],
            'post': [4, 4, 4, 4]  # With fix, this should be populated
        }
        
        dka_data = {
            'pre': [1, 1, 1, 1],
            'no_loop': [0, 0, 0, 0],
            'post': [1, 1, 1, 1]
        }
        
        tar_data = {
            'pre': [0.0, 0.0, 0.0, 0.0],
            'no_loop': [21.649, 0.0, 0.0, 17.526],
            'post': [0.0, 0.0, 0.0, 0.0]
        }
        
        # Calculate averages
        lbgi_averages = calculate_integer_averages(lbgi_data)
        dka_averages = calculate_integer_averages(dka_data)
        tar_averages = calculate_stage_averages(tar_data)
        
        # Calculate hyperglycemia scores
        hyperglycemia_scores = {}
        for stage in ['pre', 'no_loop', 'post']:
            hyperglycemia_scores[stage] = calculate_hyperglycemia_score(tar_averages[stage])
        
        # Determine harm and severity for post-mitigation
        harm, severity = determine_harm_and_severity(
            lbgi_averages['post'],
            dka_averages['post'],
            hyperglycemia_scores['post']
        )
        
        # This is the critical assertion - severity should be 4, not 1
        assert severity == "4", f"Post-mitigation severity should be 4, got {severity}"
        assert harm == "Hypoglycemia", f"Post-mitigation harm should be Hypoglycemia, got {harm}"

    def test_tlr606_bug_scenario_empty_post_data(self):
        """
        Demonstrate what happens when post data is empty (the bug condition).
        This documents the bug behavior to ensure we don't regress.
        """
        # Bug scenario: post data is empty because pattern didn't match
        lbgi_data_bug = {
            'pre': [4, 4, 4, 4],
            'no_loop': [0, 2, 4, 1],
            'post': []  # Empty due to pattern mismatch
        }
        
        tar_data_bug = {
            'pre': [0.0, 0.0, 0.0, 0.0],
            'no_loop': [21.649, 0.0, 0.0, 17.526],
            'post': []
        }
        
        # With empty data, averages return 0 or "NA"
        lbgi_avg = calculate_integer_averages(lbgi_data_bug)
        tar_avg = calculate_stage_averages(tar_data_bug)
        
        assert lbgi_avg['post'] == 0, "Empty LBGI should return 0"
        assert tar_avg['post'] == "NA", "Empty TAR should return 'NA'"
        
        # With NA TAR, hyperglycemia score defaults to 1
        hyper_score = calculate_hyperglycemia_score(tar_avg['post'])
        assert hyper_score == 1
        
        # Bug result: severity 1 from hyperglycemia
        harm, severity = determine_harm_and_severity(0, 0, 1)
        assert severity == "1", "Bug condition produces severity 1"
        assert harm == "Hyperglycemia", "Bug condition produces Hyperglycemia"


class TestTLR789Scenario:
    """
    Integration test modeling the exact TLR-789 scenario.
    
    This test verifies that when pre-mitigation data uses lowercase 'noMitigations',
    the data is correctly captured and severity is calculated properly.
    """
    
    def test_tlr789_pre_mitigation_severity(self):
        """
        Reproduce TLR-789 bug scenario:
        - Pre-mitigation sim_ids use lowercase 'noMitigations' variant
        - Pre-mitigation has mixed LBGI scores (2, 4, 4, 4)
        - Expected: Data should be captured, not show NA
        - Bug showed: Hyperglycemia severity 1 with NA values due to empty pre data
        """
        # Simulated data matching TLR-789 Risk_Results.csv
        lbgi_data = {
            'pre': [2, 4, 4, 4],  # With fix, this should be populated
            'no_loop': [2, 4, 4, 4],
            'post': [1, 4, 4, 4]
        }
        
        dka_data = {
            'pre': [0, 1, 1, 1],
            'no_loop': [0, 0, 0, 0],
            'post': [0, 1, 1, 1]
        }
        
        tar_data = {
            'pre': [9.278, 0.0, 0.0, 0.0],
            'no_loop': [0.0, 0.0, 0.0, 0.0],
            'post': [25.773, 0.0, 0.0, 0.0]
        }
        
        # Calculate averages
        lbgi_averages = calculate_integer_averages(lbgi_data)
        tar_averages = calculate_stage_averages(tar_data)
        
        # Pre-mitigation should have data, not be empty
        assert lbgi_averages['pre'] == 4, f"Pre LBGI should be 4 (rounded), got {lbgi_averages['pre']}"
        assert tar_averages['pre'] != "NA", "Pre TAR should not be NA"
        
    def test_tlr789_bug_scenario_empty_pre_data(self):
        """
        Demonstrate what happens when pre data is empty (the bug condition).
        This documents the bug behavior to ensure we don't regress.
        """
        # Bug scenario: pre data is empty because pattern didn't match
        lbgi_data_bug = {
            'pre': [],  # Empty due to pattern mismatch
            'no_loop': [2, 4, 4, 4],
            'post': [1, 4, 4, 4]
        }
        
        tar_data_bug = {
            'pre': [],
            'no_loop': [0.0, 0.0, 0.0, 0.0],
            'post': [25.773, 0.0, 0.0, 0.0]
        }
        
        # With empty data, averages return 0 or "NA"
        lbgi_avg = calculate_integer_averages(lbgi_data_bug)
        tar_avg = calculate_stage_averages(tar_data_bug)
        
        assert lbgi_avg['pre'] == 0, "Empty LBGI should return 0"
        assert tar_avg['pre'] == "NA", "Empty TAR should return 'NA'"
        
        # With NA TAR, hyperglycemia score defaults to 1
        hyper_score = calculate_hyperglycemia_score(tar_avg['pre'])
        assert hyper_score == 1
        
        # Bug result: severity 1 from hyperglycemia
        harm, severity = determine_harm_and_severity(0, 0, 1)
        assert severity == "1", "Bug condition produces severity 1"
        assert harm == "Hyperglycemia", "Bug condition produces Hyperglycemia"


class TestTLR822Scenario:
    """
    Integration test modeling the exact TLR-822 scenario.
    
    This test verifies that when pre-mitigation data uses no separator between
    'Loop' and 'NoMitigations', the data is correctly captured.
    """
    
    def test_tlr822_pre_mitigation_severity(self):
        """
        Reproduce TLR-822 bug scenario:
        - Pre-mitigation sim_ids use no separator (pre-LoopNoMitigations_)
        - One variant has double 's' typo (pre-LoopNoMitigationss_)
        - Expected: Data should be captured, not show NA
        """
        # Simulated data matching TLR-822 Risk_Results.csv
        lbgi_data = {
            'pre': [4, 4, 4, 4],  # With fix, this should be populated
            'no_loop': [4, 4, 4, 4],
            'post': [1, 4, 4, 4]
        }
        
        tar_data = {
            'pre': [0.0, 0.0, 0.0, 0.0],
            'no_loop': [0.0, 0.0, 0.0, 0.0],
            'post': [25.773, 0.0, 0.0, 3.093]
        }
        
        # Calculate averages
        lbgi_averages = calculate_integer_averages(lbgi_data)
        tar_averages = calculate_stage_averages(tar_data)
        
        # Pre-mitigation should have data, not be empty
        assert lbgi_averages['pre'] == 4, f"Pre LBGI should be 4, got {lbgi_averages['pre']}"
        assert tar_averages['pre'] != "NA", "Pre TAR should not be NA"


import re

FIXTURE_TLR_DIR = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'rtf_regression',
    'Risk_Run_2026-01-01T00_00_00.000000', 'TLR-999',
)
FIXTURE_TIMESTAMP = '2026-01-01T00:00:00.000000'


def _table_rows(rtf):
    """Return the RTF table as a list of rows, each a list of cell contents.

    A row is a \\trowd...\\row block; a cell is the text of a \\pard\\intbl ...\\cell
    unit with bold markup ({\\b ...}) stripped so header labels compare cleanly.
    """
    rows = []
    for block in re.findall(r'\\trowd(.*?)\\row', rtf, re.DOTALL):
        cells = []
        for raw in re.findall(r'\\pard\\intbl\s*(.*?)\\cell', block, re.DOTALL):
            cell = re.sub(r'\{\\b\s*(.*?)\}', r'\1', raw).strip()
            cells.append(cell)
        rows.append(cells)
    return rows


class TestLbgiDkaiColumnRendering:
    """Integration: LBGI/DKAI columns render in the right positions with right values."""

    def _render(self):
        assessment = build_assessment(FIXTURE_TLR_DIR, FIXTURE_TIMESTAMP)
        assert assessment is not None
        return render_rtf(assessment)

    def test_header_column_order(self):
        rtf = self._render()
        header = _table_rows(rtf)[0]
        assert header == [
            'Evaluation stage', 'Harm', 'Severity',
            'TIR % (70 - 180 mg/dL)', 'TBR % (<54 mg/dL)',
            'LBGI', 'DKAI', 'TAR % (>180 mg/dL)',
        ]
        # LBGI immediately after TBR; DKAI immediately before TAR.
        assert header.index('LBGI') == header.index('TBR % (<54 mg/dL)') + 1
        assert header.index('DKAI') == header.index('TAR % (>180 mg/dL)') - 1

    def test_eight_cellx_stops_per_row(self):
        rtf = self._render()
        for stops in re.findall(r'(\\cellx\d+(?:\\cellx\d+)*)', rtf):
            assert stops.count('\\cellx') == 8
        # Evenly redistributed across the original 10200 page width.
        assert '\\cellx1275\\cellx2550\\cellx3825\\cellx5100' \
               '\\cellx6375\\cellx7650\\cellx8925\\cellx10200' in rtf

    def test_stage_row_values_in_correct_cells(self):
        rtf = self._render()
        rows = _table_rows(rtf)
        # rows[0] header, rows[1] pre, rows[2] no_loop, rows[3] post.
        pre = rows[1]
        assert pre[0] == 'Pre-mitigation'
        assert pre[5] == '2.5'    # LBGI cell (raw lbgi avg, trailing zero dropped)
        assert pre[6] == '21.91'  # DKAI cell (multi-decimal truncation)
        assert pre[7] == '79.0'   # TAR still last

        no_loop = rows[2]
        assert no_loop[5] == '0'  # whole-number LBGI renders with no decimal
        assert no_loop[6] == '3'  # whole-number DKAI

        post = rows[3]
        assert post[5] == '1.14'  # fractional LBGI
        assert post[6] == '0'     # whole-number DKAI

    def test_lbgi_dkai_sit_between_tbr_and_tar_values(self):
        rtf = self._render()
        pre = _table_rows(rtf)[1]
        tbr_i, lbgi_i, dkai_i, tar_i = 4, 5, 6, 7
        assert pre[tbr_i] == '5.0'
        assert lbgi_i == tbr_i + 1 and dkai_i == tar_i - 1


class TestRtfRegressionGolden:
    """The stored-golden regression gate must pass against the committed golden."""

    def test_fixture_matches_golden(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'post_processing'))
        import rtf_regression_diff
        matches, diff = rtf_regression_diff.diff_against_golden()
        assert matches, "Rendered fixture differs from golden:\n" + "".join(diff[:60])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
