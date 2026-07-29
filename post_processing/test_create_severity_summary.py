"""
Unit tests for create_severity_summary.py

Focuses on round_half_up() and calculate_integer_averages() to verify
conservative (round-half-up) rounding behavior for risk severity scores.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from create_severity_summary import round_half_up, calculate_integer_averages


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
