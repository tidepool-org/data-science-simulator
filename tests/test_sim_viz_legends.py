__author__ = "Claude"

"""
Unit tests for sim_viz.py legend configuration.

Tests verify that legend configurations are properly defined and applied
consistently across all visualization functions.
"""

import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

import matplotlib.dates as mdates

from tidepool_data_science_simulator.visualization.sim_viz import (
    LEGEND_CONFIG,
    LEGEND_CONFIG_DENSE,
    MidnightDateFormatter,
    configure_datetime_axis,
    plot_sim_results,
    plot_sim_icgm_paired,
    plot_sim_results_missing_insulin,
)


class TestLegendConfigurationConstants:
    """Tests for the centralized legend configuration dictionaries."""

    def test_legend_config_has_required_keys(self):
        """Verify LEGEND_CONFIG contains all required keys."""
        required_keys = ['fontsize', 'framealpha', 'loc']
        for key in required_keys:
            assert key in LEGEND_CONFIG, f"LEGEND_CONFIG missing required key: {key}"

    def test_legend_config_dense_has_required_keys(self):
        """Verify LEGEND_CONFIG_DENSE contains all required keys."""
        required_keys = ['fontsize', 'framealpha', 'loc', 'ncol']
        for key in required_keys:
            assert key in LEGEND_CONFIG_DENSE, f"LEGEND_CONFIG_DENSE missing required key: {key}"

    def test_legend_fontsize_is_reasonable(self):
        """Verify fontsize is within readable range (6-10 pt)."""
        assert 6 <= LEGEND_CONFIG['fontsize'] <= 10, \
            f"LEGEND_CONFIG fontsize {LEGEND_CONFIG['fontsize']} outside readable range 6-10"
        assert 6 <= LEGEND_CONFIG_DENSE['fontsize'] <= 10, \
            f"LEGEND_CONFIG_DENSE fontsize {LEGEND_CONFIG_DENSE['fontsize']} outside readable range 6-10"

    def test_legend_dense_has_multiple_columns(self):
        """Verify dense legend uses multiple columns to reduce vertical footprint."""
        assert LEGEND_CONFIG_DENSE['ncol'] >= 2, \
            "LEGEND_CONFIG_DENSE should use at least 2 columns"

    def test_legend_framealpha_allows_visibility(self):
        """Verify framealpha is set to allow some data visibility."""
        assert 0.5 <= LEGEND_CONFIG['framealpha'] <= 1.0, \
            "framealpha should be between 0.5 and 1.0 for readability"

    def test_dense_fontsize_not_larger_than_standard(self):
        """Verify dense legend fontsize is not larger than standard."""
        assert LEGEND_CONFIG_DENSE['fontsize'] <= LEGEND_CONFIG['fontsize'], \
            "Dense legend fontsize should not exceed standard fontsize"


class TestLegendConsistency:
    """Tests to verify legends are applied consistently across functions."""

    @pytest.fixture
    def mock_results_single_sim(self):
        """Create minimal mock simulation results for testing."""
        t0 = datetime.datetime(2019, 8, 15, 12, 0, 0)
        times = pd.date_range(start=t0, periods=10, freq='5min')

        df = pd.DataFrame({
            'time': times,
            'bg': np.random.uniform(80, 180, 10),
            'bg_sensor': np.random.uniform(80, 180, 10),
            'sbr': np.random.uniform(0.5, 1.5, 10),
            'temp_basal': np.random.uniform(0, 2, 10),
            'true_bolus': np.zeros(10),
            'reported_bolus': np.zeros(10),
            'iob': np.random.uniform(0, 3, 10),
            'ei': np.random.uniform(0, 0.2, 10),
            'true_carb_value': np.zeros(10),
            'reported_carb_value': np.zeros(10),
        }, index=times)

        return {'test_sim': df}

    @pytest.fixture
    def mock_results_missing_insulin(self):
        """Create mock results for missing insulin plot function."""
        t0 = datetime.datetime(2019, 8, 15, 12, 0, 0)
        times = pd.date_range(start=t0, periods=10, freq='5min')

        df = pd.DataFrame({
            'time': times,
            'bg': np.random.uniform(80, 180, 10),
            'bg_sensor': np.random.uniform(80, 180, 10),
            'sbr': np.random.uniform(0.5, 1.5, 10),
            'temp_basal': np.random.uniform(0, 2, 10),
            'bolus': np.zeros(10),
            'delivered_basal_insulin': np.random.uniform(0, 0.1, 10),
            'undelivered_basal_insulin': np.zeros(10),
        })

        return {'test_sim': df}

    def test_plot_sim_results_creates_figure(self, mock_results_single_sim):
        """Verify plot_sim_results creates a figure without errors."""
        plt.close('all')
        # This should not raise any exceptions
        plot_sim_results(mock_results_single_sim, save=False)

        # Check that a figure was created
        assert len(plt.get_fignums()) > 0, "No figure was created"
        plt.close('all')

    def test_plot_sim_results_legends_have_consistent_properties(self, mock_results_single_sim):
        """Verify all legends in plot_sim_results use the centralized config."""
        plt.close('all')

        # Need to patch plt.show to prevent blocking
        original_show = plt.show
        plt.show = lambda: None

        try:
            plot_sim_results(mock_results_single_sim, save=False)

            fig = plt.gcf()
            axes = fig.get_axes()

            # Check BG and Carbs charts use standard config
            for i in [0, 2]:  # BG and Carbs axes
                legend = axes[i].get_legend()
                if legend is not None:
                    # Verify fontsize is from LEGEND_CONFIG
                    for text in legend.get_texts():
                        assert text.get_fontsize() == LEGEND_CONFIG['fontsize'], \
                            f"Axis {i} legend fontsize mismatch"

            # Check Insulin chart uses dense config
            legend = axes[1].get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    assert text.get_fontsize() == LEGEND_CONFIG_DENSE['fontsize'], \
                        "Insulin chart legend fontsize should use LEGEND_CONFIG_DENSE"
        finally:
            plt.show = original_show
            plt.close('all')


class TestLegendReadability:
    """Tests to verify legend sizing promotes readability."""

    def test_insulin_legend_not_oversized(self):
        """Verify insulin chart legend uses smaller font than previous 12pt."""
        # Previously the insulin chart used size 12, which was too large
        assert LEGEND_CONFIG_DENSE['fontsize'] < 12, \
            "Dense legend fontsize should be smaller than previous 12pt"

    def test_all_legends_within_readable_range(self):
        """Verify all legend fontsizes are within the readable range of 6-10pt."""
        readable_min = 6
        readable_max = 10

        assert readable_min <= LEGEND_CONFIG['fontsize'] <= readable_max, \
            f"Standard legend fontsize {LEGEND_CONFIG['fontsize']} not in readable range"
        assert readable_min <= LEGEND_CONFIG_DENSE['fontsize'] <= readable_max, \
            f"Dense legend fontsize {LEGEND_CONFIG_DENSE['fontsize']} not in readable range"


class TestMidnightDateFormatter:
    """Tests for the MidnightDateFormatter class."""

    def test_formatter_shows_time_for_non_midnight(self):
        """Verify formatter shows only time (HH:MM) for non-midnight times."""
        formatter = MidnightDateFormatter()

        # Test 14:00 (2 PM) - should show only time
        dt_2pm = datetime.datetime(2019, 8, 15, 14, 0, 0)
        num_2pm = mdates.date2num(dt_2pm)
        result = formatter(num_2pm)

        assert result == "14:00", f"Expected '14:00', got '{result}'"
        assert "\n" not in result, "Non-midnight time should not contain newline (date)"

    def test_formatter_shows_date_at_midnight(self):
        """Verify formatter shows time and date at midnight."""
        formatter = MidnightDateFormatter()

        # Test midnight - should show time and date
        dt_midnight = datetime.datetime(2019, 8, 16, 0, 0, 0)
        num_midnight = mdates.date2num(dt_midnight)
        result = formatter(num_midnight)

        assert "00:00" in result, f"Midnight should show '00:00', got '{result}'"
        assert "\n" in result, "Midnight should contain newline for date"
        assert "8/16" in result, f"Midnight should show date '8/16', got '{result}'"

    def test_formatter_handles_various_times(self):
        """Verify formatter handles various times correctly."""
        formatter = MidnightDateFormatter()

        test_cases = [
            (datetime.datetime(2019, 8, 15, 6, 0), "06:00", False),   # 6 AM
            (datetime.datetime(2019, 8, 15, 12, 0), "12:00", False),  # Noon
            (datetime.datetime(2019, 8, 15, 18, 0), "18:00", False),  # 6 PM
            (datetime.datetime(2019, 8, 16, 0, 0), "00:00", True),    # Midnight
        ]

        for dt, expected_time, should_have_date in test_cases:
            num = mdates.date2num(dt)
            result = formatter(num)
            assert expected_time in result, f"Time {dt} should contain '{expected_time}'"
            if should_have_date:
                assert "\n" in result, f"Midnight {dt} should have date"
            else:
                assert "\n" not in result, f"Non-midnight {dt} should not have date"


class TestConfigureDatetimeAxis:
    """Tests for the configure_datetime_axis helper function."""

    def test_configure_sets_hour_locator(self):
        """Verify configure_datetime_axis sets HourLocator."""
        fig, ax = plt.subplots()
        configure_datetime_axis(ax, interval_hours=2)

        locator = ax.xaxis.get_major_locator()
        assert isinstance(locator, mdates.HourLocator), \
            f"Expected HourLocator, got {type(locator)}"

        plt.close(fig)

    def test_configure_sets_custom_formatter(self):
        """Verify configure_datetime_axis sets MidnightDateFormatter."""
        fig, ax = plt.subplots()
        configure_datetime_axis(ax, interval_hours=2)

        formatter = ax.xaxis.get_major_formatter()
        assert isinstance(formatter, MidnightDateFormatter), \
            f"Expected MidnightDateFormatter, got {type(formatter)}"

        plt.close(fig)

    def test_configure_accepts_custom_interval(self):
        """Verify configure_datetime_axis accepts custom hour interval."""
        fig, ax = plt.subplots()

        # Should not raise any exceptions
        configure_datetime_axis(ax, interval_hours=4)

        plt.close(fig)


class TestXAxisInPlotSimResults:
    """Tests to verify x-axis formatting is applied in plot_sim_results."""

    @pytest.fixture
    def mock_results_24hr(self):
        """Create mock simulation results spanning 24+ hours."""
        t0 = datetime.datetime(2019, 8, 15, 11, 30, 0)
        # Create data spanning ~25 hours to cross midnight
        times = pd.date_range(start=t0, periods=300, freq='5min')

        df = pd.DataFrame({
            'time': times,
            'bg': np.random.uniform(80, 180, 300),
            'bg_sensor': np.random.uniform(80, 180, 300),
            'sbr': np.random.uniform(0.5, 1.5, 300),
            'temp_basal': np.random.uniform(0, 2, 300),
            'true_bolus': np.zeros(300),
            'reported_bolus': np.zeros(300),
            'iob': np.random.uniform(0, 3, 300),
            'ei': np.random.uniform(0, 0.2, 300),
            'true_carb_value': np.zeros(300),
            'reported_carb_value': np.zeros(300),
        }, index=times)

        return {'test_sim': df}

    def test_plot_sim_results_uses_hour_locator(self, mock_results_24hr):
        """Verify plot_sim_results applies HourLocator to x-axis."""
        plt.close('all')
        original_show = plt.show
        plt.show = lambda: None

        try:
            plot_sim_results(mock_results_24hr, save=False)
            fig = plt.gcf()
            axes = fig.get_axes()

            # Check the shared x-axis (bottom chart)
            locator = axes[2].xaxis.get_major_locator()
            assert isinstance(locator, mdates.HourLocator), \
                f"Expected HourLocator on x-axis, got {type(locator)}"
        finally:
            plt.show = original_show
            plt.close('all')

    def test_plot_sim_results_uses_midnight_formatter(self, mock_results_24hr):
        """Verify plot_sim_results applies MidnightDateFormatter to x-axis."""
        plt.close('all')
        original_show = plt.show
        plt.show = lambda: None

        try:
            plot_sim_results(mock_results_24hr, save=False)
            fig = plt.gcf()
            axes = fig.get_axes()

            # Check the shared x-axis formatter
            formatter = axes[2].xaxis.get_major_formatter()
            assert isinstance(formatter, MidnightDateFormatter), \
                f"Expected MidnightDateFormatter, got {type(formatter)}"
        finally:
            plt.show = original_show
            plt.close('all')
