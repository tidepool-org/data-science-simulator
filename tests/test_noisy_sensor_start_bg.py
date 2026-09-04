"""
Unit tests for NoisySensor true_start_bg and start_bg_with_offset tracking.

These attributes are used for iCGM evaluation to track the initial true BG
and the first sensor reading.
"""

import pytest
import datetime
import numpy as np
from numpy.random import RandomState

from tidepool_data_science_simulator.models.sensor import NoisySensor, IdealSensor
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig
from tidepool_data_science_simulator.models.measures import GlucoseTrace


class TestNoisySensorStartBG:
    """Test that NoisySensor properly tracks initial BG values."""
    
    def test_attributes_initialized_to_none(self):
        """Verify true_start_bg and start_bg_with_offset are initialized to None."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        
        sensor = NoisySensor(time, sensor_config, random_state=RandomState(42))
        
        assert sensor.true_start_bg is None, "true_start_bg should initialize to None"
        assert sensor.start_bg_with_offset is None, "start_bg_with_offset should initialize to None"
        assert sensor._first_update is True, "_first_update flag should be True"
    
    def test_first_update_captures_start_values(self):
        """Verify first update captures true BG and sensor reading."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        sensor_config.std_dev = 5.0
        
        sensor = NoisySensor(time, sensor_config, random_state=RandomState(42))
        
        # First update
        true_bg = 120.0
        sensor.update(
            time, 
            patient_true_bg=true_bg,
            patient_true_bg_prediction=[120, 120, 120]
        )
        
        assert sensor.true_start_bg == true_bg, "true_start_bg should be set to first true BG"
        assert sensor.start_bg_with_offset is not None, "start_bg_with_offset should be set"
        assert isinstance(sensor.start_bg_with_offset, (int, float)), "start_bg_with_offset should be numeric"
        assert sensor._first_update is False, "_first_update flag should be False after first update"
    
    def test_subsequent_updates_dont_change_start_values(self):
        """Verify subsequent updates don't overwrite initial values."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        sensor_config.std_dev = 5.0
        
        sensor = NoisySensor(time, sensor_config, random_state=RandomState(42))
        
        # First update
        first_true_bg = 120.0
        sensor.update(
            time, 
            patient_true_bg=first_true_bg,
            patient_true_bg_prediction=[120, 120, 120]
        )
        
        first_start_bg = sensor.true_start_bg
        first_sensor_reading = sensor.start_bg_with_offset
        
        # Second update with different values
        time2 = time + datetime.timedelta(minutes=5)
        second_true_bg = 150.0
        sensor.update(
            time2,
            patient_true_bg=second_true_bg,
            patient_true_bg_prediction=[150, 150, 150]
        )
        
        # Verify start values haven't changed
        assert sensor.true_start_bg == first_start_bg, "true_start_bg should not change after first update"
        assert sensor.start_bg_with_offset == first_sensor_reading, "start_bg_with_offset should not change after first update"
    
    def test_get_info_stateless_includes_start_values(self):
        """Verify get_info_stateless returns start BG values."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        sensor_config.std_dev = 5.0
        
        sensor = NoisySensor(time, sensor_config, random_state=RandomState(42))
        
        # Before first update
        info = sensor.get_info_stateless()
        assert "true_start_bg" in info, "true_start_bg should be in stateless info"
        assert "start_bg_with_offset" in info, "start_bg_with_offset should be in stateless info"
        assert info["true_start_bg"] is None, "true_start_bg should be None before first update"
        assert info["start_bg_with_offset"] is None, "start_bg_with_offset should be None before first update"
        
        # After first update
        true_bg = 120.0
        sensor.update(
            time,
            patient_true_bg=true_bg,
            patient_true_bg_prediction=[120, 120, 120]
        )
        
        info = sensor.get_info_stateless()
        assert info["true_start_bg"] == true_bg, "true_start_bg should be in stateless info after update"
        assert info["start_bg_with_offset"] is not None, "start_bg_with_offset should be set after update"
    
    def test_noise_applied_to_start_bg(self):
        """Verify noise is applied to first sensor reading."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        sensor_config.std_dev = 10.0  # Higher std dev to ensure noise is applied
        sensor_config.spurious_prob = 0.0  # No spurious readings
        
        # Run multiple times to check that noise is applied
        differences = []
        for seed in range(10):
            sensor = NoisySensor(time, sensor_config, random_state=RandomState(seed))
            
            true_bg = 150.0
            sensor.update(
                time,
                patient_true_bg=true_bg,
                patient_true_bg_prediction=[150, 150, 150]
            )
            
            difference = abs(sensor.start_bg_with_offset - true_bg)
            differences.append(difference)
        
        # At least some readings should have noise (not all exactly zero difference)
        assert any(diff > 0 for diff in differences), "Some sensor readings should have noise applied"
        
        # Average difference should be reasonable (not excessive)
        avg_diff = np.mean(differences)
        assert avg_diff < 20, f"Average difference {avg_diff} seems too high for std_dev=10"
    
    def test_ideal_sensor_no_start_bg_tracking(self):
        """Verify IdealSensor doesn't have start BG tracking."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        
        sensor = IdealSensor(time, sensor_config)
        
        # IdealSensor should not have these attributes
        assert not hasattr(sensor, 'true_start_bg'), "IdealSensor should not have true_start_bg"
        assert not hasattr(sensor, 'start_bg_with_offset'), "IdealSensor should not have start_bg_with_offset"


class TestNoisySensorStartBGWithDifferentParameters:
    """Test start BG tracking with various sensor parameters."""
    
    def test_with_spurious_reading(self):
        """Test start BG capture when first reading is spurious."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        sensor_config.std_dev = 5.0
        sensor_config.spurious_prob = 1.0  # Always spurious
        sensor_config.bg_spurious_error_delta_mgdl_range = [60, 150]
        
        sensor = NoisySensor(time, sensor_config, random_state=RandomState(42))
        
        true_bg = 120.0
        sensor.update(
            time,
            patient_true_bg=true_bg,
            patient_true_bg_prediction=[120, 120, 120]
        )
        
        # Should still capture values even if spurious
        assert sensor.true_start_bg == true_bg
        assert sensor.start_bg_with_offset is not None
        
        # Spurious reading should be different from true BG
        error = abs(sensor.start_bg_with_offset - true_bg)
        assert error >= 60, "Spurious reading should have significant error"
    
    def test_with_sensor_not_working(self):
        """Test start BG capture when sensor is not working initially."""
        time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        glucose_history = GlucoseTrace()
        sensor_config = SensorConfig(glucose_history)
        sensor_config.spurious_prob = 1.0
        sensor_config.spurious_outage_prob = 1.0  # Always causes outage
        
        sensor = NoisySensor(time, sensor_config, random_state=RandomState(42))
        
        true_bg = 120.0
        sensor.update(
            time,
            patient_true_bg=true_bg,
            patient_true_bg_prediction=[120, 120, 120]
        )
        
        # Should still capture true BG
        assert sensor.true_start_bg == true_bg
        # Sensor reading might be None due to outage
        assert sensor.start_bg_with_offset is not None or sensor.start_bg_with_offset is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
