"""
Unit tests for NoisySensor configuration override functionality.

Tests that scenarios can successfully override sensor type and parameters
in base configurations.
"""

import pytest
import json
import os
import tempfile
import shutil

# Add the simulator to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2


class TestNoisySensorOverride:
    """Tests for NoisySensor configuration overrides."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        reusable_dir = os.path.join(temp_dir, "reusable")
        simulations_dir = os.path.join(reusable_dir, "simulations")
        glucose_dir = os.path.join(reusable_dir, "glucose")
        metabolism_dir = os.path.join(reusable_dir, "metabolism_settings")
        loop_settings_dir = os.path.join(reusable_dir, "loop_settings")
        
        os.makedirs(simulations_dir)
        os.makedirs(glucose_dir)
        os.makedirs(metabolism_dir)
        os.makedirs(loop_settings_dir)
        
        # Create glucose history file
        # History must end at the same instant as base_config's time_to_calculate_at
        # (8/15/2019 12:00:00) -- VirtualPatient.init() requires this to match.
        glucose_history = {
            "datetime": {"0": "8/15/2019 11:55:00", "1": "8/15/2019 12:00:00"},
            "value": {"0": 110, "1": 110}
        }
        with open(os.path.join(glucose_dir, "flat_110.json"), "w") as f:
            json.dump(glucose_history, f)
        
        # Create metabolism settings file
        metabolism_settings = {
            "patient_insulin_type": "rapid_acting_adult",
            "basal_rate": {
                "start_times": ["0:00:00"],
                "values": [1.0]
            },
            "carb_insulin_ratio": {
                "start_times": ["0:00:00"],
                "values": [10.0]
            },
            "insulin_sensitivity_factor": {
                "start_times": ["0:00:00"],
                "values": [50.0]
            }
        }
        with open(os.path.join(metabolism_dir, "test_v1.json"), "w") as f:
            json.dump(metabolism_settings, f)
        
        # Create loop settings file
        loop_settings = {
            "model": "rapid_acting_adult",
            "momentum_data_interval": 15,
            "suspend_threshold": 70,
            "max_basal_rate": 4.0,
            "max_bolus": 10.0
        }
        with open(os.path.join(loop_settings_dir, "test_v1.json"), "w") as f:
            json.dump(loop_settings, f)
        
        # Create base simulation config WITH sensor type and empty parameters
        # Empty parameters dict allows new keys to be added via overrides
        base_config = {
            "sim_id": "base_test",
            "time_to_calculate_at": "8/15/2019 12:00:00",
            "duration_hours": 1.0,
            "patient": {
                "sensor": {
                    "glucose_history": "reusable.glucose.flat_110",
                    "type": "IdealSensor",
                    "parameters": {}
                },
                "pump": {
                    "metabolism_settings": "reusable.metabolism_settings.test_v1",
                    "bolus_entries": [],
                    "carb_entries": [],
                    "target_range": {
                        "start_times": ["0:00:00"],
                        "lower_values": [70],
                        "upper_values": [90]
                    }
                },
                "patient_model": {
                    "metabolism_settings": "reusable.metabolism_settings.test_v1",
                    "glucose_history": "reusable.glucose.flat_110",
                    "bolus_entries": [],
                    "carb_entries": [],
                    "physical_activity_entries": []
                }
            },
            "controller": {
                "id": "swift",
                "settings": "reusable.loop_settings.test_v1",
                "automation_control_timeline": []
            }
        }
        with open(os.path.join(simulations_dir, "base_test.json"), "w") as f:
            json.dump(base_config, f)
        
        yield temp_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_noisy_sensor_type_override(self, temp_config_dir):
        """Test that sensor type can be overridden to NoisySensor."""
        # Create scenario config that overrides sensor type
        scenario_config = {
            "metadata": {
                "risk-id": "TEST-001",
                "simulation_id": "TEST-001-noisy",
                "risk_description": "Test NoisySensor override",
                "config_format_version": "v1.0"
            },
            "base_config": "reusable.simulations.base_test",
            "override_config": [
                {
                    "sim_id": "test_noisy_sensor",
                    "patient": {
                        "sensor": {
                            "type": "NoisySensor",
                            "parameters": {
                                "std_dev": 5.0
                            }
                        }
                    }
                }
            ]
        }
        
        scenario_path = os.path.join(temp_config_dir, "test_scenario.json")
        with open(scenario_path, "w") as f:
            json.dump(scenario_config, f)
        
        # Parse the scenario
        parser = ScenarioParserV2(
            path_to_json_config=scenario_path,
            pointer_object_dir=temp_config_dir
        )
        
        # Get simulations
        sims = parser.get_sims()
        
        # Verify simulation was created
        assert "test_noisy_sensor" in sims
        sim = sims["test_noisy_sensor"]
        
        # Verify sensor type is NoisySensor
        sensor = sim.virtual_patient.sensor
        assert sensor.name == "BasicNoisySensor"
        assert sensor.sensor_config.std_dev == 5.0
    
    def test_noisy_sensor_all_parameters(self, temp_config_dir):
        """Test that all NoisySensor parameters can be configured."""
        scenario_config = {
            "metadata": {
                "risk-id": "TEST-002",
                "simulation_id": "TEST-002-full-noisy",
                "risk_description": "Test all NoisySensor parameters",
                "config_format_version": "v1.0"
            },
            "base_config": "reusable.simulations.base_test",
            "override_config": [
                {
                    "sim_id": "test_full_noisy_sensor",
                    "patient": {
                        "sensor": {
                            "type": "NoisySensor",
                            "parameters": {
                                "std_dev": 4.0,
                                "spurious_prob": 0.05,
                                "spurious_outage_prob": 0.02,
                                "time_delta_crunch_prob": 0.01,
                                "bg_spurious_error_delta_mgdl_range": [50, 100],
                                "not_working_time_minutes_range": [5, 30],
                                "cgm_offset_minutes_range": [1, 3]
                            }
                        }
                    }
                }
            ]
        }
        
        scenario_path = os.path.join(temp_config_dir, "test_scenario_full.json")
        with open(scenario_path, "w") as f:
            json.dump(scenario_config, f)
        
        parser = ScenarioParserV2(
            path_to_json_config=scenario_path,
            pointer_object_dir=temp_config_dir
        )
        
        sims = parser.get_sims()
        sim = sims["test_full_noisy_sensor"]
        sensor = sim.virtual_patient.sensor
        
        # Verify all parameters
        assert sensor.name == "BasicNoisySensor"
        assert sensor.sensor_config.std_dev == 4.0
        assert sensor.sensor_config.spurious_prob == 0.05
        assert sensor.sensor_config.spurious_outage_prob == 0.02
        assert sensor.sensor_config.time_delta_crunch_prob == 0.01
        assert sensor.sensor_config.bg_spurious_error_delta_mgdl_range == [50, 100]
        assert sensor.sensor_config.not_working_time_minutes_range == [5, 30]
        assert sensor.sensor_config.cgm_offset_minutes_range == [1, 3]
    
    def test_ideal_sensor_remains_default(self, temp_config_dir):
        """Test that IdealSensor is used when type is not overridden."""
        scenario_config = {
            "metadata": {
                "risk-id": "TEST-003",
                "simulation_id": "TEST-003-ideal",
                "risk_description": "Test IdealSensor default",
                "config_format_version": "v1.0"
            },
            "base_config": "reusable.simulations.base_test",
            "override_config": [
                {
                    "sim_id": "test_ideal_sensor"
                }
            ]
        }
        
        scenario_path = os.path.join(temp_config_dir, "test_scenario_ideal.json")
        with open(scenario_path, "w") as f:
            json.dump(scenario_config, f)
        
        parser = ScenarioParserV2(
            path_to_json_config=scenario_path,
            pointer_object_dir=temp_config_dir
        )
        
        sims = parser.get_sims()
        sim = sims["test_ideal_sensor"]
        sensor = sim.virtual_patient.sensor
        
        # Verify sensor type is IdealSensor
        assert sensor.name == "IdealSensor"


class TestSensorConfigValidation:
    """Tests for sensor parameter validation."""
    
    def test_std_dev_validation_valid(self):
        """Test valid std_dev values pass validation."""
        parser = ScenarioParserV2()
        # Should not raise
        parser.validate_sensor_std_dev(0.0)
        parser.validate_sensor_std_dev(5.0)
        parser.validate_sensor_std_dev(50.0)
        parser.validate_sensor_std_dev(25)  # int should work too
    
    def test_std_dev_validation_invalid(self):
        """Test invalid std_dev values raise errors."""
        parser = ScenarioParserV2()
        
        with pytest.raises(ValueError):
            parser.validate_sensor_std_dev(-1.0)
        
        with pytest.raises(ValueError):
            parser.validate_sensor_std_dev(51.0)
        
        with pytest.raises(ValueError):
            parser.validate_sensor_std_dev("five")
    
    def test_spurious_prob_validation(self):
        """Test spurious probability validation."""
        parser = ScenarioParserV2()
        
        # Valid values
        parser.validate_sensor_spurious_prob(0.0)
        parser.validate_sensor_spurious_prob(0.5)
        parser.validate_sensor_spurious_prob(1.0)
        
        # Invalid values
        with pytest.raises(ValueError):
            parser.validate_sensor_spurious_prob(-0.1)
        
        with pytest.raises(ValueError):
            parser.validate_sensor_spurious_prob(1.1)
    
    def test_bg_error_range_validation(self):
        """Test BG error range validation."""
        parser = ScenarioParserV2()
        
        # Valid ranges
        parser.validate_sensor_bg_error_range([50, 100])
        parser.validate_sensor_bg_error_range([0, 500])
        
        # Invalid: not a list
        with pytest.raises(ValueError):
            parser.validate_sensor_bg_error_range(50)
        
        # Invalid: wrong length
        with pytest.raises(ValueError):
            parser.validate_sensor_bg_error_range([50])
        
        # Invalid: min >= max
        with pytest.raises(ValueError):
            parser.validate_sensor_bg_error_range([100, 50])


class TestBaseConfigSensorFields:
    """Tests to verify base configs have required sensor fields."""
    
    def test_base_median_has_sensor_fields(self):
        """Verify base_median_1dotx.json has type and parameters fields."""
        base_path = "/Users/shawnfoster/PycharmProjects/data-science-simulator-v2/data-science-simulator"
        config_path = os.path.join(
            base_path,
            "scenario_configs/tidepool_risk_v2/reusable/simulations/1xComparator/base_median_1dotx.json"
        )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sensor_config = config["patient"]["sensor"]
        
        assert "type" in sensor_config, "Missing 'type' field in sensor config"
        assert "parameters" in sensor_config, "Missing 'parameters' field in sensor config"
        assert sensor_config["type"] == "IdealSensor"
        assert sensor_config["parameters"] == {}
    
    def test_activity_preset_has_sensor_fields(self):
        """Verify activity preset configs have type and parameters fields."""
        base_path = "/Users/shawnfoster/PycharmProjects/data-science-simulator-v2/data-science-simulator"
        config_path = os.path.join(
            base_path,
            "scenario_configs/tidepool_risk_v2/reusable/simulations/activity_presets/ap_bike_median_2_0_v1.json"
        )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sensor_config = config["patient"]["sensor"]
        
        assert "type" in sensor_config, "Missing 'type' field in sensor config"
        assert "parameters" in sensor_config, "Missing 'parameters' field in sensor config"
        assert sensor_config["type"] == "IdealSensor"
        assert sensor_config["parameters"] == {}
    
    def test_base_2_0_has_sensor_fields(self):
        """Verify base 2.0 configs have type and parameters fields."""
        base_path = "/Users/shawnfoster/PycharmProjects/data-science-simulator-v2/data-science-simulator"
        config_path = os.path.join(
            base_path,
            "scenario_configs/tidepool_risk_v2/reusable/simulations/base/base_median_2_0_v1.json"
        )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sensor_config = config["patient"]["sensor"]
        
        assert "type" in sensor_config, "Missing 'type' field in sensor config"
        assert "parameters" in sensor_config, "Missing 'parameters' field in sensor config"
        assert sensor_config["type"] == "IdealSensor"
        assert sensor_config["parameters"] == {}
    
    def test_custom_preset_has_sensor_fields(self):
        """Verify custom preset configs have type and parameters fields."""
        base_path = "/Users/shawnfoster/PycharmProjects/data-science-simulator-v2/data-science-simulator"
        config_path = os.path.join(
            base_path,
            "scenario_configs/tidepool_risk_v2/reusable/simulations/custom_presets/preset_170_median_2_0_v1.json"
        )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sensor_config = config["patient"]["sensor"]
        
        assert "type" in sensor_config, "Missing 'type' field in sensor config"
        assert "parameters" in sensor_config, "Missing 'parameters' field in sensor config"
        assert sensor_config["type"] == "IdealSensor"
        assert sensor_config["parameters"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
