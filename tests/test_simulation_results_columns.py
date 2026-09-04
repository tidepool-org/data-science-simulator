"""
Test for simulation results DataFrame column structure.

Verifies that carb entry tracking fields are only present for reported (pump) data,
not for true (patient model) data, since entry time metadata is only meaningful
for Loop/pump interactions.
"""

__author__ = "Shawn Foster"

import datetime
import pytest

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import DoNothingController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.makedata.make_patient import (
    DATETIME_DEFAULT, get_canonical_risk_patient_config, get_canonical_risk_pump_config,
    get_canonical_sensor_config
)
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config

from tidepool_data_science_simulator.models.events import CarbTimeline
from tidepool_data_science_simulator.models.measures import Carb


class TestSimulationResultsColumns:
    """Test that simulation results have correct column structure for carb tracking."""

    def test_true_carb_entry_metadata_not_in_results(self):
        """
        Verify that true_carb_entry_time and related version control fields
        are NOT present in simulation results.
        
        These fields only make sense for reported (pump) carbs since entry_time,
        sync_identifier, sync_version, etc. are Loop-specific metadata. The patient's
        body doesn't track when carbs were "entered" - only the pump/Loop does.
        """
        # Setup minimal simulation
        t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=120)
        t0, sensor_config = get_canonical_sensor_config(start_value=120)
        t0, controller_config = get_canonical_controller_config()
        t0, pump_config = get_canonical_risk_pump_config()

        # Add a carb entry to both patient and pump
        carb_time = t0
        carb_entry_time = t0 - datetime.timedelta(minutes=15)  # Entry before consumption
        
        carb_obj = Carb(
            value=50.0,
            units="g",
            duration_minutes=180,
            entry_time=carb_entry_time,
            sync_identifier="test-carb-123",
            sync_version=0
        )
        
        patient_config.carb_event_timeline = CarbTimeline(
            datetimes=[carb_time],
            events=[carb_obj]
        )
        pump_config.carb_event_timeline = CarbTimeline(
            datetimes=[carb_time],
            events=[carb_obj]
        )

        pump = ContinuousInsulinPump(pump_config, t0)
        sensor = IdealSensor(t0, sensor_config)
        controller = DoNothingController(t0, controller_config)

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config
        )

        sim = Simulation(
            time=t0,
            duration_hrs=1,
            virtual_patient=vp,
            controller=controller,
            sim_id="test_columns"
        )

        sim.run()
        results_df = sim.get_results_df()

        # These columns should NOT be present (removed as they don't make conceptual sense)
        removed_columns = [
            "true_carb_entry_time",
            "true_carb_sync_identifier",
            "true_carb_sync_version",
            "true_carb_user_created_date",
            "true_carb_user_updated_date",
            "true_carb_superceded_date",
        ]
        
        for col in removed_columns:
            assert col not in results_df.columns, f"Column '{col}' should not be in results"

    def test_reported_carb_entry_metadata_present_in_results(self):
        """
        Verify that reported_carb_entry_time and related version control fields
        ARE present in simulation results.
        
        These fields are meaningful for pump/Loop carb entries where we need to
        track when entries were made and their version history.
        """
        # Setup minimal simulation
        t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=120)
        t0, sensor_config = get_canonical_sensor_config(start_value=120)
        t0, controller_config = get_canonical_controller_config()
        t0, pump_config = get_canonical_risk_pump_config()

        pump = ContinuousInsulinPump(pump_config, t0)
        sensor = IdealSensor(t0, sensor_config)
        controller = DoNothingController(t0, controller_config)

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config
        )

        sim = Simulation(
            time=t0,
            duration_hrs=1,
            virtual_patient=vp,
            controller=controller,
            sim_id="test_columns"
        )

        sim.run()
        results_df = sim.get_results_df()

        # These columns SHOULD be present (for pump/reported carbs)
        expected_columns = [
            "reported_carb_entry_time",
            "reported_carb_sync_identifier",
            "reported_carb_sync_version",
            "reported_carb_user_created_date",
            "reported_carb_user_updated_date",
            "reported_carb_superceded_date",
            "reported_carb_operation",
        ]
        
        for col in expected_columns:
            assert col in results_df.columns, f"Column '{col}' should be in results"

    def test_true_carb_basic_fields_still_present(self):
        """
        Verify that basic true_carb fields (value, duration, operation) are still present.
        
        These represent actual physiological carb consumption which is meaningful
        for the patient model.
        """
        # Setup minimal simulation
        t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=120)
        t0, sensor_config = get_canonical_sensor_config(start_value=120)
        t0, controller_config = get_canonical_controller_config()
        t0, pump_config = get_canonical_risk_pump_config()

        pump = ContinuousInsulinPump(pump_config, t0)
        sensor = IdealSensor(t0, sensor_config)
        controller = DoNothingController(t0, controller_config)

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config
        )

        sim = Simulation(
            time=t0,
            duration_hrs=1,
            virtual_patient=vp,
            controller=controller,
            sim_id="test_columns"
        )

        sim.run()
        results_df = sim.get_results_df()

        # Basic true_carb fields should still be present
        expected_true_carb_columns = [
            "true_carb_value",
            "true_carb_duration",
            "true_carb_operation",
        ]
        
        for col in expected_true_carb_columns:
            assert col in results_df.columns, f"Column '{col}' should be in results"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
