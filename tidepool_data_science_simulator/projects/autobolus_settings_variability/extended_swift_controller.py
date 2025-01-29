__author__ = "Shawn Foster"

import datetime
from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController


class ExtendedSwiftController(SwiftLoopController):
    """
    Extended Swift Loop controller that handles multi-day schedules properly.
    Wraps the original SwiftLoopController without modifying its code.
    """

    def prepare_inputs(self, virtual_patient):
        """
        Override prepare_inputs to extend schedule timelines properly.

        Parameters
        ----------
        virtual_patient : VirtualPatient
            The virtual patient model

        Returns
        -------
        dict
            Inputs for the Swift Loop Algorithm with extended schedules
        """
        # First get base inputs from parent class
        base_inputs = super().prepare_inputs(virtual_patient)

        # Get the simulation time range
        current_time = self.time
        simulation_end = current_time + datetime.timedelta(hours=24)  # At least 24h ahead

        # Get schedule inputs for full duration
        basal_values, basal_starts, basal_ends = (
            virtual_patient.pump.pump_config.basal_schedule.get_loop_swift_inputs()
        )

        isf_values, isf_starts, isf_ends = (
            virtual_patient.pump.pump_config.insulin_sensitivity_schedule.get_loop_swift_inputs()
        )

        cir_values, cir_starts, cir_ends = (
            virtual_patient.pump.pump_config.carb_ratio_schedule.get_loop_swift_inputs()
        )

        tr_min_values, tr_max_values, tr_starts, tr_ends = (
            virtual_patient.pump.pump_config.target_range_schedule.get_loop_swift_inputs()
        )

        # Update