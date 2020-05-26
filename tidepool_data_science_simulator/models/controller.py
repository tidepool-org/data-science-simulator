__author__ = "Cameron Summers"

import pickle as pk
import datetime
import copy
import numpy as np

from src.models.simulation import SimulationComponent
from src.models.measures import GlucoseTrace

from pyloopkit.loop_data_manager import update
from pyloopkit.dose import DoseType


class DoNothingController(SimulationComponent):
    """
    A controller that does nothing, which means that pump schedules
    are the only modulation.
    """

    def __init__(self, time, controller_config):
        self.name = "Do Nothing"
        self.time = time
        self.controller_config = controller_config

    def get_state(self):
        return None

    def update(self, time, **kwargs):
        # Do nothing
        pass


class LoopController(SimulationComponent):
    """
    A controller class for the Pyloopkit algorithm
    """

    def __init__(self, time, loop_config, simulation_config):

        self.name = "PyLoopkit v0.1"
        self.time = time
        self.loop_config = copy.deepcopy(loop_config)
        self.recommendations = None

        # This is a hack to get this working quickly, it's too coupled to the input file format
        #  Future: Collect the information for the various simulation components
        self.simulation_config = copy.deepcopy(simulation_config)

        # The are not used at the moment, but will be once we decouple from simulation config.
        self.model = loop_config["model"]
        self.momentum_data_interval = loop_config["momentum_data_interval"]
        self.suspend_threshold = loop_config["suspend_threshold"]
        self.dynamic_carb_absorption_enabled = loop_config[
            "dynamic_carb_absorption_enabled"
        ]
        self.retrospective_correction_integration_interval = loop_config[
            "retrospective_correction_integration_interval"
        ]
        self.recency_interval = loop_config["recency_interval"]
        self.retrospective_correction_grouping_interval = loop_config[
            "retrospective_correction_grouping_interval"
        ]
        self.rate_rounder = loop_config["rate_rounder"]
        self.insulin_delay = loop_config["insulin_delay"]
        self.carb_delay = loop_config["carb_delay"]
        self.default_absorption_times = loop_config["default_absorption_times"]
        self.max_basal_rate = loop_config["max_basal_rate"]
        self.max_bolus = loop_config["max_bolus"]
        self.retrospective_correction_enabled = loop_config[
            "retrospective_correction_enabled"
        ]

        self.ctr = -5  # TODO remove once we feel refactor is good

    def get_state(self):

        # TODO: make this a class with convenience functions
        return self.recommendations

    def prepare_inputs(self, virtual_patient):
        """
        Collect inputs to the loop update call for the current time.

        Note: TODO: MVP needs to conform to the current pyloopkit interface which
                needs a lot of info. In the future, expose pyloopkit interface
                that takes minimal state info for computing at time

        Parameters
        ----------
        virtual_patient: Virtual Pa

        Returns
        -------

        """
        glucose_dates, glucose_values = virtual_patient.bg_history.get_loop_format()
        loop_inputs_dict = copy.deepcopy(self.simulation_config)
        loop_update_dict = {
            "time_to_calculate_at": self.time,
            "glucose_dates": glucose_dates,
            "glucose_values": glucose_values,
        }
        loop_inputs_dict.update(loop_update_dict)

        return loop_inputs_dict

    def update(self, time, **kwargs):
        """
        Using the virtual patient state, get the next action and apply it to patient,
        e.g. via pump.
        """
        self.time = time

        virtual_patient = kwargs["virtual_patient"]

        loop_inputs_dict = self.prepare_inputs(virtual_patient)

        # TODO remove once we feel refactor is good
        # Debugging Code for refactor
        # import os
        # from src.utils import findDiff
        # save_dir = "/Users/csummers/tmp"
        # in_fp = os.path.join(save_dir, "tmp_inputs_{}.pk".format(self.ctr))
        # other_inputs = pk.load(open(in_fp, "rb"))
        # print(findDiff(loop_inputs_dict, other_inputs))
        # assert other_inputs == loop_inputs_dict
        # out_fp = os.path.join(save_dir, "tmp_outputs_{}.pk".format(self.ctr))
        # other_outputs = pk.load(open(out_fp, "rb"))

        loop_algorithm_output = update(loop_inputs_dict)
        loop_algorithm_output.get("recommended_temp_basal")

        # TODO remove once we feel refactor is good
        # assert other_outputs == loop_algorithm_output
        # self.ctr += 5

        self.modulate_temp_basal(virtual_patient, loop_algorithm_output)
        self.recommendations = loop_algorithm_output

    def modulate_temp_basal(self, virtual_patient, loop_algorithm_output):
        """
        Set temp basal on the virtual patient's pump.

        Parameters
        ----------
        virtual_patient
        loop_algorithm_output
        """

        # Update the virtual_patient with any recommendations from loop
        if loop_algorithm_output.get("recommended_temp_basal") is not None:
            loop_temp_basal, duration = loop_algorithm_output.get(
                "recommended_temp_basal"
            )
            virtual_patient.pump.set_temp_basal(loop_temp_basal, "U")
            self.simulation_config["dose_values"].append(
                virtual_patient.pump.active_temp_basal.value
            )
        else:
            # If no recommendations, set a temp basal to the scheduled basal rate
            scheduled_basal_rate = virtual_patient.pump.get_state().scheduled_basal_rate
            virtual_patient.pump.set_temp_basal(scheduled_basal_rate.value, "U")
            self.simulation_config["dose_values"].append(
                virtual_patient.pump.get_state().scheduled_basal_rate.value
            )

        # Append dose info to simulation config.
        self.simulation_config["dose_types"].append(DoseType.tempbasal)
        self.simulation_config["dose_start_times"].append(self.time)

        next_time = self.time + datetime.timedelta(minutes=5)
        self.simulation_config["dose_end_times"].append(
            next_time
        )  # TODO: is this supposed to be 5 or 30 minutes?


class LoopControllerDisconnector(LoopController):
    """
    Loop controller that probabilistically loses connection disallowing
    setting of temp basals.
    """

    def __init__(self, time, loop_config, simulation_config, connect_prob):
        super().__init__(time, loop_config, simulation_config)

        self.name = "PyLoopkit v0.1, P(Connect)={}".format(connect_prob)
        self.original_time = copy.copy(time)
        self.connect_prob = connect_prob

    def is_connected(self):
        """
        Determine probabilistically if Loop is connected

        Returns
        -------
        bool
        """
        is_connected = False
        u = np.random.random()
        if u < self.connect_prob:
            is_connected = True

        return is_connected

    def update(self, time, **kwargs):
        """
        Update the state of the controller and do actions.

        Parameters
        ----------
        time: datetime
        kwargs: VirtualPatient
        """

        self.time = time

        if self.is_connected():

            virtual_patient = kwargs["virtual_patient"]

            loop_inputs_dict = self.prepare_inputs(virtual_patient)
            loop_algorithm_output = update(loop_inputs_dict)
            loop_algorithm_output.get("recommended_temp_basal")

            self.modulate_temp_basal(virtual_patient, loop_algorithm_output)
            self.recommendations = loop_algorithm_output

        else:
            # Disconnected. Do Nothing.
            pass
