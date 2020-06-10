__author__ = "Cameron Summers"

import pickle as pk
import datetime
import copy
import numpy as np

from tidepool_data_science_simulator.models.simulation import SimulationComponent, EventTimeline
from tidepool_data_science_simulator.models.measures import GlucoseTrace, Bolus, TempBasal

from pyloopkit.loop_data_manager import update


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

        self.bolus_event_timeline = loop_config.bolus_event_timeline
        self.temp_basal_event_timeline = loop_config.temp_basal_event_timeline
        self.carb_event_timeline = loop_config.carb_event_timeline

        self.num_hours_history = 6  # how many hours of recent events to pass to Loop

        # self.ctr = -5  # TODO remove once we feel refactor is good

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
        virtual_patient:

        Returns
        -------
        dict
            Inputs for Pyloopkit algo
        """
        glucose_dates, glucose_values = virtual_patient.bg_history.get_loop_inputs()
        loop_inputs_dict = copy.deepcopy(self.simulation_config)

        bolus_dose_types, bolus_dose_values, bolus_start_times, bolus_end_times = \
            self.bolus_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        temp_basal_dose_types, temp_basal_dose_values, temp_basal_start_times, temp_basal_end_times = \
            self.temp_basal_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        carb_values, carb_start_times, carb_durations = \
            self.carb_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        basal_rate_values, basal_rate_start_times, basal_rate_durations = \
            virtual_patient.pump.pump_config.basal_schedule.get_loop_inputs()

        # TODO NOW: add bolus and temp basal events
        loop_update_dict = {
            "time_to_calculate_at": self.time,
            "glucose_dates": glucose_dates,
            "glucose_values": glucose_values,
            "dose_types": bolus_dose_types + temp_basal_dose_types,
            "dose_values": bolus_dose_values + temp_basal_dose_values,
            "dose_start_times": bolus_start_times + temp_basal_start_times,
            "dose_end_times": bolus_end_times + temp_basal_end_times,
            "carb_dates": carb_start_times,
            "carb_values": carb_values,
            "carb_absorption_times": carb_durations,
            "basal_rate_values": basal_rate_values,
            "basal_rate_minutes": basal_rate_durations,
            "basal_rate_start_times": basal_rate_start_times
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

        # Loop knows about any events reported on pump.
        self.bolus_event_timeline = virtual_patient.pump.bolus_event_timeline
        self.carb_event_timeline = virtual_patient.pump.carb_event_timeline

        loop_inputs_dict = self.prepare_inputs(virtual_patient)

        # TODO remove once we feel refactor is good
        # Debugging Code for refactor
        # import os
        # from tidepool_data_science_simulator.utils import findDiff
        # save_dir = "/Users/csummers/tmp"
        # in_fp = os.path.join(save_dir, "tmp_inputs_{}.pk".format(self.ctr))
        # other_inputs = pk.load(open(in_fp, "rb"))
        # print(findDiff(loop_inputs_dict, other_inputs))
        # # assert other_inputs == loop_inputs_dict
        # out_fp = os.path.join(save_dir, "tmp_outputs_{}.pk".format(self.ctr))
        # other_outputs = pk.load(open(out_fp, "rb"))

        loop_algorithm_output = update(loop_inputs_dict)

        # TODO remove once we feel refactor is good
        # assert other_outputs == loop_algorithm_output
        # self.ctr += 5

        self.apply_loop_recommendations(virtual_patient, loop_algorithm_output)

    def apply_loop_recommendations(self, virtual_patient, loop_algorithm_output):
        """
        Apply the recommendations from the pyloopkit algo.

        Parameters
        ----------
        virtual_patient
        loop_algorithm_output
        """

        bolus_rec = self.get_recommended_bolus(loop_algorithm_output)
        temp_basal_rec = self.get_recommended_temp_basal(loop_algorithm_output)

        if bolus_rec is not None and virtual_patient.does_accept_bolus_recommendation(bolus_rec):
            self.set_bolus_recommendation_event(virtual_patient, bolus_rec)
        elif temp_basal_rec is not None:
            self.modulate_temp_basal(virtual_patient, temp_basal_rec)
        else:
            # Should only happen if temp basal rate recommendation is same as the scheduled
            # basal rate.
            virtual_patient.pump.deactivate_temp_basal()

        self.recommendations = loop_algorithm_output

    def get_recommended_bolus(self, loop_algorithm_output):
        """
        Extract bolus recommendation from pyloopkit recommendations output.

        Parameters
        ----------
        loop_algorithm_output: dict

        Returns
        -------
        Bolus
        """

        bolus = None
        bolus_value = loop_algorithm_output.get('recommended_bolus')[0]  # TODO: potential error here
        if bolus_value > 0:
            bolus = Bolus(bolus_value, "U")

        return bolus

    def get_recommended_temp_basal(self, loop_algorithm_output):
        """
        Extract temp basal from pyloopkit recommendations output.

        Parameters
        ----------
        loop_algorithm_output: dict

        Returns
        -------
        TempBasal
        """

        temp_basal = None
        if loop_algorithm_output.get("recommended_temp_basal") is not None:
            loop_temp_basal, duration = loop_algorithm_output.get("recommended_temp_basal")
            temp_basal = TempBasal(self.time, loop_temp_basal, duration, "U/hr")

        return temp_basal

    def set_bolus_recommendation_event(self, virtual_patient, bolus):
        """
        Add the accepted bolus event to the virtual patient's timeline.

        Parameters
        ----------
        virtual_patient
        bolus
        """

        # Add to patient timeline
        virtual_patient.bolus_event_timeline.add_event(self.time, bolus)

        # Log in pump, which Loop will read at update
        virtual_patient.pump.bolus_event_timeline.add_event(self.time, bolus)

    def modulate_temp_basal(self, virtual_patient, temp_basal):
        """
        Set temp basal on the virtual patient's pump.

        Parameters
        ----------
        virtual_patient
        temp_basal
        """
        virtual_patient.pump.set_temp_basal(temp_basal)

        # Log in loop timeline
        self.temp_basal_event_timeline.add_event(self.time, temp_basal)


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
            super().update(time, **kwargs)
