__author__ = "Cameron Summers"

import pickle as pk
import datetime
import copy
import numpy as np
import pandas as pd

from tidepool_data_science_simulator.models.simulation import SimulationComponent
from tidepool_data_science_simulator.models.measures import GlucoseTrace, Bolus, TempBasal
from pytz import utc
from pyloopkit.loop_data_manager import update


class BaseControllerClass(SimulationComponent):

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @classmethod
    def get_classname(cls):
        return cls.__name__


class DoNothingController(BaseControllerClass):
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


class LoopController(BaseControllerClass):
    """
    A controller class for the Pyloopkit algorithm
    """
    def __repr__(self):
        return "PyLoopkit_v0.1"

    def __str__(self):
        return "PyLoopkit_v0.1"

    def __init__(self, time, controller_config):

        self.name = "PyLoopkit v0.1"
        self.time = time
        self.controller_config = copy.deepcopy(controller_config)
        self.recommendations = None

        self.bolus_event_timeline = controller_config.bolus_event_timeline
        self.temp_basal_event_timeline = controller_config.temp_basal_event_timeline
        self.carb_event_timeline = controller_config.carb_event_timeline

        self.num_hours_history = 8  # how many hours of recent events to pass to Loop

    def get_state(self):

        return ControllerState(
            pyloopkit_recommendations=self.recommendations
        )

    def get_dose_event_timelines(self, virtual_patient):
        """
        Retrieve dose data to pass to Pyloopkit.

        Parameters
        ----------
        virtual_patient

        Returns
        -------
        (BolusTimeline, CarbTimeline, TempBasalTimeline)
        """
        # Loop data store for doses is technically on pump in simulator.
        bolus_event_timeline, carb_event_timeline, temp_basal_event_timeline = \
            virtual_patient.get_pump_events()

        # Merge Controller-specific data store, e.g. user inputs manual bolus while pump is inactive
        bolus_event_timeline.merge_timeline(self.bolus_event_timeline)
        carb_event_timeline.merge_timeline(self.carb_event_timeline)
        temp_basal_event_timeline.merge_timeline(self.temp_basal_event_timeline)

        return bolus_event_timeline, carb_event_timeline, temp_basal_event_timeline

    def prepare_inputs(self, virtual_patient):
        """
        Collect inputs to the loop update call for the current time.

        Parameters
        ----------
        virtual_patient:

        Returns
        -------
        dict
            Inputs for Pyloopkit algo
        """
        glucose_dates, glucose_values = virtual_patient.sensor.get_loop_inputs()

        bolus_event_timeline, carb_event_timeline, temp_basal_event_timeline = self.get_dose_event_timelines(virtual_patient)

        bolus_dose_types, bolus_dose_values, bolus_start_times, bolus_end_times, bolus_delivered_units = \
            bolus_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        temp_basal_dose_types, temp_basal_dose_values, temp_basal_start_times, temp_basal_end_times, temp_basal_delivered_units = \
            temp_basal_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        carb_values, carb_start_times, carb_durations = \
            carb_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        basal_rate_values, basal_rate_start_times, basal_rate_durations = \
            virtual_patient.pump.pump_config.basal_schedule.get_loop_inputs()

        isf_values, isf_start_times, isf_end_times = \
            virtual_patient.pump.pump_config.insulin_sensitivity_schedule.get_loop_inputs()

        cir_values, cir_start_times, cir_durations = \
            virtual_patient.pump.pump_config.carb_ratio_schedule.get_loop_inputs()

        tr_min_values, tr_max_values, tr_start_times, tr_end_times = \
            virtual_patient.pump.pump_config.target_range_schedule.get_loop_inputs()

        last_temp_basal = None
        if len(temp_basal_dose_types) > 0:
            last_temp_basal = [temp_basal_dose_types[-1], temp_basal_start_times[-1], temp_basal_end_times[-1], temp_basal_dose_values[-1]]

        loop_inputs_dict = {
            "time_to_calculate_at": self.time,
            "glucose_dates": glucose_dates,
            "glucose_values": glucose_values,

            "dose_types": bolus_dose_types + temp_basal_dose_types,
            "dose_values": bolus_dose_values + temp_basal_dose_values,
            "dose_start_times": bolus_start_times + temp_basal_start_times,
            "dose_end_times": bolus_end_times + temp_basal_end_times,
            "dose_delivered_units": bolus_delivered_units + temp_basal_delivered_units,

            "carb_dates": carb_start_times,
            "carb_values": carb_values,
            "carb_absorption_times": carb_durations,

            "basal_rate_values": basal_rate_values,
            "basal_rate_minutes": basal_rate_durations,
            "basal_rate_start_times": basal_rate_start_times,

            "carb_ratio_values": cir_values,
            "carb_ratio_start_times": cir_start_times,

            "sensitivity_ratio_values": isf_values,
            "sensitivity_ratio_start_times": isf_start_times,
            "sensitivity_ratio_end_times": isf_end_times,

            "target_range_minimum_values": tr_min_values,
            "target_range_maximum_values": tr_max_values,
            "target_range_start_times": tr_start_times,
            "target_range_end_times": tr_end_times,

            "last_temporary_basal": last_temp_basal,

            "settings_dictionary": self.controller_config.controller_settings
        }

        return loop_inputs_dict

    def update(self, time, **kwargs):
        """
        Using the virtual patient state, get the next action and apply it to patient,
        e.g. via pump.
        """
        self.time = time
        virtual_patient = kwargs["virtual_patient"]

        if virtual_patient.pump is not None:
            loop_inputs_dict = self.prepare_inputs(virtual_patient)
            loop_algorithm_output = update(loop_inputs_dict)

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
            if temp_basal_rec.scheduled_duration_minutes == 0 and temp_basal_rec.value == 0:
                # In pyloopkit this is a "cancel"
                virtual_patient.pump.deactivate_temp_basal()
            else:
                self.modulate_temp_basal(virtual_patient, temp_basal_rec)
        else:
            pass  # no recommendations

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
        Add the accepted bolus event to the virtual patient's timeline to
        be applied at the next update.

        Parameters
        ----------
        virtual_patient
        bolus
        """
        # Note: due to having the patient go "first" in the update loop we have
        #       to set the bolus at the next update time.
        next_time = self.time + datetime.timedelta(minutes=5)

        virtual_patient.add_event(next_time, bolus)

    def modulate_temp_basal(self, virtual_patient, temp_basal):
        """
        Set temp basal on the virtual patient's pump.

        Parameters
        ----------
        virtual_patient
        temp_basal
        """
        virtual_patient.pump.set_temp_basal(temp_basal)


class LoopControllerDisconnector(LoopController):
    """
    Loop controller that probabilistically loses connection disallowing
    setting of temp basals.
    """

    def __init__(self, time, controller_config, connect_prob):
        super().__init__(time, controller_config)

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


class LoopReplay(LoopController):

    def __init__(self, time, controller_config):
        super().__init__(time, controller_config)

        self.name = "Jaeb Replay in PyLoopkit v0.1"
        self.active_temp_basal = None

    def get_dose_event_timelines(self, virtual_patient):
        """
        Retrieve dose data to pass to Pyloopkit without merging in controller timelines.

        Parameters
        ----------
        virtual_patient

        Returns
        -------
        (BolusTimeline, CarbTimeline, TempBasalTimeline)
        """
        # Loop data store for doses is technically on pump in simulator.
        bolus_event_timeline, carb_event_timeline, temp_basal_event_timeline = \
            virtual_patient.get_pump_events()

        return bolus_event_timeline, carb_event_timeline, temp_basal_event_timeline

    def prepare_inputs(self, virtual_patient):
        """
        Collect inputs to the loop update call for the current time.

        Parameters
        ----------
        virtual_patient:

        Returns
        -------
        dict
            Inputs for Pyloopkit algo
        """
        glucose_dates_with_timezone, glucose_values = virtual_patient.sensor.get_loop_inputs()
        glucose_dates = [date.replace(tzinfo=None) for date in glucose_dates_with_timezone]

        bolus_event_timeline, carb_event_timeline, temp_basal_event_timeline = self.get_dose_event_timelines(virtual_patient)

        bolus_dose_types, bolus_dose_values, bolus_start_times, bolus_end_times, bolus_delivered_units = \
            bolus_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        temp_basal_dose_types, temp_basal_dose_values, temp_basal_start_times, temp_basal_end_times, temp_basal_delivered_units = \
            temp_basal_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        time_naive = self.time.replace(tzinfo=None)

        shouldbedeliveredunits = abs((temp_basal_dose_values[-1] / 12) * \
                int(int((time_naive - temp_basal_start_times[-1]).total_seconds()/60) / 5) - \
                temp_basal_delivered_units[-1])

        if temp_basal_end_times[-1] > time_naive and shouldbedeliveredunits > 0.05:
            temp_basal_delivered_units[-1] = (temp_basal_dose_values[-1] / 12) * \
                int(int((time_naive - temp_basal_start_times[-1]).total_seconds()/60) / 5)

        if ((temp_basal_end_times[-1] - temp_basal_start_times[-1]).total_seconds() / 60) % 30 != 0 and \
                temp_basal_end_times[-1] == time_naive:
            temp_basal_end_times[-1] = temp_basal_start_times[-1] + \
                                       datetime.timedelta(
                                           minutes=(((temp_basal_end_times[-1] - temp_basal_start_times[
                                               -1]).total_seconds() / 1800) + 1) * 30)

        carb_values, carb_start_times, carb_durations = \
            carb_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        basal_rate_values, basal_rate_start_times, basal_rate_durations = \
            virtual_patient.pump.pump_config.basal_schedule.get_loop_inputs()

        isf_values, isf_start_times, isf_end_times = \
            virtual_patient.pump.pump_config.insulin_sensitivity_schedule.get_loop_inputs()

        cir_values, cir_start_times, cir_durations = \
            virtual_patient.pump.pump_config.carb_ratio_schedule.get_loop_inputs()

        tr_min_values, tr_max_values, tr_start_times, tr_end_times = \
            virtual_patient.pump.pump_config.target_range_schedule.get_loop_inputs()

        last_temp_basal = None
        if len(temp_basal_dose_types) > 0:
            last_temp_basal = [temp_basal_dose_types[-1], temp_basal_start_times[-1], temp_basal_end_times[-1], temp_basal_dose_values[-1]]

        loop_inputs_dict = {
            "time_to_calculate_at": time_naive,
            "glucose_dates": glucose_dates,
            "glucose_values": glucose_values,

            "dose_types": bolus_dose_types + temp_basal_dose_types,
            "dose_values": bolus_dose_values + temp_basal_dose_values,
            "dose_start_times": bolus_start_times + temp_basal_start_times,
            "dose_end_times": bolus_end_times + temp_basal_end_times,
            "dose_delivered_units": bolus_delivered_units + temp_basal_delivered_units,

            "carb_dates": carb_start_times,
            "carb_values": carb_values,
            "carb_absorption_times": carb_durations,

            "basal_rate_values": basal_rate_values,
            "basal_rate_minutes": basal_rate_durations,
            "basal_rate_start_times": basal_rate_start_times,

            "carb_ratio_values": cir_values,
            "carb_ratio_start_times": cir_start_times,

            "sensitivity_ratio_values": isf_values,
            "sensitivity_ratio_start_times": isf_start_times,
            "sensitivity_ratio_end_times": isf_end_times,

            "target_range_minimum_values": tr_min_values,
            "target_range_maximum_values": tr_max_values,
            "target_range_start_times": tr_start_times,
            "target_range_end_times": tr_end_times,

            "last_temporary_basal": last_temp_basal,

            "settings_dictionary": self.controller_config.controller_settings
        }

        return loop_inputs_dict

    def apply_loop_recommendations(self, virtual_patient, loop_algorithm_output):
        """
        Instead of applying loop recommendations, save them to the controller timelines.

        Parameters
        __________
        virtual_patient
        loop_algorithm_output

        """
        bolus_value = None
        temp_basal_value = None
        bolus_rec = self.get_recommended_bolus(loop_algorithm_output)
        temp_basal_rec = self.get_recommended_temp_basal(loop_algorithm_output)

        if bolus_rec is not None and virtual_patient.does_accept_bolus_recommendation(bolus_rec):
            self.bolus_event_timeline.add_event(self.time, bolus_rec)
        elif temp_basal_rec is not None:
            if temp_basal_rec.scheduled_duration_minutes == 0 and temp_basal_rec.value == 0:
                # In pyloopkit this is a "cancel"
                if self.active_temp_basal is not None:
                    self.deactivate_recorded_temp_basal()
            else:
                self.record_temp_basal(virtual_patient, temp_basal_rec)
        else:
            pass  # no recommendations

        self.recommendations = loop_algorithm_output

    def deactivate_recorded_temp_basal(self):
        self.active_temp_basal.actual_end_time = self.time
        self.active_temp_basal.actual_duration_minutes = (self.time -
                                                          self.active_temp_basal.start_time).total_seconds() / 60
        self.active_temp_basal.active = False
        self.active_temp_basal = None

    def record_temp_basal(self, virtual_patient, temp_basal_rec):
        is_valid, message = virtual_patient.pump.is_valid_temp_basal(temp_basal_rec)

        if is_valid:
            if self.active_temp_basal is not None:
                self.deactivate_recorded_temp_basal()

            self.active_temp_basal = temp_basal_rec
            self.temp_basal_event_timeline.add_event(self.time, temp_basal_rec)

        else:
            raise ValueError("Temp basal request is invalid. {}".format(message))


class ControllerState(object):
    def __init__(self,
                 pyloopkit_recommendations
                 ):
        self.pyloopkit_recommendations = pyloopkit_recommendations
