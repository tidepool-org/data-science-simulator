__author__ = "Cameron Summers"

"""
Classes for simulation specific code.
"""

import multiprocessing
import copy
import datetime
import pandas as pd
import copy
from numpy.random import RandomState


from tidepool_data_science_simulator.models.measures import Bolus, Carb


class SimulationComponent(object):
    """
    A class with abstract and convenience methods for use in the simulation.
    """

    def get_state(self):
        raise NotImplementedError

    def update(self, time, **kwargs):
        raise NotImplementedError

    def get_time_delta_minutes(self, end_time):
        tdelta = end_time - self.time
        return tdelta.total_seconds() / 60


class SimulationState(object):
    """
    A class for holding the state of the simulation at any given time.
    """

    def __init__(self, patient_state, controller_state, randint):
        """
        Parameters
        ----------
        patient_state: VirtualPatientState
        controller_state
        """

        self.patient_state = patient_state
        self.controller_state = controller_state
        self.randint = randint

    def __repr__(self):

        return "BG: {:.2f}, IOB: {:.2f} Temp Basal: {}".format(
            self.patient_state.bg,
            self.patient_state.iob,
            self.patient_state.pump_state.temp_basal_rate,
        )


class Simulation(multiprocessing.Process):
    """
    A class that organizes the elements of the simulation through time and
    tracks results. Separation of Concerns: This class owns time tracking and
    member objects should be using minimal time logic, e.g. indexing historical
    or future information.
    """

    def __init__(
        self,
        time,
        duration_hrs,
        virtual_patient,
        controller,
        sim_id,
        multiprocess=False,
        random_state=None,
    ):

        # To enable multiprocessing
        super().__init__()
        self.queue = multiprocessing.Queue()
        self.multiprocess = multiprocess

        self.random_state = random_state
        if random_state is None:
            self.random_state = RandomState(0)

        self.sim_id = sim_id
        self.start_time = copy.deepcopy(time)
        self.time = time

        self.duration_hrs = duration_hrs
        self.virtual_patient = virtual_patient
        self.controller = controller

        self.simulation_results = dict()

        # Get things setup for t=0
        self.init()

    def init(self):
        """
        Initialize the simulation
        """
        # Setup steady state basal and t0 glucose
        self.virtual_patient.init()

        # Set any temp basals at t=0
        self.controller.update(self.time, virtual_patient=self.virtual_patient)

        # Store info at t=0
        self.store_state()

    def update(self, time):
        """
        Main feedback loop between patient and controller.

        Parameters
        ----------
        time: datetime
        """
        # Set patient state at time from prediction at time - 1
        self.virtual_patient.update(time)

        # Get and set on patient the next action from controller,
        #   e.g. temp basal, at time
        self.controller.update(time, virtual_patient=self.virtual_patient)

    def step(self):
        """
        Move the simulation time forward one step, which is 5 minutes.
        """
        next_time = self.time + datetime.timedelta(minutes=5)

        self.time = next_time
        self.update(next_time)

    def run(self, early_stop_datetime=None):
        """
        Run the simulation until it's finished.

        Parameters
        ----------
        early_stop_datetime: datetime
            Optional stop time for the simulation.
        """

        while not (self.is_finished() or early_stop_datetime == self.time):
            self.step()
            self.store_state()

        if self.multiprocess:
            self.queue.put(self.get_results_df())

        return self.simulation_results

    def store_state(self):
        """
        Store the current state of the simulation in the results.
        """
        self.simulation_results[self.time] = SimulationState(
            patient_state=self.virtual_patient.get_state(),
            controller_state=self.controller.get_state(),
            randint=self.random_state.randint(0, 1e6)
        )

    def is_finished(self):
        """
        Determines if the simulation has finished running.

        Returns
        -------
        bool:
            True if the simulation has passed the specified length
        """

        seconds_passed = (self.time - self.start_time).total_seconds()
        hours_frac_passed = seconds_passed / 3600.0

        return hours_frac_passed >= self.duration_hrs

    def get_results_df(self):
        """
        Get results as a dataframe object.

        Returns
        -------
        pd.DataFrame
            The time series result of the simulation
        """
        data = []
        for time, simulation_state in self.simulation_results.items():

            # Patient stuff
            true_bolus = simulation_state.patient_state.bolus
            if true_bolus is None:
                true_bolus = Bolus(0, "U")

            true_carb = simulation_state.patient_state.carb
            if true_carb is None:
                true_carb = Carb(0, "g", 0)

            # Pump stuff
            pump_state = simulation_state.patient_state.pump_state
            temp_basal_value = None
            temp_basal_time_remaining = None
            pump_sbr = None
            pump_isf = None
            pump_cir = None
            delivered_basal_insulin = None
            undelivered_basal_insulin = None
            if pump_state is not None:
                temp_basal_value = pump_state.get_temp_basal_rate_value(
                    default=None
                )
                temp_basal_time_remaining = pump_state.get_temp_basal_minutes_left(
                    time
                )
                pump_sbr = pump_state.scheduled_basal_rate
                pump_isf = pump_state.scheduled_insulin_sensitivity_factor
                pump_cir = pump_state.scheduled_carb_insulin_ratio
                delivered_basal_insulin = pump_state.delivered_basal_insulin
                undelivered_basal_insulin = pump_state.undelivered_basal_insulin

                reported_bolus = pump_state.bolus
                if reported_bolus is None:
                    reported_bolus = Bolus(0, "U")

                reported_carb = pump_state.carb
                if reported_carb is None:
                    reported_carb = Carb(0, "g", 0)

            row = {
                "time": time,
                "bg": simulation_state.patient_state.bg,
                "bg_sensor": simulation_state.patient_state.sensor_bg,
                "iob": simulation_state.patient_state.iob,
                "temp_basal": temp_basal_value,
                "temp_basal_time_remaining": temp_basal_time_remaining,
                "sbr": simulation_state.patient_state.sbr.value,
                "cir": simulation_state.patient_state.cir.value,
                "isf": simulation_state.patient_state.isf.value,
                "pump_sbr": pump_sbr.value,
                "pump_isf": pump_isf.value,
                "pump_cir": pump_cir.value,
                "true_bolus": true_bolus.value,
                "true_carb_value": true_carb.value,
                "true_carb_duration": true_carb.duration_minutes,
                "reported_bolus": reported_bolus.value,
                "reported_carb_value": reported_carb.value,
                "reported_carb_duration": reported_carb.duration_minutes,
                "delivered_basal_insulin": delivered_basal_insulin,
                "undelivered_basal_insulin": undelivered_basal_insulin,
                "randint": simulation_state.randint
            }

            # Controller stuff
            for horizon_minutes in [15, 30, 60, 90]:
                time_horizon_ago = time - datetime.timedelta(minutes=horizon_minutes)
                if time_horizon_ago in self.simulation_results and hasattr(self.simulation_results[time_horizon_ago].controller_state,
                               "pyloopkit_recommendations"):
                    predicted_bgs_from_horizon_ago = self.simulation_results[time_horizon_ago].controller_state.pyloopkit_recommendations.get("predicted_glucose_values")

                    try:
                        predicted_bg_at_horizon = predicted_bgs_from_horizon_ago[int(horizon_minutes / 5)]
                        loop_prediction_mae_true = abs(predicted_bg_at_horizon - simulation_state.patient_state.bg)
                        loop_prediction_mae_sensor = abs(predicted_bg_at_horizon - simulation_state.patient_state.sensor_bg)
                        row.update({
                            "loop_prediction_abs_error_{}_ago_true".format(horizon_minutes): loop_prediction_mae_true,
                            "loop_prediction_abs_error_{}_ago_sensor".format(horizon_minutes): loop_prediction_mae_sensor,
                        })
                    except:
                        pass

            data.append(row)

        df = pd.DataFrame(data)
        df.set_index("time", inplace=True)
        return df

    def get_info_stateless(self):

        stateless_info = {
            "sim_id": self.sim_id,
            "duration_hrs": self.duration_hrs,
            "start_time": self.start_time.isoformat(),
            "multiprocess": self.multiprocess,
            "patient": self.virtual_patient.get_info_stateless(),
            "controller": self.controller.get_info_stateless()
        }
        return stateless_info


class SettingSchedule24Hr(SimulationComponent):
    """
    A class for settings schedules on a 24 hour cycle.
    """
    def __init__(self, time, name, start_times=None, values=None, duration_minutes=None):
        """
        Parameters
        ----------
        time: datetime
            Current time

        name: str
            Setting name

        start_times: list
            List of datetime.time objects

        values: list
            List of objects

        duration_minutes: list
            List of ints
        """
        self.time = time
        self.name = name
        self.schedule_durations = {}
        self.schedule = {}

        # All the same length
        assert (
            len(start_times) + len(values) + len(duration_minutes)
            == len(start_times) * 3
        )

        for start_time, value, duration_minutes in zip(
            start_times, values, duration_minutes
        ):

            start_datetime = datetime.datetime.combine(
                datetime.datetime.today(), start_time
            )
            end_datetime = (
                start_datetime
                + datetime.timedelta(minutes=duration_minutes)
                - datetime.timedelta(seconds=1)
            )
            end_time = end_datetime.time()
            self.schedule[(start_time, end_time)] = value
            self.schedule_durations[(start_time, end_time)] = duration_minutes

    def get_state(self):
        """
        Get the value object at the current time, e.g. carb ratio or target range

        Returns
        -------
        object
            The object at the current time
        """

        for (start_time, end_time), value in self.schedule.items():
            current_time = self.time.time()
            if start_time <= current_time <= end_time:
                return value

        raise Exception("Could not find setting for time {}".format(self.time))

    def validate_schedule(self):
        """
        Ensure there are no overlapping segments,
         units are consistent,
         datetimes have no gaps for 24 hrs
        """
        raise NotImplementedError

    def update(self, time, **kwargs):
        """
        Set the new time.

        Parameters
        ----------
        time: datetime
        """

        self.time = time

    def get_loop_inputs(self):

        values = []
        start_times = []
        end_times = []
        for (start_time, end_time), setting in self.schedule.items():
            values.append(setting.value)
            start_times.append(start_time)
            end_times.append(end_time)

        return values, start_times, end_times

    def get_info_stateless(self):

        stateless_info = {
            "schedule": [
                {
                    "setting": str(setting),
                    "start_time": start_time.strftime('%H:%M:%S'),
                    "end_time": end_time.strftime('%H:%M:%S')
                }
                for (start_time, end_time), setting in self.schedule.items()
            ]
        }
        return stateless_info


class BasalSchedule24hr(SettingSchedule24Hr):

    def __init__(self, time, start_times, values, duration_minutes):
        super().__init__(time, "Basal Rate Schedule", start_times, values, duration_minutes)

    def get_loop_inputs(self):

        values = []
        start_times = []
        durations = []
        for (start_time, end_time), setting in self.schedule.items():
            values.append(setting.value)
            start_times.append(start_time)
            durations.append(self.schedule_durations[(start_time, end_time)])

        return values, start_times, durations


class TargetRangeSchedule24hr(SettingSchedule24Hr):

    def __init__(self, time, start_times, values, duration_minutes):
        super().__init__(time, "Target Range Schedule", start_times, values, duration_minutes)

    def get_loop_inputs(self):

        min_values = []
        max_values = []
        start_times = []
        end_times = []
        for (start_time, end_time), target_range in self.schedule.items():
            min_values.append(target_range.min_value)
            max_values.append(target_range.max_value)
            start_times.append(start_time)
            end_times.append(end_time)

        return min_values, max_values, start_times, end_times


class SingleSettingSchedule24Hr(SimulationComponent):
    """
    Convenience class for creating single value setting schedules.
    """
    def __init__(self, time, name, setting):

        self.time = time
        self.name = name
        self.setting = setting

    def get_state(self):

        return self.setting

    def update(self, time, **kwargs):

        pass  # stateless

    def get_loop_inputs(self, use_durations):

        values = [self.setting.value]
        start_times = [datetime.time(0, 0, 0)]

        end_times = [datetime.time(23, 59, 59)]
        durations = [1440]

        if use_durations:
            end = durations
        else:
            end = end_times

        return values, start_times, end
