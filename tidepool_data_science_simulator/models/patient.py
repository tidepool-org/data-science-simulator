__author__ = "Cameron Summers"

import copy
import numpy as np
from numpy.random import RandomState
import datetime

from tidepool_data_science_simulator.models.simulation import SimulationComponent
from tidepool_data_science_simulator.makedata.scenario_parser import PatientConfig
from tidepool_data_science_simulator.models.measures import Carb, Bolus, ManualBolus
from tidepool_data_science_simulator.models.events import (
    MealModel, Action, ActionTimeline, BolusTimeline, CarbTimeline, TempBasalTimeline
)
from tidepool_data_science_simulator.utils import get_bernoulli_trial_uniform_step_prob

from tidepool_data_science_simulator.models.state import VirtualPatientState
# ============= Patient Stuff ===================


class VirtualPatient(SimulationComponent):
    """
    Patient class for simulation
    """

    def __init__(self, time, pump, sensor, metabolism_model, patient_config, random_state=None):
        """
        Parameters
        ----------
        time: datetime
            t=0
        pump: Pump
        sensor: Sensor
        metabolism_model: SimpleMetabolismModel
        patient_config: PatientConfig
        """

        # This time is expected to be t=0 for simulation
        self.time = time

        self.name = "Virtual Patient"

        self.random_state = random_state
        if random_state is None:
            self.random_state = RandomState(0)

        self.pump = pump
        self.sensor = sensor
        self.metabolism_model = metabolism_model

        self.patient_config = copy.deepcopy(patient_config)
        self._validate_config()

        self.set_random_values()

        self.bg_history = self.patient_config.glucose_history
        self.iob_history = []  # TODO: make trace obj, not list

        self.bg_prediction = None
        self.iob_prediction = None

        self.bg_current = None
        self.iob_current = None
        self.sbr_iob = None

        self.carb_event_timeline = patient_config.carb_event_timeline
        self.bolus_event_timeline = patient_config.bolus_event_timeline

        # TODO: prediction horizon should probably come from simple metabolism model
        prediction_horizon_hrs = 8
        self.num_prediction_steps = int(prediction_horizon_hrs * 60 / 5)

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def _validate_config(self):

        assert hasattr(self.patient_config, "recommendation_accept_prob"), "No recommendation_accept_prob in patient config"
        assert hasattr(self.patient_config, "glucose_history"), "No glucose_history in patient config"
        assert hasattr(self.patient_config, "basal_schedule"), "No basal_schedule in patient config"
        assert hasattr(self.patient_config, "carb_ratio_schedule"), "No carb_ratio_schedule in patient config"
        assert hasattr(self.patient_config, "carb_event_timeline"), "No carb_event_timeline in patient config"
        assert hasattr(self.patient_config, "bolus_event_timeline"), "No bolus_event_timeline in patient config"
        assert hasattr(self.patient_config, "action_timeline"), "No action_timeline in patient config"

    def init(self):
        """
        This is similar to the "initial scenario" setup in the original code.

        Establish conditions for t=0 and get the initial prediction
        """
        # Note: Initializing this class currently makes assumptions about the state at time,
        #       e.g. basal rate has been constant, bg is set and not predicted
        # TODO: Ideally set initial state based previous prediction_horizon_hrs up to t=0

        bg_time, self.bg_current = self.patient_config.glucose_history.get_last()
        assert bg_time == self.time

        # Scenario doesn't give iob info so derive from steady state pump basal rate
        iob_steady_state = self.get_steady_state_basal_iob(
            self.pump.get_state().scheduled_basal_rate
        )

        # Only use iob up to t=time-1 since we account for basal insulin at t=0
        self.iob_init = np.append(iob_steady_state[1:], 0)
        self.iob_current = self.iob_init[0]

        self.sbr_iob = iob_steady_state[0]

        # Initialize the pump for t=0
        self.pump.init()

        # Establish prediction based on events at t0
        self.predict()

        # Set current values at t=0
        self.bg_current = self.bg_prediction[0]
        self.iob_current = self.iob_prediction[0]

    def get_state(self):
        """
        Get the current state of the patient.

        Returns
        -------
        VirtualPatientState
        """
        sensor_state = self.sensor.get_state()  # todo: put whole state below

        pump_state = None
        if self.pump is not None:
            pump_state = self.pump.get_state()

        patient_state = VirtualPatientState(
            bg=self.bg_current,
            bg_prediction=self.bg_prediction,
            sensor_bg=sensor_state.sensor_bg,
            sensor_bg_prediction=sensor_state.sensor_bg_prediction,
            iob=self.iob_current,
            iob_prediction=self.iob_prediction,
            sbr=self.patient_config.basal_schedule.get_state(),
            isf=self.patient_config.insulin_sensitivity_schedule.get_state(),
            cir=self.patient_config.carb_ratio_schedule.get_state(),
            pump_state=pump_state,
            bolus=self.bolus_event_timeline.get_event(self.time),
            carb=self.carb_event_timeline.get_event(self.time),
            actions=self.patient_config.action_timeline.get_event(self.time)
        )

        return patient_state

    def get_info_stateless(self):

        stateless_info = {
            "name": self.name,
            "sensor": self.sensor.get_info_stateless(),
            "pump": self.pump.get_info_stateless(),
            "config": self.patient_config.get_info_stateless()
        }
        return stateless_info

    def get_actions(self):
        """
        Get actions that the user has performed which are not boluses or meals.
        Possible actions: set change, battery change, user deletes all data, exercise.

        Returns
        -------
        [Action]
        """
        actions = self.patient_config.action_timeline.get_event(self.time)

        return actions

    def get_user_inputs(self):
        """
        Get user inputs (boluses, carbs) that directly affect Loop calculations.

        Returns
        _______
        (Insulin Event, Carb Event)
        """
        # Get boluses at time
        bolus = self.bolus_event_timeline.get_event(self.time)

        # Get carbs at time
        carb = self.carb_event_timeline.get_event(self.time)

        return bolus, carb

    def update(self, time, **kwargs):
        """
        Predict the future state of the patient given the current state.

        Order is important:
        Pump: Get the insulin delivery that affects the patient prediction
        -> Patient: Update bg prediction based on events from pump and elsewhere
        -> Sensor: Update sensor based on bg from patient

        Parameters
        ----------
        time: datetime
        """
        self.time = time

        self.patient_config.basal_schedule.update(time)
        self.patient_config.insulin_sensitivity_schedule.update(time)
        self.patient_config.carb_ratio_schedule.update(time)

        if self.pump is not None:
            self.pump.update(time)

        # TODO: Adding in framework for actions other than boluses and carbs
        user_action = self.get_actions()
        if user_action is not None:
            user_action.execute(self)

        self.predict()
        self.update_from_prediction(time)

        # Update sensor after patient prediction to use the current bg
        self.sensor.update(time,
                           patient_true_bg=self.bg_current,
                           patient_true_bg_prediction=self.bg_prediction
                           )

    def update_from_prediction(self, time):
        """
        Set the patient state to the result of the metabolism simulation.
        """
        self.bg_current = self.bg_prediction[0]
        self.iob_current = self.iob_prediction[0]

        self.bg_history.append(time, self.bg_current)
        self.iob_history.append(self.iob_current)

    def get_steady_state_basal_iob(self, basal_rate):
        """
        Get insulin on board prediction for a given basal rate assuming
        a steady state of basal insulin prior.

        Parameters
        ----------
        basal_rate: BasalRate
            The assumed steady state basal rate

        Returns
        -------
        np.array
            The insulin on board from t=0 to t=prediction_horizon_hrs assuming a steady state
            basal rate since t=-prediction_horizon_hrs
        """
        metabolism_model_instance = self.instantiate_metabolism_model()
        return metabolism_model_instance.get_iob_from_sbr(basal_rate.value)

    def get_basal_insulin_amount_since_update(self):
        """
        Get the absolute and relative insulin amount delivered since the last update

        Returns
        -------
        (float, float)
        """
        abs_insulin_amount = 0
        if self.pump is not None:
            abs_insulin_amount = self.pump.deliver_basal(self.pump.basal_insulin_delivered_last_update)

        patient_egp_basal_value_equivalent = self.patient_config.basal_schedule.get_state().get_insulin_in_interval()
        rel_insulin_amount = abs_insulin_amount - patient_egp_basal_value_equivalent

        return abs_insulin_amount, rel_insulin_amount

    def get_total_insulin_and_carb_amounts(self):
        """
        Get the basal/bolus insulin and carb amounts at current time.

        Returns
        -------
        (float, float, float)
        """
        abs_insulin_amount, rel_insulin_amount = self.get_basal_insulin_amount_since_update()

        bolus, carb = self.get_user_inputs()

        # accept_recommendation boluses will be handled after forecast generation
        if bolus is not None and bolus.value != 'accept_recommendation':
            delivered_bolus = self.deliver_bolus(bolus)
            rel_insulin_amount += delivered_bolus.value
            abs_insulin_amount += delivered_bolus.value

        carb_amount = 0
        if carb is not None:
            carb_amount = carb.value

        return abs_insulin_amount, rel_insulin_amount, carb_amount

    def deliver_bolus(self, bolus):
        """
        Deliver a bolus either manually or via the pump. It's expected that the incoming
        bolus is the correct type for the circumstance, e.g. a manual bolus if the
        user is not wearing a pump

        Parameters
        ----------
        bolus

        Returns
        -------
        Bolus
        """

        if isinstance(bolus, ManualBolus):
            bolus = self.deliver_manual_bolus(bolus)
        else:
            bolus = self.pump.deliver_bolus(bolus)

        return bolus

    def deliver_manual_bolus(self, bolus):
        """
        Deliver a manual bolus. Override to model differently.

        Parameters
        ----------
        bolus

        Returns
        -------
        ManualBolus
        """
        if not isinstance(bolus, ManualBolus):
            raise TypeError("Cannot manually deliver a bolus not of ManualBolus type.")

        return bolus

    def get_pump_events(self):
        """
        Read events off the pump.

        Returns
        -------
        (BolusTimeline, CarbTimeline, TempBasalTimeline)
        """
        pump_bolus_event_timeline = BolusTimeline()
        pump_carb_event_timeline = CarbTimeline()
        pump_temp_basal_event_timeline = TempBasalTimeline()

        if self.pump is not None:
            pump_bolus_event_timeline = self.pump.bolus_event_timeline
            pump_carb_event_timeline = self.pump.carb_event_timeline
            pump_temp_basal_event_timeline = self.pump.temp_basal_event_timeline

        return pump_bolus_event_timeline, pump_carb_event_timeline, pump_temp_basal_event_timeline

    def add_event(self, time_of_event, event):

        # Add event to patient and pump timeline
        if isinstance(event, Bolus):
            self.bolus_event_timeline.add_event(time_of_event, event)
            self.pump.bolus_event_timeline.add_event(time_of_event, event)
        elif isinstance(event, Carb):
            self.bolus_event_timeline.add_event(time_of_event, event)
            self.pump.bolus_event_timeline.add_event(time_of_event, event)
        else:
            raise Exception("Unsupported event to add.")

    def predict(self):
        """
        Using state at t=time, ie
         1. Insulin delivered via basal since the last update, between t=time-1 and t=time
         2. Insulin administered as a bolus at t=time
         3. Carbs administered at t=time

         predict the horizon for bg and insulin on board
        """
        abs_insulin_amount, rel_insulin_amount, carb_amount = self.get_total_insulin_and_carb_amounts()

        # Initialize zero change
        combined_delta_bg_pred = np.zeros(self.num_prediction_steps)
        iob_pred = np.zeros(self.num_prediction_steps)

        # Apply insulin and carbs to get change in bg relative to endogenous glucose production
        if rel_insulin_amount != 0 or carb_amount > 0:  # NOTE: Insulin can be negative
            # This gives results for t=time -> t=time+prediction_horizon_hrs
            combined_delta_bg_pred, _ = self.run_metabolism_model(
                rel_insulin_amount, carb_amount
            )

        # Apply the absolute amount of insulin to get the insulin on board
        if abs_insulin_amount != 0:
            # TODO: Is it possible to avoid running this twice with a change in the
            #       metabolism model?
            _, iob_pred = self.run_metabolism_model(
                abs_insulin_amount, carb_amount=0
            )

        # Update bg prediction with delta bgs
        if self.bg_prediction is None:  # At initialization t=0
            self.bg_prediction = self.bg_current + np.cumsum(combined_delta_bg_pred)
            self.iob_prediction = self.iob_init + iob_pred
        else:
            # Get shifted predictions for the next time
            bg_pred_prev_shifted = np.append(
                self.bg_prediction[1:], self.bg_prediction[-1]
            )
            delta_bg_pred_next_t = np.cumsum(combined_delta_bg_pred)
            self.bg_prediction = bg_pred_prev_shifted + delta_bg_pred_next_t

            iob_pred_shifted = np.append(
                self.iob_prediction[1:], 0
            )
            self.iob_prediction = iob_pred + iob_pred_shifted

        pass

    def run_metabolism_model(self, insulin_amount, carb_amount):
        """
        Get the predicted effects of insulin and carbs

        Parameters
        ----------
        insulin_amount: float
            Insulin amount

        carb_amount: float
            carb amount

        Returns
        -------
        (np.array, np.array)
            Delta bg and iob from model
        """

        metabolism_model_instance = self.instantiate_metabolism_model()

        combined_delta_bg, t_min, insulin_amount, iob = metabolism_model_instance.run(
            insulin_amount=insulin_amount, carb_amount=carb_amount, five_min=True,
        )

        return combined_delta_bg, iob

    def instantiate_metabolism_model(self):
        """
        Get instance of metabolism model from the current state of settings.

        Returns
        -------
        SimpleMetabolismModel
        """

        isf = self.patient_config.insulin_sensitivity_schedule.get_state()
        cir = self.patient_config.carb_ratio_schedule.get_state()

        metabolism_model_instance = self.metabolism_model(
            insulin_sensitivity_factor=isf.value, carb_insulin_ratio=cir.value
        )

        return metabolism_model_instance

    def does_accept_bolus_recommendation(self, bolus):
        """
        This models in a basic way whether a patient accepts a recommended bolus or not. Overrides
        of this function should consider the size of the bolus and other state info
        like time of day.

        Parameters
        ----------
        bolus: Bolus
            The recommended bolus

        Returns
        -------
        bool
            True if patient accepts
        """
        does_accept = False
        u = self.random_values["uniform"][0]

        if u <= self.patient_config.recommendation_accept_prob and \
                bolus.value >= self.patient_config.min_bolus_rec_threshold and \
                self.has_eaten_recently(within_time_minutes=self.patient_config.recommendation_meal_attention_time_minutes):
            does_accept = True

        return does_accept

    def has_eaten_recently(self, within_time_minutes):
        recent_carb_events = self.carb_event_timeline.get_recent_event_times(self.time, num_hours_history=within_time_minutes / 60)

        has_eaten_recently = False
        if len(recent_carb_events) > 0:
            has_eaten_recently = True

        return has_eaten_recently

    def stop_pump_session(self):
        """
        Stop a pump session
        """

        if self.pump is None:
            raise Exception("Trying to stop a pump session, but there is no pump")

        self.pump = None

    def start_pump_session(self, pump_class, pump_config):
        """
        Start a new pump session.

        Parameters
        ----------
        pump_class
        pump_config
        """
        if self.pump is not None:
            raise Exception("Trying to start pump session during existing session.")

        self.pump = pump_class(time=self.time, pump_config=pump_config)

    def set_random_values(self):
        """
        Draw all random values that may be needed to keep streams in sync across sims
        """
        self.random_values = {
            "uniform": self.random_state.uniform(0, 1, 100),
        }

    def __repr__(self):

        return "BG: {:.2f}. IOB: {:.2f}. BR {:.2f}".format(self.bg_current, self.iob_current, self.pump.get_basal_rate().value)


class VirtualPatientCarbBolusAccept(VirtualPatient):
    """
    A vitual patient that does not accept small Loop bolus recommendations. This will encourage recommendations
    that are related to Carb entries or other insulin-deficient scenarios.
    """

    def does_accept_bolus_recommendation(self, bolus):
        does_accept = False
        u = self.random_values["uniform"][1]

        min_bolus_rec_threshold = self.patient_config.min_bolus_rec_threshold
        if u <= self.patient_config.recommendation_accept_prob and bolus.value >= min_bolus_rec_threshold:
            does_accept = True

        return does_accept


class VirtualPatientModel(VirtualPatient):
    """
    A virtual patient that probabilistically eats meals and probabilistically gives
    insulin/carb corrections.
    """

    def __init__(
        self,
        time,
        pump,
        sensor,
        metabolism_model,
        patient_config,
        id=None,
        random_state=None,
    ):
        super().__init__(time, pump, sensor, metabolism_model, patient_config, random_state=random_state)
        self._validate_config()

        if id is None:
            id = np.random.randint(0, 1e6)

        self.name = "VP-{}".format(id)

        # No meals
        # self.meal_model = [
        #     MealModel("Breakfast", datetime.time(hour=7), datetime.time(hour=10), 0.98),
        # ]

        self.meal_model = [
            MealModel("Breakfast", datetime.time(hour=7), datetime.time(hour=10), 0.98),
            MealModel("Snack", datetime.time(hour=10), datetime.time(hour=11), 0.05),
            MealModel("Lunch", datetime.time(hour=11), datetime.time(hour=13), 0.98),
            MealModel("Snack", datetime.time(hour=14), datetime.time(hour=15), 0.05),
            MealModel("Dinner", datetime.time(hour=17), datetime.time(hour=21), 0.999),
        ]

        num_trials = int(self.patient_config.correct_bolus_delay_minutes / 5.0)
        self.correct_bolus_step_prob = get_bernoulli_trial_uniform_step_prob(
            num_trials, 1.0
        )

        num_trials = int(self.patient_config.correct_carb_delay_minutes / 5.0)
        self.correct_carb_step_prob = get_bernoulli_trial_uniform_step_prob(
            num_trials, 1.0
        )

        self.correct_carb_wait_time_min = 30
        self.correct_carb_wait_minutes = 0

        self.meal_wait_minutes = 0

    def _validate_config(self):

        super()._validate_config()

        assert hasattr(self.patient_config, "correct_bolus_bg_threshold"), "No correct_bolus_bg_threshold in patient config"
        assert hasattr(self.patient_config, "correct_bolus_delay_minutes"), "No correct_bolus_delay_minutes in patient config"
        assert hasattr(self.patient_config, "correct_carb_bg_threshold"), "No correct_carb_bg_threshold in patient config"
        assert hasattr(self.patient_config, "correct_carb_delay_minutes"), "No correct_carb_delay_minutes in patient config"
        assert hasattr(self.patient_config, "carb_count_noise_percentage"), "No carb_count_noise_percentage in patient config"
        assert hasattr(self.patient_config, "report_carb_probability"), "No report_carb_probability in patient config"

        assert hasattr(self.patient_config, "prebolus_minutes_choices"), "No prebolus_minutes_choices in patient config"
        assert np.array([v % 5 == 0 for v in self.patient_config.prebolus_minutes_choices]).all(), "Prebolus minutes not a multiple of 5"

        assert hasattr(self.patient_config, "carb_reported_minutes_choices"), "No carb_reported_minutes_choices in patient config"
        assert np.array([v % 5 == 0 for v in self.patient_config.carb_reported_minutes_choices]).all(), "Prebolus minutes not a multiple of 5"

        assert hasattr(self.patient_config, "correct_carb_delay_minutes"), "No correct_carb_delay_minutes in patient config"

    def get_info_stateless(self):

        stateless_info = super().get_info_stateless()
        stateless_info.update({
            "correct_bolus_bg_threshold": self.patient_config.correct_bolus_bg_threshold,
            "correct_bolus_delay_minutes": self.patient_config.correct_bolus_delay_minutes,
            "correct_carb_bg_threshold": self.patient_config.correct_carb_bg_threshold,
            "correct_carb_delay_minutes": self.patient_config.correct_carb_delay_minutes,
            "carb_count_noise_percentage": self.patient_config.carb_count_noise_percentage,
            "recommendation_accept_prob": self.patient_config.recommendation_accept_prob,
            "min_bolus_rec_threshold": self.patient_config.min_bolus_rec_threshold,
            "report_bolus_probability": self.patient_config.report_bolus_probability,
            "report_carb_probability": self.patient_config.report_carb_probability,
            "recommendation_meal_attention_time_minutes": self.patient_config.recommendation_meal_attention_time_minutes,
            "prebolus_minutes_choices": self.patient_config.prebolus_minutes_choices,
            "carb_reported_minutes_choices": self.patient_config.carb_reported_minutes_choices
        })
        return stateless_info

    def get_user_inputs(self):
        """
        Get carb and insulin inputs.
        """
        meal = self.get_meal()

        meal_carb = None
        meal_carb_estimate = None
        if meal is not None:
            if meal.name == "Snack":
                random_key = "snack"
            else:
                random_key = "meal"
            meal_carb = Carb(value=self.random_values["{}_carbs".format(random_key)][0],
                             units="g",
                             duration_minutes=self.random_values["{}_carb_durations".format(random_key)][0])
            meal_carb_estimate = Carb(value=self.random_values["{}_carb_estimates".format(random_key)][0],
                             units="g",
                             duration_minutes=self.random_values["{}_carb_duration_estimates".format(random_key)][0])

        correction_carb, correction_carb_estimate = self.get_correction_carb()

        total_carb = self.combine_carbs(meal_carb, correction_carb)
        total_carb_estimate = self.combine_carbs(meal_carb_estimate, correction_carb_estimate)

        # This is set from Loop controller
        total_bolus = self.bolus_event_timeline.get_event(self.time)

        if total_carb is not None:
            carb_time = self.time + datetime.timedelta(minutes=self.random_values["prebolus_offset_minutes"])
            carb_time_reported = self.time + datetime.timedelta(minutes=self.random_values["carb_reported_minutes"])
            self.carb_event_timeline.add_event(carb_time, total_carb)  # Actual carbs go to patient predict
            self.report_carb(total_carb_estimate, carb_time_reported)  # Reported carbs go to Loop

        return total_bolus, total_carb

    def report_carb(self, estimated_carb, carb_time_estimated):
        """
        Probabilistically report carb to pump. Controller knows about pump events.

        Parameters
        ----------
        carb
        """
        u = self.random_values["uniform"][2]

        if u <= self.patient_config.report_carb_probability:
            self.pump.carb_event_timeline.add_event(carb_time_estimated, estimated_carb)

    def get_meal(self):
        """
        Check all the meals and see if user is to eat one.

        Returns
        -------
        MealModel
        """
        meal = None
        u = self.random_values["uniform"][4]

        for meal_model in self.meal_model:
            if self.meal_wait_minutes == 0 and meal_model.is_meal_time(self.time) and u < meal_model.step_prob:
                meal = meal_model
                self.meal_wait_minutes = (datetime.datetime.combine(self.time.date(), meal_model.time_end) - self.time).total_seconds() / 60
                break

        return meal

    def get_correction_carb(self):
        """
        Get a correction carb according to the user's parameterized behavior.

        Returns
        -------
        Carb
        """

        u = self.random_values["uniform"][5]
        correction_carb_value = self.random_values["correct_carbs"][0]
        correction_carb_value_estimate = self.random_values["correct_carb_estimates"][0]

        carb = None
        carb_estimate = None
        if (
            self.bg_current <= self.patient_config.correct_carb_bg_threshold
            and u <= self.correct_carb_step_prob
            and self.correct_carb_wait_minutes == 0
        ):
            carb = Carb(value=correction_carb_value, units="g", duration_minutes=3 * 60)
            carb_estimate = Carb(value=correction_carb_value_estimate, units="g", duration_minutes=3 * 60)
            self.correct_carb_wait_minutes = self.correct_carb_wait_time_min

        return carb, carb_estimate

    def get_correction_bolus(self):
        """
        Get a correction bolus according to the user's parameterized behavior.

        Returns
        -------
        Bolus
        """
        u = self.random_values["uniform"][6]

        correction_bolus = None
        if (
            self.bg_current >= self.patient_config.correct_bolus_bg_threshold
            and u <= self.correct_bolus_step_prob
        ):
            isf = self.patient_config.insulin_sensitivity_schedule.get_state()
            target_range = self.pump.pump_config.target_range_schedule.get_state()
            target_bg = (
                target_range.min_value
                + (target_range.max_value - target_range.min_value) / 2
            )

            # TODO: Fast placeholder for demo, I don't think it's a correct implementation
            #       Really this should be a pump-specific function so we can model different bolus calculators
            insulin_amt = max(
                0,
                (self.bg_current - target_bg) / isf.value - (self.iob_current - self.sbr_iob),
            )

            correction_bolus = Bolus(value=insulin_amt, units="U")

        return correction_bolus

    def combine_carbs(self, meal_carb, correction_carb):
        """
        Combine carbs if they occur at the same time.

        Parameters
        ----------
        meal_carb: Carb
        correction_carb: Carb

        Returns
        -------
        Carb
        """

        if meal_carb is not None and correction_carb is not None:
            # TODO use measure __add__() here instead?
            carb = Carb(
                value=meal_carb.value + correction_carb.value,
                units="g",
                duration_minutes=(meal_carb.duration_minutes + correction_carb.duration_minutes) / 2,
            )
        elif meal_carb is not None:
            carb = meal_carb
        elif correction_carb is not None:
            carb = correction_carb
        else:
            carb = None

        return carb

    def delete_pump_event_history(self):
        self.patient_config.action_timeline.add_action(self.time, "delete_pump_event_history")

    def set_random_values(self):
        """
        Draw all random values that may be needed to keep streams in sync across sims
        """
        super().set_random_values()  # Set values needed for inherited class

        num_values = 100

        meal_carb_values = self.random_state.uniform(20, 60, num_values)
        meal_carb_estimates = []
        for carb_value in meal_carb_values:
            meal_carb_estimates.append(
                max(0,
                    int(self.random_state.normal(carb_value, carb_value * self.patient_config.carb_count_noise_percentage))
                )
            )

        meal_carb_durations = self.random_state.choice(range(2 * 60, 5 * 60), num_values)
        meal_carb_duration_estimates = []
        for duration in meal_carb_durations:
            meal_carb_duration_estimates.append(
                max(120,
                    int(self.random_state.normal(duration, duration * self.patient_config.carb_count_noise_percentage))
                )
            )

        snack_carb_values = self.random_state.uniform(5, 15, num_values)
        snack_carb_estimates = []
        for carb_value in snack_carb_values:
            snack_carb_estimates.append(
                max(0,
                    int(self.random_state.normal(carb_value,
                                                 carb_value * self.patient_config.carb_count_noise_percentage))
                    )
            )

        snack_carb_durations = self.random_state.choice(range(2 * 60, 5 * 60), num_values)
        snack_carb_duration_estimates = []
        for duration in snack_carb_durations:
            snack_carb_duration_estimates.append(
                max(120,
                    int(self.random_state.normal(duration, duration * self.patient_config.carb_count_noise_percentage))
                    )
            )

        correct_carbs = self.random_state.uniform(5, 10, num_values)
        correct_carb_estimates = []
        for carb_value in correct_carbs:
            correct_carb_estimates.append(
                max(0,
                    int(self.random_state.normal(carb_value, carb_value * self.patient_config.carb_count_noise_percentage))
                )
            )

        prebolus_minutes = int(self.random_state.choice(self.patient_config.prebolus_minutes_choices))
        carb_reported_minutes = int(self.random_state.choice(self.patient_config.carb_reported_minutes_choices))

        self.random_values.update({
            "meal_carbs": meal_carb_values,
            "meal_carb_estimates": meal_carb_estimates,
            "meal_carb_durations": meal_carb_durations,
            "meal_carb_duration_estimates": meal_carb_duration_estimates,
            "snack_carbs": snack_carb_values,
            "snack_carb_estimates": snack_carb_estimates,
            "snack_carb_durations": snack_carb_durations,
            "snack_carb_duration_estimates": snack_carb_duration_estimates,
            "correct_carbs": correct_carbs,
            "correct_carb_estimates": correct_carb_estimates,
            "prebolus_offset_minutes": prebolus_minutes,
            "carb_reported_minutes": carb_reported_minutes
        })

    def update(self, time, **kwargs):
        super().update(time, **kwargs)
        self.correct_carb_wait_minutes = max(0, self.correct_carb_wait_minutes - 5)
        self.meal_wait_minutes = max(0, self.meal_wait_minutes - 5)
        self.set_random_values()