__author__ = "Cameron Summers"

import copy
import numpy as np
import datetime

from tidepool_data_science_simulator.models.simulation import SimulationComponent, EventTimeline, ActionTimeline
from tidepool_data_science_simulator.makedata.scenario_parser import PatientConfig
from tidepool_data_science_simulator.models.measures import Carb, Bolus
from tidepool_data_science_simulator.models.events import MealModel, Action
from tidepool_data_science_simulator.utils import get_bernoulli_trial_uniform_step_prob


# ============= Patient Stuff ===================
class VirtualPatientState(object):
    """
    A class of instantaneous patient information.
    """

    def __init__(
        self,
        bg,
        sensor_bg,
        bg_prediction,
        pump_state,
        iob,
        iob_prediction,
        sensor_bg_prediction,
        sbr,
        isf,
        cir,
        bolus,
        carb,
        actions=None
    ):

        self.bg = bg
        self.sensor_bg = sensor_bg
        self.bg_prediction = bg_prediction
        self.sensor_bg_prediction = sensor_bg_prediction
        self.iob = iob
        self.iob_prediction = iob_prediction
        self.pump_state = pump_state
        self.sbr = sbr
        self.isf = isf
        self.cir = cir
        self.bolus = bolus
        self.carb = carb
        self.actions = actions


class VirtualPatient(SimulationComponent):
    """
    Patient class for simulation
    """

    def __init__(self, time, pump, sensor, metabolism_model, patient_config):
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

        self.pump = pump
        self.sensor = sensor
        self.metabolism_model = metabolism_model

        self.patient_config = copy.deepcopy(patient_config)

        self.bg_history = self.patient_config.glucose_history
        self.iob_history = []  # TODO: make trace obj, not list

        self.bg_prediction = None
        self.iob_prediction = None

        self.bg_current = None
        self.iob_current = None
        self.sbr_iob = None

        self.carb_event_timeline = patient_config.carb_event_timeline
        self.bolus_event_timeline = patient_config.bolus_event_timeline
        self.action_event_timeline = patient_config.action_event_timeline

        # TODO: prediction horizon should probably come from simple metabolism model
        prediction_horizon_hrs = 8
        self.num_prediction_steps = int(prediction_horizon_hrs * 60 / 5)

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
            pump_state=self.pump.get_state(),
            bolus=self.bolus_event_timeline.get_event_value(self.time),
            carb=self.carb_event_timeline.get_event_value(self.time),
            actions=self.action_event_timeline.get_event(self.time)
        )

        return patient_state

    def get_actions(self):
        """
        Get actions that the user has performed which are not boluses or meals.
        Possible actions: set change, battery change, user deletes all data, exercise.

        Returns
        -------
        [Action]
        """
        actions = self.action_event_timeline.get_event(self.time)

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

        if bolus is not None:
            delivered_bolus = self.pump.deliver_bolus(bolus)
            rel_insulin_amount += delivered_bolus.value
            abs_insulin_amount += delivered_bolus.value

        carb_amount = 0
        if carb is not None:
            carb_amount = carb.value

        return abs_insulin_amount, rel_insulin_amount, carb_amount

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
        u = np.random.random()

        if u <= self.patient_config.recommendation_accept_prob:
            does_accept = True

        return does_accept

    def __repr__(self):

        return "BG: {:.2f}. IOB: {:.2f}. BR {:.2f}".format(self.bg_current, self.iob_current, self.pump.get_basal_rate().value)


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
        remember_meal_bolus_prob=1.0,  # TODO: move these into config object. Separate config object?
        correct_bolus_bg_threshold=180,
        correct_bolus_delay_minutes=30,
        correct_carb_bg_threshold=80,
        correct_carb_delay_minutes=10,
        carb_count_noise_percentage=0.1,
        id=None,
    ):
        super().__init__(time, pump, sensor, metabolism_model, patient_config)

        if id is None:
            id = np.random.randint(0, 1000000)

        self.name = "VP-{}".format(id)

        self.meal_model = [
            MealModel("Breakfast", datetime.time(hour=7), datetime.time(hour=10), 0.98),
            MealModel("Lunch", datetime.time(hour=11), datetime.time(hour=13), 0.98),
            MealModel("Dinner", datetime.time(hour=17), datetime.time(hour=21), 0.999),
        ]

        self.report_carb_probability = 1.0  # TODO: move into config object
        self.report_bolus_probability = 1.0

        self.remember_meal_bolus_prob = remember_meal_bolus_prob

        self.correct_bolus_bg_threshold = correct_bolus_bg_threshold
        num_trials = int(correct_bolus_delay_minutes / 5.0)
        self.correct_bolus_step_prob = get_bernoulli_trial_uniform_step_prob(
            num_trials, 1.0
        )

        self.correct_carb_bg_threshold = correct_carb_bg_threshold
        num_trials = int(correct_carb_delay_minutes / 5.0)
        self.correct_carb_step_prob = get_bernoulli_trial_uniform_step_prob(
            num_trials, 1.0
        )

        self.carb_count_noise_percentage = carb_count_noise_percentage

        # TODO: Actually need something more robust than this method to avoid duplicate meal
        #       events. Case example: user eats breakfast, skips lunch and dinner, then eats
        #       breakfast again, this will prevent that.
        # Why is this not a plausible scenario?
        self.last_meal = None
        self.patient_actions = Action()

    def get_events(self):
        """
        Get carb and insulin inputs.
        """
        meal = self.get_meal()

        meal_carb = None
        if meal is not None:
            meal_carb = meal.get_carb()

        # TODO maybe don't combine since different durations and better accounting
        #   instead let predict() run on multiple carbs at a time.
        correction_carb = self.get_correction_carb()
        total_carb = self.combine_carbs(meal_carb, correction_carb)

        meal_bolus = self.get_meal_bolus(meal_carb)
        correction_bolus = self.get_correction_bolus()
        total_bolus = self.combine_boluses(meal_bolus, correction_bolus)

        if total_carb is not None:
            self.carb_event_timeline.add_event(self.time, total_carb)
            self.report_carb(total_carb)

        if total_bolus is not None:
            self.bolus_event_timeline.add_event(self.time, total_bolus)
            self.report_bolus(total_bolus)

        return total_bolus, total_carb

    def report_carb(self, carb):
        """
        Probabilistically report carb to pump. Controller knows about pump events.

        Parameters
        ----------
        carb
        """
        u = np.random.random()
        if u <= self.report_carb_probability:
            self.pump.carb_event_timeline.add_event(self.time, carb)

    def report_bolus(self, bolus):
        """
        Probabilistically report bolus to pump. Controller knows about pump events.
        This case is more for an MDI or Afrezza type situation.

        Parameters
        ----------
        bolus
        """
        u = np.random.random()
        if u <= self.report_bolus_probability:
            self.pump.bolus_event_timeline.add_event(self.time, bolus)

    def get_meal(self):
        """
        Check all the meals and see if user is to eat one.

        Returns
        -------
        MealModel
        """
        meal = None
        u = np.random.uniform()

        for meal_model in self.meal_model:
            if meal_model.is_meal_time(self.time) and u < meal_model.step_prob:

                if self.last_meal != meal_model:
                    meal = meal_model
                    self.last_meal = meal
                    break

        return meal

    def get_correction_carb(self):
        """
        Get a correction carb according to the user's parameterized behavior.

        Returns
        -------
        Carb
        """

        u = np.random.random()

        carb = None
        if (
            self.bg_current <= self.correct_carb_bg_threshold
            and u <= self.correct_carb_step_prob
        ):
            carb = Carb(value=10, units="g", duration_minutes=3 * 60)

        return carb

    def get_correction_bolus(self):
        """
        Get a correction bolus according to the user's parameterized behavior.

        Returns
        -------
        Bolus
        """
        u = np.random.random()

        correction_bolus = None
        if (
            self.bg_current >= self.correct_bolus_bg_threshold
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

    def get_meal_bolus(self, carb):
        """
        Get the bolus for a meal based whether the user remembered
        to bolus and user estimation capabilities.

        Parameters
        ----------
        carb: Carb

        Returns
        -------
        Bolus
        """

        # Meal bolus
        bolus = None
        if carb is not None:
            u = np.random.random()

            if u <= self.remember_meal_bolus_prob:

                estimated_carb = self.estimate_meal_carb(carb)
                cir = self.pump.pump_config.carb_ratio_schedule.get_state()

                bolus = Bolus(value=cir.calculate_bolus(estimated_carb), units="U")
            else:
                print("{} Forgot bolus".format(self.name))

        return bolus

    def combine_boluses(self, meal_bolus, correction_bolus):
        """
        Combine boluses if they occur at the same time.

        Parameters
        ----------
        meal_bolus: Bolus
        correction_bolus: Bolus

        Returns
        -------
        Bolus
        """
        if meal_bolus is not None and correction_bolus is not None:
            # TODO use measure __add__() here instead
            bolus = Bolus(value=correction_bolus.value + meal_bolus.value, units="U")
        elif meal_bolus is not None:
            bolus = meal_bolus
        elif correction_bolus is not None:
            bolus = correction_bolus
        else:
            bolus = None

        return bolus

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
        self.action_event_timeline.add_action(self.time, "delete_pump_event_history")

    def estimate_meal_carb(self, carb):
        """
        Estimate the meal's carb based on a normal distribution centered
        on the real value and variance parameter in the user model.

        Parameters
        ----------
        carb: Carb
            The actual carb for the meal

        Returns
        -------
        Carb
            The estimated carb for the meal
        """

        estimated_carb = Carb(
            value=int(
                np.random.normal(
                    carb.value, carb.value * self.carb_count_noise_percentage
                )
            ),
            units="g",
            duration_minutes=carb.duration_minutes,
        )
        return estimated_carb
