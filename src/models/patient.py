__author__ = "Cameron Summers"

import copy
import numpy as np
import datetime

from src.models.simulation import SimulationComponent
from src.makedata.scenario_parser import PatientConfig
from src.models.measures import Carb, Bolus
from src.models.events import MealModel
from src.utils import get_bernoulli_trial_uniform_step_prob


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
        isf,
        cir,
    ):

        self.bg = bg
        self.sensor_bg = sensor_bg
        self.bg_prediction = bg_prediction
        self.sensor_bg_prediction = sensor_bg_prediction
        self.iob = iob
        self.iob_prediction = iob_prediction
        self.pump_state = pump_state
        self.isf = isf
        self.cir = cir


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

        # TODO: prediction horizon should probably come from simple metabolism model
        prediction_horizon_hrs = 8
        self.num_prediction_steps = int(prediction_horizon_hrs * 60 / 5)

    def init(self):
        """
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
        self.iob_current = iob_steady_state[0]
        self.sbr_iob = iob_steady_state[0]

    def get_state(self):
        """
        Get the current state of the patient.

        Returns
        -------
        VirtualPatientState
        """

        patient_state = VirtualPatientState(
            bg=self.bg_current,
            sensor_bg=self.sensor.get_bg(self.bg_current),
            bg_prediction=self.bg_prediction,
            pump_state=self.pump.get_state(),
            iob=self.iob_current,
            iob_prediction=self.iob_prediction,
            sensor_bg_prediction=self.sensor.get_bg_trace(self.bg_prediction),
            isf=self.patient_config.insulin_sensitivity_schedule.get_state(),
            cir=self.patient_config.carb_ratio_schedule.get_state(),
        )

        return patient_state

    def get_actions(self):
        """
        Get events from configuration that influence the internal metabolism model.

        Returns
        -------
        (Insulin Event, Carb Event)
        """
        # Get boluses at time
        bolus = self.patient_config.insulin_events.get_event(self.time)

        # Get carbs at time
        carb = self.patient_config.carb_events.get_event(self.time)

        return bolus, carb

    def update(self, time, **kwargs):
        """
        Predict the future state of the patient given the current state.

        Parameters
        ----------
        time: datetime
        """
        self.time = time

        # Update member simulation components
        self.pump.update(time)
        self.sensor.update(time)

        self.predict()

    def update_from_prediction(self, time):
        """
        Set the patient state to the result of the metabolism simulation.
        """
        self.bg_current = self.sensor.get_bg(self.bg_prediction[0])
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

    def get_net_basal_insulin(self):
        """
        Get the insulin from the difference between the pump and the patient,
        which could come from a temp basal or a pump scheduled basal.

        Returns
        -------
        float

        """
        patient_basal_rate = self.patient_config.basal_schedule.get_state()

        pump_state = self.pump.get_state()
        pump_basal_rate = pump_state.get_basal_rate()

        patient_true_basal_amt = patient_basal_rate.get_insulin_in_interval()
        pump_basal_amt = pump_basal_rate.get_insulin_in_interval()

        net_basal_insulin_amt = pump_basal_amt - patient_true_basal_amt

        return net_basal_insulin_amt

    def predict(self):
        """
        Using state at t=time, predict the patient state from t=time+1 to
        t=time + prediction_horizon_hrs + 1 based on the basal/bolus/carbs.
        """
        # Insulin from basal state
        # Should be zero if no temp basal or if pump basal matches patient basal
        insulin_amount = self.get_net_basal_insulin()

        # TODO: NOTE: Here is where we'd add additional events,
        #       e.g. exercise, sensor compression, pump site change, etc.
        bolus, carb = self.get_actions()

        if bolus is not None:
            delivered_bolus = self.pump.deliver_insulin(bolus)
            insulin_amount += delivered_bolus.value

        carb_amount = 0
        if carb is not None:
            carb_amount = carb.value

        # Initialize zero change
        combined_delta_bg_pred = np.zeros(self.num_prediction_steps)
        iob_pred = np.zeros(self.num_prediction_steps)

        # Apply insulin and carbs according to metabolism model
        if insulin_amount != 0 or carb_amount > 0:  # NOTE: Insulin can be negative
            # This gives results for t=time -> t=time+prediction_horizon_hrs
            combined_delta_bg_pred, iob_pred = self.run_metabolism_model(
                insulin_amount, carb_amount
            )

        # Update bg prediction with delta bgs
        if self.bg_prediction is None:  # At initialization t=0
            self.bg_prediction = self.bg_current + np.cumsum(combined_delta_bg_pred)
            self.bg_prediction = np.append(
                self.bg_prediction[1:], self.bg_prediction[-1]
            )

            self.iob_prediction = self.iob_current + iob_pred

            # TODO: need this if events, but feels awkward here
            self.iob_current = self.iob_prediction[0]
            self.iob_prediction = np.append(
                self.iob_prediction[1:], self.iob_prediction[-1]
            )
        else:
            # Get shifted predictions for the next time
            bg_pred_prev_shifted = np.append(
                self.bg_prediction[1:], self.bg_prediction[-1]
            )
            delta_bg_pred_next_t = np.cumsum(np.append(combined_delta_bg_pred[1:], 0))
            self.bg_prediction = bg_pred_prev_shifted + delta_bg_pred_next_t

            iob_pred_prev_shifted = np.append(
                self.iob_prediction[1:], self.iob_prediction[-1]
            )
            iob_pred_next_t = np.append(iob_pred[1:], 0)
            self.iob_prediction = iob_pred_prev_shifted + iob_pred_next_t
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

        combined_delta_bg, t_min, carb_amount, insulin_amount, iob = metabolism_model_instance.run(
            insulin_amount=insulin_amount, carb_amount=carb_amount
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

        # TODO: hack due to non-explicit carb bolusing
        #  Is insulin always in scenario config???
        # metab model run() function will compute this if insulin_amount is np.nan,
        # but let's move away from this and update run() function to take only explicit input
        # if carb_amount > 0:
        #     insulin_amount += cir.calculate_bolus(carb_amount)

        metabolism_model_instance = self.metabolism_model(
            insulin_sensitivity_factor=isf.value, carb_insulin_ratio=cir.value
        )

        return metabolism_model_instance

    def __repr__(self):

        return "BG: {:.2f}. IOB: {:.2f}".format(self.bg_current, self.iob_current)


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
        remember_meal_bolus_prob=1.0,
        correct_bolus_bg_threshold=180,
        correct_bolus_delay_minutes=30,
        correct_carb_bg_threshold=80,
        correct_carb_delay_minutes=10,
        carb_count_noise_percentage=0.1,
        id="",
    ):
        super().__init__(time, pump, sensor, metabolism_model, patient_config)

        self.name = "VP-{}".format(id)

        self.meal_model = [
            MealModel("Breakfast", datetime.time(hour=7), datetime.time(hour=10), 0.98),
            MealModel("Lunch", datetime.time(hour=11), datetime.time(hour=13), 0.98),
            MealModel("Dinner", datetime.time(hour=17), datetime.time(hour=21), 0.999),
        ]

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
        self.last_meal = None

    def get_actions(self):
        """
        Get carb and insulin actions.
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

        return total_bolus, total_carb

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
            target_range = self.patient_config.target_range_schedule.get_state()
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
                print("Forgot bolus")

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
