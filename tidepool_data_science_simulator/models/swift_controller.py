
import datetime
import json
import logging
import os


from tidepool_data_science_simulator.models.measures import Bolus, TempBasal
from tidepool_data_science_simulator.models.controller import AutomationControlTimeline, LoopController

from tidepool_data_science_simulator import USE_LOCAL_PYLOOPKIT

from loop_to_python_api.api import (
    get_loop_recommendations,
    get_prediction_values_and_dates,
    get_glucose_velocity_values_and_dates,
    get_active_carbs,
    get_active_insulin,
)

logger = logging.getLogger(__name__)


class SwiftLoopController(LoopController):
    """
    Loop controller class that intefaces with the Swift verion of Loop.
    """

    def __repr__(self):
        return "SwiftLoopKit"

    def __str__(self):
        return "SwiftLoopKit.1"

    def __init__(self, time, controller_config, automation_control_timeline=AutomationControlTimeline([], []),
                 loop_algo_io_dir=None):
        super().__init__(time, controller_config, automation_control_timeline)
        self.name = "SwiftLoopKit v0.1"
        self.loop_algo_io_dir = loop_algo_io_dir
        self.pump_history_initialized = False  # Track whether pump history has been populated


    def prepare_inputs(self, virtual_patient):
        """
        Collect inputs to the loop update call for the current time.

        Parameters
        ----------
        virtual_patient:

        Returns
        -------
        dict
            Inputs for the Swift Loop Algorithm
        """
        glucose_dates, glucose_values = virtual_patient.sensor.get_loop_inputs()

        bolus_event_timeline, carb_event_timeline, temp_basal_event_timeline = self.get_dose_event_timelines(virtual_patient)

        bolus_dose_types, bolus_dose_values, bolus_start_times, bolus_end_times, bolus_delivered_units = \
            bolus_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        temp_basal_dose_types, temp_basal_dose_values, temp_basal_start_times, temp_basal_end_times, temp_basal_delivered_units = \
            temp_basal_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        carb_values, carb_start_times, carb_durations = \
            carb_event_timeline.get_loop_inputs(self.time, num_hours_history=self.num_hours_history)

        basal_rate_values, basal_rate_start_times, basal_rate_end_times = \
            virtual_patient.pump.pump_config.basal_schedule.get_loop_swift_inputs()

        isf_values, isf_start_times, isf_end_times = \
            virtual_patient.pump.pump_config.insulin_sensitivity_schedule.get_loop_swift_inputs()

        cir_values, cir_start_times, cir_end_times = \
            virtual_patient.pump.pump_config.carb_ratio_schedule.get_loop_swift_inputs()

        tr_min_values, tr_max_values, tr_start_times, tr_end_times = \
            virtual_patient.pump.pump_config.target_range_schedule.get_loop_swift_inputs()
        
        ##########################
        # Create the Swift Loop input structure
        ##########################
        format_string = r'%Y-%m-%dT%H:%M:%SZ'
        t_now = self.time

        data = {}

        # SETTINGS
        settings_dictionary = self.controller_config.controller_settings

        data['predictionStart'] = t_now.strftime(format=format_string)
        # Read the controller insulin model from config; fall back to 'novolog' for
        # backward compatibility with configs that pre-date the 'model' setting.
        data['recommendationInsulinType'] = settings_dictionary.get('model', 'novolog')
        # maxBasalRate / maxBolus are required by the Swift AlgorithmInputFixture
        # decoder (non-optional `decode`), so index them directly -- a missing one
        # is a genuine config error and the KeyError should surface it clearly.
        data['maxBasalRate'] = settings_dictionary['max_basal_rate']
        data['maxBolus'] = settings_dictionary['max_bolus']
        data['suspendThreshold'] = settings_dictionary['suspend_threshold']
        # automaticBolusApplicationFactor is optional on the Swift side (Double?,
        # defaults to nil) and is only meaningful in automaticBolus mode. Include
        # it only when the config actually provides partial_application_factor;
        # when omitted, Swift falls back to nil rather than an arbitrary default.
        # (The recommendationType guard below keys off the same setting.)
        if 'partial_application_factor' in settings_dictionary:
            data['automaticBolusApplicationFactor'] = settings_dictionary['partial_application_factor']
        # useMidAbsorptionISF is an optional flag on the Swift side
        # (decodeIfPresent ... ?? false); mirror that default when the config
        # omits it instead of raising.
        data['useMidAbsorptionISF'] = settings_dictionary.get('use_mid_absorption_isf', False)
        # Default to 2.0 for backward compatibility if not specified
        data['maxActiveInsulinMultiplier'] = settings_dictionary.get('max_active_insulin_multiplier', 2.0)

        if settings_dictionary.get('partial_application_factor'):
            data['recommendationType'] = 'automaticBolus'
            data['includePositiveVelocityAndRC'] = False
        else:
            data['recommendationType'] = 'tempBasal'
            data['includePositiveVelocityAndRC'] = True

        # If includePositiveVelocityAndRC is set in the settings, override the default value
        if settings_dictionary.get('includePositiveVelocityAndRC'):
            data['includePositiveVelocityAndRC'] = settings_dictionary['include_positive_velocity_and_RC']

        # BASAL RATE
        data_entries = []
        for start_time, end_time, value in zip(basal_rate_start_times, basal_rate_end_times, basal_rate_values):    
            data_entry = { "endDate" : end_time.strftime(format_string),
                "startDate" : start_time.strftime(format_string),
                "value" : value }
            data_entries.append(data_entry)

        data['basal'] = data_entries 

        # SENSITIVITY
        data_entries = []
        for start_time, end_time, value in zip(isf_start_times, isf_end_times, isf_values):
            data_entry = { "endDate" : end_time.strftime(format_string),
                "startDate" : start_time.strftime(format_string),
                "value" : value }
            data_entries.append(data_entry)

        data['sensitivity'] = data_entries
        
        # CARB RATIO
        data_entries = []
        for start_time, end_time, value in zip(cir_start_times, cir_end_times, cir_values):
            data_entry = { "endDate" : end_time.strftime(format_string),
                "startDate" : start_time.strftime(format_string),
                "value" : value }
            data_entries.append(data_entry)

        data['carbRatio'] = data_entries

        # TARGET
        data_entries = []
        for start_time, end_time, lower_bound, upper_bound in zip(
            tr_start_times, tr_end_times, tr_min_values, tr_max_values
        ):
            data_entry = { "endDate" : end_time.strftime(format_string),
                "startDate" : start_time.strftime(format_string),
                "lowerBound" : lower_bound,
                "upperBound" : upper_bound }
            data_entries.append(data_entry)

        data['target'] = data_entries

        # GLUCOSE
        history = []
        for value, date in zip(glucose_values, glucose_dates):
            entry = {
                'date' : date.strftime(format=format_string),  
                'value' : value
            }
            history.append(entry)

        data['glucoseHistory'] = history

        # CARB ENTRIES
        history = []
        for value, date, absorption_time in zip(carb_values, carb_start_times, carb_durations):
            entry = {
                'date' : date.strftime(format=format_string),  
                'grams' : value,
                'absorptionTime' : absorption_time * 60
            }
            history.append(entry)

        data['carbEntries'] = history

        # DOSES
        dose_types = bolus_dose_types + temp_basal_dose_types
        dose_values = bolus_dose_values + temp_basal_dose_values
        dose_start_times = bolus_start_times + temp_basal_start_times
        dose_end_times = bolus_end_times + temp_basal_end_times
    
        history = []
        for value, dose_start_time, dose_end_time, dose_type in zip(dose_values, dose_start_times, dose_end_times, dose_types):
            dose_type = dose_type.name.replace('tempbasal', 'basal')

            if dose_type == 'bolus':
                dose_start_time = dose_start_time + datetime.timedelta(seconds=1)
                dose_end_time = dose_end_time + datetime.timedelta(seconds=2)
            elif dose_type == 'basal':
                value = value / 12
                dose_start_time = dose_start_time + datetime.timedelta(seconds=3)

            entry = {
                'startDate' : dose_start_time.strftime(format=format_string),  
                'endDate' : dose_end_time.strftime(format=format_string),  
                'volume' : value,
                'type' : dose_type
            }

            history.append(entry)
        history = sorted(history, key=lambda x: x["startDate"], reverse=True)
        data['doses'] = history

        return data

    
    def _compute_prediction_output(self, loop_inputs_dict: dict) -> dict:
        """Compute the Swift Loop prediction/effect/COB payload for the current step.

        Calls the Swift prediction API with the *same* input dict already built for the
        recommendation call (DRY -- no second ``prepare_inputs()`` build). Uses the
        ``*_values_and_dates`` wrappers, which size to the real series length rather than
        the raw entry points' fixed ``len=72`` default.

        Parameters
        ----------
        loop_inputs_dict : dict
            The Swift Loop input structure produced by ``prepare_inputs()``.

        Returns
        -------
        dict | None
            Payload with ``predicted_glucose_values`` / ``predicted_glucose_dates``
            (lists), ``glucose_effect_velocity_values`` (list, counteraction/ICE),
            ``active_carbs`` (float COB) and ``active_insulin`` (float IOB); or ``None``
            if the prediction API call fails.
        """
        try:
            pred_values, pred_dates = get_prediction_values_and_dates(loop_inputs_dict)
            ice_values, _ice_dates = get_glucose_velocity_values_and_dates(loop_inputs_dict)
            return {
                "predicted_glucose_values": pred_values,
                "predicted_glucose_dates": pred_dates,
                "glucose_effect_velocity_values": ice_values,
                "active_carbs": get_active_carbs(loop_inputs_dict),
                "active_insulin": get_active_insulin(loop_inputs_dict),
            }
        except Exception as e:
            # Explicit: log and continue with no prediction data for this step.
            # Never swallow silently (workflow section 4 / AC #4).
            logger.warning("Loop prediction extraction failed at %s: %s", self.time, e)
            return None

    def get_loop_recommendations(self, time, virtual_patient=None):
        """
        Get recommendations from the Loop Algorithm, based on
        virtual_patient dosing and glucose.
        """
        self.time = time
        # Reset each step so stale predictions never leak into a step that
        # doesn't produce a fresh recommendation.
        self.prediction_output = None

        automation_control_event = self.automation_control_timeline.get_event(time)

        if automation_control_event is not None:
            self.open_loop = not automation_control_event.dosing_enabled

        if virtual_patient.pump is not None:
            # On first activation of Loop, populate the pump's historical dose data
            # This ensures Loop has access to pre-Loop basal doses for accurate IOB
            if not self.pump_history_initialized:
                virtual_patient.pump.populate_historical_basal_doses(
                    current_time=time,
                    num_hours_history=self.num_hours_history
                )
                self.pump_history_initialized = True
            
            loop_inputs_dict = self.prepare_inputs(virtual_patient)

            # Construct file paths based on whether directory is set
            format_string = r'%Y-%m-%dT%H:%M:%SZ'
            timestamp_str = self.time.strftime(format_string)
            
            if self.loop_algo_io_dir is not None:
                input_filename = os.path.join(self.loop_algo_io_dir, f"loop_algo_input_{timestamp_str}.json")
                output_filename = os.path.join(self.loop_algo_io_dir, f"loop_algo_output_{timestamp_str}.json")
            else:
                # Fallback to current directory for backward compatibility
                input_filename = f"loop_algo_input_{timestamp_str}.json"
                output_filename = f"loop_algo_output_{timestamp_str}.json"
            
            # Write out input dict to file
            with open(input_filename, 'w') as f:
                json.dump(loop_inputs_dict, f, indent=4)
            
            # Get Loop recommendations
            swift_output = get_loop_recommendations(loop_inputs_dict)
            swift_output_decode = swift_output.decode('utf-8')
            swift_output_json = json.loads(swift_output_decode)

            # Write out output dict to file
            with open(output_filename, 'w') as f:
                json.dump(swift_output_json, f, indent=4)

            # Compute the prediction/effect/COB payload from the SAME input dict
            # (DRY), but ONLY when Loop produced a recommendation. Degenerate inputs
            # (e.g. glucoseTooOld) make Loop return null AND make the prediction/COB
            # entry points hard-trap the process -- an uncatchable native crash. A
            # valid recommendation means the prediction machinery is safe to call.
            # This is also the perf gate: DoNothing/OpenLoop controllers never reach here.
            if swift_output_json is not None:
                self.prediction_output = self._compute_prediction_output(loop_inputs_dict)

            return swift_output_json
        

    def apply_loop_recommendations(self, virtual_patient, loop_algorithm_output):
        """
        Apply the recommendations from the pyloopkit algo.

        Parameters
        ----------
        virtual_patient
        loop_algorithm_output
        """                
        manual_data = loop_algorithm_output.get('manual')
        automatic_data = loop_algorithm_output.get('automatic')

        if manual_data:
            manual_bolus_rec = manual_data['amount']
            if virtual_patient.does_accept_bolus_recommendation(manual_bolus_rec):
                self.set_bolus_recommendation_event(virtual_patient, manual_bolus_rec)

        elif automatic_data:
            autobolus_rec = automatic_data.get('bolusUnits')
            temp_basal_data = automatic_data.get('basalAdjustment')
            
            if autobolus_rec:
                self.set_bolus_recommendation_event(virtual_patient, Bolus(autobolus_rec, "U"))
            
            if temp_basal_data is not None:
                units_per_hour = temp_basal_data.get('unitsPerHour') or 0
      
                temp_basal = TempBasal(self.time, units_per_hour, 30, "U/hr")
                self.modulate_temp_basal(virtual_patient, temp_basal)
        else: 
            pass

        self.recommendations = loop_algorithm_output
