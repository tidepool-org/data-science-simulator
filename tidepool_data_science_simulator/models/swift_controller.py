
import datetime
import json

from tidepool_data_science_simulator.models.measures import Bolus, TempBasal
from tidepool_data_science_simulator.models.controller import AutomationControlTimeline, LoopController

from tidepool_data_science_simulator import USE_LOCAL_PYLOOPKIT

from loop_to_python_api.api import get_loop_recommendations

class SwiftLoopController(LoopController):
    """
    Loop controller class that intefaces with the Swift verion of Loop.
    """

    def __repr__(self):
        return "SwiftLoopKit"

    def __str__(self):
        return "SwiftLoopKit.1"

    def __init__(self, time, controller_config, automation_control_timeline=AutomationControlTimeline([], [])):
        super().__init__(time, controller_config, automation_control_timeline)
        self.name = "SwiftLoopKit v0.1"


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
        data['recommendationInsulinType'] = 'novolog'
        data['maxBasalRate'] = settings_dictionary['max_basal_rate']
        data['maxBolus'] = settings_dictionary['max_bolus']
        data['suspendThreshold'] = settings_dictionary['suspend_threshold']
        data['automaticBolusApplicationFactor'] = settings_dictionary['partial_application_factor']
        data['useMidAbsorptionISF'] = settings_dictionary['use_mid_absorption_isf']
              
        if settings_dictionary.get('partial_application_factor'):
            data['recommendationType'] = 'automaticBolus' 
        else:
            data['recommendationType'] = 'tempBasal'

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

    
    def get_loop_recommendations(self, time, virtual_patient=None):
        """
        Get recommendations from the Loop Algorithm, based on
        virtual_patient dosing and glucose.
        """
        self.time = time

        automation_control_event = self.automation_control_timeline.get_event(time)

        if automation_control_event is not None:
            self.open_loop = not automation_control_event.dosing_enabled

        if virtual_patient.pump is not None:
            loop_inputs_dict = self.prepare_inputs(virtual_patient)
                        
            swift_output_automatic = get_loop_recommendations(loop_inputs_dict)
            swift_output_decode_automatic = swift_output_automatic.decode('utf-8')
            swift_output_json_automatic = json.loads(swift_output_decode_automatic)

            loop_inputs_dict['recommendationType'] = 'manualBolus'
            loop_inputs_dict['includePositiveVelocityAndRC'] = False
            swift_output_manual = get_loop_recommendations(loop_inputs_dict)
            swift_output_decode_manual = swift_output_manual.decode('utf-8')
            swift_output_json_manual = json.loads(swift_output_decode_manual)
            
            swift_output_json = swift_output_json_automatic | swift_output_json_manual
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
                bolus = Bolus(manual_bolus_rec, "U")
                self.set_bolus_recommendation_event(virtual_patient, bolus)            

        if automatic_data:
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