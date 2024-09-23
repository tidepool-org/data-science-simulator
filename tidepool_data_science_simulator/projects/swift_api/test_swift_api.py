__author__ = "Mark Connolly"

from datetime import datetime, timedelta, time
import json

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation, TargetRangeSchedule24hr
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController, SwiftLoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import (
  DATETIME_DEFAULT, get_canonical_risk_patient_config, get_canonical_risk_pump_config,
    get_canonical_sensor_config
)

from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import Bolus, Carb, TargetRange

from loop_to_python_api.api import get_loop_recommendations

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

def loop_input_to_swift(loop_input):
    format_string = r'%Y-%m-%dT%H:%M:%SZ'
    t_now = loop_input['time_to_calculate_at']

    data = {}

    # SETTINGS
    settings_dictionary = loop_input['settings_dictionary']

    data['predictionStart'] = t_now.strftime(format=format_string)
    data['recommendationInsulinType'] = 'novolog'
    data['maxBasalRate'] = settings_dictionary['max_basal_rate']
    data['maxBolus'] = settings_dictionary['max_bolus']
    data['suspendThreshold'] = settings_dictionary['suspend_threshold']
    
    data['recommendationType'] = 'automaticBolus'
    if settings_dictionary['partial_application_factor']:
        data['recommendationType'] = 'automaticBolus'

    # BASAL RATE
    start_times = loop_input['basal_rate_start_times']
    durations = loop_input['basal_rate_minutes']
    values = loop_input['basal_rate_values']

    for start_time, duration, value in zip(start_times, durations, values):
        start_date = datetime(t_now.year, t_now.month, t_now.day, 
            start_time.hour, start_time.minute, start_time.second
        )
        end_date = min(start_date + timedelta(minutes=duration), t_now)

        data_entry = { "endDate" : end_date.strftime(format_string),
            "startDate" : start_date.strftime(format_string),
            "value" : value }

    data['basal'] = [data_entry] 

    # SENSITIVITY
    start_times = loop_input['sensitivity_ratio_start_times']
    end_times = loop_input['sensitivity_ratio_end_times']
    values = loop_input['sensitivity_ratio_values']

    for start_time, end_time, value in zip(start_times, end_times, values):
        start_date = datetime(t_now.year, t_now.month, t_now.day, 
            start_time.hour, start_time.minute, start_time.second
        )
        end_date = datetime(t_now.year, t_now.month, t_now.day, 
            end_time.hour, end_time.minute, end_time.second
        )
        data_entry = { "endDate" : end_date.strftime(format_string),
            "startDate" : start_date.strftime(format_string),
            "value" : value }

    data['sensitivity'] = [data_entry]

    # CARB RATIO
    start_times = loop_input['carb_ratio_start_times']
    end_times = loop_input['carb_ratio_end_times']
    values = loop_input['carb_ratio_values']

    for start_time, end_time, value in zip(start_times, end_times, values):
        start_date = datetime(t_now.year, t_now.month, t_now.day, 
            start_time.hour, start_time.minute, start_time.second
        )
        end_date = datetime(t_now.year, t_now.month, t_now.day, 
            end_time.hour, end_time.minute, end_time.second
        )
        data_entry = { "endDate" : end_date.strftime(format_string),
            "startDate" : start_date.strftime(format_string),
            "value" : value }

    data['carbRatio'] = [data_entry]

    # TARGET
    start_times = loop_input['target_range_start_times']
    end_times = loop_input['target_range_end_times']
    lower_bounds = loop_input['target_range_minimum_values']
    upper_bounds = loop_input['target_range_maximum_values']

    for start_time, end_time, lower_bound, upper_bound in zip(
        start_times, end_times, lower_bounds, upper_bounds
    ):
        start_date = datetime(t_now.year, t_now.month, t_now.day, 
            start_time.hour, start_time.minute, start_time.second
        )
        end_date = datetime(t_now.year, t_now.month, t_now.day, 
            end_time.hour, end_time.minute, end_time.second
        )

        data_entry = { "endDate" : end_date.strftime(format_string),
            "startDate" : start_date.strftime(format_string),
            "lowerBound" : lower_bound,
            "upperBound" : upper_bound }

    data['target'] = [data_entry]

    # GLUCOSE
    glucose_values = loop_input['glucose_values']
    glucose_dates = loop_input['glucose_dates']

    history = []
    for value, date in zip(glucose_values, glucose_dates):
        entry = {
            'date' : date.strftime(format=format_string),  
            'value' : value
        }
        history.append(entry)

    data['glucoseHistory'] = history

    # CARB ENTRIES
    carb_values = loop_input['carb_values']
    carb_dates = loop_input['carb_dates']
    carb_absorption_times = loop_input['carb_absorption_times'] 

    history = []
    for value, date, absorption_time in zip(carb_values, carb_dates, carb_absorption_times):
        entry = {
            'date' : date.strftime(format=format_string),  
            'grams' : value,
            'absorptionTime' : absorption_time * 60
        }
        history.append(entry)

    data['carbEntries'] = history

    # TEMP BASAL DOSES
    dose_values = loop_input['dose_values']
    dose_start_times = loop_input['dose_start_times']
    dose_end_times = loop_input['dose_end_times']

    history = []
    for value, dose_start_time, dose_end_time in zip(dose_values, dose_start_times, dose_end_times):
        entry = {
            'startDate' : dose_start_time.strftime(format=format_string),  
            'endDate' : dose_end_time.strftime(format=format_string),  
            'volume' : value/12,
            'type' : 'basal'
        }
        history.append(entry)

    data['doses'] = history

    return data

def test_swift_api():
    """
    Extract data from simulator and feed into LoopAlgorithmToPython
    """
    target = 120

    t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=250)
    t0, sensor_config = get_canonical_sensor_config(start_value=250)
    t0, controller_config = get_canonical_controller_config()
    t0, pump_config = get_canonical_risk_pump_config()

    bolus_timeline = BolusTimeline(datetimes=[t0], events=[Bolus(1.0, "U")])
    patient_config.bolus_event_timeline = bolus_timeline
    pump_config.bolus_event_timeline = bolus_timeline

    true_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(20.0, "U", 180)])
    patient_config.carb_event_timeline = true_carb_timeline
    reported_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 240)])
    pump_config.carb_event_timeline = reported_carb_timeline

    new_target_range_schedule = \
        TargetRangeSchedule24hr(
            t0,
            start_times=[time(0, 0, 0)],
            values=[TargetRange(target, target, "mg/dL")],
            duration_minutes=[1440]
        )
    pump_config.target_range_schedule = new_target_range_schedule

    pump = ContinuousInsulinPump(pump_config, t0)
    sensor = IdealSensor(t0, sensor_config)

    controller = SwiftLoopController(t0, controller_config)

    vp = VirtualPatient(
        time=DATETIME_DEFAULT,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )

    sim_id = "test_swift_api"
    sim = Simulation(
        time=t0,
        duration_hrs=4,
        virtual_patient=vp,
        controller=controller,
        sim_id=sim_id
    )
    sim.run()
    
    sim_results_df = sim.get_results_df()

    plot_sim_results({sim_id: sim_results_df})
    # loop_input = sim.controller.prepare_inputs(virtual_patient=vp)

    # for _ in range(0, 20):
    #     sim.step()
    #     sim.store_state()

    # loop_input = sim.controller.prepare_inputs(virtual_patient=vp)

    # swift_input = loop_input_to_swift(loop_input)
    # swift_output = get_loop_recommendations(swift_input)

    # swift_output_decode = swift_output.decode('utf-8')
    # swift_output_json = json.loads(swift_output_decode)

    # print(json.dumps(swift_output_json, indent=2))



if __name__ == "__main__":
    # with open('../LoopAlgorithmToPython/python_tests/test_files/loop_algorithm_input.json', 'r') as f:
    #     loop_algorithm_input = json.load(f)

    # loop_output = get_loop_recommendations(loop_algorithm_input)
    # data = json.loads(loop_output)
    # print(data)
    test_swift_api()