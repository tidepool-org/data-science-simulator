__author__ = "Cameron Summers"

from datetime import timedelta

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.sensor import NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ControllerConfig
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.models.measures import Bolus, TempBasal

from pyloopkit.loop_data_manager import update


def test_do_nothing():
    t0, controller = get_canonical_controller(controller_class=DoNothingController)

    assert controller.get_state() is None
    assert isinstance(controller.controller_config, ControllerConfig)


def test_loop_controller():
    t0, controller = get_canonical_controller(controller_class=LoopController)
    t0, patient = get_canonical_risk_patient()
    patient.bolus_event_timeline.add_event(t0, Bolus(1.2, "U"))

    patient.init()
    test_results = dict()

    assert controller.get_state().pyloopkit_recommendations is None

    controller.update(t0, virtual_patient=patient)
    assert controller.recommendations is not None

    # Test prepare_inputs
    next_time = t0 + timedelta(minutes=5)
    patient.update(next_time)
    controller.time = next_time
    loop_inputs_dict = controller.prepare_inputs(patient)
    assert loop_inputs_dict['target_range_minimum_values'] is not None
    assert loop_inputs_dict['time_to_calculate_at'] == next_time
    assert len(set(loop_inputs_dict['glucose_values'])) != 1

    bolus_timeline = patient.pump.bolus_event_timeline
    bolus_timeline.merge_timeline(controller.bolus_event_timeline)
    temp_basal_timeline = patient.pump.temp_basal_event_timeline
    temp_basal_timeline.merge_timeline(controller.temp_basal_event_timeline)
    carb_timeline = patient.pump.carb_event_timeline
    carb_timeline.merge_timeline(controller.carb_event_timeline)

    bolus_dose_types, bolus_dose_values, bolus_start_times, bolus_end_times, bolus_delivered_units = \
        bolus_timeline.get_loop_inputs(next_time, num_hours_history=controller.num_hours_history)

    temp_basal_dose_types, temp_basal_dose_values, temp_basal_start_times, temp_basal_end_times, temp_basal_delivered_units = \
        temp_basal_timeline.get_loop_inputs(next_time, num_hours_history=controller.num_hours_history)

    carb_values, carb_start_times, carb_durations = \
        carb_timeline.get_loop_inputs(next_time, num_hours_history=controller.num_hours_history)

    assert loop_inputs_dict['dose_types'] == bolus_dose_types + temp_basal_dose_types
    assert loop_inputs_dict['dose_values'] == bolus_dose_values + temp_basal_dose_values
    assert loop_inputs_dict['dose_start_times'] == bolus_start_times + temp_basal_start_times
    assert loop_inputs_dict['dose_end_times'] == bolus_end_times + temp_basal_end_times
    assert loop_inputs_dict['sensitivity_ratio_values'] == [150.0]
    assert loop_inputs_dict['target_range_minimum_values'] == [100]
    assert loop_inputs_dict['target_range_maximum_values'] == [120]

    loop_algorithm_output = update(loop_inputs_dict)
    controller.apply_loop_recommendations(patient, loop_algorithm_output)
    assert patient.pump.has_active_temp_basal() is False

    while next_time != t0 + timedelta(hours=8):
        next_time = next_time + timedelta(minutes=5)
        patient.update(next_time)
        controller.time = next_time
        loop_inputs_dict = controller.prepare_inputs(patient)
        loop_algorithm_output = update(loop_inputs_dict)
        controller.apply_loop_recommendations(patient, loop_algorithm_output)

        recommended_temp_basal = controller.get_recommended_temp_basal(loop_algorithm_output)
        if recommended_temp_basal is not None and \
                patient.pump.is_valid_temp_basal(recommended_temp_basal):
            if recommended_temp_basal.scheduled_duration_minutes == 0 and recommended_temp_basal.value == 0:
                assert patient.pump.has_active_temp_basal() is False
            else:
                assert patient.pump.has_active_temp_basal()

    assert len(patient.bolus_event_timeline.events) > 0
