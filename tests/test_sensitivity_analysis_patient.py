__author__ = "Jason Meno"

from tidepool_data_science_simulator.models.patient_for_icgm_sensitivity_analysis import VirtualPatientISA
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.pump import Omnipod
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor import IdealSensor

# %%
def test_sensitivity_analysis_patient():
    """Tests the VirtualPatientISA class's handling of different analysis types"""

    scenario_csv_filepath = "tests/data/Scenario-test-0-carbs-0-insulin.tsv"
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    analysis_type_list = ['temp_basal_only', 'correction_bolus', 'meal_bolus']
    all_results = {}
    for analysis_type in analysis_type_list:
        print("Running: {}".format(analysis_type))

        controller = LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config(),
        )
        controller.num_hours_history = 8  # Force 8 hours to look for historical boluses

        pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())
        sensor = IdealSensor(time=t0, sensor_config=sim_parser.get_sensor_config())

        vp = VirtualPatientISA(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=sim_parser.get_patient_config(),
            t0=t0,
            analysis_type=analysis_type
        )

        assert vp.analysis_type == analysis_type

        simulation = Simulation(
            time=t0,
            duration_hrs=8.0,
            virtual_patient=vp,
            controller=controller,
        )

        simulation.run()

        results_df = simulation.get_results_df()
        all_results[analysis_type] = results_df

    assert all_results['temp_basal_only'].true_carb_value.sum() == 0
    assert all_results['correction_bolus'].true_carb_value.sum() == 0
    assert all_results['meal_bolus'].true_carb_value[0] == 30

    assert all_results['temp_basal_only'].true_bolus.sum() == 0
    assert all_results['correction_bolus'].true_bolus[1:].sum() == 0
    # CAS - Fix/update 2020-08-12 - Bolus recs come in on next patient time
    assert all_results['meal_bolus'].true_bolus[2:].sum() == 0



