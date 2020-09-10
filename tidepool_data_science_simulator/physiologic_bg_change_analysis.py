__author__ = "Cameron Summers"

import pdb
import time
import copy
import numpy as np

print("***************** Warning: Overriding Pyloopkit with Local Copy ***************************")
import sys
sys.path.insert(0, "/mnt/cameronsummers/dev/PyLoopKit/")


from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import ActionTimeline
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatientModelCarbBolusAccept
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing, save_df, get_sim_results_save_dir

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_sensor_config, \
    get_pump_config, get_variable_risk_patient_config

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index

from numpy.random import RandomState


@timing
def compare_physiologic_bg_change_cap(save_dir, save_results):
    """
    Compare two controllers for a given scenario file:
        1. No controller, ie no insulin modulation except for pump schedule
        2. Loop controller

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    prng = RandomState(1234567890)

    t0, controller_config = get_canonical_controller_config()

    no_cap_controller_config = copy.deepcopy(controller_config)
    no_cap_loop_controller = LoopController(time=t0, controller_config=no_cap_controller_config)

    controllers = [no_cap_loop_controller]
    for max_rate in [3.0, 5.0, 7.0, 9.0]:
        capped_controller_config = copy.deepcopy(controller_config)
        capped_controller_config.controller_settings["max_physiologic_slope"] = max_rate
        capped_loop_controller = LoopController(time=t0, controller_config=capped_controller_config)
        capped_loop_controller.name = "PyloopKit_BG_Change_Max={}".format(max_rate)
        controllers.append(capped_loop_controller)

    num_patients = 30
    virtual_patients = []
    for i in range(num_patients):

        t0, sensor_config = get_canonical_sensor_config()
        sensor_config.std_dev = prng.uniform(5, 15)
        sensor = NoisySensor(time=t0, sensor_config=sensor_config)

        t0, patient_config = get_variable_risk_patient_config(prng)

        patient_config.recommendation_accept_prob = prng.uniform(0.8, 0.99)
        patient_config.action_timeline = ActionTimeline()

        t0, pump_config = get_pump_config(prng)
        pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)

        vp = VirtualPatientModelCarbBolusAccept(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
            remember_meal_bolus_prob=prng.uniform(0.9, 1.0),
            correct_bolus_bg_threshold=prng.uniform(140, 190),
            correct_bolus_delay_minutes=prng.uniform(20, 40),
            correct_carb_bg_threshold=prng.uniform(70, 90),
            correct_carb_delay_minutes=prng.uniform(5, 15),
            carb_count_noise_percentage=prng.uniform(0.1, 0.25)
        )
        vp.name = "vp{}".format(i)
        vp.patient_config.min_bolus_rec_threshold = prng.uniform(0.4, 0.6)
        virtual_patients.append(vp)

    sims = {}
    for controller in controllers:
        for vp in virtual_patients:
            sim_id = "{}_{}".format(vp.name, controller.name)

            sim = Simulation(
                time=t0,
                duration_hrs=4*7*24, # 4 weeks
                virtual_patient=vp,
                controller=controller,
                multiprocess=True,
            )
            sim.seed = 1234
            sims[sim_id] = sim

    num_sims = len(sims)
    sim_ctr = 1
    start_time = time.time()
    num_procs = 20
    running_sims = {}
    for sim_id, sim in sims.items():
        print("Running: {}. {} of {}".format(sim_id, sim_ctr, num_sims))
        sim.start()
        sim_ctr += 1
        running_sims[sim_id] = sim

        if len(running_sims) >= num_procs:
            all_results = {id: sim.queue.get() for id, sim in running_sims.items()}
            [sim.join() for id, sim in running_sims.items()]
            running_sims = {}

            for sim_id, results_df in all_results.items():
                lbgi, hbgi, brgi = blood_glucose_risk_index(results_df['bg'])
                print(sim_id, lbgi, hbgi, brgi)

                if save_results:
                    save_df(results_df, sim_id, save_dir)

    print("Run time sec:", time.time() - start_time)
    #plot_sim_results(all_results, save=False)

if __name__ == "__main__":
    results_dir = get_sim_results_save_dir()
    compare_physiologic_bg_change_cap(save_dir=results_dir, save_results=True)

    # TODO:
    #   Delays for carb/bolus times
    #   Random seed alignment for meal carbs?
