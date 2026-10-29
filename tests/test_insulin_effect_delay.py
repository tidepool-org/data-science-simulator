from functools import partial

import numpy as np
from numpy.random import RandomState

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.models.insulin_effect_delay import (
    ACTION_DELAY_BY_SPREAD,
    chain_density,
    draw_timing_factors,
    EffectDelayInsulinModel,
    EffectDelayMetabolismModel,
)


def test_curve():

    dt = 0.05
    t_min = np.arange(0, 721, dt)
    palerm = (np.exp(-t_min / 70) - np.exp(-t_min / 55)) / (70 - 55)
    rate = chain_density(55, 70, 31.6, t_min)

    # Independent check: pass the Palerm curve through a third exponential step numerically
    delay = np.exp(-t_min / 31.6) / 31.6
    numerical = np.convolve(palerm, delay)[:len(t_min)] * dt
    assert np.max(np.abs(rate - numerical)) < 0.002 * rate.max()
    assert abs(t_min[np.argmax(rate)] - 99.5) < 0.5  # peak glucose lowering, minutes
    assert abs(np.sum(rate) * dt - 1) < 1e-3  # unit area by 12 h


def test_timing_factor():

    factors = draw_timing_factors(20000, 0.26, RandomState(0))
    assert abs(np.median(factors) - 1) < 0.01
    assert abs(np.std(factors) / np.mean(factors) - 0.26) < 0.01

    # A factor of 1.2 stretches the whole curve: half the effect is done 1.2 times later
    halfway = []
    for k in (1.0, 1.2):
        _, _, bg, _ = EffectDelayInsulinModel(isf=50, cir=10, timing_factor=k).run(num_hours=12, insulin_amount=1, five_min=False)
        halfway.append(np.argmax(bg <= bg[-1] / 2))
    assert abs(halfway[1] / halfway[0] - 1.2) < 0.01


def test_simulation_runs():

    sim_parser = ScenarioParserCSV("tests/test_data/Scenario-0-simulation-template - inputs.tsv")
    t0 = sim_parser.get_simulation_start_time()
    vp = VirtualPatient(
        time=t0,
        pump=ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config()),
        sensor=IdealSensor(time=t0, sensor_config=sim_parser.get_sensor_config()),
        metabolism_model=partial(EffectDelayMetabolismModel, timing_factor=1.2, tau3=ACTION_DELAY_BY_SPREAD[0.26]),
        patient_config=sim_parser.get_patient_config(),
    )
    _, controller_config = get_canonical_controller_config()
    controller = LoopController(time=t0, controller_config=controller_config)

    Simulation(time=t0, duration_hrs=2.0, virtual_patient=vp, controller=controller, sim_id="Test Sim").run()

    assert np.isfinite(vp.bg_current)
    assert vp.instantiate_metabolism_model().insulin_model._timing_factor == 1.2
