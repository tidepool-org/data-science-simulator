from functools import partial

import numpy as np
from numpy.random import RandomState
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import (
    get_canonical_risk_patient_config, get_canonical_risk_pump_config, get_canonical_sensor_config)
from tidepool_data_science_simulator.models.controller import DoNothingController
from tidepool_data_science_simulator.models.events import BolusTimeline
from tidepool_data_science_simulator.models.measures import Bolus
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
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


def run_bolus_sim(metabolism_model):

    t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=250)
    _, sensor_config = get_canonical_sensor_config(start_value=250)
    _, controller_config = get_canonical_controller_config()
    _, pump_config = get_canonical_risk_pump_config()
    patient_config.bolus_event_timeline = BolusTimeline(datetimes=[t0], events=[Bolus(1.0, "U")])
    pump_config.bolus_event_timeline = BolusTimeline(datetimes=[t0], events=[Bolus(1.0, "U")])

    vp = VirtualPatient(
        time=t0,
        pump=ContinuousInsulinPump(pump_config, t0),
        sensor=IdealSensor(t0, sensor_config),
        metabolism_model=metabolism_model,
        patient_config=patient_config,
    )
    sim = Simulation(time=t0, duration_hrs=2.0, virtual_patient=vp,
                     controller=DoNothingController(t0, controller_config), sim_id="Test Sim")
    sim.run()
    return vp, sim.get_results_df()["bg"].to_numpy()


def test_simulation_runs():

    vp, bg_new = run_bolus_sim(partial(EffectDelayMetabolismModel, timing_factor=1.2, tau3=ACTION_DELAY_BY_SPREAD[0.26]))
    _, bg_palerm = run_bolus_sim(SimpleMetabolismModel)

    assert np.all(np.isfinite(bg_new))
    assert vp.instantiate_metabolism_model().insulin_model._timing_factor == 1.2
    assert bg_new[-1] > bg_palerm[-1]  # delayed curve has lowered glucose less by 2 h
