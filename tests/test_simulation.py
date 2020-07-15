__author__ = "Eden Grown-Haeberli"

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_patient import DATETIME_DEFAULT

def test_simulation():

    t0, sim = get_canonical_simulation()

    # Initialization checks
    assert sim.time == DATETIME_DEFAULT
    assert sim.duration_hrs == 8
    #assert sim.simulation_results[t0]

    # Run checks
