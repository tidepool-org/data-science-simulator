__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.makedata.scenario_parser import ControllerConfig


def test_do_nothing():
    t0, controller = get_canonical_controller(controller_class=DoNothingController)

    assert controller.get_state() is None
    assert isinstance(controller.controller_config, ControllerConfig)


def test_loop_controller():
    t0, controller = get_canonical_controller(controller_class=LoopController)
