__author__ = "Cameron Summers"

from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline
from tidepool_data_science_simulator.makedata.scenario_parser import ControllerConfig

from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.makedata.make_patient import DATETIME_DEFAULT


def get_canonical_controller_config(t0=DATETIME_DEFAULT):

    controller_settings = {
        "model": [360.0, 65],
        "momentum_data_interval": 15,
        "suspend_threshold": 70,
        "dynamic_carb_absorption_enabled": True,
        "retrospective_correction_integration_interval": True,
        "minimum_autobolus": 0.0,
        "maximum_autobolus": 0.0,
        "partial_application_factor": None,
        "recency_interval": 15,
        "retrospective_correction_grouping_interval": 30,
        "rate_rounder": 0.05,
        "insulin_delay": 10,
        "carb_delay": 10,
        "default_absorption_times": [30.0, 120.0, 180.0, 240.0, 300.0],
        "max_basal_rate": 35,
        "max_bolus": 30,
        "retrospective_correction_enabled": True,
        "partial_application_factor": 0.4,
        "use_mid_absorption_isf": False
    }
    controller_config = ControllerConfig(
        bolus_event_timeline=BolusTimeline(),
        carb_event_timeline=CarbTimeline(),
        controller_settings=controller_settings
    )

    return t0, controller_config


def get_canonical_controller(t0, controller_class, controller_config=None):

    if controller_config is None:
        t0, controller_config = get_canonical_controller_config(t0)

    controller = controller_class(t0, controller_config)

    return t0, controller