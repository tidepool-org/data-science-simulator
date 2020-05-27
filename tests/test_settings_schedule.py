__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_simulator.models.simulation import SettingSchedule24Hr
from tidepool_data_science_simulator.models.measures import BasalRate


def test_basal_rate_retrieval():

    basal_rate_start_times = [datetime.time(hour=0, minute=0, second=0)]
    basal_rate_minutes = [1440]
    basal_rate_values = [0.3]

    test_start_time = datetime.datetime.fromisoformat("2020-01-01 00:00:00")

    basal_schedule = SettingSchedule24Hr(
        time=test_start_time,
        name="Basal",
        start_times=basal_rate_start_times,
        values=[BasalRate(value, 'U/hr') for value in basal_rate_values],
        duration_minutes=basal_rate_minutes,
    )

    assert basal_schedule.get_state().value == 0.3

    # Multiple values
    basal_rate_start_times = [
        datetime.time(hour=0, minute=0, second=0),
        datetime.time(hour=12, minute=0, second=0)
    ]
    basal_rate_minutes = [720, 720]
    basal_rate_values = [0.3, 0.4]

    test_start_time = datetime.datetime.fromisoformat("2020-01-01 00:00:00")
    basal_schedule = SettingSchedule24Hr(
        time=test_start_time,
        name="Basal",
        start_times=basal_rate_start_times,
        values=[BasalRate(value, 'U/hr') for value in basal_rate_values],
        duration_minutes=basal_rate_minutes,
    )

    assert basal_schedule.get_state().value == 0.3

    basal_schedule.update(datetime.datetime.fromisoformat("2020-01-01 13:00:00"))
    assert basal_schedule.get_state().value == 0.4

    basal_schedule.update(datetime.datetime.fromisoformat("2020-01-01 23:59:59"))
    assert basal_schedule.get_state().value == 0.4

    basal_schedule.update(datetime.datetime.fromisoformat("2020-01-01 01:00:00"))
    assert basal_schedule.get_state().value == 0.3


