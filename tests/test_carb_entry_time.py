"""
Unit tests for carb entry time separation feature.

This module tests the ability to specify separate entry_time and start_time
for carb entries, mirroring Loop's handling of userCreatedDate vs startDate.

Tests cover:
1. Carb class entry_time attribute
2. JSON parsing of entry_time field
3. CarbTimeline input_time filtering
4. Validation of entry_time values
5. Backward compatibility (no entry_time specified)
"""

import pytest
import datetime
import logging

from tidepool_data_science_simulator.models.measures import Carb
from tidepool_data_science_simulator.models.events import CarbTimeline
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2, DATETIME_FORMAT


class TestCarbClassEntryTime:
    """Tests for the Carb class entry_time attribute."""
    
    def test_carb_without_entry_time(self):
        """Test that Carb works without entry_time (backward compatibility)."""
        carb = Carb(50.0, "g", 180)
        
        assert carb.value == 50.0
        assert carb.units == "g"
        assert carb.duration_minutes == 180
        assert carb.entry_time is None
        assert carb.get_entry_time() is None
        assert carb.has_separate_entry_time() is False
    
    def test_carb_with_entry_time(self):
        """Test that Carb correctly stores entry_time."""
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        carb = Carb(50.0, "g", 180, entry_time=entry_time)
        
        assert carb.value == 50.0
        assert carb.entry_time == entry_time
        assert carb.get_entry_time() == entry_time
        assert carb.has_separate_entry_time() is True
    
    def test_carb_repr_without_entry_time(self):
        """Test Carb string representation without entry_time."""
        carb = Carb(50.0, "g", 180)
        repr_str = repr(carb)
        
        assert "50.0 g" in repr_str
        assert "180min" in repr_str
        assert "entry_time" not in repr_str
    
    def test_carb_repr_with_entry_time(self):
        """Test Carb string representation with entry_time."""
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        carb = Carb(50.0, "g", 180, entry_time=entry_time)
        repr_str = repr(carb)
        
        assert "50.0 g" in repr_str
        assert "180min" in repr_str
        assert "entry_time" in repr_str


class TestCarbTimelineEntryTime:
    """Tests for CarbTimeline with entry_time (input_time) support."""
    
    def test_add_event_with_input_time(self):
        """Test adding carb event with separate input_time."""
        timeline = CarbTimeline()
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        carb = Carb(50.0, "g", 180, entry_time=entry_time)
        
        timeline.add_event(time=start_time, event=carb, input_time=entry_time)
        
        # Event should be keyed by start_time
        assert start_time in timeline.events
        assert timeline.events[start_time] == carb
        
        # Input time should be tracked
        assert timeline.events_input[carb] == entry_time
    
    def test_add_event_without_input_time(self):
        """Test adding carb event without input_time (defaults to event time)."""
        timeline = CarbTimeline()
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = Carb(50.0, "g", 180)
        
        timeline.add_event(time=start_time, event=carb)
        
        # Event should be keyed by start_time
        assert start_time in timeline.events
        
        # Input time should default to start_time
        assert timeline.events_input[carb] == start_time
    
    def test_get_recent_event_times_filters_by_input_time(self):
        """Test that get_recent_event_times respects input_time for filtering."""
        timeline = CarbTimeline()
        
        # Carb consumed at 12:00, entered at 12:30
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        carb = Carb(50.0, "g", 180, entry_time=entry_time)
        
        timeline.add_event(time=start_time, event=carb, input_time=entry_time)
        
        # At 12:15, the carb should NOT be visible (entry not yet made)
        query_time_before_entry = datetime.datetime(2019, 8, 15, 12, 15, 0)
        recent_before = timeline.get_recent_event_times(
            time=query_time_before_entry, 
            num_hours_history=6
        )
        assert start_time not in recent_before
        
        # At 12:45, the carb SHOULD be visible (entry has been made)
        query_time_after_entry = datetime.datetime(2019, 8, 15, 12, 45, 0)
        recent_after = timeline.get_recent_event_times(
            time=query_time_after_entry, 
            num_hours_history=6
        )
        assert start_time in recent_after
    
    def test_get_recent_event_times_pre_bolus_scenario(self):
        """Test pre-bolus scenario where entry_time < start_time."""
        timeline = CarbTimeline()
        
        # Carb will be consumed at 12:30, entered at 12:00 (pre-bolus)
        start_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = Carb(50.0, "g", 180, entry_time=entry_time)
        
        timeline.add_event(time=start_time, event=carb, input_time=entry_time)
        
        # At 12:15, the carb SHOULD be visible (entry was made at 12:00)
        query_time = datetime.datetime(2019, 8, 15, 12, 15, 0)
        recent = timeline.get_recent_event_times(
            time=query_time, 
            num_hours_history=6
        )
        assert start_time in recent


class TestScenarioParserCarbEntryTime:
    """Tests for ScenarioParserV2 carb entry time parsing."""
    
    @pytest.fixture
    def parser(self):
        """Create a ScenarioParserV2 instance for testing."""
        return ScenarioParserV2()
    
    def test_parse_carb_entry_without_entry_time(self, parser):
        """Test parsing carb entry without entry_time (backward compatibility)."""
        carb_entries = [
            {
                "start_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "duration": 180
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        assert start_time in timeline.events
        
        carb = timeline.events[start_time]
        assert carb.value == 50.0
        assert carb.duration_minutes == 180
        # entry_time should equal start_time when not specified
        assert carb.entry_time == start_time
    
    def test_parse_carb_entry_with_entry_time(self, parser):
        """Test parsing carb entry with explicit entry_time."""
        carb_entries = [
            {
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:30:00",
                "value": 50.0,
                "duration": 180
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        
        assert start_time in timeline.events
        
        carb = timeline.events[start_time]
        assert carb.value == 50.0
        assert carb.entry_time == entry_time
        
        # Check that input_time is properly set in timeline
        assert timeline.events_input[carb] == entry_time
    
    def test_parse_carb_entry_pre_bolus(self, parser):
        """Test parsing pre-bolus carb entry (entry before consumption)."""
        carb_entries = [
            {
                "start_time": "8/15/2019 12:30:00",
                "entry_time": "8/15/2019 12:00:00",
                "value": 50.0
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        start_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        
        assert start_time in timeline.events
        
        carb = timeline.events[start_time]
        assert carb.entry_time == entry_time
        assert timeline.events_input[carb] == entry_time
    
    def test_parse_multiple_carb_entries_mixed(self, parser):
        """Test parsing multiple carb entries with mixed entry_time usage."""
        carb_entries = [
            {
                "start_time": "8/15/2019 08:00:00",
                "value": 30.0
            },
            {
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:30:00",
                "value": 50.0
            },
            {
                "start_time": "8/15/2019 18:00:00",
                "entry_time": "8/15/2019 17:45:00",
                "value": 60.0
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        # Check first entry (no entry_time)
        time1 = datetime.datetime(2019, 8, 15, 8, 0, 0)
        assert time1 in timeline.events
        carb1 = timeline.events[time1]
        assert carb1.entry_time == time1  # Should default to start_time
        
        # Check second entry (late entry)
        time2 = datetime.datetime(2019, 8, 15, 12, 0, 0)
        entry_time2 = datetime.datetime(2019, 8, 15, 12, 30, 0)
        carb2 = timeline.events[time2]
        assert carb2.entry_time == entry_time2
        
        # Check third entry (pre-bolus)
        time3 = datetime.datetime(2019, 8, 15, 18, 0, 0)
        entry_time3 = datetime.datetime(2019, 8, 15, 17, 45, 0)
        carb3 = timeline.events[time3]
        assert carb3.entry_time == entry_time3


class TestCarbEntryTimeValidation:
    """Tests for carb entry time validation."""
    
    @pytest.fixture
    def parser(self):
        """Create a ScenarioParserV2 instance for testing."""
        return ScenarioParserV2()
    
    def test_validate_reasonable_time_difference(self, parser, caplog):
        """Test that reasonable time differences don't trigger warnings."""
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)  # 30 min difference
        
        with caplog.at_level(logging.WARNING):
            parser.validate_carb_entry_times(start_time, entry_time)
        
        assert "may not reflect realistic user behavior" not in caplog.text
    
    def test_validate_large_time_difference_warns(self, parser, caplog):
        """Test that large time differences trigger warnings."""
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        entry_time = datetime.datetime(2019, 8, 15, 20, 0, 0)  # 8 hour difference
        
        with caplog.at_level(logging.WARNING):
            parser.validate_carb_entry_times(start_time, entry_time)
        
        assert "may not reflect realistic user behavior" in caplog.text
        assert "8.0 hours" in caplog.text
    
    def test_validate_negative_time_difference(self, parser, caplog):
        """Test pre-bolus scenario (entry before consumption) within limits."""
        start_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)  # 30 min before
        
        with caplog.at_level(logging.WARNING):
            parser.validate_carb_entry_times(start_time, entry_time)
        
        # Should not warn for reasonable pre-bolus
        assert "may not reflect realistic user behavior" not in caplog.text


class TestCarbEntryTimeIntegration:
    """Integration tests for carb entry time in simulation context."""
    
    @pytest.fixture
    def parser(self):
        """Create a ScenarioParserV2 instance for testing."""
        return ScenarioParserV2()
    
    def test_late_entry_visibility_in_timeline(self, parser):
        """Test that late carb entries are only visible after entry_time."""
        carb_entries = [
            {
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:30:00",
                "value": 50.0,
                "duration": 180
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        
        # Simulate Loop querying at different times
        times_to_check = [
            (datetime.datetime(2019, 8, 15, 12, 15, 0), False, "before entry"),
            (datetime.datetime(2019, 8, 15, 12, 30, 0), True, "at entry"),
            (datetime.datetime(2019, 8, 15, 13, 0, 0), True, "after entry"),
        ]
        
        for query_time, should_be_visible, description in times_to_check:
            recent = timeline.get_recent_event_times(
                time=query_time, 
                num_hours_history=6
            )
            
            if should_be_visible:
                assert start_time in recent, f"Carb should be visible {description}"
            else:
                assert start_time not in recent, f"Carb should NOT be visible {description}"
    
    def test_get_loop_inputs_respects_entry_time(self, parser):
        """Test that get_loop_inputs method respects entry_time filtering."""
        carb_entries = [
            {
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:30:00",
                "value": 50.0,
                "duration": 180
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        # Query before entry_time - should return empty
        query_time_before = datetime.datetime(2019, 8, 15, 12, 15, 0)
        carb_values, carb_start_times, carb_durations = timeline.get_loop_inputs(
            time=query_time_before, 
            num_hours_history=6
        )
        assert len(carb_values) == 0
        
        # Query after entry_time - should return the carb
        query_time_after = datetime.datetime(2019, 8, 15, 13, 0, 0)
        carb_values, carb_start_times, carb_durations = timeline.get_loop_inputs(
            time=query_time_after, 
            num_hours_history=6
        )
        assert len(carb_values) == 1
        assert carb_values[0] == 50.0
        assert carb_durations[0] == 180


class TestStateEntryTimeMethods:
    """Tests for entry_time methods in VirtualPatientState and PumpState."""
    
    def test_virtual_patient_state_get_carb_entry_time_with_entry_time(self):
        """Test VirtualPatientState.get_carb_entry_time() when entry_time is set."""
        from tidepool_data_science_simulator.models.state import VirtualPatientState
        
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        carb = Carb(50.0, "g", 180, entry_time=entry_time)
        
        state = VirtualPatientState(carb=carb)
        
        assert state.get_carb_entry_time() == entry_time
    
    def test_virtual_patient_state_get_carb_entry_time_without_entry_time(self):
        """Test VirtualPatientState.get_carb_entry_time() when entry_time is not set."""
        from tidepool_data_science_simulator.models.state import VirtualPatientState
        
        carb = Carb(50.0, "g", 180)  # No entry_time
        state = VirtualPatientState(carb=carb)
        
        assert state.get_carb_entry_time() is None
    
    def test_virtual_patient_state_get_carb_entry_time_no_carb(self):
        """Test VirtualPatientState.get_carb_entry_time() when no carb is set."""
        from tidepool_data_science_simulator.models.state import VirtualPatientState
        
        state = VirtualPatientState()  # No carb
        
        assert state.get_carb_entry_time() is None
    
    def test_pump_state_get_carb_entry_time_with_entry_time(self):
        """Test PumpState.get_carb_entry_time() when entry_time is set."""
        from tidepool_data_science_simulator.models.state import PumpState
        
        entry_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        carb = Carb(50.0, "g", 180, entry_time=entry_time)
        
        state = PumpState(carb=carb)
        
        assert state.get_carb_entry_time() == entry_time
    
    def test_pump_state_get_carb_entry_time_without_entry_time(self):
        """Test PumpState.get_carb_entry_time() when entry_time is not set."""
        from tidepool_data_science_simulator.models.state import PumpState
        
        carb = Carb(50.0, "g", 180)  # No entry_time
        state = PumpState(carb=carb)
        
        assert state.get_carb_entry_time() is None
    
    def test_pump_state_get_carb_entry_time_no_carb(self):
        """Test PumpState.get_carb_entry_time() when no carb is set."""
        from tidepool_data_science_simulator.models.state import PumpState
        
        state = PumpState()  # No carb
        
        assert state.get_carb_entry_time() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
