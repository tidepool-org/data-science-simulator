"""
Unit tests for Carb Editing functionality.

Tests the version-based supercession model for carb entries, mirroring Loop's
CarbStore edit handling.
"""

import unittest
import datetime
import uuid

from tidepool_data_science_simulator.models.measures import Carb, CarbOperation
from tidepool_data_science_simulator.models.events import CarbTimeline
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2


class TestCarbClassVersioning(unittest.TestCase):
    """Test Carb class with versioning attributes."""
    
    def test_carb_default_versioning(self):
        """Test that default Carb has v0 CREATE operation."""
        carb = Carb(50, "g", 180)
        
        self.assertEqual(carb.sync_version, 0)
        self.assertEqual(carb.operation, CarbOperation.CREATE)
        self.assertIsNone(carb.superceded_date)
        self.assertIsNone(carb.user_updated_date)
        self.assertTrue(carb.is_active())
    
    def test_carb_with_sync_identifier(self):
        """Test Carb with explicit sync_identifier."""
        sync_id = "test-carb-123"
        carb = Carb(
            50, "g", 180,
            sync_identifier=sync_id,
            sync_version=0
        )
        
        self.assertEqual(carb.get_sync_identifier(), sync_id)
        self.assertEqual(carb.get_sync_version(), 0)
    
    def test_carb_versioning_attributes(self):
        """Test all versioning attributes."""
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = Carb(
            50, "g", 180,
            entry_time=entry_time,
            sync_identifier="carb-1",
            sync_version=0,
            user_created_date=entry_time,
            user_updated_date=None,
            superceded_date=None,
            operation=CarbOperation.CREATE
        )
        
        self.assertEqual(carb.get_entry_time(), entry_time)
        self.assertEqual(carb.get_sync_identifier(), "carb-1")
        self.assertEqual(carb.get_sync_version(), 0)
        self.assertEqual(carb.get_user_created_date(), entry_time)
        self.assertIsNone(carb.get_user_updated_date())
        self.assertIsNone(carb.get_superceded_date())
        self.assertEqual(carb.get_operation(), CarbOperation.CREATE)
    
    def test_carb_is_active(self):
        """Test is_active() method."""
        # Active carb
        carb = Carb(50, "g", 180, operation=CarbOperation.CREATE)
        self.assertTrue(carb.is_active())
        
        # Superceded carb
        carb_superceded = Carb(
            50, "g", 180,
            superceded_date=datetime.datetime.now(),
            operation=CarbOperation.CREATE
        )
        self.assertFalse(carb_superceded.is_active())
        
        # Deleted carb
        carb_deleted = Carb(50, "g", 180, operation=CarbOperation.DELETE)
        self.assertFalse(carb_deleted.is_active())
    
    def test_carb_is_visible_at_time(self):
        """Test is_visible_at_time() method."""
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        superceded_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        
        carb = Carb(
            50, "g", 180,
            entry_time=entry_time,
            superceded_date=superceded_time,
            operation=CarbOperation.CREATE
        )
        
        # Before entry - not visible
        before_entry = datetime.datetime(2019, 8, 15, 11, 59, 0)
        self.assertFalse(carb.is_visible_at_time(before_entry))
        
        # After entry, before superceded - visible
        during_active = datetime.datetime(2019, 8, 15, 12, 15, 0)
        self.assertTrue(carb.is_visible_at_time(during_active))
        
        # After superceded - not visible
        after_superceded = datetime.datetime(2019, 8, 15, 13, 0, 0)
        self.assertFalse(carb.is_visible_at_time(after_superceded))
    
    def test_create_edited_version(self):
        """Test create_edited_version() method."""
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        original = Carb(
            50, "g", 180,
            entry_time=entry_time,
            sync_identifier="carb-1",
            sync_version=0,
            user_created_date=entry_time
        )
        
        edit_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        edited = original.create_edited_version(
            edit_time=edit_time,
            new_value=75.0
        )
        
        # Check edited version
        self.assertEqual(edited.value, 75.0)
        self.assertEqual(edited.get_sync_identifier(), "carb-1")  # Same ID
        self.assertEqual(edited.get_sync_version(), 1)  # Incremented
        self.assertEqual(edited.get_user_created_date(), entry_time)  # Preserved
        self.assertEqual(edited.get_user_updated_date(), edit_time)  # Set
        self.assertIsNone(edited.get_superceded_date())  # Active
        self.assertEqual(edited.get_operation(), CarbOperation.UPDATE)


class TestCarbTimelineVersioning(unittest.TestCase):
    """Test CarbTimeline with version tracking."""
    
    def test_add_event_tracks_versions(self):
        """Test that add_event tracks versions in all_carb_versions."""
        timeline = CarbTimeline()
        
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        
        carb = Carb(
            50, "g", 180,
            entry_time=entry_time,
            sync_identifier="carb-1",
            sync_version=0
        )
        
        timeline.add_event(start_time, carb, input_time=entry_time)
        
        self.assertIn("carb-1", timeline.all_carb_versions)
        self.assertEqual(len(timeline.all_carb_versions["carb-1"]), 1)
        self.assertEqual(timeline.all_carb_versions["carb-1"][0], carb)
    
    def test_add_carb_edit_supercedes_previous(self):
        """Test that add_carb_edit marks previous version as superceded."""
        timeline = CarbTimeline()
        
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        edit_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        
        # Add original
        original = Carb(
            50, "g", 180,
            entry_time=entry_time,
            sync_identifier="carb-1",
            sync_version=0,
            user_created_date=entry_time
        )
        timeline.add_event(start_time, original, input_time=entry_time)
        
        # Create and add edit
        edited = Carb(
            75, "g", 180,
            entry_time=edit_time,
            sync_identifier="carb-1",
            sync_version=1,
            user_created_date=entry_time,
            user_updated_date=edit_time,
            operation=CarbOperation.UPDATE
        )
        timeline.add_carb_edit(start_time, edited)
        
        # Check original is superceded
        self.assertEqual(original.superceded_date, edit_time)
        
        # Check both versions are tracked
        self.assertEqual(len(timeline.all_carb_versions["carb-1"]), 2)
    
    def test_get_active_carb_at_query_time(self):
        """Test getting active carb version at different times."""
        timeline = CarbTimeline()
        
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        edit_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        
        # Add original (50g)
        original = Carb(
            50, "g", 180,
            entry_time=entry_time,
            sync_identifier="carb-1",
            sync_version=0,
            user_created_date=entry_time
        )
        timeline.add_event(start_time, original, input_time=entry_time)
        
        # Add edit (75g)
        edited = Carb(
            75, "g", 180,
            entry_time=edit_time,
            sync_identifier="carb-1",
            sync_version=1,
            user_created_date=entry_time,
            user_updated_date=edit_time,
            operation=CarbOperation.UPDATE
        )
        timeline.add_carb_edit(start_time, edited)
        
        # Before edit: should see 50g
        query_before_edit = datetime.datetime(2019, 8, 15, 12, 15, 0)
        active_before = timeline.get_active_carb_at_query_time(start_time, query_before_edit)
        self.assertEqual(active_before.value, 50)
        
        # After edit: should see 75g
        query_after_edit = datetime.datetime(2019, 8, 15, 13, 0, 0)
        active_after = timeline.get_active_carb_at_query_time(start_time, query_after_edit)
        self.assertEqual(active_after.value, 75)
    
    def test_get_recent_event_times_filters_correctly(self):
        """Test that get_recent_event_times filters by version timing."""
        timeline = CarbTimeline()
        
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        edit_time = datetime.datetime(2019, 8, 15, 12, 30, 0)
        delete_time = datetime.datetime(2019, 8, 15, 14, 0, 0)
        
        # Add original
        original = Carb(
            50, "g", 180,
            entry_time=entry_time,
            sync_identifier="carb-1",
            sync_version=0,
            user_created_date=entry_time
        )
        timeline.add_event(start_time, original, input_time=entry_time)
        
        # Add delete
        deleted = Carb(
            50, "g", 180,
            entry_time=delete_time,
            sync_identifier="carb-1",
            sync_version=1,
            user_created_date=entry_time,
            user_updated_date=delete_time,
            operation=CarbOperation.DELETE
        )
        timeline.add_carb_edit(start_time, deleted)
        
        # Before delete: should include carb
        query_before_delete = datetime.datetime(2019, 8, 15, 13, 0, 0)
        events_before = timeline.get_recent_event_times(query_before_delete, num_hours_history=6)
        self.assertIn(start_time, events_before)
        
        # After delete: should not include carb
        query_after_delete = datetime.datetime(2019, 8, 15, 15, 0, 0)
        events_after = timeline.get_recent_event_times(query_after_delete, num_hours_history=6)
        self.assertNotIn(start_time, events_after)
    
    def test_get_all_versions(self):
        """Test get_all_versions method."""
        timeline = CarbTimeline()
        
        entry_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        
        # Add carb with ID
        carb = Carb(
            50, "g", 180,
            entry_time=entry_time,
            sync_identifier="carb-1",
            sync_version=0
        )
        timeline.add_event(start_time, carb)
        
        # Get all versions
        all_versions = timeline.get_all_versions()
        self.assertIn("carb-1", all_versions)
        
        # Get versions for specific ID
        carb_versions = timeline.get_all_versions("carb-1")
        self.assertEqual(len(carb_versions), 1)
        
        # Get versions for non-existent ID
        no_versions = timeline.get_all_versions("non-existent")
        self.assertEqual(len(no_versions), 0)


class TestScenarioParserCarbEditing(unittest.TestCase):
    """Test JSON parsing of carb entries with edits."""
    
    def setUp(self):
        self.parser = ScenarioParserV2()
    
    def test_parse_carb_with_no_edits(self):
        """Test parsing carb entry without edits (backward compatible)."""
        carb_entries = [
            {
                "start_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "duration": 180
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        # Should have one event
        self.assertEqual(len(timeline.events), 1)
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = timeline.get_event(start_time)
        
        self.assertEqual(carb.value, 50.0)
        self.assertEqual(carb.get_sync_version(), 0)
        self.assertEqual(carb.get_operation(), CarbOperation.CREATE)
        self.assertIsNotNone(carb.get_sync_identifier())  # Auto-generated
    
    def test_parse_carb_with_explicit_id(self):
        """Test parsing carb entry with explicit ID."""
        carb_entries = [
            {
                "id": "my-carb-id",
                "start_time": "8/15/2019 12:00:00",
                "value": 50.0
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = timeline.get_event(start_time)
        
        self.assertEqual(carb.get_sync_identifier(), "my-carb-id")
    
    def test_parse_carb_with_single_edit(self):
        """Test parsing carb entry with one edit."""
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "duration": 180,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "value": 75.0}
                ]
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        # Should have v0 in timeline
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = timeline.get_event(start_time)
        
        # Check all versions are tracked
        versions = timeline.get_all_versions("carb-1")
        self.assertEqual(len(versions), 2)
        
        # Check v0 is superceded
        v0 = [v for v in versions if v.sync_version == 0][0]
        self.assertIsNotNone(v0.superceded_date)
        self.assertEqual(v0.value, 50.0)
        
        # Check v1 is active
        v1 = [v for v in versions if v.sync_version == 1][0]
        self.assertIsNone(v1.superceded_date)
        self.assertEqual(v1.value, 75.0)
        self.assertEqual(v1.get_operation(), CarbOperation.UPDATE)
    
    def test_parse_carb_with_multiple_edits(self):
        """Test parsing carb entry with multiple edits."""
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "value": 75.0},
                    {"edit_time": "8/15/2019 13:00:00", "value": 80.0},
                    {"edit_time": "8/15/2019 13:30:00", "value": 85.0}
                ]
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        # Should have 4 versions (v0, v1, v2, v3)
        versions = timeline.get_all_versions("carb-1")
        self.assertEqual(len(versions), 4)
        
        # Check versions are correct
        versions_sorted = sorted(versions, key=lambda c: c.sync_version)
        self.assertEqual(versions_sorted[0].value, 50.0)  # v0
        self.assertEqual(versions_sorted[1].value, 75.0)  # v1
        self.assertEqual(versions_sorted[2].value, 80.0)  # v2
        self.assertEqual(versions_sorted[3].value, 85.0)  # v3
        
        # Only v3 should be active
        for v in versions_sorted[:-1]:
            self.assertIsNotNone(v.superceded_date)
        self.assertIsNone(versions_sorted[-1].superceded_date)
    
    def test_parse_carb_with_delete(self):
        """Test parsing carb entry with delete operation."""
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "operation": "delete"}
                ]
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        versions = timeline.get_all_versions("carb-1")
        self.assertEqual(len(versions), 2)
        
        # Check delete operation
        v1 = [v for v in versions if v.sync_version == 1][0]
        self.assertEqual(v1.get_operation(), CarbOperation.DELETE)
        self.assertFalse(v1.is_active())
    
    def test_parse_carb_edit_with_start_time_change(self):
        """Test parsing carb entry where edit changes start_time."""
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "edits": [
                    {
                        "edit_time": "8/15/2019 12:30:00",
                        "start_time": "8/15/2019 12:05:00",
                        "value": 50.0
                    }
                ]
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        versions = timeline.get_all_versions("carb-1")
        self.assertEqual(len(versions), 2)
        
        # v1 should have new start_time
        v1 = [v for v in versions if v.sync_version == 1][0]
        self.assertEqual(v1.get_operation(), CarbOperation.UPDATE)
    
    def test_parse_carb_edit_inherits_values(self):
        """Test that edits inherit values from previous version if not specified."""
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "duration": 240,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "value": 75.0}
                    # duration not specified, should inherit 240
                ]
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        versions = timeline.get_all_versions("carb-1")
        v1 = [v for v in versions if v.sync_version == 1][0]
        
        self.assertEqual(v1.value, 75.0)  # Changed
        self.assertEqual(v1.duration_minutes, 240)  # Inherited
    
    def test_parse_preserves_user_created_date(self):
        """Test that user_created_date is preserved across edits."""
        original_entry_time = "8/15/2019 12:00:00"
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": original_entry_time,
                "entry_time": original_entry_time,
                "value": 50.0,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "value": 75.0},
                    {"edit_time": "8/15/2019 13:00:00", "value": 80.0}
                ]
            }
        ]
        
        timeline = self.parser.carb_entries_to_timeline(carb_entries)
        
        original_dt = datetime.datetime(2019, 8, 15, 12, 0, 0)
        
        # All versions should have same user_created_date
        for version in timeline.get_all_versions("carb-1"):
            self.assertEqual(version.get_user_created_date(), original_dt)


class TestCarbEditIntegration(unittest.TestCase):
    """Integration tests for carb editing in simulation context."""
    
    def test_loop_inputs_reflect_active_version(self):
        """Test that get_loop_inputs returns correct version based on time."""
        parser = ScenarioParserV2()
        
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "duration": 180,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "value": 75.0}
                ]
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        # Query before edit: should see 50g
        time_before_edit = datetime.datetime(2019, 8, 15, 12, 15, 0)
        values, times, durations = timeline.get_loop_inputs(time_before_edit)
        
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 50.0)
        
        # Query after edit: should see 75g
        time_after_edit = datetime.datetime(2019, 8, 15, 13, 0, 0)
        values, times, durations = timeline.get_loop_inputs(time_after_edit)
        
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], 75.0)
    
    def test_deleted_carb_not_in_loop_inputs(self):
        """Test that deleted carbs are excluded from Loop inputs."""
        parser = ScenarioParserV2()
        
        carb_entries = [
            {
                "id": "carb-1",
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:00:00",
                "value": 50.0,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "operation": "delete"}
                ]
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        # Before delete: should include carb
        time_before = datetime.datetime(2019, 8, 15, 12, 15, 0)
        values, times, durations = timeline.get_loop_inputs(time_before)
        self.assertEqual(len(values), 1)
        
        # After delete: should be empty
        time_after = datetime.datetime(2019, 8, 15, 13, 0, 0)
        values, times, durations = timeline.get_loop_inputs(time_after)
        self.assertEqual(len(values), 0)
    
    def test_multiple_carbs_with_edits(self):
        """Test multiple carb entries with different edit patterns."""
        parser = ScenarioParserV2()
        
        carb_entries = [
            {
                "id": "breakfast",
                "start_time": "8/15/2019 08:00:00",
                "entry_time": "8/15/2019 08:00:00",
                "value": 40.0
            },
            {
                "id": "lunch",
                "start_time": "8/15/2019 12:00:00",
                "entry_time": "8/15/2019 12:00:00",
                "value": 60.0,
                "edits": [
                    {"edit_time": "8/15/2019 12:30:00", "value": 80.0}
                ]
            },
            {
                "id": "snack",
                "start_time": "8/15/2019 15:00:00",
                "entry_time": "8/15/2019 15:00:00",
                "value": 20.0,
                "edits": [
                    {"edit_time": "8/15/2019 15:30:00", "operation": "delete"}
                ]
            }
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        # Query at 16:00 - after all edits
        query_time = datetime.datetime(2019, 8, 15, 16, 0, 0)
        values, times, durations = timeline.get_loop_inputs(query_time)
        
        # Should see: breakfast (40g), lunch (80g), NOT snack (deleted)
        self.assertEqual(len(values), 2)
        self.assertIn(40.0, values)  # breakfast
        self.assertIn(80.0, values)  # lunch (edited)
        self.assertNotIn(20.0, values)  # snack (deleted)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing carb configurations."""
    
    def test_existing_carb_format_works(self):
        """Test that existing carb JSON format without new fields works."""
        parser = ScenarioParserV2()
        
        # Old format - no id, entry_time, or edits
        carb_entries = [
            {"start_time": "8/15/2019 12:00:00", "value": 50.0, "duration": 180}
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = timeline.get_event(start_time)
        
        self.assertEqual(carb.value, 50.0)
        self.assertEqual(carb.duration_minutes, 180)
        self.assertTrue(carb.is_active())
        self.assertEqual(carb.get_sync_version(), 0)
        self.assertIsNotNone(carb.get_sync_identifier())  # Auto-generated
    
    def test_carb_without_duration_uses_default(self):
        """Test that carbs without duration use default 180 minutes."""
        parser = ScenarioParserV2()
        
        carb_entries = [
            {"start_time": "8/15/2019 12:00:00", "value": 50.0}
        ]
        
        timeline = parser.carb_entries_to_timeline(carb_entries)
        
        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)
        carb = timeline.get_event(start_time)
        
        self.assertEqual(carb.duration_minutes, 180)


if __name__ == '__main__':
    unittest.main()
