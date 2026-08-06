"""
Unit tests for gui_runner.py, the single validated entry point Phase 3's
Streamlit GUI (and any other consumer) calls to run a risk assessment.

build_risk_sim_generator, run_simulations, build_assessment_result, and
ConfigValidator are monkeypatched throughout -- these are exercised for real
in the separately-approved Phase 3 integration test plan; these tests target
gui_runner's own orchestration logic (finalize-once-per-TLR-dir, progress
reporting, cancellation, validation gating) in isolation.
"""

import json
import os
import shutil
import tempfile
import threading
from types import SimpleNamespace

import pytest

from tidepool_data_science_simulator.projects.risk import gui_runner


# ---------------------------------------------------------------------------
# _find_pointer_object_dir
# ---------------------------------------------------------------------------

def test_find_pointer_object_dir_finds_reusable_sibling():
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "reusable"))
        collection_dir = os.path.join(root, "loop_risk_v2_0", "loop_risk_v2_exploratory")
        os.makedirs(collection_dir)
        assert gui_runner._find_pointer_object_dir(collection_dir) == root
    finally:
        shutil.rmtree(root)


def test_find_pointer_object_dir_returns_none_when_not_found():
    root = tempfile.mkdtemp()
    try:
        nested = os.path.join(root, "a", "b")
        os.makedirs(nested)
        assert gui_runner._find_pointer_object_dir(nested, max_levels_up=2) is None
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# validate_config_dir
# ---------------------------------------------------------------------------

class _FakeValidationError:
    def __init__(self, msg):
        self.error_message = msg


class _FakeValidationWarning:
    def __init__(self, msg):
        self.warning_message = msg


class _FakeConfigValidator:
    """Records the directories validate_directory was called with."""
    calls = []
    canned_results = {}

    def __init__(self, pointer_object_dir=None):
        self.pointer_object_dir = pointer_object_dir

    def validate_directory(self, directory, recursive=True):
        _FakeConfigValidator.calls.append(directory)
        return _FakeConfigValidator.canned_results.get(directory, {})


@pytest.fixture(autouse=False)
def fake_validator(monkeypatch):
    _FakeConfigValidator.calls = []
    _FakeConfigValidator.canned_results = {}
    monkeypatch.setattr(gui_runner, "ConfigValidator", _FakeConfigValidator)
    return _FakeConfigValidator


def test_validate_config_dir_aggregates_errors_and_warnings(fake_validator):
    root = tempfile.mkdtemp()
    try:
        fake_validator.canned_results = {
            root: {
                "a.json": (False, [_FakeValidationError("bad")], []),
                "b.json": (True, [], [_FakeValidationWarning("meh")]),
                "c.json": (True, [], []),
            }
        }
        result = gui_runner.validate_config_dir(root)
        assert result.is_valid is False
        assert list(result.errors_by_file.keys()) == ["a.json"]
        assert list(result.warnings_by_file.keys()) == ["b.json"]
    finally:
        shutil.rmtree(root)


def test_validate_config_dir_scopes_to_target_risk_dir(fake_validator):
    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "TLR-HF-1"))
        os.makedirs(os.path.join(root, "TLR-ABC-1"))
        fake_validator.canned_results = {
            os.path.join(root, "TLR-HF-1"): {"x.json": (True, [], [])},
        }
        result = gui_runner.validate_config_dir(root, target_risk_dir="TLR-HF")
        assert fake_validator.calls == [os.path.join(root, "TLR-HF-1")]
        assert result.is_valid is True
    finally:
        shutil.rmtree(root)


def test_validate_config_dir_empty_results_is_valid(fake_validator):
    root = tempfile.mkdtemp()
    try:
        fake_validator.canned_results = {}
        result = gui_runner.validate_config_dir(root)
        assert result.is_valid is True
        assert result.errors_by_file == {}
    finally:
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# run_risk_assessment
# ---------------------------------------------------------------------------

def _fake_sim_suite():
    return {"sim1": SimpleNamespace(controller=SimpleNamespace())}


@pytest.fixture
def run_env(monkeypatch):
    """Common monkeypatching for run_risk_assessment tests."""
    save_dir = tempfile.mkdtemp()

    monkeypatch.setattr(gui_runner, "validate_config_dir",
                         lambda config_dir, target_risk_dir=None: gui_runner.ConfigValidationResult(True, {}, {}))
    monkeypatch.setattr(gui_runner, "create_save_dir", lambda: save_dir)
    monkeypatch.setattr(gui_runner, "get_timestamp", lambda: "2026-07-21T00:00:00")
    monkeypatch.setattr(gui_runner, "run_simulations", lambda *a, **kw: ({}, None))
    monkeypatch.setattr(gui_runner, "plot_sim_results", lambda *a, **kw: None)

    yield save_dir, monkeypatch
    shutil.rmtree(save_dir, ignore_errors=True)


# build_assessment_result returns an AssessmentOutcome (TRSET-28), so the fakes
# below return the real dataclass rather than a bare assessment-or-None -- a fake
# with the wrong shape would pass while the GUI's own status branch broke.
def _ok_outcome(assessment=None):
    return gui_runner.AssessmentOutcome(assessment or SimpleNamespace(), "ok")


def _no_assessment_outcome(status="empty", detail="nothing ran in this directory"):
    return gui_runner.AssessmentOutcome(None, status, detail)


def test_happy_path_finalizes_once_per_risk_dir(run_env, monkeypatch):
    save_dir, monkeypatch = run_env

    generator_items = [
        ("TLR-1", "scenario_a.json", _fake_sim_suite()),
        ("TLR-1", "scenario_b.json", _fake_sim_suite()),
        ("TLR-2", "scenario_a.json", _fake_sim_suite()),
    ]
    monkeypatch.setattr(gui_runner, "build_risk_sim_generator", lambda *a, **kw: iter(generator_items))
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1", "TLR-2"])

    assessment_calls = []

    def fake_build_assessment_result(tlr_dir, timestamp):
        assessment_calls.append(tlr_dir)
        return _ok_outcome(SimpleNamespace(simulation_id=os.path.basename(tlr_dir)))

    monkeypatch.setattr(gui_runner, "build_assessment_result", fake_build_assessment_result)

    progress_calls = []
    result = gui_runner.run_risk_assessment(
        "unused_config_dir",
        progress_callback=lambda n, total, name: progress_calls.append((n, total, name)),
    )

    assert len(assessment_calls) == 2  # once per risk dir, not once per scenario file
    assert [r.risk_dir_name for r in result.risk_dir_results] == ["TLR-1", "TLR-2"]
    assert progress_calls == [(1, 2, "TLR-1"), (2, 2, "TLR-2")]
    assert result.cancelled is False
    # TLR-1 ran 2 scenario files -> 2 pngs; TLR-2 ran 1 -> 1 png
    assert len(result.risk_dir_results[0].png_paths) == 2
    assert len(result.risk_dir_results[1].png_paths) == 1


def test_cancel_mid_run_stops_before_finalizing_in_progress_dir(run_env, monkeypatch):
    save_dir, monkeypatch = run_env

    cancel_event = threading.Event()
    generator_items = [
        ("TLR-1", "scenario_a.json", _fake_sim_suite()),
        ("TLR-1", "scenario_b.json", _fake_sim_suite()),
        ("TLR-2", "scenario_a.json", _fake_sim_suite()),
    ]

    def generator():
        for i, item in enumerate(generator_items):
            if i == 2:
                cancel_event.set()
            yield item

    monkeypatch.setattr(gui_runner, "build_risk_sim_generator", lambda *a, **kw: generator())
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1", "TLR-2"])

    assessment_calls = []

    def fake_build_assessment_result(tlr_dir, timestamp):
        assessment_calls.append(tlr_dir)
        return _ok_outcome()

    monkeypatch.setattr(gui_runner, "build_assessment_result", fake_build_assessment_result)

    result = gui_runner.run_risk_assessment("unused_config_dir", cancel_event=cancel_event)

    assert result.cancelled is True
    # TLR-1 finished before cancel was observed; TLR-2 never started -> only TLR-1 finalized
    assert assessment_calls == [os.path.join(save_dir, "TLR-1")]


def test_raises_when_validation_has_errors(monkeypatch):
    monkeypatch.setattr(
        gui_runner, "validate_config_dir",
        lambda config_dir, target_risk_dir=None: gui_runner.ConfigValidationResult(
            False, {"bad.json": ["error"]}, {}
        ),
    )

    def _should_not_be_called(*a, **kw):
        raise AssertionError("build_risk_sim_generator should not run when validation fails")

    monkeypatch.setattr(gui_runner, "build_risk_sim_generator", _should_not_be_called)

    with pytest.raises(ValueError, match="validation errors"):
        gui_runner.run_risk_assessment("unused_config_dir")


def test_none_assessment_is_included_not_dropped(run_env, monkeypatch):
    save_dir, monkeypatch = run_env

    generator_items = [("TLR-1", "scenario_a.json", _fake_sim_suite())]
    monkeypatch.setattr(gui_runner, "build_risk_sim_generator", lambda *a, **kw: iter(generator_items))
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1"])
    monkeypatch.setattr(gui_runner, "build_assessment_result",
                        lambda tlr_dir, timestamp: _no_assessment_outcome())

    result = gui_runner.run_risk_assessment("unused_config_dir")

    assert len(result.risk_dir_results) == 1
    assert result.risk_dir_results[0].risk_dir_name == "TLR-1"
    assert result.risk_dir_results[0].assessment is None


class TestAssessmentStatusIsSurfaced:
    """TRSET-28: a GUI could only say "no data" for a directory whose data was in
    fact corrupt, because both conditions arrived as a bare None."""

    def _run_with(self, monkeypatch, outcome):
        monkeypatch.setattr(gui_runner, "build_risk_sim_generator",
                            lambda *a, **kw: iter([("TLR-1", "scenario_a.json", _fake_sim_suite())]))
        monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1"])
        monkeypatch.setattr(gui_runner, "build_assessment_result",
                            lambda tlr_dir, timestamp: outcome)
        result = gui_runner.run_risk_assessment("unused_config_dir")
        return result.risk_dir_results[0]

    def test_an_empty_directory_reports_empty(self, run_env, monkeypatch):
        save_dir, monkeypatch = run_env

        risk_dir_result = self._run_with(monkeypatch, _no_assessment_outcome("empty", "nothing ran"))

        assert risk_dir_result.assessment_status == "empty"
        assert risk_dir_result.assessment_detail == "nothing ran"

    def test_a_malformed_directory_reports_malformed(self, run_env, monkeypatch):
        save_dir, monkeypatch = run_env

        risk_dir_result = self._run_with(
            monkeypatch, _no_assessment_outcome("malformed", "3 files present, none usable")
        )

        assert risk_dir_result.assessment_status == "malformed"
        assert risk_dir_result.assessment_detail == "3 files present, none usable"

    def test_the_two_are_distinguishable(self, run_env, monkeypatch):
        """The whole point: both used to be assessment=None and nothing else."""
        save_dir, monkeypatch = run_env

        empty = self._run_with(monkeypatch, _no_assessment_outcome("empty"))
        malformed = self._run_with(monkeypatch, _no_assessment_outcome("malformed", "unreadable"))

        assert empty.assessment is malformed.assessment is None
        assert empty.assessment_status != malformed.assessment_status

    def test_a_usable_directory_reports_ok(self, run_env, monkeypatch):
        save_dir, monkeypatch = run_env

        risk_dir_result = self._run_with(monkeypatch, _ok_outcome())

        assert risk_dir_result.assessment_status == "ok"
        assert risk_dir_result.assessment is not None

    def test_the_new_fields_default_so_positional_construction_still_works(self):
        """Appended after the existing fields, so a consumer building this
        positionally the old way is unaffected."""
        risk_dir_result = gui_runner.RiskDirRunResult("TLR-1", None, ["a.png"], {})

        assert risk_dir_result.assessment_status == "ok"
        assert risk_dir_result.assessment_detail == ""


def test_target_risk_dir_passed_through_to_generator(run_env, monkeypatch):
    save_dir, monkeypatch = run_env

    captured_kwargs = {}

    def fake_generator(config_dir, override_config_save_dir=None, target_risk_dir=None):
        captured_kwargs["target_risk_dir"] = target_risk_dir
        return iter([])

    monkeypatch.setattr(gui_runner, "build_risk_sim_generator", fake_generator)
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: [])

    gui_runner.run_risk_assessment("unused_config_dir", target_risk_dir="TLR-HF")

    assert captured_kwargs["target_risk_dir"] == "TLR-HF"


# ---------------------------------------------------------------------------
# trace_paths (TRSET-23)
# ---------------------------------------------------------------------------

def test_trace_paths_grouped_by_scenario_file_and_keyed_by_sim_id(run_env, monkeypatch):
    """Each scenario config file maps to {sim_id: <save_dir>/<risk_dir>/<sim_id>.tsv}.

    The mapping is derived from run_simulations' own full_results dict, so the
    sim_ids are exactly the ones that ran (and that save_df wrote a .tsv for).
    """
    save_dir, monkeypatch = run_env

    sims_by_scenario = {
        "Simulation-Configuration-TLR-1_Adolescent_profile.json": {
            "pre-Loop_NoMitigations_t1_adolescent": None,
            "pre-noLoop_t1_adolescent": None,
        },
        "Simulation-Configuration-TLR-1_Median_profile.json": {
            "pre-Loop_NoMitigations_t1_median": None,
        },
    }
    generator_items = [("TLR-1", name, _fake_sim_suite()) for name in sims_by_scenario]
    monkeypatch.setattr(gui_runner, "build_risk_sim_generator", lambda *a, **kw: iter(generator_items))
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1"])
    monkeypatch.setattr(gui_runner, "build_assessment_result",
                        lambda tlr_dir, timestamp: _ok_outcome())
    monkeypatch.setattr(
        gui_runner, "run_simulations",
        lambda sims, **kw: (sims_by_scenario[kw["name"]], None),
    )

    result = gui_runner.run_risk_assessment("unused_config_dir")

    trace_paths = result.risk_dir_results[0].trace_paths
    assert set(trace_paths) == set(sims_by_scenario)
    risk_dir_path = os.path.join(save_dir, "TLR-1")
    for scenario_name, sim_ids in sims_by_scenario.items():
        assert trace_paths[scenario_name] == {
            sim_id: os.path.join(risk_dir_path, f"{sim_id}.tsv") for sim_id in sim_ids
        }


def test_trace_paths_are_reset_between_risk_dirs(run_env, monkeypatch):
    """A second TLR dir must not inherit the first one's trace paths -- the same
    per-dir reset png_paths gets."""
    save_dir, monkeypatch = run_env

    per_scenario_sims = {
        "scenario_a.json": {"pre-Loop_NoMitigations_t1_median": None},
        "scenario_b.json": {"pre-noLoop_t1_median": None},
    }
    generator_items = [
        ("TLR-1", "scenario_a.json", _fake_sim_suite()),
        ("TLR-2", "scenario_b.json", _fake_sim_suite()),
    ]
    monkeypatch.setattr(gui_runner, "build_risk_sim_generator", lambda *a, **kw: iter(generator_items))
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1", "TLR-2"])
    monkeypatch.setattr(gui_runner, "build_assessment_result",
                        lambda tlr_dir, timestamp: _ok_outcome())
    monkeypatch.setattr(
        gui_runner, "run_simulations",
        lambda sims, **kw: (per_scenario_sims[kw["name"]], None),
    )

    result = gui_runner.run_risk_assessment("unused_config_dir")

    first, second = result.risk_dir_results
    assert list(first.trace_paths) == ["scenario_a.json"]
    assert list(second.trace_paths) == ["scenario_b.json"]
    # Each dir's paths point into its own results dir.
    assert os.path.dirname(first.trace_paths["scenario_a.json"]["pre-Loop_NoMitigations_t1_median"]) == \
        os.path.join(save_dir, "TLR-1")
    assert os.path.dirname(second.trace_paths["scenario_b.json"]["pre-noLoop_t1_median"]) == \
        os.path.join(save_dir, "TLR-2")


def test_stage_identity_helpers_are_reexported_for_gui_consumers():
    """TRSET-23: the GUI reads stage identity through this entry point rather
    than replicating the post_processing/ sys.path setup. severity_model stays
    the single source of truth -- these are re-exports, not redefinitions."""
    import severity_model

    assert gui_runner.classify_sim_id is severity_model.classify_sim_id
    assert gui_runner.STAGE_ORDER is severity_model.STAGE_ORDER
    assert gui_runner.STAGE_DISPLAY is severity_model.STAGE_DISPLAY


# ---------------------------------------------------------------------------
# metadata.json (TRSET-7)
# ---------------------------------------------------------------------------

def test_run_writes_metadata_json_with_the_assessment_timestamp(run_env, monkeypatch):
    """The run's save_dir carries metadata.json in the shape (and with the
    timestamp) create_severity_summary reads, so the directory is a valid input
    to it -- here and via the existing CLI."""
    save_dir, monkeypatch = run_env

    assessment_timestamps = []
    monkeypatch.setattr(gui_runner, "build_risk_sim_generator",
                        lambda *a, **kw: iter([("TLR-1", "scenario_a.json", _fake_sim_suite())]))
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1"])
    monkeypatch.setattr(
        gui_runner, "build_assessment_result",
        lambda tlr_dir, timestamp: assessment_timestamps.append(timestamp) or _ok_outcome(),
    )

    gui_runner.run_risk_assessment("unused_config_dir")

    metadata_path = os.path.join(save_dir, gui_runner.METADATA_FILENAME)
    assert os.path.isfile(metadata_path), "run did not write metadata.json"
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    # Same key create_severity_summary reads, same value the assessments are
    # dated with -- an exported summary matches what the GUI displayed.
    assert metadata == {"timestamp": "2026-07-21T00:00:00"}
    assert assessment_timestamps == ["2026-07-21T00:00:00"]


def test_metadata_json_is_written_before_any_simulation_runs(run_env, monkeypatch):
    """Written up front, off the run's own save_dir/timestamp -- not appended at
    the end, so a cancelled run still exports."""
    save_dir, monkeypatch = run_env
    metadata_path = os.path.join(save_dir, gui_runner.METADATA_FILENAME)

    seen_during_run = []

    def _observing_run_simulations(sims, **kwargs):
        seen_during_run.append(os.path.isfile(metadata_path))
        return {}, None

    monkeypatch.setattr(gui_runner, "build_risk_sim_generator",
                        lambda *a, **kw: iter([("TLR-1", "scenario_a.json", _fake_sim_suite())]))
    monkeypatch.setattr(gui_runner, "_list_target_risk_dirs", lambda *a, **kw: ["TLR-1"])
    monkeypatch.setattr(gui_runner, "build_assessment_result",
                        lambda tlr_dir, timestamp: _ok_outcome())
    monkeypatch.setattr(gui_runner, "run_simulations", _observing_run_simulations)

    gui_runner.run_risk_assessment("unused_config_dir")

    assert seen_during_run == [True]


def test_severity_summary_renderer_is_reexported_for_gui_consumers():
    """TRSET-7: the GUI writes the RTF summaries through this entry point, same
    seam as the stage-identity re-exports. A re-export, not a wrapper -- the RTF
    output stays byte-identical to the CLI's."""
    import create_severity_summary

    assert gui_runner.process_results_directory is create_severity_summary.process_results_directory
