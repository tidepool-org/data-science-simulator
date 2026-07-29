"""
Unit tests for gui_runner.py, the single validated entry point Phase 3's
Streamlit GUI (and any other consumer) calls to run a risk assessment.

build_risk_sim_generator, run_simulations, build_assessment, and
ConfigValidator are monkeypatched throughout -- these are exercised for real
in the separately-approved Phase 3 integration test plan; these tests target
gui_runner's own orchestration logic (finalize-once-per-TLR-dir, progress
reporting, cancellation, validation gating) in isolation.
"""

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

    def fake_build_assessment(tlr_dir, timestamp):
        assessment_calls.append(tlr_dir)
        return SimpleNamespace(simulation_id=os.path.basename(tlr_dir))

    monkeypatch.setattr(gui_runner, "build_assessment", fake_build_assessment)

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
    monkeypatch.setattr(gui_runner, "build_assessment",
                         lambda tlr_dir, timestamp: assessment_calls.append(tlr_dir))

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
    monkeypatch.setattr(gui_runner, "build_assessment", lambda tlr_dir, timestamp: None)

    result = gui_runner.run_risk_assessment("unused_config_dir")

    assert len(result.risk_dir_results) == 1
    assert result.risk_dir_results[0].risk_dir_name == "TLR-1"
    assert result.risk_dir_results[0].assessment is None


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
