"""
Unit tests for build_risk_sim_generator's TLR-* directory selection.

Covers the bugfix to tidepool_data_science_simulator/projects/risk/loop_risk_v2_0.py:
a syntax error (unterminated string literal) previously made the module
unimportable. The underlying filter is now an explicit `target_risk_dir`
keyword argument instead of a hardcoded, silent "TLR-HF" default.

ScenarioParserV2 is monkeypatched out since these tests target only the
directory-selection logic, not scenario parsing.
"""

import os
import tempfile
import shutil

import pytest

from tidepool_data_science_simulator.projects.risk.loop_risk_v2_0 import (
    build_risk_sim_generator,
)


@pytest.fixture
def risk_dirs_tree(monkeypatch):
    """Build a temp dir with TLR-* subdirs, each holding one dummy .json file."""
    root = tempfile.mkdtemp()

    def _make_dir(name):
        d = os.path.join(root, name)
        os.mkdir(d)
        with open(os.path.join(d, f"{name}-Simulation-Configuration.json"), "w") as fh:
            fh.write("{}")
        return d

    _make_dir("TLR-HF-1")
    _make_dir("TLR-HF-2")
    _make_dir("TLR-ABC-1")
    os.mkdir(os.path.join(root, "not-a-risk-dir"))

    class _FakeParser:
        def __init__(self, path_to_json_config):
            self.path_to_json_config = path_to_json_config

        def get_sims(self, override_json_save_dir=None):
            return {}

    monkeypatch.setattr(
        "tidepool_data_science_simulator.projects.risk.loop_risk_v2_0.ScenarioParserV2",
        _FakeParser,
    )

    yield root
    shutil.rmtree(root)


def test_target_risk_dir_none_processes_all_tlr_dirs(risk_dirs_tree):
    results = list(build_risk_sim_generator(risk_dirs_tree))
    risk_dir_names = {risk_dir_name for risk_dir_name, _, _ in results}
    assert risk_dir_names == {"TLR-HF-1", "TLR-HF-2", "TLR-ABC-1"}


def test_target_risk_dir_filters_to_matching_substring(risk_dirs_tree):
    results = list(build_risk_sim_generator(risk_dirs_tree, target_risk_dir="TLR-HF"))
    risk_dir_names = {risk_dir_name for risk_dir_name, _, _ in results}
    assert risk_dir_names == {"TLR-HF-1", "TLR-HF-2"}


def test_target_risk_dir_with_no_match_yields_nothing(risk_dirs_tree):
    results = list(build_risk_sim_generator(risk_dirs_tree, target_risk_dir="TLR-DOES-NOT-EXIST"))
    assert results == []


def test_non_tlr_directories_are_never_included(risk_dirs_tree):
    results = list(build_risk_sim_generator(risk_dirs_tree))
    risk_dir_names = {risk_dir_name for risk_dir_name, _, _ in results}
    assert "not-a-risk-dir" not in risk_dir_names
