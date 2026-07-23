"""Tests for the scenario-configs pointer-object-dir env seam.

`scenario_configs/` is a top-level repo directory that is NOT shipped with the
installed package, so the parser's in-tree default for the reusable/ pointer
tree only resolves under an editable/dev checkout. Under a non-editable install
(the packaged bundle) `LOOP_RISK_GUI_SCENARIO_CONFIGS_ROOT` -- the same env seam
the GUI's LIBRARY_ROOT uses -- points it at a vendored config tree.

These cover both the standalone resolver and ScenarioParserV2's use of it,
env-set and env-unset, plus the explicit call-site override.
"""

import os

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import (
    ScenarioParserV2,
    resolve_pointer_object_dir,
    _SCENARIO_CONFIGS_ROOT_ENV,
)

ENV = _SCENARIO_CONFIGS_ROOT_ENV


def _in_tree_default():
    """The module-relative default (repo_root/scenario_configs/tidepool_risk_v2)."""
    parser_module_dir = os.path.dirname(
        os.path.abspath(
            ScenarioParserV2.__init__.__globals__["__file__"]
        )
    )
    return os.path.normpath(
        os.path.join(parser_module_dir, "..", "..", "scenario_configs", "tidepool_risk_v2")
    )


def test_resolver_unset_returns_in_tree_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert os.path.normpath(resolve_pointer_object_dir()) == _in_tree_default()


def test_resolver_set_returns_root_joined_with_tidepool_risk_v2(monkeypatch):
    monkeypatch.setenv(ENV, "/vendored/scenario_configs")
    assert os.path.normpath(resolve_pointer_object_dir()) == os.path.normpath(
        "/vendored/scenario_configs/tidepool_risk_v2"
    )


def test_resolver_semantics_match_gui_root_contract(monkeypatch):
    """Env value is a root that CONTAINS tidepool_risk_v2/ -- identical to the
    GUI's LOOP_RISK_GUI_SCENARIO_CONFIGS_ROOT contract."""
    monkeypatch.setenv(ENV, "/some/root")
    assert resolve_pointer_object_dir().replace(os.sep, "/").endswith(
        "/some/root/tidepool_risk_v2"
    )


def test_parser_picks_up_env_set_after_import(monkeypatch):
    """A default-constructed parser resolves the env at instantiation, so an env
    set after this module was imported (e.g. by the bundle launcher) is honored."""
    monkeypatch.setenv(ENV, "/vendored/scenario_configs")
    parser = ScenarioParserV2()
    assert os.path.normpath(parser.pointer_object_dir) == os.path.normpath(
        "/vendored/scenario_configs/tidepool_risk_v2"
    )


def test_parser_unset_uses_in_tree_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    parser = ScenarioParserV2()
    assert os.path.normpath(parser.pointer_object_dir) == _in_tree_default()


def test_explicit_pointer_object_dir_overrides_env(monkeypatch):
    """An explicit call-site value always wins over the env seam."""
    monkeypatch.setenv(ENV, "/vendored/scenario_configs")
    parser = ScenarioParserV2(pointer_object_dir="/explicit/override")
    assert parser.pointer_object_dir == "/explicit/override"
