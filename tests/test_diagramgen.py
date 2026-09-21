"""
Unit tests for the architecture diagram generator's pure parts.

Everything here runs without a simulation: config parsing, identity and
rendering rules, and the static pass. The behaviour that needs a real run --
tracing, the four cross-package edges, the drift gate -- is covered by
``test_diagramgen_integration.py``.
"""

import os
import textwrap

import pytest

from tidepool_data_science_simulator.diagramgen import config as diagram_config
from tidepool_data_science_simulator.diagramgen import coverage as diagram_coverage
from tidepool_data_science_simulator.diagramgen import manifest as diagram_manifest
from tidepool_data_science_simulator.diagramgen import mermaid, naming, staticscan
from tidepool_data_science_simulator.diagramgen.yamlmini import MiniYamlError, load_mini_yaml


# -- yamlmini -------------------------------------------------------------


def test_mini_yaml_reads_scalars_lists_and_comments():
    parsed = load_mini_yaml(
        textwrap.dedent(
            """\
            # leading comment
            version: 1

            module_prefixes:
              - alpha
              - beta    # trailing comment
            name: "quoted value"
            """
        )
    )
    assert parsed == {
        "version": "1",
        "module_prefixes": ["alpha", "beta"],
        "name": "quoted value",
    }


def test_mini_yaml_keeps_hash_inside_a_value():
    assert load_mini_yaml("key: build#1") == {"key": "build#1"}


@pytest.mark.parametrize(
    "text",
    [
        "key:\n\t- item\n",                    # tab indentation
        "key: [a, b]\n",                       # flow sequence
        "key: &anchor\n",                      # anchor
        "outer:\n  inner: value\n",            # nested mapping
        "- item\n",                            # list with no key
        "key: 1\nkey: 2\n",                    # duplicate key
        "---\nkey: 1\n",                       # multi-document
        "bare line\n",                         # neither key nor item
    ],
)
def test_mini_yaml_rejects_constructs_outside_the_subset(text):
    with pytest.raises(MiniYamlError):
        load_mini_yaml(text)


# -- config ---------------------------------------------------------------


@pytest.fixture
def committed_allowlist():
    return diagram_config.load_allowlist(_committed(".docs/architecture/allowlist.yml"))


@pytest.fixture
def committed_exclusions():
    return diagram_config.load_exclusions(_committed(".docs/architecture/exclusions.yml"))


def _committed(rel_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, rel_path)


def test_allowlist_carries_the_four_packages(committed_allowlist):
    assert committed_allowlist.module_prefixes == frozenset(
        [
            "tidepool_data_science_simulator",
            "loop_to_python_api",
            "tidepool_data_science_models",
            "tidepool_data_science_metrics",
        ]
    )


def test_named_step_matching_is_qualified_or_bare(committed_allowlist):
    # Dotted entries pin the class.
    assert committed_allowlist.matches_named_step("Simulation.update")
    assert not committed_allowlist.matches_named_step("Pump.update")
    # Bare entries match any controller implementation, which is what keeps the
    # `"controller": null` stage's DoNothingController visible.
    assert committed_allowlist.matches_named_step("DoNothingController.get_loop_recommendations")
    assert committed_allowlist.matches_named_step("SwiftLoopController.apply_loop_recommendations")


def test_missing_required_key_is_rejected(tmp_path):
    path = tmp_path / "allowlist.yml"
    path.write_text("module_prefixes:\n  - alpha\n")
    with pytest.raises(diagram_config.ConfigError):
        diagram_config.load_allowlist(str(path))


def test_unconditional_directories_are_excluded_without_being_listed(committed_exclusions):
    # A search for loop_risk_v2_0.py returns three copies; only the package one counts.
    assert committed_exclusions.is_excluded("build/lib/x.py")
    assert committed_exclusions.is_excluded("venv/lib/python3.12/site-packages/x.py")
    assert committed_exclusions.is_excluded("venv.bak.20260610/lib/x.py")
    assert committed_exclusions.is_excluded(".claude/worktrees/w/x.py")
    assert committed_exclusions.is_excluded("notebooks/x.py")
    assert not committed_exclusions.is_excluded("tidepool_data_science_simulator/run.py")


def test_new_files_default_to_included(committed_exclusions):
    assert not committed_exclusions.is_excluded("tidepool_data_science_simulator/models/brand_new.py")


# -- naming ---------------------------------------------------------------


def test_node_ids_depend_only_on_the_qualified_name():
    first = naming.Node("pkg", "pkg.mod", "Thing")
    second = naming.Node("pkg", "pkg.mod", "Thing")
    other = naming.Node("pkg", "pkg.mod", "Other")
    assert first.mermaid_id == second.mermaid_id
    assert first.mermaid_id != other.mermaid_id


def test_rendered_labels_are_package_qualified_never_absolute():
    node = naming.Node(
        "tidepool_data_science_models",
        "tidepool_data_science_models.models.simple_metabolism_model",
        "SimpleMetabolismModel",
    )
    assert node.label == "SimpleMetabolismModel"
    assert node.sublabel == "models.simple_metabolism_model"
    assert not node.qualified.startswith("/")


def test_redact_path_replaces_home():
    home = os.path.expanduser("~")
    assert naming.redact_path(os.path.join(home, "PycharmProjects", "x")) == os.path.join("~", "PycharmProjects", "x")


def test_repo_relative_prefers_a_relative_path_inside_the_repo(tmp_path):
    inside = tmp_path / "a" / "b.json"
    inside.parent.mkdir()
    inside.write_text("{}")
    assert naming.repo_relative(str(inside), str(tmp_path)) == os.path.join("a", "b.json")


# -- mermaid --------------------------------------------------------------


def test_normalized_body_strips_the_provenance_header():
    text = mermaid.render_header(["commit abc123", "generated 2026-01-01"]) + "\nflowchart LR\n    a --> b\n"
    assert mermaid.normalized_body(text) == "flowchart LR\n    a --> b\n"


def test_normalized_body_ignores_a_changed_header():
    body = "\nflowchart LR\n    a --> b\n"
    first = mermaid.render_header(["commit aaa"]) + body
    second = mermaid.render_header(["commit bbb", "extra line"]) + body
    assert mermaid.normalized_body(first) == mermaid.normalized_body(second)


def test_data_flow_edges_are_sorted_by_caller_then_callee():
    caller_b = naming.Node("pkg_b", "pkg_b.mod", "")
    caller_a = naming.Node("pkg_a", "pkg_a.mod", "")
    target = naming.Node("pkg_c", "pkg_c.mod", "")
    edges = [(caller_b, target, ["z"]), (caller_a, target, ["y"])]
    rendered = mermaid.render_data_flow([], edges, ["header"])
    assert rendered.index(caller_a.mermaid_id + " -.->") < rendered.index(caller_b.mermaid_id + " -.->")


# -- static pass ----------------------------------------------------------


def test_static_pass_distinguishes_a_call_from_a_binding(tmp_path):
    source = tmp_path / "thing.py"
    source.write_text(
        textwrap.dedent(
            """\
            from other_pkg.models import Widget, make_widget

            def build():
                return Widget

            def run():
                return make_widget()
            """
        )
    )
    exclusions = diagram_config.Exclusions(path_prefixes=(), source_path="none", sha256="0")
    result = staticscan.scan_repo(str(tmp_path), exclusions, ["other_pkg"])

    by_kind = {(site.enclosing, site.kind): site for site in result.sites}
    assert by_kind[("build", "bind")].target_symbol == "Widget"
    assert by_kind[("run", "call")].target_symbol == "make_widget"


def test_static_pass_ignores_same_package_references(tmp_path):
    package = tmp_path / "other_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "thing.py").write_text("from other_pkg.models import Widget\n\ndef go():\n    return Widget()\n")
    exclusions = diagram_config.Exclusions(path_prefixes=(), source_path="none", sha256="0")
    result = staticscan.scan_repo(str(tmp_path), exclusions, ["other_pkg"])
    assert result.sites == []


# -- coverage classification ----------------------------------------------


class _FakeRun(object):
    def __init__(self, records=(), executed=(), loaded=()):
        self.records = list(records)
        self.executed_functions = set(executed)
        self.loaded_modules = frozenset(loaded)


class _FakeScan(object):
    def __init__(self, sites):
        self.sites = list(sites)
        self.files_scanned = len(self.sites)
        self.files_excluded = 0
        self.unparsable = []


def _site(**kwargs):
    defaults = dict(
        rel_path="a/b.py",
        module="a.b",
        enclosing="Thing.method",
        target_module="other.models",
        target_symbol="Widget",
        kind="bind",
        lineno=1,
    )
    defaults.update(kwargs)
    return staticscan.ReferenceSite(**defaults)


def test_a_binding_in_a_function_that_ran_counts_as_exercised():
    classified = diagram_coverage.classify_sites(
        _FakeScan([_site()]),
        _FakeRun(executed=[("a.b", "Thing.method")]),
    )
    assert classified[0][1] == diagram_coverage.EXERCISED


def test_a_call_in_a_function_that_ran_but_was_not_reached_is_not_taken():
    classified = diagram_coverage.classify_sites(
        _FakeScan([_site(kind="call", target_symbol="make_widget")]),
        _FakeRun(executed=[("a.b", "Thing.method")]),
    )
    assert classified[0][1] == diagram_coverage.NOT_TAKEN


def test_a_site_in_a_module_never_imported_says_so():
    classified = diagram_coverage.classify_sites(
        _FakeScan([_site(kind="call", enclosing="<module>")]),
        _FakeRun(),
    )
    assert classified[0][1] == diagram_coverage.MODULE_NOT_IMPORTED


def test_a_dotted_call_is_one_call_not_a_call_plus_bindings(tmp_path):
    """`m.models.make(...)` must not also record `m` and `m.models` as bindings."""
    source = tmp_path / "thing.py"
    source.write_text(
        textwrap.dedent(
            """\
            import other_pkg as m

            def run():
                return m.models.make_widget()
            """
        )
    )
    exclusions = diagram_config.Exclusions(path_prefixes=(), source_path="none", sha256="0")
    result = staticscan.scan_repo(str(tmp_path), exclusions, ["other_pkg"])

    assert [site.kind for site in result.sites] == ["call"]
    assert result.sites[0].target_symbol == "make_widget"


def test_a_package_inside_an_unrelated_repo_reports_no_vcs(tmp_path, monkeypatch):
    """An enclosing repository must not lend its commit to a vendored package.

    `data-science-metrics` installs under `venv/lib/.../site-packages/`, inside
    the simulator's own working tree. `git rev-parse` resolves there, so without
    a tracked-files check the manifest would report the simulator's commit as
    the metrics package's commit.
    """
    calls = []

    def fake_git(args, cwd):
        calls.append(list(args))
        if args[0] == "--version":
            return "git version 2.54.0", None
        if args[:2] == ["rev-parse", "--show-toplevel"]:
            return str(tmp_path), None
        if args[0] == "ls-files":
            return None, "did not match any file(s) known to git"
        raise AssertionError("commit was resolved for an untracked package: {}".format(args))

    monkeypatch.setattr(diagram_manifest, "_git", fake_git)
    record = diagram_manifest.package_provenance("json", str(tmp_path))

    assert record["vcs"]["kind"] == "none"
    assert "not tracked" in record["vcs"]["reason"]
    assert not any(call[:2] == ["rev-parse", "HEAD"] for call in calls)
