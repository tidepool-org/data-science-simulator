"""
Integration tests for the architecture diagram generator.

Everything here is real: the real ``ScenarioParserV2``, a real
``SwiftLoopController``, the real ``libLoopAlgorithmToPython.dylib``, the real
``SimpleMetabolismModel`` and the real metrics functions. Nothing is mocked --
mocking the Swift boundary would test the tracer against a fiction, and the
whole point of the generator is that the figures are evidence of executed
behavior.

The fixture scenario is a short-duration structural twin of the TLR-552 median
reference: same base config, same ``"controller": {"id": "swift"}``, the same
three stages including the ``"controller": null`` one, at one hour each. It
lives under ``tests/`` rather than ``scenario_configs/`` so a
``loop_risk_v2_0.py`` directory walk cannot sweep it into a risk run.

Requires macOS: the Swift algorithm ships as a ``.dylib``.
"""

import json
import os
import sys

import pytest

from tidepool_data_science_simulator.diagramgen import cli
from tidepool_data_science_simulator.diagramgen.mermaid import normalized_body

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="libLoopAlgorithmToPython.dylib is macOS-only; the generator cannot run elsewhere.",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_SCENARIO = os.path.join(
    REPO_ROOT, "tests", "test_data", "diagramgen", "Simulation-Configuration-TRSET51-fixture.json"
)
PA_FIXTURE_SCENARIO = os.path.join(
    REPO_ROOT, "tests", "test_data", "diagramgen", "Simulation-Configuration-TRSET57-pa-fixture.json"
)
ALLOWLIST = os.path.join(REPO_ROOT, ".docs", "architecture", "allowlist.yml")
EXCLUSIONS = os.path.join(REPO_ROOT, ".docs", "architecture", "exclusions.yml")

NULL_CONTROLLER_STAGE = "pre-NoLoop_t1_median"
SEQUENCE_STAGE = "post-Loop_WithMitigations_t1_median"


def _generate(output_dir):
    return cli.generate(
        output_dir=str(output_dir),
        scenario_path=FIXTURE_SCENARIO,
        allowlist_path=ALLOWLIST,
        exclusions_path=EXCLUSIONS,
        sequence_stage=SEQUENCE_STAGE,
        repo_root=REPO_ROOT,
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """One generation run, shared by the assertions that only read its output."""
    output_dir = tmp_path_factory.mktemp("diagramgen")
    manifest = _generate(output_dir)
    return output_dir, manifest


def _read(output_dir, filename):
    with open(os.path.join(str(output_dir), filename), encoding="utf-8") as handle:
        return handle.read()


def _trace_records(output_dir):
    records = []
    with open(os.path.join(str(output_dir), cli.TRACE_FILENAME), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


# -- 1. all five artifacts ------------------------------------------------


def test_all_five_artifacts_are_emitted(generated):
    output_dir, _manifest = generated
    for filename in (
        cli.DATA_FLOW_FILENAME,
        cli.SEQUENCE_FILENAME,
        cli.MANIFEST_FILENAME,
        cli.TRACE_FILENAME,
        cli.COVERAGE_FILENAME,
    ):
        path = os.path.join(str(output_dir), filename)
        assert os.path.isfile(path), "{} was not emitted".format(filename)
        assert os.path.getsize(path) > 0, "{} is empty".format(filename)


# -- 2. the four cross-package edges --------------------------------------


def test_data_flow_contains_the_four_cross_package_edges(generated):
    output_dir, _manifest = generated
    records = _trace_records(output_dir)

    observed = {
        (record["caller"]["package"], record["callee"]["package"])
        for record in records
        if record["caller"]["package"] != record["callee"]["package"]
    }
    assert ("tidepool_data_science_simulator", "loop_to_python_api") in observed
    assert ("loop_to_python_api", "native") in observed
    assert ("tidepool_data_science_simulator", "tidepool_data_science_models") in observed
    assert ("tidepool_data_science_simulator", "tidepool_data_science_metrics") in observed


def test_data_flow_diagram_renders_the_ctypes_dylib_edge(generated):
    output_dir, _manifest = generated
    diagram = _read(output_dir, cli.DATA_FLOW_FILENAME)

    # The native boundary is the edge the figure exists for. `sys.setprofile`'s
    # `c_call` event does not report it -- a ctypes `_FuncPtr` is not a
    # `PyCFunction` -- so this asserts the `sys.monitoring` probe is working.
    assert "libLoopAlgorithmToPython.dylib" in diagram
    assert "getLoopRecommendations" in diagram
    assert "Swift shared library" in diagram


def test_rendered_identifiers_never_leak_a_filesystem_path(generated):
    output_dir, _manifest = generated
    for filename in (cli.DATA_FLOW_FILENAME, cli.SEQUENCE_FILENAME):
        text = _read(output_dir, filename)
        assert os.path.expanduser("~") not in text
        assert "/Users/" not in text
        assert ".dylib\"" not in text.replace("libLoopAlgorithmToPython.dylib<br/>", "")


# -- 3. concrete class resolution -----------------------------------------


def test_sequence_diagram_resolves_concrete_classes(generated):
    output_dir, _manifest = generated
    sequence = _read(output_dir, cli.SEQUENCE_FILENAME)

    # The call site is typed against LoopController; the figure must say what
    # actually ran.
    assert "as SwiftLoopController" in sequence
    assert "as SimpleMetabolismModel" in sequence
    assert "as LoopController" not in sequence


def test_sequence_diagram_covers_one_full_control_cycle(generated):
    output_dir, manifest = generated
    sequence = _read(output_dir, cli.SEQUENCE_FILENAME)

    assert manifest["sequence_diagram"]["stage"] == SEQUENCE_STAGE
    for message in ("update", "get_loop_recommendations", "getLoopRecommendations", "apply_loop_recommendations"):
        assert message in sequence


# -- 4. the negative assertion --------------------------------------------


def test_null_controller_stage_is_traced_with_no_loop_edge(generated):
    """The ``"controller": null`` stage proves the tracer observes, not asserts.

    Its records must be present -- so the stage really ran under the tracer --
    and must contain no edge into the Loop algorithm or the Swift library.
    """
    output_dir, _manifest = generated
    records = [r for r in _trace_records(output_dir) if r["stage"] == NULL_CONTROLLER_STAGE]

    assert records, "the null-controller stage produced no trace records at all"
    callee_packages = {record["callee"]["package"] for record in records}
    assert "loop_to_python_api" not in callee_packages
    assert "native" not in callee_packages

    # And it is the same stage: the controller resolved to DoNothingController.
    resolved = {record["callee"]["resolved_class"] for record in records}
    assert "DoNothingController" in resolved
    assert "SwiftLoopController" not in resolved


# -- 5. diagram-generation only -------------------------------------------


def test_the_run_writes_nothing_outside_the_output_directory(tmp_path, monkeypatch):
    from tidepool_data_science_simulator.utils import DATA_DIR

    results_dir = os.path.join(DATA_DIR, "results")
    before = _snapshot(results_dir)

    work_dir = tmp_path / "cwd"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)

    output_dir = tmp_path / "out"
    _generate(output_dir)

    assert _snapshot(results_dir) == before, "the generator wrote into the simulator's results directory"

    # No TSV, no <sim_id>.json, and no Swift per-cycle I/O left in the working
    # directory: the controller's loop_algo_io_dir is redirected to a scratch
    # directory that is removed when the run finishes.
    stray = sorted(os.listdir(str(work_dir)))
    assert stray == [], "the generator left files in the working directory: {}".format(stray)

    emitted = sorted(os.listdir(str(output_dir)))
    assert emitted == sorted(
        [
            cli.DATA_FLOW_FILENAME,
            cli.SEQUENCE_FILENAME,
            cli.MANIFEST_FILENAME,
            cli.TRACE_FILENAME,
            cli.COVERAGE_FILENAME,
        ]
    )
    assert not any(name.endswith(".tsv") for name in emitted)


def _snapshot(directory):
    if not os.path.isdir(directory):
        return None
    return sorted(os.listdir(directory))


def test_trace_records_shape_but_never_values(generated):
    """No glucose array and no LBGI/DKAI scalar may reach disk."""
    output_dir, _manifest = generated
    records = _trace_records(output_dir)

    permitted_keys = {
        "seq", "stage", "phase", "timestep", "kind",
        "caller", "callee", "arity", "arg_types", "ts_ns", "pid",
    }
    permitted_endpoint_keys = {"package", "module", "resolved_class", "qualname", "attributed"}

    for record in records:
        unexpected = set(record) - permitted_keys
        assert not unexpected, "unexpected key in a trace record: {}".format(unexpected)
        for endpoint in ("caller", "callee"):
            assert set(record[endpoint]) <= permitted_endpoint_keys
        # arg_types carries type *names*, never the values themselves.
        assert all(isinstance(name, str) for name in record["arg_types"])
        assert len(record["arg_types"]) <= record["arity"]

    metrics_records = [r for r in records if r["callee"]["package"] == "tidepool_data_science_metrics"]
    assert metrics_records, "the metrics edge was never exercised"
    for record in metrics_records:
        # The glucose array reaches these functions; only its type name is kept.
        assert record["arg_types"], "the metrics call recorded no argument shape at all"
        assert all(name.isidentifier() for name in record["arg_types"]), (
            "a trace record carries something that is not a bare type name: {}".format(record["arg_types"])
        )


# -- 6. determinism -------------------------------------------------------


def test_two_runs_produce_byte_identical_normalized_bodies(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _generate(first)
    _generate(second)

    for filename in (cli.DATA_FLOW_FILENAME, cli.SEQUENCE_FILENAME):
        assert normalized_body(_read(first, filename)) == normalized_body(_read(second, filename)), (
            "{} is not deterministic between runs".format(filename)
        )

    # The unnormalized text differs, because the header carries a timestamp --
    # which is exactly why the gate strips it.
    assert _read(first, cli.DATA_FLOW_FILENAME) != _read(second, cli.DATA_FLOW_FILENAME)


# -- TRSET-57: null-stage sequence diagram and PA twin ---------------------


def _generate_with(output_dir, scenario=FIXTURE_SCENARIO, stage=SEQUENCE_STAGE, cycle=None):
    kwargs = {} if cycle is None else {"sequence_cycle": cycle}
    return cli.generate(
        output_dir=str(output_dir),
        scenario_path=scenario,
        allowlist_path=ALLOWLIST,
        exclusions_path=EXCLUSIONS,
        sequence_stage=stage,
        repo_root=REPO_ROOT,
        **kwargs
    )


def test_it1_null_stage_first_cycle_sequence(tmp_path):
    manifest = _generate_with(tmp_path, stage=NULL_CONTROLLER_STAGE, cycle="first-cycle")
    sequence = _read(tmp_path, cli.SEQUENCE_FILENAME)
    # The header's package-provenance list names loop_to_python_api whatever
    # the stage; the participants and messages are in the body.
    body = normalized_body(sequence)

    assert "DoNothingController" in body
    assert "get_loop_recommendations" in body
    assert "loop_to_python_api" not in body
    assert "libLoopAlgorithmToPython.dylib" not in body
    assert "apply_loop_recommendations" not in body

    assert manifest["sequence_diagram"]["cycle_rule"] == "first-cycle"
    assert "%% sequence cycle rule: first-cycle" in sequence
    assert "first run-phase timestep of the stage" in sequence
    assert "reference scenario" not in sequence


def test_default_header_and_manifest_record_first_recommendation(generated):
    output_dir, manifest = generated
    sequence = _read(output_dir, cli.SEQUENCE_FILENAME)
    assert manifest["sequence_diagram"]["cycle_rule"] == "first-recommendation"
    assert "%% sequence cycle rule: first-recommendation" in sequence
    assert "non-empty recommendation" in sequence


def test_it2_pa_loop_stage_matches_the_committed_architecture(tmp_path):
    from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

    # Precondition: the PA scenario really has an activity configured, so the
    # comparison below is not vacuous.
    sims = ScenarioParserV2(path_to_json_config=PA_FIXTURE_SCENARIO).get_sims()
    timeline = sims[SEQUENCE_STAGE].virtual_patient.patient_config.pa_timeline
    assert timeline is not None and len(timeline.events) > 0, "PA fixture has no physical activity"

    base_dir = tmp_path / "base"
    pa_dir = tmp_path / "pa"
    _generate_with(base_dir)
    _generate_with(pa_dir, scenario=PA_FIXTURE_SCENARIO)

    assert normalized_body(_read(pa_dir, cli.SEQUENCE_FILENAME)) == normalized_body(
        _read(base_dir, cli.SEQUENCE_FILENAME)
    )


def test_it3_default_invocation_passes_the_drift_gate(tmp_path):
    out = tmp_path / "default"
    assert cli.main(["--scenario", FIXTURE_SCENARIO, "--output-dir", str(out),
                     "--allowlist", ALLOWLIST, "--exclusions", EXCLUSIONS,
                     "--sequence-stage", SEQUENCE_STAGE]) == 0
    assert cli.main(["--scenario", FIXTURE_SCENARIO, "--output-dir", str(out),
                     "--allowlist", ALLOWLIST, "--exclusions", EXCLUSIONS,
                     "--sequence-stage", SEQUENCE_STAGE, "--check"]) == 0


def test_it4_first_recommendation_on_a_null_stage_fails_and_writes_nothing(tmp_path):
    out = tmp_path / "untouched"
    out.mkdir()
    with pytest.raises(SystemExit) as excinfo:
        _generate_with(out, stage=NULL_CONTROLLER_STAGE)
    assert "apply_loop_recommendations" in str(excinfo.value)
    assert os.listdir(str(out)) == []


# -- the full reference scenario ------------------------------------------


@pytest.mark.slow
def test_full_reference_scenario(tmp_path):
    """The real 3 x 23 h reference run. Minutes, not seconds; run on demand."""
    output_dir = tmp_path / "reference"
    manifest = cli.generate(
        output_dir=str(output_dir),
        scenario_path=cli.DEFAULT_SCENARIO,
        allowlist_path=ALLOWLIST,
        exclusions_path=EXCLUSIONS,
        sequence_stage=cli.DEFAULT_SEQUENCE_STAGE,
        repo_root=REPO_ROOT,
    )

    stages = {stage["sim_id"]: stage for stage in manifest["scenario"]["stages"]}
    assert set(stages) == {
        "pre-Loop_NoMitigations_t1_median",
        "pre-NoLoop_t1_median",
        "post-Loop_WithMitigations_t1_median",
    }
    assert all(stage["duration_hrs"] == 23.0 for stage in stages.values())
    assert stages["pre-NoLoop_t1_median"]["controller"] == "DoNothingController"

    # The pointer the scenario names exists under two directories; provenance
    # must say which one was loaded.
    pointers = manifest["scenario"]["resolved_pointer_files"]
    assert any(path.endswith("reusable/simulations/base/base_median_2_0_v1.json") for path in pointers)
    assert not any("base_urai" in path for path in pointers)

    # The fixture is a structural twin: the same architecture, so the same figure.
    fixture_dir = tmp_path / "fixture"
    _generate(fixture_dir)
    assert normalized_body(_read(output_dir, cli.DATA_FLOW_FILENAME)) == normalized_body(
        _read(fixture_dir, cli.DATA_FLOW_FILENAME)
    )
