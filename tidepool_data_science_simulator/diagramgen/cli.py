"""
Command line entry point for the architecture diagram generator.

    python -m tidepool_data_science_simulator.diagramgen
    python -m tidepool_data_science_simulator.diagramgen --check

``--check`` regenerates into a scratch directory and exits non-zero if the
normalized body of a committed *figure* differs. It compares bodies with the
``%%`` provenance header stripped: the manifest's SHAs change on every commit to
any of four packages, and a gate that fired on those would fire constantly.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

from tidepool_data_science_simulator.diagramgen import __version__
from tidepool_data_science_simulator.diagramgen.config import load_allowlist, load_exclusions
from tidepool_data_science_simulator.diagramgen.coverage import (
    classify_sites,
    render_coverage_report,
    static_bind_edges,
)
from tidepool_data_science_simulator.diagramgen.manifest import build_manifest
from tidepool_data_science_simulator.diagramgen.mermaid import (
    normalized_body,
    render_data_flow,
    render_timestep_sequence,
    select_sequence_timestep,
    SEQUENCE_CYCLE_FIRST_CYCLE,
    SEQUENCE_CYCLE_FIRST_RECOMMENDATION,
    SEQUENCE_CYCLE_RULES,
)
from tidepool_data_science_simulator.diagramgen.naming import Node, repo_relative, top_package
from tidepool_data_science_simulator.diagramgen.runner import run_traced
from tidepool_data_science_simulator.diagramgen.staticscan import scan_repo
from tidepool_data_science_simulator.diagramgen.tracer import PHASE_RUN

__all__ = ["main", "generate"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, ".docs", "architecture")

DEFAULT_SCENARIO = os.path.join(
    REPO_ROOT,
    "scenario_configs", "tidepool_risk_v2", "loop_risk_v2_0", "loop_risk_v2_2_0_full",
    "TLR-552", "Simulation-Configuration-TLR-552_Median_profile.json",
)

# The stage the sequence diagram is cut from: the post-mitigation Loop stage, the
# one that exercises the full control cycle.
DEFAULT_SEQUENCE_STAGE = "post-Loop_WithMitigations_t1_median"

DATA_FLOW_FILENAME = "data_flow.mmd"
SEQUENCE_FILENAME = "timestep_sequence.mmd"
MANIFEST_FILENAME = "manifest.json"
TRACE_FILENAME = "trace.jsonl"
COVERAGE_FILENAME = "coverage_report.md"

# Every artifact a generation run emits.
ARTIFACT_FILENAMES = (
    DATA_FLOW_FILENAME,
    SEQUENCE_FILENAME,
    MANIFEST_FILENAME,
    TRACE_FILENAME,
    COVERAGE_FILENAME,
)

# Files whose normalized body the drift gate compares: the figures, and only the
# figures. The manifest and the trace carry commit SHAs, timestamps and PIDs by
# design. The coverage report carries source line numbers, which move whenever
# anything above them is edited -- gating it would fire the architecture drift
# gate on a reformatted comment, and a gate that cries wolf gets switched off.
GATED_FILES = (DATA_FLOW_FILENAME, SEQUENCE_FILENAME)


def _node_factory(module, qualified_symbol):
    """Build a diagram node from a static reference site's module and symbol.

    A symbol whose first segment starts with an upper-case letter is treated as
    a class, which is what makes the figure say ``ScenarioParserV2`` rather than
    the module it lives in.
    """
    head = qualified_symbol.split(".", 1)[0] if qualified_symbol else ""
    component = head if head[:1].isupper() else ""
    return Node(top_package(module), module, component)


_CYCLE_DESCRIPTIONS = {
    SEQUENCE_CYCLE_FIRST_RECOMMENDATION: (
        "control cycle: first one at which the controller returned a non-empty recommendation "
        "(timestep index {})"
    ),
    SEQUENCE_CYCLE_FIRST_CYCLE: (
        "control cycle: first run-phase timestep of the stage, whether or not the controller "
        "recommended anything (timestep index {})"
    ),
}


def _write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def generate(
    output_dir,
    scenario_path,
    allowlist_path,
    exclusions_path,
    sequence_stage,
    repo_root=REPO_ROOT,
    sequence_cycle=SEQUENCE_CYCLE_FIRST_RECOMMENDATION,
):
    """Run the scenario, emit all five artifacts into ``output_dir``.

    Returns the manifest dict.
    """
    allowlist = load_allowlist(allowlist_path)
    exclusions = load_exclusions(exclusions_path)

    run_result = run_traced(scenario_path, allowlist)
    scan_result = scan_repo(repo_root, exclusions, allowlist.module_prefixes)
    classified = classify_sites(scan_result, run_result)

    manifest = build_manifest(
        repo_root=repo_root,
        scenario_path=scenario_path,
        run_result=run_result,
        allowlist=allowlist,
        exclusions=exclusions,
        scan_result=scan_result,
        packages=allowlist.module_prefixes,
    )

    header_lines = _provenance_header(manifest)
    bind_edges = static_bind_edges(classified, _node_factory)

    data_flow = render_data_flow(run_result.records, bind_edges, header_lines)

    timestep = select_sequence_timestep(run_result.records, sequence_stage, PHASE_RUN, sequence_cycle)
    if timestep is None:
        if sequence_cycle == SEQUENCE_CYCLE_FIRST_CYCLE:
            raise SystemExit(
                "Stage {!r} has no run-phase timestep in the trace; the sequence diagram has "
                "nothing to show. Check the stage name.".format(sequence_stage)
            )
        raise SystemExit(
            "No control cycle in stage {!r} reached apply_loop_recommendations; the sequence "
            "diagram has nothing to show. Check that the scenario's controller is configured."
            .format(sequence_stage)
        )
    sequence_header = header_lines + [
        "stage: {}".format(sequence_stage),
        "sequence cycle rule: {}".format(sequence_cycle),
        _CYCLE_DESCRIPTIONS[sequence_cycle].format(timestep),
    ]
    sequence = render_timestep_sequence(
        run_result.records, sequence_stage, PHASE_RUN, timestep, sequence_header
    )

    traced_edges = {
        (record.caller.qualified, record.callee.qualified)
        for record in run_result.records
        if record.caller.package != record.callee.package
    }
    coverage = render_coverage_report(
        classified, scan_result, run_result, allowlist, exclusions, len(traced_edges)
    )

    manifest["sequence_diagram"] = {
        "stage": sequence_stage,
        "timestep": timestep,
        "cycle_rule": sequence_cycle,
    }

    os.makedirs(output_dir, exist_ok=True)
    _write(os.path.join(output_dir, DATA_FLOW_FILENAME), data_flow)
    _write(os.path.join(output_dir, SEQUENCE_FILENAME), sequence)
    _write(os.path.join(output_dir, COVERAGE_FILENAME), coverage)
    _write(os.path.join(output_dir, MANIFEST_FILENAME), json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with open(os.path.join(output_dir, TRACE_FILENAME), "w", encoding="utf-8") as handle:
        for record in run_result.records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True))
            handle.write("\n")

    return manifest


def _provenance_header(manifest):
    """Provenance lines stamped into each figure as ``%%`` comments."""
    lines = [
        "Generated by tidepool_data_science_simulator.diagramgen v{}".format(__version__),
        "Do not edit by hand. Regenerate with:",
        "    python -m tidepool_data_science_simulator.diagramgen",
        "Drift gate: python -m tidepool_data_science_simulator.diagramgen --check",
        "",
        "This figure records cross-package calls OBSERVED during a real, in-process run",
        "of the scenario named below. It is evidence of executed behavior, not an",
        "assertion about permitted structure.",
        "",
        "scenario: {}".format(manifest["scenario"]["path"]),
        "generated (UTC): {}".format(manifest["generator"]["generated_utc"]),
        "allowlist sha256: {}".format(manifest["config"]["allowlist"]["sha256"]),
        "",
        "package provenance:",
    ]
    for package in manifest["packages"]:
        if not package.get("available"):
            lines.append("    {}: unavailable ({})".format(package["package"], package.get("reason")))
            continue
        vcs = package.get("vcs", {})
        version = package.get("distribution_version") or "unknown"
        if vcs.get("kind") == "git" and vcs.get("commit"):
            state = "dirty" if vcs.get("dirty") else "clean"
            lines.append("    {}: {} ({})".format(package["package"], vcs["commit"], state))
        elif vcs.get("kind") == "unavailable":
            lines.append(
                "    {}: commit UNRESOLVED (git unusable on this host); "
                "distribution version {}".format(package["package"], version)
            )
        else:
            lines.append(
                "    {}: no git working tree; distribution version {}".format(package["package"], version)
            )
    return lines


def _check(
    output_dir,
    scenario_path,
    allowlist_path,
    exclusions_path,
    sequence_stage,
    repo_root,
    sequence_cycle=SEQUENCE_CYCLE_FIRST_RECOMMENDATION,
):
    """Regenerate into a scratch directory and diff the gated bodies."""
    scratch = tempfile.mkdtemp(prefix="trset51-check-")
    try:
        generate(
            scratch, scenario_path, allowlist_path, exclusions_path, sequence_stage, repo_root, sequence_cycle
        )
        drifted = []
        for filename in GATED_FILES:
            committed_path = os.path.join(output_dir, filename)
            if not os.path.isfile(committed_path):
                drifted.append((filename, "missing from {}".format(repo_relative(output_dir, repo_root))))
                continue
            with open(committed_path, encoding="utf-8") as handle:
                committed = normalized_body(handle.read())
            with open(os.path.join(scratch, filename), encoding="utf-8") as handle:
                regenerated = normalized_body(handle.read())
            if committed != regenerated:
                drifted.append((filename, "normalized body differs"))
        return drifted
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m tidepool_data_science_simulator.diagramgen",
        description="Generate trace-based architecture figures for the TRSET position paper.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where the artifacts are written.")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="Scenario configuration JSON to trace.")
    parser.add_argument("--allowlist", default=None, help="Capture allowlist (defaults to <output-dir>/allowlist.yml).")
    parser.add_argument(
        "--exclusions", default=None, help="Static-pass exclusion list (defaults to <output-dir>/exclusions.yml)."
    )
    parser.add_argument(
        "--sequence-stage",
        default=DEFAULT_SEQUENCE_STAGE,
        help="Scenario stage the single-timestep sequence diagram is cut from.",
    )
    parser.add_argument(
        "--sequence-cycle",
        choices=SEQUENCE_CYCLE_RULES,
        default=SEQUENCE_CYCLE_FIRST_RECOMMENDATION,
        help="Control cycle the sequence diagram is cut from: the first cycle whose controller "
        "returned a recommendation (default), or the first run-phase cycle of the stage "
        "(needed for a null-controller stage, which never recommends).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate and exit non-zero if a committed figure's normalized body has drifted.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    output_dir = os.path.abspath(args.output_dir)
    allowlist_path = args.allowlist or os.path.join(DEFAULT_OUTPUT_DIR, "allowlist.yml")
    exclusions_path = args.exclusions or os.path.join(DEFAULT_OUTPUT_DIR, "exclusions.yml")

    if args.check:
        drifted = _check(
            output_dir, args.scenario, allowlist_path, exclusions_path, args.sequence_stage, REPO_ROOT,
            args.sequence_cycle,
        )
        if drifted:
            sys.stderr.write("Architecture figures have drifted from the code:\n")
            for filename, reason in drifted:
                sys.stderr.write("  {}: {}\n".format(filename, reason))
            sys.stderr.write(
                "Regenerate with: python -m tidepool_data_science_simulator.diagramgen\n"
            )
            return 1
        sys.stdout.write("Architecture figures are up to date.\n")
        return 0

    manifest = generate(
        output_dir, args.scenario, allowlist_path, exclusions_path, args.sequence_stage, REPO_ROOT,
        args.sequence_cycle,
    )
    sys.stdout.write(
        "Wrote {} artifacts to {}\n".format(len(ARTIFACT_FILENAMES), repo_relative(output_dir, REPO_ROOT))
    )
    sys.stdout.write(
        "  {} records, {} stages, sequence cut from {} timestep {}\n".format(
            manifest["capture"]["records"],
            len(manifest["scenario"]["stages"]),
            manifest["sequence_diagram"]["stage"],
            manifest["sequence_diagram"]["timestep"],
        )
    )
    return 0
