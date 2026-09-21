"""
Mermaid emission for the two figures.

Determinism rules, because the ``--check`` drift gate depends on all of them:

* Node ids come from a stable hash of the qualified name, never insertion order.
* Edges are emitted sorted by ``(caller, callee)``; symbol lists inside a label
  are sorted too.
* Provenance is stamped as a ``%%`` comment header and the gate compares only the
  normalized body with that header stripped. The manifest's SHAs change on every
  commit to any of four packages and would otherwise fire the gate constantly.
* No timestamp, duration or PID reaches a figure. Those live in ``trace.jsonl``.
"""

import collections

from tidepool_data_science_simulator.diagramgen.naming import NATIVE_PACKAGE

__all__ = [
    "HEADER_PREFIX",
    "normalized_body",
    "render_data_flow",
    "render_header",
    "render_timestep_sequence",
    "select_sequence_timestep",
]

HEADER_PREFIX = "%%"

PACKAGE_TITLES = {
    "tidepool_data_science_simulator": "data-science-simulator",
    "loop_to_python_api": "LoopAlgorithmToPython (loop_to_python_api)",
    "tidepool_data_science_models": "data-science-models",
    "tidepool_data_science_metrics": "data-science-metrics",
    NATIVE_PACKAGE: "Swift shared library",
}

# Rendering order for the package subgraphs: the runtime path reads left to
# right, with the results path last.
PACKAGE_ORDER = (
    "tidepool_data_science_simulator",
    "tidepool_data_science_models",
    "loop_to_python_api",
    NATIVE_PACKAGE,
    "tidepool_data_science_metrics",
)


def _package_sort_key(package):
    try:
        return (0, PACKAGE_ORDER.index(package))
    except ValueError:
        return (1, package)


def _escape(text):
    """Make a string safe inside a Mermaid quoted label."""
    return text.replace('"', "&quot;").replace("#", "&#35;")


def _short_symbol(qualname):
    """Last meaningful segment of a qualified name, for an edge label."""
    if qualname.endswith(".__init__"):
        return qualname.rsplit(".", 2)[-2] + "()"
    return qualname.rsplit(".", 1)[-1]


def render_header(lines):
    """Render provenance lines as a ``%%`` comment block."""
    return "\n".join("{} {}".format(HEADER_PREFIX, line).rstrip() for line in lines)


def normalized_body(text):
    """Strip the provenance header and normalise whitespace for diffing."""
    kept = [
        line.rstrip()
        for line in text.splitlines()
        if not line.lstrip().startswith(HEADER_PREFIX)
    ]
    while kept and not kept[0]:
        kept.pop(0)
    while kept and not kept[-1]:
        kept.pop()
    return "\n".join(kept) + "\n"


# -- data flow ------------------------------------------------------------


def collect_data_flow(records, static_bind_edges):
    """Group traced cross-package calls, plus construction bindings, into edges.

    Returns ``(nodes_by_package, edges)`` where ``edges`` maps
    ``(caller, callee)`` to a dict describing the edge.
    """
    edges = collections.OrderedDict()
    nodes = set()

    traced = collections.defaultdict(lambda: {"symbols": set(), "kind": "call", "attributed": False})
    for record in records:
        if record.caller.package == record.callee.package:
            continue
        key = (record.caller, record.callee)
        entry = traced[key]
        entry["symbols"].add(_short_symbol(record.callee_qualname))
        entry["attributed"] = entry["attributed"] or record.caller_attributed
        if record.kind == "native_call":
            entry["kind"] = "native_call"

    for key, entry in traced.items():
        edges[key] = entry
        nodes.add(key[0])
        nodes.add(key[1])

    for caller, callee, symbols in static_bind_edges:
        key = (caller, callee)
        if key in edges:
            # Already observed as a call; the stronger evidence wins.
            continue
        edges[key] = {"symbols": set(symbols), "kind": "bind", "attributed": False}
        nodes.add(caller)
        nodes.add(callee)

    by_package = collections.defaultdict(list)
    for node in nodes:
        by_package[node.package].append(node)
    for package in by_package:
        by_package[package].sort()

    ordered = collections.OrderedDict()
    for key in sorted(edges, key=lambda pair: (pair[0].qualified, pair[1].qualified)):
        ordered[key] = edges[key]
    return by_package, ordered


def render_data_flow(records, static_bind_edges, header_lines):
    """Render the cross-package data flow diagram."""
    by_package, edges = collect_data_flow(records, static_bind_edges)

    lines = [render_header(header_lines), "", "flowchart LR"]

    for package in sorted(by_package, key=_package_sort_key):
        title = PACKAGE_TITLES.get(package, package)
        lines.append('    subgraph sg_{}["{}"]'.format(package.replace(".", "_"), _escape(title)))
        lines.append("        direction TB")
        for node in by_package[package]:
            label = _escape(node.label)
            sublabel = _escape(node.sublabel)
            text = "{}<br/>{}".format(label, sublabel) if sublabel else label
            lines.append('        {}["{}"]'.format(node.mermaid_id, text))
        lines.append("    end")

    lines.append("")
    for (caller, callee), entry in edges.items():
        label = _escape("<br/>".join(sorted(entry["symbols"])))
        if entry["kind"] == "bind":
            arrow = "-.->"
            label = "binds (construction)<br/>" + label
        else:
            arrow = "==>" if entry["kind"] == "native_call" else "-->"
        if entry["attributed"]:
            label = "results path<br/>" + label
        lines.append('    {} {}|"{}"| {}'.format(caller.mermaid_id, arrow, label, callee.mermaid_id))

    lines.append("")
    lines.append("    classDef nativeBox stroke-width:3px")
    native_ids = sorted(node.mermaid_id for node in by_package.get(NATIVE_PACKAGE, []))
    if native_ids:
        lines.append("    class {} nativeBox".format(",".join(native_ids)))

    return "\n".join(lines).rstrip() + "\n"


# -- sequence -------------------------------------------------------------


def select_sequence_timestep(records, stage, run_phase):
    """Pick the control cycle the sequence diagram is cut from.

    The first timestep of ``stage`` at which the controller returns a non-empty
    recommendation -- observed as ``apply_loop_recommendations`` being reached,
    since ``Simulation.update`` only calls it when the recommendation is
    truthy. That is the first fully-exercised control cycle, past warm-up. The
    ``Simulation.init()`` call at t=0 is excluded for free: it runs during the
    construction phase, before the loop, and has a different shape.

    Returns the timestep index, or ``None`` when no cycle qualifies.
    """
    candidates = collections.defaultdict(list)
    for record in records:
        if record.stage == stage and record.phase == run_phase and record.timestep is not None:
            candidates[record.timestep].append(record)

    for timestep in sorted(candidates):
        for record in candidates[timestep]:
            if record.callee_qualname.rsplit(".", 1)[-1] == "apply_loop_recommendations":
                return timestep
    return None


def render_timestep_sequence(records, stage, run_phase, timestep, header_lines):
    """Render the single-timestep sequence diagram."""
    selected = [
        record
        for record in records
        if record.stage == stage and record.phase == run_phase and record.timestep == timestep
    ]

    participants = []
    seen = set()
    for record in selected:
        for node in (record.caller, record.callee):
            if node.qualified not in seen:
                seen.add(node.qualified)
                participants.append(node)

    lines = [render_header(header_lines), "", "sequenceDiagram", "    autonumber"]
    for node in participants:
        lines.append('    participant {} as {}'.format(node.mermaid_id, _escape(node.participant_label)))

    lines.append("")
    for record in selected:
        lines.append(
            "    {} ->> {}: {}".format(
                record.caller.mermaid_id,
                record.callee.mermaid_id,
                _escape(_short_symbol(record.callee_qualname)),
            )
        )

    return "\n".join(lines).rstrip() + "\n"
