"""
Coverage of the static cross-package reference set by the traced run.

The report exists to make the figures falsifiable: an edge the static pass can
see but the trace never exercised is stated, not quietly dropped. The header
carries the excluded-file count and the exclusion-list SHA so a jump in the
unexercised count is traceable to a stale exclusion list rather than mistaken
for an architecture change.

Static reachability cannot see dynamic dispatch -- the parser resolves
controllers by string key (``"id": "swift"``) -- so this states its method
rather than claiming completeness.
"""

import collections

from tidepool_data_science_simulator.diagramgen import __version__

__all__ = ["classify_sites", "render_coverage_report", "static_bind_edges"]

_HEADER_MARKER = "<!-- provenance:"

EXERCISED = "exercised"
FUNCTION_NOT_EXECUTED = "function-not-executed"
MODULE_NOT_IMPORTED = "module-not-imported"
NOT_TAKEN = "not-taken"

_STATUS_TEXT = {
    EXERCISED: "exercised",
    NOT_TAKEN: "not taken",
    FUNCTION_NOT_EXECUTED: "enclosing function never ran",
    MODULE_NOT_IMPORTED: "module never imported",
}

_REPORTED_STATUSES = (EXERCISED, NOT_TAKEN, FUNCTION_NOT_EXECUTED, MODULE_NOT_IMPORTED)


def _traced_index(records):
    """Index traced edges for matching against static reference sites.

    Keyed by ``(caller module, callee module, callee symbol)``. The symbol is
    the first segment of the callee's qualified name, so a traced
    ``SimpleMetabolismModel.__init__`` matches a static ``SimpleMetabolismModel(...)``.
    """
    index = collections.defaultdict(set)
    for record in records:
        symbol = record.callee_qualname.split(".", 1)[0]
        index[(record.caller.module, record.callee.module, symbol)].add(record.callee_qualname)
    return index


def classify_sites(scan_result, run_result):
    """Label every static reference site against what the run actually did."""
    index = _traced_index(run_result.records)
    executed = {(module, qualname) for module, qualname in run_result.executed_functions}

    loaded = run_result.loaded_modules

    classified = []
    for site in scan_result.sites:
        symbol = site.target_symbol.split(".", 1)[0]
        enclosing_ran = (site.module, site.enclosing) in executed
        if index.get((site.module, site.target_module, symbol)):
            status = EXERCISED
        elif site.kind == "bind" and enclosing_ran:
            # A binding is not a branch: if the function that names the symbol
            # ran, the name was bound. `ScenarioParserV2` handing
            # `SimpleMetabolismModel` to `VirtualPatient` is exercised even
            # though no call event carries it.
            status = EXERCISED
        elif enclosing_ran:
            status = NOT_TAKEN
        elif site.enclosing == "<module>" and site.module not in loaded:
            status = MODULE_NOT_IMPORTED
        else:
            status = FUNCTION_NOT_EXECUTED
        classified.append((site, status))
    return classified


def static_bind_edges(classified, node_factory):
    """Construction-time binding edges that belong on the data flow diagram.

    Restricted to bindings whose enclosing function actually executed during the
    traced run. ``ScenarioParserV2.build_components_from_config`` binds
    ``SimpleMetabolismModel`` and did run; the same binding in a dozen
    exploratory project scripts did not, and stays off the figure.
    """
    grouped = collections.defaultdict(set)
    for site, status in classified:
        if site.kind != "bind" or status != EXERCISED:
            continue
        caller = node_factory(site.module, site.enclosing)
        callee = node_factory(site.target_module, site.target_symbol)
        if caller.package == callee.package:
            continue
        grouped[(caller, callee)].add(site.target_symbol)

    def by_qualified_pair(item):
        caller, callee = item[0]
        return (caller.qualified, callee.qualified)

    return [
        (caller, callee, sorted(symbols))
        for (caller, callee), symbols in sorted(grouped.items(), key=by_qualified_pair)
    ]


def render_coverage_report(classified, scan_result, run_result, allowlist, exclusions, traced_edge_count):
    """Render ``coverage_report.md``."""
    by_status = collections.Counter(status for _site, status in classified)
    unexercised = [(site, status) for site, status in classified if status != EXERCISED]

    lines = []
    # Volatile measures live in the header, not the body. The excluded-file
    # count moves whenever any Python file is added anywhere in the repository,
    # which is not an architecture change; keeping it out of the body means a
    # reader comparing two reports sees only what actually changed.
    lines.append("{}".format(_HEADER_MARKER))
    lines.append("     generator: {}".format(__version__))
    lines.append("     allowlist sha256:  {}".format(allowlist.sha256))
    lines.append("     exclusions sha256: {}".format(exclusions.sha256))
    lines.append("     python files scanned:  {}".format(scan_result.files_scanned))
    lines.append("     python files excluded: {}".format(scan_result.files_excluded))
    lines.append("     python files that failed to parse: {}".format(len(scan_result.unparsable)))
    lines.append("     allowlisted functions that executed: {}".format(len(run_result.executed_functions)))
    lines.append("-->")
    lines.append("")
    lines.append("# Cross-package coverage report")
    lines.append("")
    lines.append(
        "Generated by `tidepool_data_science_simulator.diagramgen`. This report diffs the "
        "cross-package reference sites a whole-repo AST pass can see against the edges a real, "
        "in-process run of the reference scenario actually exercised."
    )
    lines.append("")

    lines.append("## Scope of the static pass")
    lines.append("")
    lines.append("| Measure | Value |")
    lines.append("| --- | --- |")
    lines.append("| Cross-package reference sites found | {} |".format(len(scan_result.sites)))
    lines.append("| Cross-package edges observed in the run | {} |".format(traced_edge_count))
    lines.append("")
    lines.append(
        "File counts and the exclusion-list SHA are in this file's provenance header. The excluded "
        "count is expected to dwarf the scanned count: `build/`, `.claude/worktrees/`, `venv*/` and "
        "`notebooks/` are excluded unconditionally, and the curated list in `exclusions.yml` removes "
        "the design-and-development data science the simulator carries deliberately outside this "
        "description. A jump in the unexercised count below should be checked against that SHA "
        "before it is read as an architecture change."
    )
    lines.append("")

    lines.append("## Result")
    lines.append("")
    lines.append("| Status | Sites |")
    lines.append("| --- | --- |")
    for status in _REPORTED_STATUSES:
        lines.append("| {} | {} |".format(_STATUS_TEXT[status], by_status.get(status, 0)))
    lines.append("")

    lines.append("## Unexercised cross-package reference sites")
    lines.append("")
    if not unexercised:
        lines.append("None. Every enumerated cross-package reference site was exercised by the run.")
    else:
        lines.append("| Location | Line | Target | Kind | Status |")
        lines.append("| --- | --- | --- | --- | --- |")
        for site, status in unexercised:
            lines.append(
                "| `{}` `{}` | {} | `{}.{}` | {} | {} |".format(
                    site.rel_path,
                    site.enclosing,
                    site.lineno,
                    site.target_module,
                    site.target_symbol,
                    site.kind,
                    _STATUS_TEXT[status],
                )
            )
    lines.append("")

    lines.append("## Method, and what this report does not claim")
    lines.append("")
    lines.append(
        "- The static pass resolves a reference only when the symbol is imported by name into the "
        "file that uses it. It does not resolve dynamic dispatch: `ScenarioParserV2` picks a "
        "controller from a string key (`\"id\": \"swift\"`), and no AST pass can see that."
    )
    lines.append(
        "- Method calls on instances (for example `model.run(...)` where `model` was passed in) are "
        "not enumerated, because the static pass cannot know the receiver's type. Those edges still "
        "appear in the trace and on the figures; they simply have no static counterpart to diff "
        "against."
    )
    lines.append(
        "- The `ctypes` boundary into `libLoopAlgorithmToPython.dylib` is observed at runtime but "
        "has no static counterpart here: the call sites live in `loop_to_python_api`, outside this "
        "repository, and the static pass walks this repository only."
    )
    lines.append(
        "- A `bind` site is counted as exercised when its enclosing function ran: binding a name is "
        "not a branch, so there is no separate call event to observe. "
        "`ScenarioParserV2.build_components_from_config` handing `SimpleMetabolismModel` to "
        "`VirtualPatient` is the case this covers."
    )
    lines.append(
        "- \"Enclosing function never ran\" and \"module never imported\" are statements about this "
        "scenario, not about the code being dead."
    )
    lines.append(
        "- New files default to included. Nothing is skipped silently; every exclusion is either "
        "unconditional (see `config.UNCONDITIONAL_DIR_EXCLUSIONS`) or listed in `exclusions.yml`."
    )
    lines.append("")

    if scan_result.unparsable:
        lines.append("## Files that failed to parse")
        lines.append("")
        for path, error in scan_result.unparsable:
            lines.append("- `{}`: {}".format(path, error))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
