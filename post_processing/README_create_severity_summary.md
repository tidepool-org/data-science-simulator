# Risk Severity Summary Generator — failure contract

## Description

`create_severity_summary.py` renders the `risk_summary_<sim_id>.rtf` documents for
a risk-run results directory. **Bugfix:** `process_results_directory()` used to
print a diagnostic and return `None` on every failure path, so no caller could
tell a no-op run from a successful one, and the CLI printed `Done!` and exited
`0` regardless. It now **raises `SeveritySummaryError`** when a directory cannot
be summarized at all, **returns a `SummaryResult`** naming what it wrote and
which directories it skipped, only treats `TLR-*` **directories** (sorted) as run
directories, and never dates a summary with the placeholder `"Unknown"`.

## Example usage

```bash
# Exits 0 only if at least one summary document was written.
python create_severity_summary.py /path/to/Risk_Run_<timestamp>
```

Programmatic use:

```python
from create_severity_summary import SeveritySummaryError, process_results_directory

try:
    result = process_results_directory(save_dir)
except SeveritySummaryError as exc:
    ...                                  # unusable directory: report, don't proceed
for tlr_dir, reason in result.skipped:   # legitimate partial outcome
    print(f"no summary for {tlr_dir}: {reason}")
print(f"wrote {len(result.written)} summaries")
```

### What raises vs. what is reported

| Condition | Before | Now |
|---|---|---|
| `metadata.json` missing | print + return `None` | raises `SeveritySummaryError` |
| `metadata.json` unparseable | raw `JSONDecodeError` | raises `SeveritySummaryError` |
| no `timestamp` / `run_timestamp` key | summary dated `"Unknown"` | raises `SeveritySummaryError` |
| no `TLR-*` subdirectories | print + return `None` | raises `SeveritySummaryError` |
| one `TLR-*` dir yields no assessment | bare `continue` | reported in `result.skipped` |
| stray file named `TLR-*.txt` | passed to `build_assessment` | ignored (directories only) |
| nothing written | CLI prints `Done!`, exits `0` | CLI exits `1` |

A single `TLR-*` directory with no usable data stays a *partial outcome*, not an
error — a real run can legitimately contain one, and the GUI already renders that
state — so it is reported rather than raised.

## Validation

17 tests added (`test_create_severity_summary.py`): each raising condition, the
`run_timestamp` fallback key still being honored, per-directory skips, mixed
runs, sorted processing order, the stray-`TLR-*`-file case, and all four CLI exit
codes. One test asserts the written RTF is **byte-identical** to `render_rtf`'s
output, alongside the pre-existing 32-test renderer suite — RTF bytes are
unchanged. Full suite: 556 passed / 31 failed / 5 skipped, the 31 being the
documented pre-existing set (13 in `test_validate_configs_integration.py`, 18 in
untracked local scratch files). The GUI's suite is unaffected: 138 passed / 7
skipped.

## Cautions / limitations

**Behavior change, not a signature change — migration:** any caller that relied
on `process_results_directory` returning quietly on a bad directory must now
catch `SeveritySummaryError`. The two in-repo callers are handled: the CLI
converts it to exit code `1`, and the GUI's export path
(`loop-risk-simulator-gui/export_bundle.py`) pre-checks both fatal conditions
itself and surfaces any exception as an error in the UI. Ad-hoc scripts outside
these repos may need the `try`/`except` shown above.

`gui_runner.METADATA_FILENAME` is now imported from this module rather than
re-declared, so the writer and reader of `metadata.json` cannot drift apart; the
constant is still importable from `gui_runner` for consumers.

Regression risk **Medium** — shared post-processing entry point used by both the
CLI and the GUI export — contained by RTF output being byte-identical and the
change touching only orchestration. Not a breaking change to any signature or
schema, so no migration of stored data applies; a revert is a single-commit
revert of this change. `create_severity_summary_ORIGINAL.py` retains the old
behavior and is deliberately untouched (legacy reference copy).
