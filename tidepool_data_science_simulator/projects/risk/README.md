## [bugfix] `loop_risk_v2_0.py` syntax error + explicit single-directory filter

**Description:** `build_risk_sim_generator()` in `loop_risk_v2_0.py` contained an unterminated string literal (`if "TLR-HF not in risk_dir_name:`), a syntax error that made the whole module unimportable. The line was leftover debug scaffolding that silently restricted every run to only `TLR-HF*` directories. Fixed the syntax error and replaced the hardcoded filter with an explicit `target_risk_dir` keyword argument, so single-directory runs are an intentional caller choice instead of a silent default.

**Example usage:**
```python
from tidepool_data_science_simulator.projects.risk.loop_risk_v2_0 import build_risk_sim_generator

# Run every TLR-* directory (previously silently restricted to TLR-HF* only)
for risk_dir_name, scenario_json_name, sim_suite in build_risk_sim_generator(scenario_dir):
    ...

# Run a single directory on purpose
for risk_dir_name, scenario_json_name, sim_suite in build_risk_sim_generator(
    scenario_dir, target_risk_dir="TLR-HF"
):
    ...
```

**Validation:** Added `tests/test_build_risk_sim_generator.py` (4 tests, `ScenarioParserV2` monkeypatched out) covering: no filter processes all `TLR-*` dirs, a substring filter restricts to matches, a non-matching filter yields nothing, non-`TLR-*` directories are always excluded. Full existing suite run for regression: 19 pre-existing failures found, all in files unrelated to this change (confirmed via grep — none reference this module) and pre-dating this fix; 291 passed, 5 skipped otherwise.

**Cautions/limitations:** The `if __name__ == "__main__":` block still has no CLI flag for `target_risk_dir` (kwarg-only, by design — this repo has no `argparse` in this file today; adding a flag was scoped out as a separate small enhancement, not part of this bugfix).

**Regression risk:** Low — isolated function, single package; nothing in the codebase called it successfully before this fix (syntax error blocked import).

---

## [bugfix] TRSET-80: main-loop failures are reported, never silent

**Description:** A failure in one scenario used to be silent or to stop the whole run: a malformed JSON ended every scenario after it, metrics errors left sim_id-only rows, dummy and NaN rows were written, and the exit code was always 0. Now each JSON is handled on its own, every failure is recorded and the run carries on, nothing fake is written, and a TLR with any failure gets no RTF severity summary. `run_simulations` raises `SimulationRunError` after the batch if any sim failed.

**Example usage:**
```python
from tidepool_data_science_simulator.run import run_simulations, SimulationRunError

try:
    full_results, summary_df = run_simulations(sims, save_dir)
except SimulationRunError as e:
    e.failures         # [{sim_id, stage, exception_type, message, traceback}, ...]
    e.full_results     # results of the sims that succeeded
    e.summary_results  # their summary rows; failed sims have none
```
```
python -m tidepool_data_science_simulator.projects.risk.loop_risk_v2_0; echo $?
# 3 of 4 scenarios succeeded; 1 failed; see <run dir>/failures.log
```

**Exit codes:** `0` everything succeeded. `1` some failures, with partial results. `2` nothing succeeded, or a fatal error (results directory, scenario listing, or the run summary could not be written).

**Run outputs:**
- `metadata.json` key order: `timestamp`, `scenarios_attempted`, `scenarios_succeeded`, other run-level fields (`severity_summaries_skipped`), `failures_log` (file name `failures.log`, in the run directory), `failures` last. `failures` is `[]` when nothing failed. Each entry has `tlr`, `json`, `sim_id` (null for parse and plot failures), `stage` (`parse`, `run`, `metrics`, `plot` or `save`), `exception_type`, `message`.
- `failures.log` is written on every run with the full tracebacks, independent of the debug-logging setting.
- A JSON with any failure contributes no rows to `Risk_Results_*.csv` (all of its sims, not only the failed one). A run with no successful JSON writes no `Risk_Results_*.csv`. No dummy rows; the CSV schemas are unchanged.
- `run_simulations` writes its `summary_results_<json>.csv` only when no sim failed, so an incomplete one is never left on disk.
- **RTF skipping:** `process_results_directory(results_dir, failed_tlrs=None)` skips every TLR in `failed_tlrs` (TLR directory name to number of failed JSONs) with the reason `incomplete TLR: N failed`, recorded in `metadata.json`. An incomplete severity summary can be misleading, so none is written. Other TLRs still get their RTF.

**Validation:** `tests/test_main_loop_failure_reporting.py` injects one failure at each stage (parse, run, metrics, plot, save) into a 2-TLR fixture corpus and checks the other scenarios complete, no row is written for the failed JSON, the failure is in `metadata.json` and `failures.log` with the right stage, and the exit code. It also covers all-fail (exit 2, no CSV), a partial TLR (RTF skipped, other TLR kept, exit 1) and the key order. `tests/test_run_simulations_failures.py` covers `run_simulations`. A real run on 2 TLRs x 2 profiles gave output identical to before (Risk_Results, per-sim TSVs, summary CSVs).

**Cautions/limitations:**
- A sim whose child process crashes inside `Simulation.run` still hangs the run at `queue.get()`; this is out of scope here (TRSET-26).
- The GUI (`gui_runner.py`) is unchanged: a sim failure surfaces as a raw `SimulationRunError` traceback until a follow-up ticket adds an error display.
- A plot failure drops the JSON from `Risk_Results` and skips its TLR's RTF, even though its per-profile summary CSV exists.
- A TLR whose filenames carry no `TLR-NNN` ID is still skipped by the severity summary for that unrelated, existing reason.

**Breaking change:** `run_simulations` now raises `SimulationRunError` when a sim fails, instead of leaving a NaN or sim_id-only row. Callers (the older research scripts, `gui_runner.py`, `loop_risk_v2_0-test.py`) that need the old behavior catch it and read `e.full_results` / `e.summary_results`. The risk script's exit code is no longer always 0, and incomplete TLRs get no RTF.

**Regression risk:** Medium. Successful runs are byte-identical to before. Rollback: revert the commit.
