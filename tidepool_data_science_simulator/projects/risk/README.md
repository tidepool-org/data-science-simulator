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
