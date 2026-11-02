# URAI-RAI scenario generator (TRSET-74, TRSET-77)

Generates paired **URAI (Fiasp)** and **RAI (rapid-acting adult)** Swift-autobolus scenarios from the raw-format
virtual-patient pool, for an A/B study on identical inputs. Replaces `projects/autobolus_settings_variability/`.

## Run

```bash
URAI_RAI_PATIENT_POOL=/path/to/raw_files \
  python -m tidepool_data_science_simulator.projects.urai_rai_analysis.generate \
  --config tidepool_data_science_simulator/projects/urai_rai_analysis/example_run_config.json
```

`--pool-dir` overrides the environment variable; `--scenario-root` redirects output (default
`scenario_configs/tidepool_risk_v2`). A non-empty output directory is an error; nothing is overwritten.

Outputs:

- `scenario_configs/tidepool_risk_v2/loop_risk_v2_0/loop_risk_v2_urai_rai/<run_name>/` is the scenario root the risk
  runner is pointed at (set `TIDEPOOL_RISK_SCENARIOS_DIR` in `projects/risk/loop_risk_v2_0.py` to it). It holds one
  `TLR-000_<patient_id>/` folder per patient with all of that patient's scenario JSONs across cohorts, plus the run-level
  records, which never sit inside a `TLR-` folder: `manifest.csv`, `exclusions.csv`, `eligible_pool.csv` and
  `run_config.json` (a copy of the config). Results land where the runner writes them,
  `<DATA_DIR>/results/tidepool_loop_risk_v2_0/Risk_Run_<timestamp>/TLR-000_<patient_id>/`.
- `scenario_configs/tidepool_risk_v2/reusable/glucose/urai_rai/<run_name>/<patient_id>.csv` holds the glucose seeds.

### Replacement and comparator runs

```bash
# replace failed pairs; pairs.csv has the columns pair_id,reason (pair_id is "<cohort>_<patient_id>")
python -m tidepool_data_science_simulator.projects.urai_rai_analysis.generate \
  --replace <scenario root>/loop_risk_v2_urai_rai/<run_name> --replacements pairs.csv --run-name <new_run_name> \
  [--prior-run <earlier replacement run folder> ...]
```

A replacement is the next patient in the cohort's seeded order that is not in the cohort, was not a replacement in a
`--prior-run`, and was not used earlier in the same run. It gets a new derived meal seed. Output is a new run folder
holding only the replacement scenarios; the original run is read, never written. Running out of eligible patients is an
error. For the static max bolus comparator, generate again with the same config and seed, a new `run_name` and
`max_bolus.static_u` set (30 for the validation): only the controller `max_bolus`, the manifest `max_bolus_mode` column
and paths containing `<run_name>` differ.

## Design

- **Pool.** Read only; never written. Every file is read once.
- **Eligibility (before any cohort is drawn).** Each patient is excluded by the first failing rule and recorded in
  `exclusions.csv` with a detail string: E1 age below `min_age` (6); E2 no run of 137 consecutive glucose values; E3 a
  seed value outside 20-600 mg/dL (no search for a later run); E4 a missing or non-numeric basal, CIR, ISF, target or
  suspend threshold (a missing suspend threshold takes `default_suspend_threshold` when it is set, and the patient is
  flagged `default_suspend`); E5 a value outside its absolute guardrail (config `guardrails`); E6 target minimum above
  target maximum, or suspend threshold above a target minimum. A missing age, a mismatch between pump and patient-model
  settings in an eligible patient, and a meal-rule failure are hard errors, not exclusions. Nothing is written until
  every check and every scenario build has finished.
- **Cohorts.** Eight DAP cohorts in `example_run_config.json`: pediatric, adult and older adult with meals announced
  and unannounced, plus `isf_high` and `isf_low` (announced). Named filter + n (default 30), filtered from the eligible
  pool. Each cohort's eligible patients are put in a seeded order (seed from `master_seed` and the cohort name, logged);
  the first n are the cohort, and the rest are its replacement candidates. A patient may be in several cohorts. Filters:
  `age_min` (>=), `age_max` (<), `isf_min` (>=), `isf_max` (<=). Fewer eligible than n is an error unless
  `accept_smaller_group` is true; accepted shortfalls are logged and written to the manifest.
- **Pairing.** Each draw yields a URAI and a RAI scenario identical except `controller.settings.model` and
  `patient_model.metabolism_settings.patient_insulin_type` (`fiasp` / `rapid_acting_adult`, flipped together).
- **Controller.** Swift, autobolus only, from `base_median_autobolus_v1_2` / `autobolus_v1_2`; partial application
  factor 0.4 (config); `suspend_threshold` from the patient file or the default.
- **Max bolus.** Per patient: min(0.5 x TDD, 30 U), TDD = 2 x the scheduled basal integrated over the day, rounded down
  to the 0.05 U delivery increment. The fraction, multiplier, ceiling and increment are config values; the arithmetic is
  exact (`fractions.Fraction`), so on-grid values are not pushed down a step. `max_bolus.static_u` replaces the value
  for every patient (comparator runs). `tdd_estimate_u`, `max_bolus_u`, `max_bolus_clamped` and `meal80_fits` are
  written for every eligible patient and never exclude.
- **Settings.** Basal, CIR and ISF come from the patient file for pump and patient model (no variance). The generator
  stops if an eligible patient's pump and `actual_*` values differ. Target range comes from `target_range_minimum_values` /
  `target_range_maximum_values`; there is no fallback.
- **Meals.** 24 h from 2019-08-15 12:00. Per calendar day 2-4 meals between 05:00 and 22:00, clipped to the run, on the
  5-minute grid, 20-80 g, at least 60 min apart. With probability 0.25 a draw gets one seconds pair (second meal 5-55
  min after the first, counted in that day's total). Absorption time is drawn from {30, 180, 300} min and written to
  `duration`. The draw's meals are shared by both arms. All numbers are config values.
- **Announcement.** Announced cohorts: pump and patient-model carb entries identical, plus an
  `accept_recommendation` bolus at each meal time in the patient model. Unannounced cohort: carbs only in the patient
  model; no bolus entries anywhere.
- **Seeds.** `master_seed` is required. Each draw's seed is `sha256(master_seed | "draw" | cohort | patient_id)`, so
  adding a cohort changes no existing draw. Same config and seed give byte-identical outputs.

### Glucose seed selection rule

The seed is the **first run of 137 consecutive values with none missing** in the patient's `actual_blood_glucose` row,
read positionally, assigned in order to the 5-minute grid 2019-08-15 00:40 to 12:00. A blank or non-numeric cell is a
missing value. Values keep their source text (no rounding). In the current pool (887 files) the row has one blank leading
cell and 2880 values with no gaps, so the seed is the first 137 values, as before; the remaining values are not used.

### Raw file layout

Rows are `setting_name, settings, 0, 1, 2, ...`. Scalars may sit in `settings` or column `0`; schedules and series may
start in either. One rule covers settings rows: a row's values are its non-empty cells from `settings` onward, in order.
The glucose row is read positionally (above). Per-event rows (`carb_dates`, `actual_carbs`, `dose_*`) repeat once per
event and are concatenated; any other repeated row is an error.

### ISF used for cohorts

`isf` is the first (00:00) entry of the ISF schedule. Every file in the current pool has a single entry.

### Manifest columns

`file_name, patient_id, cohort, also_in_cohorts, pair_id, arm, insulin_model, meals_announced, age_band (<18, 18-59,
60+), age, isf, derived_seed, meal_schedule (JSON with absorption times), has_seconds_pair, controller_settings (JSON),
shortfall_note, default_suspend, max_bolus_mode (per_patient or static), tdd_estimate_u, max_bolus_u,
max_bolus_clamped, meal80_fits`. A replacement manifest adds `replaces_pair_id` and `replacement_reason`.
Join on `file_name`; the two arms of a pair share `pair_id`, `derived_seed`, `meal_schedule` and the max bolus.

## Change notes (TRSET-77, Bugfix, regression risk Medium)

**What changed.** The generator now follows the Data Analysis Plan: eight cohorts, E1-E6 eligibility with recorded
exclusions, a positional glucose seed, seeded cohort ordering, per-patient max bolus with settings flags, one
`TLR-000_<patient_id>` folder per patient, and replacement runs.

**Breaking change.** The config schema (required `guardrails` and `max_bolus` blocks; `controller.max_bolus` is
rejected; new `default_suspend_threshold`), the manifest schema (six new columns), the output layout (the
`TLR-<run_name>` folder of TRSET-79 is replaced by per-patient folders), the age bands (adult below 60, older adult 60
and over) and the draws (seeded ordering replaces `rng.choice`, so the same seed gives different patients than before)
all change. `RunResult` loses `scenario_dir` and gains `run_dir`, `exclusions` and `eligible_pool`. No analysis run
has been generated from the old layout, so there is nothing to migrate: regenerate with the new
`example_run_config.json`, and delete or rename any older output folder (the generator will not overwrite it).

**Validation.** On the real pool (887 files): 772 eligible, E1 89, E5 26, no other exclusions; cohort sizes 160
pediatric, 494 adult, 118 older adult, 57 `isf_high`, 98 `isf_low`; glucose seeds byte-identical to the previous rule for
every patient compared. The unit and integration tests below pass.

**Cautions.** Seeds are positional, so a pool with interior gaps will now exclude patients (E2) or move their seed to
a later run. `meal80_fits` compares the rounded-down max bolus, so a patient at the exact boundary off the 0.05 U grid
is flagged as not fitting. The TRSET-26 URAI subset and its reference hashes were captured from the old cohorts and
layout and must be regenerated.

**Rollback.** Revert the TRSET-77 commits; `generate.py` then writes the TRSET-79 layout again.

## Tests

- `tests/test_urai_rai_generator_unit.py`: synthetic records; cutoffs, layouts, meal and pair invariants, shortfall,
  CSV-to-JSON fidelity including multi-entry schedules.
- `tests/test_urai_rai_dap_reconciliation_unit.py` (TRSET-77): E1-E6, the positional seed rule, default suspend
  threshold, mismatch check, nothing written on failure, seeded ordering, max bolus and flags, layout, comparator and
  replacement runs.
- `tests/test_urai_rai_generator_integration.py`: real pool (`URAI_RAI_PATIENT_POOL`, fails if missing), real parser,
  real Swift controller. Run under the `tidepool-data-science-simulator` conda env.

## Lineage (replaces `projects/autobolus_settings_variability/`)

| Old script | Replacement |
|---|---|
| `preprocess_raw_to_scenario.py` | `patient_pool.py`. The shifted-column fix is now one token rule; no copies of the pool are written. |
| `extract_glucose_history.py` | Dropped. It renamed pool files in place and extracted rows the generator reads directly. |
| `trim_history_to_seed.py` | `scenario_builder.glucose_seed_rows`. Selection rule documented above; no longer rounds to integers. |
| `generate_meals.py` | `meals.py`. Meals are generated in memory per draw, not edited into the source CSVs, and the run length comes from config rather than another script's source text. |
| `icgm_to_scenario.py` | `scenario_builder.py` + `generate.py`. Target range now uses the correct fields. |
| `create_variance.py` | Dropped. Settings variance is not part of the URAI-RAI study. |
| `extended_swift_controller.py` | Dropped. Not needed: the generator uses the existing Swift controller with the existing base config. |
