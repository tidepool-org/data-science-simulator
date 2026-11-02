# URAI-RAI scenario generator (TRSET-74)

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

- `scenario_configs/tidepool_risk_v2/loop_risk_v2_0/loop_risk_v2_urai_rai/<run_name>/` holds `manifest.csv`,
  `run_config.json` (a copy of the config) and `TLR-<run_name>/` with one JSON per scenario. Point
  `TIDEPOOL_RISK_SCENARIOS_DIR` in `projects/risk/loop_risk_v2_0.py` at the `<run_name>` folder; the runner only reads
  JSONs inside `TLR-*` subfolders, so the results are written under `TLR-<run_name>`.
- `scenario_configs/tidepool_risk_v2/reusable/glucose/urai_rai/<run_name>/<patient_id>.csv` holds the glucose seeds.

## Design

- **Pool.** Read only; never written. Patients under `min_age` (default 6) are dropped before any cohort filter.
- **Cohorts.** Named filter + n (default 30), each filtered from the full pool and sampled independently, so a patient
  may be in several. Filters: `age_min` (>=), `age_max` (<), `isf_min` (>=), `isf_max` (<=). Fewer eligible than n is
  an error unless `accept_smaller_group` is true; accepted shortfalls are logged and written to the manifest.
- **Pairing.** Each draw yields a URAI and a RAI scenario identical except `controller.settings.model` and
  `patient_model.metabolism_settings.patient_insulin_type` (`fiasp` / `rapid_acting_adult`, flipped together).
- **Controller.** Swift, autobolus only, from `base_median_autobolus_v1_2` / `autobolus_v1_2`; partial application
  factor 0.4 and max bolus 20 (config); `suspend_threshold` from the patient file.
- **Settings.** Basal, CIR and ISF come from the patient file for pump and patient model (no variance). The loader
  fails if a patient's pump and `actual_*` values differ. Target range comes from `target_range_minimum_values` /
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

The seed is the **first 137 values** of the patient's `actual_blood_glucose` row, in column order, assigned in order to
the 5-minute grid 2019-08-15 00:40 to 12:00. Values keep their source text (no rounding). The source row has 2880 values;
the remaining values are not used.

### Raw file layout

Rows are `setting_name, settings, 0, 1, 2, ...`. Scalars may sit in `settings` or column `0`; schedules and series may
start in either. One rule covers both: a row's values are its non-empty cells from `settings` onward, in order.

### ISF used for cohorts

`isf` is the first (00:00) entry of the ISF schedule. Every file in the current pool has a single entry.

### Manifest columns

`file_name, patient_id, cohort, also_in_cohorts, pair_id, arm, insulin_model, meals_announced, age_band, age, isf,
derived_seed, meal_schedule (JSON with absorption times), has_seconds_pair, controller_settings (JSON), shortfall_note`.
Join on `file_name`; the two arms of a pair share `pair_id`, `derived_seed` and `meal_schedule`.

## Tests

- `tests/test_urai_rai_generator_unit.py`: synthetic records; cutoffs, layouts, meal and pair invariants, shortfall,
  CSV-to-JSON fidelity including multi-entry schedules.
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
