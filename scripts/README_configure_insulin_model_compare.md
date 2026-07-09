# configure_insulin_model_compare

## Description
Copies a subset of TLR scenario directories from `loop_risk_v2_2_0_full` into three
insulin model comparison directories and modifies the JSON configs to create controlled
mismatches between the patient's true insulin physiology and the Loop controller's
internal insulin model. Also adds the two missing Fiasp metabolism profiles
(`resistant_fiasp_v1.json`, `sensitive_fiasp_v1.json`).

## Destination directories and their configurations

| Directory | Patient physiology (`patient_model.metabolism_settings`) | Loop model (`controller.settings.model`) |
|---|---|---|
| `ab_URAI_ptModel` | Fiasp | rapid_acting_adult (unchanged) |
| `ab_URAI_pump` | rapid_acting_adult (unchanged) | Fiasp |
| `URAI` | Fiasp | Fiasp |

`null` controllers (no-Loop simulations) are never modified.

## Usage

```bash
# Test mode — processes TLR-549 only
python scripts/configure_insulin_model_compare.py

# Full run — processes all 97 in-scope TLR base IDs (~150 directories)
python scripts/configure_insulin_model_compare.py --all
```

## Validation
All 24 unit tests pass (`pytest tests/test_configure_insulin_model_compare.py -v`).
Tests cover all five `patient_model.metabolism_settings` cases (known profile ref,
unknown profile ref, inline dict with/without `patient_insulin_type`, absent key)
and all four `controller` cases (null, absent, string ref, inline dict). Integration
tests validate the full file pipeline on real TLR-549 source files.

## Cautions / limitations
- **SWIFT_CONTROLLER_MODEL_NAME_MAP** in `scenario_json_parser_v2.py` currently maps
  only `rapid_acting_adult` → `"novolog"`. If the Swift controller path is used,
  confirm `"fiasp"` is a supported value before running full simulations.
- The script is **idempotent**: re-running overwrites existing files in the destination
  directories without prompting.
- Five TLR IDs in scope (`TLR-1145`, `TLR-564`, `TLR-667`, `TLR-697`, `TLR-716`) have
  no matching directory in `loop_risk_v2_2_0_full` and are silently skipped.
- Patient model metabolism refs that are not one of the four standard profiles
  (adolescent, median, resistant, sensitive) are resolved from the reusable filesystem
  and inlined with `patient_insulin_type` added. Verify these inlined results if
  non-standard profiles appear in the scenarios being processed.
