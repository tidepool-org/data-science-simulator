# TRSET-40: Physical Activity Metabolism Parameters Fix

## Description

Physical activity (PA) configured in a scenario via a `reusable.` profile reference had **no
effect on glucose**: the run completed, the config looked correct, no warning was emitted, and
the output was indistinguishable from a run with no activity at all. Ten TLRs across five
scenario directories were affected (TLR-549, 1116-1121, 1151, LOOP-5976, REG-449).

Root cause: generic pointer resolution (`resolve_pointers`) consumed `physical_activity_entries`
before the PA-aware code that interprets it ever ran, in three different ways depending on the
reference form used. The fix makes `physical_activity_entries` own its own pointer resolution
end-to-end, and removes the unreachable-but-silent default fallback for the four metabolism
parameters (`w_hr`, `a`, `tau`, `n`).

## What Changed

`tidepool_data_science_simulator/makedata/scenario_json_parser_v2.py`:

1. **`resolve_pointers`** no longer touches the `physical_activity_entries` key at all (bare
   string, list-wrapped, or a list of entry dicts with `activity_ref`). Previously it special-cased
   only the bare-string form, and even then discarded the profile's sibling
   `metabolism_parameters` block, keeping only the entries list. Now the PA-specific pipeline in
   `build_model_from_config` (`extract_metabolism_params_from_pa_profiles`,
   `process_pa_entries_with_validation`, `resolve_pa_activity_ref`) resolves every reference form
   itself, exactly as it was already written to do -- it just never used to see the raw pointer.
2. **`resolve_pa_metabolism_params`** (new) is the single place the four metabolism parameters are
   resolved, using the documented precedence: explicit `model_config` > explicit
   `metabolism_settings` > PA profile parameters > default. The default tier (`w_hr=0.0, a=1.0,
   tau=60.0, n=1.0`) is now only used when there are **no** PA entries configured (inert values,
   since there's no activity to model). If PA entries are present and no tier resolves a value, it
   **raises**, naming the entries and the missing parameters, instead of silently defaulting to
   `w_hr=0.0` (which zeroed the exercise effect).
3. **`extract_metabolism_params_from_pa_profiles`** now also checks per-entry `activity_ref`
   pointers (not just bare-string/list-wrapped profile references), and no longer swallows
   non-`ValueError` exceptions when loading a referenced profile -- any failure raises, naming the
   profile.
4. **`get_patient_config`** reads `w_hr`/`a`/`tau`/`n` directly from `self.patient_model` (no `.get`
   default). `build_model_from_config` always sets these four keys (or raises), so this is now
   provably the only place the parameters are resolved.

## JSON Configuration Usage

All three reference forms below resolve to the same PA timeline and the same four metabolism
parameters.

### Bare string (the form used by all ten affected TLRs)
```json
"physical_activity_entries": "reusable.physical_activities.profiles.jogging_v1"
```

### List-wrapped
```json
"physical_activity_entries": ["reusable.physical_activities.profiles.jogging_v1"]
```

### Per-entry `activity_ref`
```json
"physical_activity_entries": [
  {"start_time": "8/15/2019 13:00:00", "activity_ref": "reusable.physical_activities.profiles.jogging_v1"}
]
```

### Referenced profile (unchanged shape)
```json
{
  "metabolism_parameters": {"w_hr": 1.0, "a": -0.002462, "tau": 0.9989, "n": 28},
  "physical_activity_entries": [
    {"start_time": "8/15/2019 13:00:00", "activity": "running", "duration": 30,
     "intensity": "moderate", "expected_hr": 152}
  ]
}
```

## Parameter Precedence

| Tier | Source | Reachable when |
|------|--------|-----------------|
| 1 | `model_config.w_hr` / `.a` / `.tau` / `.n` (top-level of `patient_model` or `pump`) | Explicitly set |
| 2 | `metabolism_settings.w_hr` / `.a` / `.tau` / `.n` | Explicitly set |
| 3 | Referenced PA profile's `metabolism_parameters` | A `reusable.` profile reference resolves to a profile carrying `metabolism_parameters` |
| 4 (default) | `w_hr=0.0, a=1.0, tau=60.0, n=1.0` | **Only** when `physical_activity_entries` is empty |

If PA entries are present and none of tiers 1-3 resolve a parameter, `build_model_from_config`
raises a `ValueError` naming the entries and the missing parameter(s).

## Behavior Change -- Expected, Not a Regression

**Any scenario that references a PA profile via a `reusable.` pointer will now produce a
different (correct) glucose trace**, because the profile's `w_hr`/`a`/`tau`/`n` now actually reach
the metabolism model instead of being silently replaced by inert defaults. This is the intended
fix, not a regression. Scenarios with `physical_activity_entries: []` (no activity configured) are
byte-identical to the pre-fix build.

The ten previously-affected TLRs (TLR-549, 1116-1121, 1151, LOOP-5976, REG-449) will need to be
re-run and their severity records revised under a follow-on ticket (not yet filed; must not start
until this fix is verified). Hypothesis for that re-run, not a finding from this change: with the
activity effect previously zeroed, those runs modeled only the risk-mitigation settings (reduced
basal, raised ISF/carb ratio, raised correction range) with none of the glucose-lowering exercise
effect those settings exist to offset -- so recorded severities may be **understated**.

## Regression Risk

**Medium-High.** `resolve_pointers` is the shared generic pointer-resolution routine used by every
section of every scenario config. The fix adds a single, narrowly-scoped early-exit
(`if k == "physical_activity_entries": continue`) rather than changing `is_config_file_pointer` or
any other generic resolution behavior, so no config section other than
`physical_activity_entries` is affected. AC 8 (byte-identical output for
`physical_activity_entries: []`) is the regression guard for that claim and is covered by the
existing `tests/test_noisy_sensor_override.py`-style parser tests plus the new
`tests/test_pa_metabolism_parameter_resolution.py`.

### Rollback

Revert the changes to `tidepool_data_science_simulator/makedata/scenario_json_parser_v2.py` in
this change (the `resolve_pointers` PA skip, `resolve_pa_metabolism_params`, the
`extract_metabolism_params_from_pa_profiles` and `get_patient_config` changes). No config file
format or schema changes accompany this fix, so a revert requires no scenario config changes.

## Validation

- All ten affected TLRs use the bare-string reference form; regression proof
  (`tests/test_pa_metabolism_parameter_resolution.py::TestTLR000PARegressionProof`) runs the real
  `loop_risk_v2_0/test/TLR-000-pa/` "Activity preset with exercise" and "Activity preset no
  exercise" configs and asserts their glucose traces diverge materially once the activity begins.
- `intensity` remains accepted-and-validated but unused in the simulation -- no behavior change.

## Cautions and Limitations

1. **`activity_ref` is untested in production configs.** `grep -rn '"activity_ref"'
   scenario_configs/` returns no matches -- no shipped config currently uses this form. It is
   covered by unit tests (not a real scenario config) since AC 1 requires parity across all three
   reference forms.
2. **The diagnostic override printer can flag a false `TYPE_MISMATCH`** for
   `physical_activity_entries` when a bare-string pointer overrides an empty-list base value (the
   print-only `diagnose_override_application` compares raw JSON types before pointer resolution).
   This is pre-existing, cosmetic-only behavior, unrelated to this fix -- `resolve_override` still
   applies the override correctly.
3. **Validator PA coverage is unchanged.** `PatientModelConfig.physical_activity_entries` is still
   typed `List[Any]`, so `validate_configs.py` cannot catch malformed PA entries ahead of time
   (tracked separately, out of scope here).

## Files Modified

| File | Changes |
|------|---------|
| `tidepool_data_science_simulator/makedata/scenario_json_parser_v2.py` | `resolve_pointers` skips `physical_activity_entries`; added `resolve_pa_metabolism_params` and `PA_METABOLISM_PARAM_DEFAULTS`; `extract_metabolism_params_from_pa_profiles` handles `activity_ref` and no longer swallows exceptions; `get_patient_config` reads parameters without independent defaults |
| `tests/test_pa_metabolism_parameter_resolution.py` | New parser-path unit tests for AC 1, 2, 4, 5, 6, and a scenario-level regression test for AC 7 |

## Commit Message
```
Fix PA metabolism parameters silently dropped during reusable-reference resolution (TRSET-40)
```
