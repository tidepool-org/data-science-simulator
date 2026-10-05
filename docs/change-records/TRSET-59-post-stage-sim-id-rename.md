# TRSET-59: change record for `84329fd` (stage-ID rename) and `e378b9d` (profile-name parse)

Recorded 2026-10-05. Both commits are already on `origin/main` (PR #65).

## 1. `84329fd` inventory (AC 1)
- 13,591 files changed, 13,591 insertions / 13,591 deletions, all under
  `scenario_configs/tidepool_risk_v2/loop_risk_v2_0/`, across 18 directories.
  **The commit was intended for `loop_risk_v2_2_0_full` only; it touched all of them.**
- `loop_risk_v2_2_0_full`: 862 files. `loop_risk_v2_510k`: 1,243. Others: mid-isf 1,294,
  removeChild 1,279, fiasp 1,270, autobolus_0_4/0_5/0_6 ~870 each, 510k_comp 670,
  1-2_comparison_base 557, insulin_parameters_exploratory 463, exploratory 661,
  swift_510k_t1only 868, nonlinear_t1 868, mid-isf_t1Only 755, autobolus_sp(_0_5) 95 each, test 5.
- Rewritten: ~13.4k `post-Loop-WithMitigations_`, 214 `post-Loop-withMitigations_`, 13
  `post-Loop-WithMitigations_adolescent_`, and 11 bare IDs -> `post-Loop_WithMitigations_*`.

## 2. Git state (AC 2)
`git merge-base --is-ancestor 84329fd origin/main` is true. `origin/sf/outlier-profile-name-json-suffix`
also contains it. The commits shipped together via PR #65; the "not pushed" recollection was wrong.

## 3. Equivalence (AC 6)
Every changed line in the commit is a `"sim_id"` line; no other field changed in any file.

## 4. Citation (AC 4)
No 510(k)-adjacent record cites `loop_risk_v2_2_0_full` or `loop_risk_v2_510k` (Shawn, 2026-10-05).
Results already produced under old names are unaffected; they are not renamed.

## 5. Consumers (AC 5)
- `post_processing/severity_model.py` `STAGE_PREFIXES['post']` accepts old and new spellings.
- `post_processing/algorithm_risk_comparison.py` `parse_sim_id`: identical output for both spellings.
- GUI `meal_config.py:218` already emits `post-Loop_WithMitigations_t1_`; some GUI tests still
  use the hyphen spelling as an opaque key (not run here).
- `diagramgen`: no reference to the spelling.
- `scripts/standardize_sim_ids.py` treated the hyphen form as canonical and would have reverted
  the rename. **Fixed in this change set**: canonical is now `post-Loop_WithMitigations_`.
- `compare_insulin_delivery` lives in the separate `utility_scripts` repo and joins on exact,
  case-sensitive `sim_id`. Runs from before and after the rename do not pair by name; it
  reports them as suspected drift (`drift_key` folds case and `-`/`_`). **The mixed-spelling
  fixture belongs in that repo and is not part of this change set.**

## 6. Drift guard (AC 7)
`tests/test_post_stage_sim_id_spelling.py`: the retired `post-Loop-WithMitigations` fails in
every collection. In `loop_risk_v2_2_0_full`, ten named TLR directories keep other spellings
(`post-LoopWithMitigations_`, `post-Loop_withMitigations_`, `post_Loop_WithMitigations_`) and are
allow-listed; a new one fails, and fixing one forces its removal from the list. The variants
exist across the other 17 collections too; normalising them is a separate sim_id change.

## 7. Rollback (AC 8)
- Revert `84329fd` alone: restores old names across all 18 directories (13.6k files) and nothing
  else. Runs from 2026-10-02 onward carry the new names and would no longer match.
- Revert `e378b9d` alone: `*_profile.json.csv` summaries return to outlier `no_data`.

## 8. `e378b9d` (AC 9, 10)
Touched `severity_model.py::extract_profile_from_filename` (+8/-1) and added a parametrized test
in `test_severity_model.py` (+13). Added here: directory-level `outlier_status != no_data` for
both suffixes, and a mutation test using the pre-fix parse.

## 9. Output effect and exposure (AC 11, 12)
- TRSET-42 harness: TLR-999 and TLR-998 byte-identical to goldens.
- Real runs in `~/data/simulator/results/tidepool_loop_risk_v2_0` (2026-10-02: 108 summaries;
  2026-10-05: 4): 26 TLR directories rendered with the pre- and post-fix parse; status `ok` and
  RTF byte-identical in all. None ends in `_profile.json.csv`: all 112 end in `_profile_v1.json.csv`,
  where the old parse already found the profile. No stored record in this sample was affected.
  Not surveyed: results held elsewhere, and any run named exactly `*_profile.json.csv`.
  Scoping input for TRSET-60.
