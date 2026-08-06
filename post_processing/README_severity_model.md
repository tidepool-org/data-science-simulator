# Severity Model — failure reporting and profile counting

## Description

`severity_model.py` computes the structured `SeverityAssessment` that both the RTF
renderer (`create_severity_summary.py`) and the GUI consume. **Bugfix (TRSET-28):**
four silent or ambiguous failure behaviors in this layer are corrected —

1. `build_assessment` returned a bare `None` for *both* an empty directory and
   real-but-malformed data, so no caller could tell "nothing ran here" from "the
   data is broken." There is now a typed `AssessmentOutcome` with
   `status ∈ {'ok', 'empty', 'malformed'}`.
2. Malformed profile data rendered to the reader as *absent* data: a corrupt CSV
   produced a regulatory document asserting the data was unavailable. `detect_outliers`
   now reports `'malformed_data'` separately from `'no_data'`, with its own text.
3. The aggregated-profile count could **overstate** what contributed — `count_profiles`
   counted files while `extract_metric_data` silently dropped malformed ones and
   averaged the remainder. The assessment now carries both N (present) and M (usable).
4. Five call sites used two different summary-CSV glob patterns, so a directory every
   downstream helper could read was reported unusable. One pattern, defined once.

Companion to [`README_create_severity_summary.md`](README_create_severity_summary.md)
(TRSET-27, the `process_results_directory` failure contract this builds on).

## Example usage

```python
from severity_model import build_assessment, build_assessment_result

# Typed: tells an empty directory from malformed data.
outcome = build_assessment_result(tlr_dir, timestamp)
if outcome.status == 'empty':
    ...                       # nothing ran here — a legitimate partial outcome
elif outcome.status == 'malformed':
    ...                       # data present and unusable — report outcome.detail
else:
    render(outcome.assessment)

# Unchanged contract, for consumers typed on Optional[SeverityAssessment]:
assessment = build_assessment(tlr_dir, timestamp)   # the assessment, or None
```

### What changed, per condition

| Condition | Before | Now |
|---|---|---|
| directory with no summary CSVs | `None` | `AssessmentOutcome(status='empty')` |
| CSVs present, none usable | **complete document, every metric `NA`/`0`** | `status='malformed'`, **no document** |
| no filename yields a `TLR-` ID | `None` (from `summary_files[0]` only) | `status='malformed'`, after trying every filename |
| CSVs match the loose pattern only | reported unusable | read, like every other helper |
| one CSV of several malformed | counted as an aggregated profile | excluded from M, discrepancy rendered |
| malformed CSV in outlier analysis | status `'no_data'` → "Data not available" | status `'malformed_data'` → its own text |
| optional column (`lbgi`/`dka_index`) absent | printed "CSV file malformed" | printed as "will report NA" (it is a designed degrade) |

### Rendered output

RTF output is **byte-identical on the clean path** (M == N, well-formed data),
verified against the pre-change module. Only degraded paths change text:

| | Before | Now |
|---|---|---|
| M == N | `2 virtual patient profiles aggregated for this summary.` | *unchanged* |
| 0 < M < N | `3 virtual patient profiles aggregated for this summary.` (named 3; 2 contributed) | `2 of 3 virtual patient profiles aggregated for this summary. 1 summary results file could not be read.` |
| malformed outlier input | `Data not available for outlier analysis.` | `Outlier analysis not performed: profile data is present but could not be read. Check data configuration.` |
| no profile complete across all three stages | `Data not available for outlier analysis.` | `Outlier analysis not performed: no profile has results for all three evaluation stages.` |
| M == 0, N > 0 | a full document of `NA`/`0` | no document; reported as malformed |

The malformed text deliberately does not name the offending file — the path stays in
the console diagnostic and in `outcome.detail`; a regulatory document should not
carry local filesystem paths.

## Validation

64 tests added (47 in `test_severity_model.py`, 17 in `test_create_severity_summary.py`):
the empty/malformed split and its `to_dict` round-trip, the `build_assessment`
wrapper contract, M vs. N including the every-required-column-is-load-bearing case,
the usable/unusable split, `get_profile_metrics`/`detect_outliers` statuses, glob
sharing plus a source-level DRY guard, simulation-ID fallback, and every rendered
string above pinned verbatim. Clean-path byte-identity was additionally verified
out-of-band by rendering the same fixture through the pre-change module and
`diff`ing (1846 bytes, identical).

A further 5 tests cover `gui_runner`'s migration (empty vs. malformed surfaced on
`RiskDirRunResult`, and positional construction still working); its eight
`build_assessment` monkeypatches now patch `build_assessment_result` and return a
real `AssessmentOutcome`, so a fake of the wrong shape cannot pass. The unpatched
path was additionally exercised out-of-band against real good/empty/malformed
directories through `gui_runner`'s own import.

Suites: `test_severity_model.py` + `test_create_severity_summary.py` +
`tests/test_gui_runner.py` — 170 passed. Full suite: 649 passed / 8 failed / 5
skipped, the 8 being pre-existing and unrelated (all in untracked local scratch
files: `test_consensus_risk_list.py`, `test_jira_risk_probabilities.py`,
`test_simulation_results_columns.py`) — verified identical on a stash of this change.

## Cautions / limitations

**Breaking changes — migration.** Two return contracts changed:

- `get_profile_metrics` now returns a `ProfileMetrics` object (`profiles`, `excluded`,
  `files_present`), not a bare `profile_data`/`None`.
- `render_outlier_results` takes two further optional arguments, `profile_count` and
  `usable_profile_count`. Both default to `None`, so an existing call renders as before.
- `detect_outliers`' status set gains `'malformed_data'` and `'incomplete_stages'`. A
  consumer branching on status must handle both, or they will fall through to whatever
  its `else` renders. The GUI reads `SeverityAssessment.outlier_status` and needs the
  new values in its own mapping; unmapped, those directories show the GUI's fallback
  text rather than the `'no_data'` text they used to show.

`build_assessment` is **unchanged** — still `Optional[SeverityAssessment]` — and
stays importable from both `severity_model` and `gui_runner`, so a consumer that
imports it keeps working.

**`gui_runner` is migrated to `build_assessment_result`.** `RiskDirRunResult` gains
two keyword-defaulted fields appended after the existing ones —
`assessment_status` (`'ok'` | `'empty'` | `'malformed'`) and `assessment_detail` —
so positional construction and existing field access are unaffected, and the GUI
can now say *why* a directory produced no assessment instead of only "no data".
`AssessmentOutcome` is re-exported from `gui_runner` for consumers that want to
branch on the outcome directly. A GUI showing a single "no data" state for
`assessment is None` still renders correctly; adopting `assessment_status` is
opt-in.

`SeverityAssessment.usable_profile_count` is additive and defaults to `None`
("not measured"), which renders as M == N — so an object built by an older
positional constructor is unaffected. `to_dict()` gains the key.

**A directory that renders today can stop rendering:** M == 0 with N > 0 is now
malformed rather than a document of `NA`/`0` values. This was approved deliberately
(TRSET-28 Decision A) as the point of the ticket, but it is the one change that
removes an output rather than correcting one.

Regression risk **Medium** — a cross-module interface (`severity_model` → renderer →
GUI) and rendered output — contained by clean-path byte-identity and by
`build_assessment` keeping its signature. A revert is a single-commit revert.
`create_severity_summary_ORIGINAL.py` retains the old behavior and is deliberately
untouched (legacy reference copy).

## Hyperglycemia zero case (separate commit)

`detect_outliers` computed its own TAR→score mapping, `hyper_score = 1 if tar < 12.0
else 2`, with no zero case — contradicting SOP intent (a score of 0 should mean the
index is truly 0) and disagreeing with `calculate_hyperglycemia_score`. It now calls
that function, so the module has one such mapping.

**The naive fix would have silently removed a detection.** With the corrected score,
`determine_harm_and_severity(0, 0, 0)` is baseline, so a profile with TAR == 0 and no
hypo/DKA risk leaves the `Hyperglycemia` harm group — and the zero-TAR outlier check
keyed on that group. A profile with 0% time-above-range among all-high-TAR peers, the
case most worth flagging, would have stopped being reported. Verified: 3 findings → 0.

The check therefore spans the `Hyperglycemia` and baseline groups together. That union
is provably the old `Hyperglycemia` group — baseline is exactly its `TAR == 0` subset
of `lbgi <= 1 and dka == 0` — so the compared population is unchanged. It is evaluated
at whichever of the two groups `harm_groups` reaches first, and built from
`profile_harms` order, so findings keep their emitted order too.

**Net effect: no rendered output changes.** The results table never used this mapping
(`build_assessment` already used `calculate_hyperglycemia_score` on averaged TAR); the
inline score fed only outlier grouping, and its `_severity` return was discarded.
Equivalence was verified by an exhaustive sweep of `detect_outliers` across the fix —
2, 3 and 4 profiles over a grid of TAR/LBGI/DKA values, comparing findings as ordered
tuples: **133,632 cases, 0 differences**. The same sweep reports 1,044 differences for
the naive fix, so it is not vacuous. 11 tests added, 3 of which fail against the naive
fix.

`BASELINE_HARM` names the baseline label, since `detect_outliers` now has to reason
about that group and a typo would silently empty the union.

## Incomplete stage coverage (separate commit)

`detect_outliers` returned `'no_data'` from two places, and the second was not
absence: profiles that parse fine but where **none carries results for all three
stages**. That is a run which produced only some stages — a real, recoverable
configuration problem — reported with the same sentence as a missing directory.

It now reports `'incomplete_stages'`, rendering *"Outlier analysis not performed: no
profile has results for all three evaluation stages."* The three degraded statuses are
now three distinct sentences, and only genuine absence keeps the original text.

Precedence, and why: `malformed_data` still outranks `incomplete_stages` — an
unreadable file is the more actionable fault and makes the rest unknowable. A
directory with *one* complete profile alongside incomplete ones is still
`single_profile`, since the gate is "none complete", not "any incomplete".

`'no_data'` now means exactly one thing: no per-profile results exist. That covers no
summary files at all, and an aggregate-only directory whose filenames identify no
profile (no `*_profile.csv`) — both genuine absence.

**Thin data is not malformed data.** Such a directory still renders a document, and
its files still count toward M, so finding 3's count line stays in its clean form and
absent stages render `NA` as before.

`render_outlier_results` falls through to the findings text for a status it does not
recognize, so a future status cannot silently render a degraded sentence.

## Excluding a bad profile instead of abandoning the analysis (separate commit)

`get_profile_metrics` returned on the **first** unreadable file, so one bad profile
cost the outlier analysis of every good one in the directory. This is the one change
in TRSET-28 that alters which outliers are **found** (Decision C).

An unreadable file is now excluded and named; the readable profiles are compared. A
real example, three good profiles and one bad:

| | Before | Now |
|---|---|---|
| Outlier Results | `Outlier analysis not performed: profile data is present but could not be read. Check data configuration.` | `Outlier profile exists. median has a Hypoglycemia score of 4 at Pre-mitigation, while other profiles have a Hypoglycemia score of 2. …` |

A genuine severity-4 hypoglycemia outlier was being suppressed by an unrelated
malformed file.

Because the findings are scoped to what survived, the statuses that **assert**
something about the profiles (`ok`, `single_profile`) carry a trailing sentence:
*"This analysis covered 3 of 4 profiles; 1 could not be read."* The count line above
already reports the drop, but the Outlier Results sentences are absolute claims read
and quoted on their own. The three statuses that say the analysis did not happen are
not scoped — that would only repeat the count line.

`get_profile_metrics` now returns a `ProfileMetrics` (`profiles` / `excluded` /
`files_present`) rather than a tuple; `files_present` is what distinguishes "nothing
there" from "everything there was excluded", which an empty `profiles` cannot. A
profile is published only once fully built, so a row-level failure part-way through
can no longer leave a half-populated profile behind — it previously did not matter
because any exception discarded the directory.

**Two precedences moved, deliberately:**

- `'malformed_data'` now means **every** per-profile file failed, not "at least one
  did". Reaching it needs a usable non-profile file to keep M ≥ 1 (otherwise
  `build_assessment_result` reports the directory malformed and writes no document).
- `incomplete_stages` now **outranks** an excluded file, reversing the earlier
  precedence. An unreadable file no longer poisons the directory, so what the reader
  needs is the state of the data that remains: readable, but stage-thin.

Equivalence on the clean path is unaffected: the `detect_outliers` sweep still
reports 133,632 cases / 0 differences against the pre-TRSET-28 module, and RTF output
is byte-identical when nothing is excluded.

### Deliberately out of scope

- `STAGE_PREFIXES` fragile prefix matching, noted in the module.
