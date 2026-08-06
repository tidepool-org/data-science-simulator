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

Suites: `test_severity_model.py` + `test_create_severity_summary.py` +
`tests/test_gui_runner.py` — 165 passed. Full suite: 644 passed / 8 failed / 5
skipped, the 8 being pre-existing and unrelated (all in untracked local scratch
files: `test_consensus_risk_list.py`, `test_jira_risk_probabilities.py`,
`test_simulation_results_columns.py`) — verified identical on a stash of this change.

## Cautions / limitations

**Breaking changes — migration.** Two return contracts changed:

- `get_profile_metrics` now returns `(profile_data, status)`, not a bare
  `profile_data`/`None`. Callers must unpack.
- `detect_outliers`' status set gains `'malformed_data'`. A consumer branching on
  status must handle it, or it will fall through to whatever its `else` renders.
  The GUI reads `SeverityAssessment.outlier_status` and needs the new value in its
  own mapping; unmapped, a malformed directory shows the GUI's fallback text rather
  than the `'no_data'` text it used to show.

`build_assessment` is **unchanged** — still `Optional[SeverityAssessment]` — so
`gui_runner.RiskDirRunResult` and the GUI repo that reads it need no change. That
also means `gui_runner` still cannot tell empty from malformed; migrating it to
`build_assessment_result` is a follow-up (it requires updating the eight
`build_assessment` monkeypatches in `tests/test_gui_runner.py`).

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

### Deliberately out of scope

- The `detect_outliers` hyperglycemia deviation (`hyper_score = 1 if tar < 12.0 else 2`,
  no zero case) — preserved exactly; a separate correctness decision.
- Splitting the third condition still inside `'no_data'`: profiles that parse but
  where none is complete across all three stages (TRSET-28 Decision B).
- Analyzing the M usable profiles instead of abandoning outlier analysis on the
  first malformed file — that changes which outliers are *found* (TRSET-28 Decision C).
- `STAGE_PREFIXES` fragile prefix matching, noted in the module.
