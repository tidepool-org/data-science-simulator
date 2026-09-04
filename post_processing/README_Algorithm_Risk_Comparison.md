# README_Algorithm_Risk_Comparison.md

## Overview

This directory contains post-processing modules used to compare risk profiles between
Tidepool Loop algorithm versions, enrich output with Jira risk probability data, and
generate a hazardous situation comparison report for regulatory review.

| Module | Script | Purpose |
|--------|--------|---------|
| Algorithm Risk Comparison | `algorithm_risk_comparison.py` | Statistical analysis (TMP-0012) comparing LBGI, DKAI, and TAR% between two algorithm versions; includes hazardous situation comparison report (Module 9) |
| Jira Risk Probabilities | `jira_risk_probabilities.py` | Fetches initial and residual risk probability values from Jira; used by `--jira` and `--hazard-comparison` |

---

## 1. Algorithm Risk Comparison (`algorithm_risk_comparison.py`)

### Description

Implements the statistical analysis plan (TMP-0012) for comparing risk profiles
between two versions of the Tidepool Loop algorithm. Analyzes LBGI (hypoglycemia),
DKAI (DKA), and TAR% (time above 180 mg/dL) using paired statistical tests to
determine whether version 2.0 maintains non-inferiority to the previously cleared
version.

### Usage

```bash
python post_processing/algorithm_risk_comparison.py \
    --v1 /path/to/Risk_Results_previously_cleared.csv \
    --v2 /path/to/Risk_Results_v2.csv \
    --output /path/to/output_directory
```

#### With Jira probability enrichment (Table 4 Excel)

```bash
python post_processing/algorithm_risk_comparison.py \
    --v1 v1.csv --v2 v2.csv --output out/ \
    --jira

# Override Jira API version (try if getting 404s on issue fetches after auth passes)
python post_processing/algorithm_risk_comparison.py \
    --v1 v1.csv --v2 v2.csv --output out/ \
    --jira --jira-api-version 3
```

#### With hazardous situation comparison report

```bash
python post_processing/algorithm_risk_comparison.py \
    --v1 v1.csv --v2 v2.csv --output out/ \
    --hazard-comparison \
    --env path/to/.env
```

`--hazard-comparison` and `--jira` can be combined. Both require Jira credentials.

#### Programmatic usage

```python
from algorithm_risk_comparison import main

acceptance = main(
    filepath_v1='Risk_Results_v1.csv',
    filepath_v2='Risk_Results_v2.csv',
    output_dir='./analysis_output'
)

if acceptance['overall']['passed']:
    print("Algorithm comparison: ACCEPTABLE")
```

```python
# Hazardous situation comparison only
from post_processing.algorithm_risk_comparison import (
    run_question3_analysis, generate_hazardous_situation_comparison
)
q3 = run_question3_analysis(df_merged)
generate_hazardous_situation_comparison(q3, output_dir='/path/to/output')
```

### Output Files

| File | Content |
|------|---------|
| `table2_overall_risk_profile.csv` | Question 1 — overall statistical comparison |
| `table3_patient_profile_stratification.csv` | Question 2 — results by patient profile |
| `table4_individual_hazardous_situations.csv` | Question 3 — aggregated severity by TLR key and scenario type |
| `table4_with_jira_probabilities.xlsx` | Two-sheet Excel: Sheet 1 = Table 4, Sheet 2 = Jira Probabilities (`--jira` only) |
| `hazardous_situation_comparison.xlsx` | Six-sheet Excel: Summary + five comparison sheets (`--hazard-comparison` only) |
| `unmatched_scenarios.csv` | Scenarios without matching pairs, if any |
| `analysis_report.txt` | Complete text summary |
| `*.png` | Scatter, histogram, Bland-Altman, and boxplot visualizations per metric |

### Hazardous Situation Comparison Report (Module 9)

The `--hazard-comparison` flag generates a six-sheet Excel workbook
(`hazardous_situation_comparison.xlsx`) identifying TLR hazardous situations with
practically significant changes between algorithm versions. Risk scores
(Severity × Probability) are pulled from Jira, classified by acceptability category,
and deduplicated across five comparison sheets with colour-coded cells.

**Acceptability categories:**

| Risk Score | Severity | Category | Colour |
|---|---|---|---|
| 1–3 | any | Acceptable | Green |
| 4 | 1–3 | Acceptable | Green |
| 4 | ≥4 | Conditionally acceptable | Yellow |
| 5–9 | any | Conditionally acceptable | Yellow |
| ≥10 | any | Unacceptable | Red |

**Sheet layout:**

| Sheet | Contents |
|---|---|
| Summary | Overall severity delta; total scenarios decreased/increased; practical-difference counts per sheet |
| Changes to Harm | TLRs where harm type changed (Hyperglycemia ↔ DKA) |
| Severity Increased | TLRs with severity threshold crossing in the upward direction |
| Severity Decreased | TLRs with severity threshold crossing in the downward direction |
| Risk Score Increased | TLRs where acceptability category worsened (V1→V2) |
| Risk Score Decreased | TLRs where acceptability category improved (V1→V2) |

**Deduplication priority:** Changes to Harm > Risk Score Increased/Decreased > Severity Increased/Decreased.

**noLoop rows** only count toward risk-score flags when the TLR's Jira issue has the
`no_automation` component set.

### Validation

Unit tests cover: sim_id parsing across naming variations, pairwise matching logic
with detailed unmatched scenario identification, all statistical calculations
(range, mean, median, std, Wilcoxon, t-test, Cohen's d), severity/harm detection
functions, acceptance criteria evaluation, all five Module 9 flag types,
deduplication priority, all five acceptability buckets (including the score-4
boundary case), and Excel output structure (sheet names, empty-sheet message, cell
content, colour formatting).

```bash
pytest tests/test_algorithm_risk_comparison.py -v
```

### Cautions and Limitations

- **Probability scores unavailable without `--jira`**: Question 3 analysis excludes
  risk score category changes from the hazardous situation comparison unless Jira
  credentials are provided; affected rows show `N/A` risk scores.
- **Naming convention variations**: Script normalizes multiple sim_id formats;
  novel patterns may require updates.
- **Non-normal data assumption**: Primary analysis uses Wilcoxon (non-parametric);
  t-test is included for reference only.
- **Two-sided testing**: Current implementation uses two-sided tests; directional
  hypotheses in the plan may warrant one-sided alternatives.
- **Matched pairs required**: Unmatched scenarios are identified in detail and
  exported to CSV but excluded from statistical analysis.
- **Harm column**: On Severity/Risk Score sheets the `Harm` column always reflects
  the V2 harm type.
- **noLoop / no_automation**: The `No Automation` flag is read from Jira issue
  components; the component name must match `NO_AUTOMATION_COMPONENT = "no_automation"`
  in `jira_risk_probabilities.py`.
- **One row per (TLR Key, Evaluation Stage) pair**: A TLR appearing in multiple
  scenario types may appear on multiple sheets.

---

## 2. Jira Risk Probabilities (`jira_risk_probabilities.py`)

### Description

Fetches `boja_prop_issue.risk_probability_value` (initial probability) and
`boja_prop_issue.risk_residual_probability_value` (residual probability) from the
Jira REST API for each unique base TLR key in the Table 4 output.

Scenario-specific suffixes (e.g. `TLR-1117_bike`, `TLR-899_01_025`) are stripped
to the base Jira key before the lookup; one row is written per unique base key.
The module uses the dedicated property endpoint
(`/issue/{key}/properties/boja_prop_issue`) and defaults to API v2 to match the
`atlassian-python-api` library used elsewhere in Tidepool's tooling.

### One-time credential setup

```bash
cp .env.example .env
# Edit .env and fill in JIRA_BASE_URL, JIRA_USERNAME, JIRA_API_TOKEN
```

Generate an API token at: https://id.atlassian.com/manage-profile/security/api-tokens

Set these variables in `.env` (already gitignored) or as shell environment variables:

| Variable | Description |
|----------|-------------|
| `JIRA_BASE_URL` | `https://tidepool.atlassian.net` |
| `JIRA_USERNAME` | Your Jira account email |
| `JIRA_API_TOKEN` | Jira API token (not your login password) |

### Verify credentials before running

Always run this first to confirm auth is working before a bulk fetch:

```bash
python post_processing/jira_risk_probabilities.py --verify-only
# INFO: Verifying Jira connection: GET https://tidepool.atlassian.net/rest/api/2/myself
# INFO: Jira connection OK – authenticated as 'Shawn Foster' (shawn@tidepool.org)
# Jira connection verified successfully.
```

If this fails, the problem is credentials or base URL — not the TLR issue keys.

### Standalone usage

```bash
# From a Table 4 CSV
python post_processing/jira_risk_probabilities.py \
    --table4 post_processing/table4_individual_hazardous_situations.csv

# From explicit keys
python post_processing/jira_risk_probabilities.py \
    --keys TLR-552 TLR-1117_bike TLR-899_01_025

# Try API v3 if v2 gives 404s on properties
python post_processing/jira_risk_probabilities.py \
    --table4 table4.csv --api-version 3

# Save to CSV
python post_processing/jira_risk_probabilities.py \
    --table4 table4.csv --output probs.csv
```

### As a library

```python
from post_processing.jira_risk_probabilities import fetch_probabilities

df = fetch_probabilities(["TLR-552", "TLR-1117_bike", "TLR-899_01_025"])
# Returns DataFrame with columns:
#   TLR Key | Initial Probability | Residual Probability
# TLR-1117_bike resolves to a single "TLR-1117" row.
```

### Debugging 404 errors

404 responses on every issue almost always indicate an auth or URL problem, not
missing issues. Work through these steps in order:

1. **Run `--verify-only`** — confirms credentials are valid and the base URL is
   reachable. A 401/403/404 here means the credentials are wrong.
2. **Check `JIRA_BASE_URL`** — must be the bare instance root with no path
   component, e.g. `https://tidepool.atlassian.net` (not `.../rest/api/2`).
3. **Try `--api-version 3`** — the module defaults to v2 (matching the
   `atlassian-python-api` library), but some Jira Cloud configurations differ.
4. **Check API token scope** — the token must have read access to the TLR project.
   Confirm in Atlassian account settings.

### Dependencies

`requests` and `python-dotenv` are already in `conda-environment.yml`.

`openpyxl` is required for Excel output (`--jira` and `--hazard-comparison`) and
is **not** in the conda environment:

```bash
pip install openpyxl
```

### Validation

```bash
pytest post_processing/test_jira_risk_probabilities.py -v
```

All Jira HTTP calls are mocked; no live network access is required. Tests cover
key normalisation, property traversal, `verify_connection`, the property-specific
endpoint URL shape, retry logic, API version passthrough, and graceful handling of
missing values and failed auth.

### Cautions and Limitations

- **Run `--verify-only` first** when troubleshooting; it separates auth failures
  from property-not-found failures before making dozens of issue requests.
- **Rate limits**: The module retries on HTTP 429 with exponential back-off, but
  sustained bulk fetches against large TLR lists may still hit Atlassian Cloud
  rate limits.
- **Property availability**: `boja_prop_issue` properties must be set on each TLR
  issue in Jira. Issues without the property return `None` values for both
  probability columns.
- **API version**: Defaults to v2 to match the `atlassian-python-api` library.
  Pass `--api-version 3` (standalone) or `--jira-api-version 3` (main script) if
  v2 gives unexpected results after auth is confirmed working.
- **openpyxl**: Excel output requires `openpyxl`. The existing CSV outputs are
  unaffected if `openpyxl` is absent and neither `--jira` nor `--hazard-comparison`
  is used.
