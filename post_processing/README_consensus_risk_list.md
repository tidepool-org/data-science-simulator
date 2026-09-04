# Consensus Risk List Generator

## Description

Generates the consensus approval list (Excel) for the TWI-0025 Clinical and
Senior Management Risk Review. Queries Jira for TLR Risk issues in the given
fix version whose post-mitigation probability = 1 (Negligible), excluding
specified components, and writes a single-sheet workbook with hyperlinked TLR
keys, pre- and post-mitigation risk scores (severity × probability) colour-coded
by acceptability, and Risk Control Measure titles resolved from the Jira field's
comma-separated reference IDs via the RCM Map document. Reuses
`jira_risk_probabilities` and `algorithm_risk_comparison.get_acceptability`.

## Example usage

```bash
# .env contains JIRA_BASE_URL, JIRA_USERNAME, JIRA_API_TOKEN
python consensus_risk_list.py --output consensus_list.xlsx

# Different fix version, pointing at an explicit .env file
python consensus_risk_list.py --output out.xlsx \
    --env ~/.tidepool_jira.env \
    --fix-version "Tidepool Loop 2.1"

# Override exclusion set
python consensus_risk_list.py --output out.xlsx \
    --exclude-components not_Apex ExcludeFromReport backend
```

Programmatic use:
```python
from consensus_risk_list import generate_consensus_list
generate_consensus_list("consensus_list.xlsx", env_path=".env")
```

## Validation

50 unit tests cover RCM parsing (comma splitting, whitespace, case, unknown
IDs), JQL construction, Jira field unwrapping, severity×probability scoring,
full row assembly at the `score=4` acceptability boundary, paginated `/search`
including 429-retry, and end-to-end workbook generation with mocked HTTP. Run
with `pytest test_consensus_risk_list.py -v`. Acceptability labels and fill
colours are delegated to `algorithm_risk_comparison.get_acceptability`, which
is itself covered by the hazardous situation comparison test suite.

## Cautions and limitations

- **Search uses `/rest/api/3/search/jql`** (cursor-paginated, no `total`),
  per Atlassian's post-CHANGE-2046 migration. The `--api-version` flag still
  controls the property-fetch endpoint but has no effect on search.
- **RCM titles are hardcoded** from the "Risk Control Measures Map, Tidepool
  Loop 2.0" document. Any new RCM added to the RCM Map must also be added to
  the `RCM_TITLES` dict; unknown IDs surface as `UNKNOWN (xyz)` with a log
  warning rather than silent omission.
- **Field IDs may drift** if Jira custom fields are recreated. Re-verify
  `FIELD_HARM`, `FIELD_HAZARD_CATEGORY`, and `FIELD_RCM` with
  `jira_risk_probabilities.py --list-fields TLR-XXX` if counts look wrong.
- **Issues missing `boja_prop_issue`** produce rows with empty
  severity/probability cells and "N/A" risk-score cells (white fill) rather
  than failing — inspect logs for `boja_prop_issue missing` warnings.
- **Component exclusion uses `NOT IN`**, so issues with *both* an allowed and
  an excluded component (e.g. `[backend, frontend]`) are excluded. This
  matches Jira's documented multi-value NOT IN semantics.
- **One property fetch per issue**: for large result sets this is the dominant
  cost. Consider parallelising `fetch_issue_property` if throughput becomes a
  problem.

## Commit message

```
Add consensus_risk_list.py: Jira → Excel consensus approval list for TWI-0025 Clinical Review (RCM titles, colour-coded risk scores).
```
