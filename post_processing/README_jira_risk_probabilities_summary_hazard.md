# jira_risk_probabilities: Summary, Hazard Category & No Automation fields

## Change description
Extended `fetch_issue_fields()` to also request the `components` field from the Jira issue endpoint. A new `no_automation` boolean key is returned — `True` if any component name in the issue's component list equals `no_automation`, `False` otherwise. The `No Automation` column is added to the output DataFrame between `Hazard Category` and `Initial Probability`.

## Usage

```python
from jira_risk_probabilities import fetch_probabilities
df = fetch_probabilities(["TLR-552", "TLR-899_01_025"])
# df columns: TLR Key | Summary | Hazard Category | No Automation | Initial Probability | Residual Probability
print(df[["TLR Key", "No Automation"]])
```

```bash
# CLI — output CSV now includes No Automation column
python jira_risk_probabilities.py --table4 table4.csv --output probs.csv
```

## Validation
Tests added in `TestFetchIssueFields`: `test_no_automation_component_present_returns_true`, `test_no_automation_component_absent_returns_false`, `test_empty_components_list_returns_false`, `test_null_components_field_returns_false`. `TestFetchProbabilitiesNewColumns` updated: `test_no_automation_false_when_component_absent` added; `test_summary_and_hazard_category_values_populated` and `test_missing_issue_fields_yields_none_columns` extended to assert `No Automation` values.

## Cautions / limitations
- The `components` field is fetched in the same request as `summary` and `Hazard Category` — no additional API call per issue.
- `no_automation` defaults to `False` (not `None`) on HTTP/network failure, consistent with a boolean sentinel.
- The check is case-sensitive: the component name must be exactly `no_automation` (see `NO_AUTOMATION_COMPONENT` constant).
- Issues with a `null` components field (Jira may omit it when empty) are handled safely and return `False`.
