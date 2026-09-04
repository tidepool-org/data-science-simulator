"""
Unit tests for consensus_risk_list.py

Tests cover:
  - parse_rcm_field() -- comma splitting, whitespace, uppercasing, None/non-str inputs
  - lookup_rcm_titles() -- known IDs, unknown IDs, empty input
  - build_consensus_jql() -- default and customised filters, JQL escaping
  - _extract_string_field() -- bare string, select dict, name dict, None
  - _compute_risk_score() -- normal product, None propagation
  - _as_int_or_none() -- int coercion, None pass-through
  - build_consensus_row() -- full property block, missing property, missing fields,
    colour-coding via get_acceptability
  - search_issues() -- single page, pagination across pages, 429 retry, empty result
  - generate_consensus_list() -- end-to-end with mocked HTTP and file output
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

import consensus_risk_list as crl
from consensus_risk_list import (
    DEFAULT_EXCLUDE_COMPONENTS,
    DEFAULT_EXCLUDE_STATUS,
    DEFAULT_FIX_VERSION,
    FIELD_HARM,
    FIELD_HAZARD_CATEGORY,
    FIELD_RCM,
    FIELD_STATUS,
    FIELD_SUMMARY,
    RCM_TITLES,
    _as_int_or_none,
    _compute_risk_score,
    _extract_string_field,
    build_consensus_jql,
    build_consensus_row,
    generate_consensus_list,
    lookup_rcm_titles,
    parse_rcm_field,
    search_issues,
    write_consensus_excel,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

BASE_URL = "https://tidepool.atlassian.net"
AUTH = ("user@example.com", "token123")
ENV_VARS = {
    "JIRA_BASE_URL": BASE_URL,
    "JIRA_USERNAME": "user@example.com",
    "JIRA_API_TOKEN": "token123",
}


def _mock_response(status_code, json_body=None, headers=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body or {}
    mock_resp.headers = headers or {}
    mock_resp.text = str(json_body or "")
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _fake_issue(key="TLR-100", summary="Test risk", status="Active",
                harm="Severe hypoglycemia", hazard="Dosing",
                rcm="A, CE, D", components=None):
    fields = {
        FIELD_SUMMARY: summary,
        FIELD_STATUS: {"name": status},  # Jira wraps status in {"name": "..."}
        FIELD_HARM: harm,
        FIELD_HAZARD_CATEGORY: hazard,
        FIELD_RCM: rcm,
        "components": components or [],
    }
    return {"key": key, "fields": fields}


def _fake_prop_block(pre_sev=3, pre_prob=4, post_sev=3, post_prob=1):
    return {
        crl.PROP_PRE_SEVERITY: str(pre_sev) if pre_sev is not None else None,
        crl.PROP_PRE_PROBABILITY: str(pre_prob) if pre_prob is not None else None,
        crl.PROP_POST_SEVERITY: str(post_sev) if post_sev is not None else None,
        crl.PROP_POST_PROBABILITY: str(post_prob) if post_prob is not None else None,
    }


# ─── parse_rcm_field ─────────────────────────────────────────────────────────

class TestParseRcmField:
    def test_single_id(self):
        assert parse_rcm_field("A") == ["A"]

    def test_comma_separated(self):
        assert parse_rcm_field("A,CE,D") == ["A", "CE", "D"]

    def test_whitespace_stripped(self):
        assert parse_rcm_field(" A ,  CE , D ") == ["A", "CE", "D"]

    def test_uppercases_lowercase(self):
        assert parse_rcm_field("a, ce, d") == ["A", "CE", "D"]

    def test_drops_empty_tokens(self):
        assert parse_rcm_field("A,,CE,") == ["A", "CE"]

    def test_none_returns_empty(self):
        assert parse_rcm_field(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_rcm_field("") == []

    def test_only_whitespace_returns_empty(self):
        assert parse_rcm_field("   , ,  ") == []

    def test_non_string_coerced(self):
        # int coerces to "123" which has no commas -> single token
        assert parse_rcm_field(123) == ["123"]

    def test_preserves_multi_char_ids(self):
        assert parse_rcm_field("LNL, DEL, GLU") == ["LNL", "DEL", "GLU"]


# ─── lookup_rcm_titles ───────────────────────────────────────────────────────

class TestLookupRcmTitles:
    def test_all_known_ids(self):
        result = lookup_rcm_titles(["A", "CE", "D"])
        assert result == [
            "Algorithm limits",
            "CGM error states have alerts and alarms",
            "Design",
        ]

    def test_unknown_id_returned_as_unknown(self):
        result = lookup_rcm_titles(["A", "XX"])
        assert result[0] == "Algorithm limits"
        assert result[1] == "UNKNOWN (XX)"

    def test_empty_input(self):
        assert lookup_rcm_titles([]) == []

    def test_order_preserved(self):
        result = lookup_rcm_titles(["GLU", "A", "HF"])
        assert result[0] == RCM_TITLES["GLU"]
        assert result[1] == RCM_TITLES["A"]
        assert result[2] == RCM_TITLES["HF"]

    def test_all_pdf_ids_present(self):
        # Ensure every ID documented in the PDF is mapped
        expected_ids = {
            "A", "AC", "CE", "D", "DEL", "G", "GLU", "HF", "IA", "IFU",
            "LNL", "PE", "PM", "PV", "Q", "REV", "RX", "SB", "SEC", "UI", "UP",
        }
        assert expected_ids.issubset(RCM_TITLES.keys())


# ─── build_consensus_jql ─────────────────────────────────────────────────────

class TestBuildConsensusJql:
    def test_default_jql_contains_expected_clauses(self):
        jql = build_consensus_jql()
        assert "project = TLR" in jql
        assert "issuetype = Risk" in jql
        assert 'fixVersion = "Tidepool Loop 2.0"' in jql
        assert 'risk_residual_probability_value' in jql
        assert '= "1"' in jql
        # Component clause MUST include the IS EMPTY branch so that unclassified
        # risks are not silently dropped by Jira's NOT IN semantics.
        assert "component IS EMPTY OR component NOT IN" in jql
        assert '"not_Apex"' in jql
        assert '"ExcludeFromReport"' in jql
        assert '"backend"' in jql
        # Status clause must appear and quote each value individually.
        assert 'status NOT IN' in jql
        assert '"Closed"' in jql
        assert '"Deprecated"' in jql
        assert "ORDER BY issuekey ASC" in jql

    def test_custom_fix_version(self):
        jql = build_consensus_jql(fix_version="Tidepool Loop 2.1")
        assert 'fixVersion = "Tidepool Loop 2.1"' in jql

    def test_custom_exclude_components(self):
        jql = build_consensus_jql(exclude_components=("foo", "bar"))
        assert '"foo", "bar"' in jql
        assert "not_Apex" not in jql

    def test_custom_exclude_status(self):
        jql = build_consensus_jql(exclude_status=("Done", "Archived"))
        assert 'status NOT IN ("Done", "Archived")' in jql
        assert "Closed" not in jql

    def test_empty_exclude_status_still_renders(self):
        # Edge case: empty tuple produces 'status NOT IN ()' which is
        # syntactically valid JQL and matches everything.  Documented here
        # so the behaviour is intentional if callers pass ().
        jql = build_consensus_jql(exclude_status=())
        assert "status NOT IN ()" in jql

    def test_custom_residual_probability(self):
        jql = build_consensus_jql(residual_probability="2")
        assert '= "2"' in jql

    def test_single_excluded_component(self):
        jql = build_consensus_jql(exclude_components=("only_one",))
        assert '"only_one"' in jql


class TestDefaultTupleShapes:
    """Regression tests: DEFAULT_EXCLUDE_COMPONENTS and DEFAULT_EXCLUDE_STATUS
    must be flat tuples of strings, not tuples-of-tuples. A stray trailing
    comma after the closing paren silently wraps the tuple in a 1-tuple
    containing a tuple, which produces a broken JQL clause."""

    def test_exclude_components_is_flat_string_tuple(self):
        assert len(DEFAULT_EXCLUDE_COMPONENTS) == 3
        for item in DEFAULT_EXCLUDE_COMPONENTS:
            assert isinstance(item, str), (
                f"DEFAULT_EXCLUDE_COMPONENTS must contain plain strings; "
                f"got {type(item).__name__}: {item!r}"
            )

    def test_exclude_status_is_flat_string_tuple(self):
        assert len(DEFAULT_EXCLUDE_STATUS) == 2
        for item in DEFAULT_EXCLUDE_STATUS:
            assert isinstance(item, str), (
                f"DEFAULT_EXCLUDE_STATUS must contain plain strings; "
                f"got {type(item).__name__}: {item!r}"
            )


# ─── _extract_string_field ───────────────────────────────────────────────────

class TestExtractStringField:
    def test_bare_string(self):
        assert _extract_string_field("hello") == "hello"

    def test_select_option_dict(self):
        assert _extract_string_field({"value": "Dosing", "id": "10001"}) == "Dosing"

    def test_name_dict(self):
        assert _extract_string_field({"name": "Closed"}) == "Closed"

    def test_value_takes_precedence_over_name(self):
        assert _extract_string_field({"value": "V", "name": "N"}) == "V"

    def test_none_returns_none(self):
        assert _extract_string_field(None) is None

    def test_number_coerced(self):
        assert _extract_string_field(42) == "42"


# ─── _compute_risk_score ─────────────────────────────────────────────────────

class TestComputeRiskScore:
    def test_normal_product(self):
        assert _compute_risk_score(3, 4) == 12.0

    def test_zero_severity(self):
        assert _compute_risk_score(0, 4) == 0.0

    def test_none_severity_returns_none(self):
        assert _compute_risk_score(None, 4) is None

    def test_none_probability_returns_none(self):
        assert _compute_risk_score(3, None) is None

    def test_both_none_returns_none(self):
        assert _compute_risk_score(None, None) is None


# ─── _as_int_or_none ─────────────────────────────────────────────────────────

class TestAsIntOrNone:
    def test_integer_pass_through(self):
        assert _as_int_or_none(3) == 3

    def test_float_truncated(self):
        assert _as_int_or_none(3.0) == 3

    def test_none_pass_through(self):
        assert _as_int_or_none(None) is None


# ─── build_consensus_row ─────────────────────────────────────────────────────

class TestBuildConsensusRow:
    def test_full_row(self):
        issue = _fake_issue(key="TLR-100", rcm="A, CE")
        prop = _fake_prop_block(pre_sev=3, pre_prob=4, post_sev=3, post_prob=1)
        row = build_consensus_row(issue, prop, BASE_URL)

        assert row["TLR Key"] == "TLR-100"
        assert row["URL"] == f"{BASE_URL}/browse/TLR-100"
        assert row["Summary"] == "Test risk"
        assert row["Harm"] == "Severe hypoglycemia"
        assert row["Hazard Category"] == "Dosing"
        assert row["Pre-mitigation Severity"] == 3
        assert row["Pre-mitigation Probability"] == 4
        assert row["Post-mitigation Severity"] == 3
        assert row["Post-mitigation Probability"] == 1
        # Pre score = 12 -> Unacceptable (red)
        assert "Unacceptable" in row["Pre-mitigation Risk Score"]
        assert row["_pre_color"] == "FFC7CE"
        # Post score = 3 -> Acceptable (green)
        assert "Acceptable" in row["Post-mitigation Risk Score"]
        assert row["_post_color"] == "C6EFCE"
        # RCM titles joined with newlines
        assert "Algorithm limits" in row["Risk Control Measures"]
        assert "CGM error states have alerts and alarms" in row["Risk Control Measures"]

    def test_missing_property_block_yields_none_cells(self):
        issue = _fake_issue(key="TLR-101")
        row = build_consensus_row(issue, None, BASE_URL)

        assert row["Pre-mitigation Severity"] is None
        assert row["Pre-mitigation Probability"] is None
        assert row["Post-mitigation Severity"] is None
        assert row["Post-mitigation Probability"] is None
        assert row["Pre-mitigation Risk Score"] == "N/A"
        assert row["Post-mitigation Risk Score"] == "N/A"
        assert row["_pre_color"] == "FFFFFF"  # Unknown (white)
        assert row["_post_color"] == "FFFFFF"

    def test_missing_rcm_yields_empty_string(self):
        issue = _fake_issue(rcm=None)
        row = build_consensus_row(issue, _fake_prop_block(), BASE_URL)
        assert row["Risk Control Measures"] == ""

    def test_conditionally_acceptable_boundary(self):
        # severity=4, probability=1 -> score=4, severity=4 -> Conditionally acceptable (yellow)
        issue = _fake_issue()
        prop = _fake_prop_block(post_sev=4, post_prob=1)
        row = build_consensus_row(issue, prop, BASE_URL)
        assert "Conditionally acceptable" in row["Post-mitigation Risk Score"]
        assert row["_post_color"] == "FFEB9C"

    def test_acceptable_boundary_score_4_severity_3(self):
        # severity=2, probability=2 -> score=4, severity<=3 -> Acceptable
        issue = _fake_issue()
        prop = _fake_prop_block(post_sev=2, post_prob=2)
        row = build_consensus_row(issue, prop, BASE_URL)
        assert "Acceptable" in row["Post-mitigation Risk Score"]
        assert row["_post_color"] == "C6EFCE"

    def test_select_option_field_unwrapped(self):
        issue = _fake_issue()
        issue["fields"][FIELD_HAZARD_CATEGORY] = {"value": "Cybersecurity", "id": "1"}
        row = build_consensus_row(issue, _fake_prop_block(), BASE_URL)
        assert row["Hazard Category"] == "Cybersecurity"

    def test_status_populated_from_name_dict(self):
        issue = _fake_issue(status="Active")
        row = build_consensus_row(issue, _fake_prop_block(), BASE_URL)
        assert row["Status"] == "Active"

    def test_missing_status_yields_empty_string(self):
        issue = _fake_issue()
        issue["fields"][FIELD_STATUS] = None
        row = build_consensus_row(issue, _fake_prop_block(), BASE_URL)
        assert row["Status"] == ""


# ─── search_issues (pagination + retry) ──────────────────────────────────────

class TestSearchIssues:
    @patch("consensus_risk_list.requests.get")
    def test_single_page(self, mock_get):
        # New endpoint: last page has no nextPageToken (isLast optional).
        mock_get.return_value = _mock_response(
            200,
            {"issues": [_fake_issue("TLR-1"), _fake_issue("TLR-2")], "isLast": True},
        )
        issues = search_issues(BASE_URL, AUTH, "dummy JQL")
        assert len(issues) == 2
        assert issues[0]["key"] == "TLR-1"
        mock_get.assert_called_once()
        # First request must NOT include nextPageToken.
        first_params = mock_get.call_args_list[0].kwargs["params"]
        assert "nextPageToken" not in first_params

    @patch("consensus_risk_list.requests.get")
    def test_pagination_across_pages(self, mock_get):
        page1 = {
            "issues": [_fake_issue(f"TLR-{i}") for i in range(100)],
            "nextPageToken": "cursor-abc",
            "isLast": False,
        }
        page2 = {
            "issues": [_fake_issue(f"TLR-{i}") for i in range(100, 150)],
            "isLast": True,
        }
        mock_get.side_effect = [_mock_response(200, page1), _mock_response(200, page2)]

        issues = search_issues(BASE_URL, AUTH, "dummy JQL")
        assert len(issues) == 150
        assert mock_get.call_count == 2
        # Second call must forward the cursor from the first response.
        assert mock_get.call_args_list[1].kwargs["params"]["nextPageToken"] == "cursor-abc"

    @patch("consensus_risk_list.requests.get")
    def test_pagination_stops_when_token_missing(self, mock_get):
        # Defensive: absence of nextPageToken terminates even without isLast flag.
        page1 = {
            "issues": [_fake_issue("TLR-1")],
            "nextPageToken": "c1",
        }
        page2 = {"issues": [_fake_issue("TLR-2")]}  # no token, no isLast
        mock_get.side_effect = [_mock_response(200, page1), _mock_response(200, page2)]

        issues = search_issues(BASE_URL, AUTH, "dummy JQL")
        assert len(issues) == 2
        assert mock_get.call_count == 2

    @patch("consensus_risk_list.requests.get")
    def test_empty_result(self, mock_get):
        mock_get.return_value = _mock_response(200, {"issues": [], "isLast": True})
        issues = search_issues(BASE_URL, AUTH, "dummy JQL")
        assert issues == []

    @patch("consensus_risk_list.time.sleep", return_value=None)
    @patch("consensus_risk_list.requests.get")
    def test_429_retry_then_success(self, mock_get, _mock_sleep):
        mock_get.side_effect = [
            _mock_response(429, headers={"Retry-After": "1"}),
            _mock_response(200, {"issues": [_fake_issue("TLR-1")], "isLast": True}),
        ]
        issues = search_issues(BASE_URL, AUTH, "dummy JQL", max_retries=3)
        assert len(issues) == 1
        assert mock_get.call_count == 2

    @patch("consensus_risk_list.requests.get")
    def test_non_200_raises(self, mock_get):
        bad = _mock_response(500, {"error": "boom"})
        bad.raise_for_status.side_effect = requests.HTTPError("500")
        mock_get.return_value = bad
        with pytest.raises(requests.HTTPError):
            search_issues(BASE_URL, AUTH, "dummy JQL")

    @patch("consensus_risk_list.requests.get")
    def test_uses_new_jql_endpoint(self, mock_get):
        mock_get.return_value = _mock_response(200, {"issues": [], "isLast": True})
        search_issues(BASE_URL, AUTH, "dummy JQL")
        called_url = mock_get.call_args_list[0].args[0]
        assert called_url.endswith("/rest/api/3/search/jql")


# ─── write_consensus_excel ───────────────────────────────────────────────────

class TestWriteConsensusExcel:
    def test_writes_file_with_expected_structure(self, tmp_path):
        from openpyxl import load_workbook

        rows = [
            build_consensus_row(
                _fake_issue("TLR-100", rcm="A"), _fake_prop_block(), BASE_URL
            )
        ]
        out = tmp_path / "out.xlsx"
        result_path = write_consensus_excel(rows, str(out))

        assert result_path == str(out)
        assert out.exists()

        wb = load_workbook(result_path)
        ws = wb["Consensus List"]

        # Header row
        header_values = [c.value for c in ws[1]]
        assert "TLR Key" in header_values
        assert "Risk Control Measures" in header_values
        assert header_values[-1] == "Risk Control Measures"

        # Data row
        assert ws.cell(row=2, column=1).value == "TLR-100"
        # Hyperlink on TLR key
        assert ws.cell(row=2, column=1).hyperlink is not None

    def test_empty_rows_writes_header_only(self, tmp_path):
        from openpyxl import load_workbook

        out = tmp_path / "empty.xlsx"
        write_consensus_excel([], str(out))

        wb = load_workbook(str(out))
        ws = wb["Consensus List"]
        assert ws.max_row == 1  # header only


# ─── generate_consensus_list (end-to-end) ────────────────────────────────────

class TestGenerateConsensusList:
    @patch.dict(os.environ, ENV_VARS, clear=True)
    @patch("consensus_risk_list.verify_connection", return_value=True)
    @patch("consensus_risk_list.fetch_issue_property")
    @patch("consensus_risk_list.requests.get")
    def test_end_to_end_writes_excel(
        self, mock_search, mock_fetch_prop, _mock_verify, tmp_path
    ):
        mock_search.return_value = _mock_response(
            200,
            {
                "issues": [
                    _fake_issue("TLR-100", rcm="A, CE"),
                    _fake_issue("TLR-101", rcm="D"),
                ],
                "isLast": True,
            },
        )
        mock_fetch_prop.side_effect = [
            _fake_prop_block(pre_sev=3, pre_prob=4, post_sev=3, post_prob=1),
            _fake_prop_block(pre_sev=4, pre_prob=3, post_sev=2, post_prob=1),
        ]

        out = tmp_path / "consensus.xlsx"
        result_path = generate_consensus_list(str(out))

        assert result_path == str(out)
        assert out.exists()
        assert mock_fetch_prop.call_count == 2

        from openpyxl import load_workbook
        wb = load_workbook(result_path)
        ws = wb["Consensus List"]
        assert ws.cell(row=2, column=1).value == "TLR-100"
        assert ws.cell(row=3, column=1).value == "TLR-101"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_env_raises(self, tmp_path):
        out = tmp_path / "out.xlsx"
        # Point env_path to a non-existent file so load_dotenv is a no-op
        with pytest.raises(EnvironmentError, match="Missing required"):
            generate_consensus_list(str(out), env_path="/nonexistent/.env")

    @patch.dict(os.environ, ENV_VARS, clear=True)
    @patch("consensus_risk_list.verify_connection", return_value=False)
    def test_bad_credentials_raises(self, _mock_verify, tmp_path):
        out = tmp_path / "out.xlsx"
        with pytest.raises(ConnectionError, match="Could not authenticate"):
            generate_consensus_list(str(out))

    @patch.dict(os.environ, ENV_VARS, clear=True)
    @patch("consensus_risk_list.verify_connection", return_value=True)
    @patch("consensus_risk_list.fetch_issue_property")
    @patch("consensus_risk_list.requests.get")
    def test_custom_exclude_status_reaches_jql(
        self, mock_search, _mock_fetch_prop, _mock_verify, tmp_path
    ):
        """Regression: prevents a recurrence of the positional-arg mix-up where
        exclude_status was passed in the exclude_components slot (and vice versa)
        during the call from generate_consensus_list to build_consensus_jql."""
        mock_search.return_value = _mock_response(
            200, {"issues": [], "isLast": True}
        )
        out = tmp_path / "out.xlsx"
        generate_consensus_list(
            str(out),
            exclude_status=("Done", "Archived"),
            exclude_components=("foo_component",),
        )
        jql = mock_search.call_args_list[0].kwargs["params"]["jql"]
        # Status values land in the status clause, not the component clause.
        assert 'status NOT IN ("Done", "Archived")' in jql
        # Component values land in the component clause, not the status clause.
        assert '"foo_component"' in jql
        assert 'status NOT IN ("foo_component"' not in jql
        assert 'component IS EMPTY OR component NOT IN ("foo_component")' in jql
