"""
Unit tests for jira_risk_probabilities.py

Tests cover:
  - TLR key normalisation (suffix stripping, deduplication)
  - Safe nested property traversal
  - verify_connection() – auth check via /myself
  - fetch_issue_property() – property-specific endpoint, retry logic,
    detailed error logging
  - _extract_field_value() – normalisation of bare strings, select dicts, None
  - list_issue_fields() – field discovery via expand=names
  - fetch_issue_fields() – summary + hazard category from issue endpoint,
    retry logic, select-field unwrapping
  - fetch_probabilities() – end-to-end with mocked HTTP, deduplication,
    missing values, credential validation, new Summary/Hazard Category columns
"""

import os
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
import requests

from jira_risk_probabilities import (
    DEFAULT_API_VERSION,
    HAZARD_CATEGORY_FIELD,
    INITIAL_PROB_FIELD,
    PROPERTY_KEY,
    RESIDUAL_PROB_FIELD,
    _extract_field_value,
    fetch_issue_fields,
    fetch_issue_property,
    fetch_probabilities,
    get_nested_property,
    list_issue_fields,
    normalize_tlr_key,
    unique_base_keys,
    verify_connection,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_response(status_code: int, json_body: dict = None, headers: dict = None) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body or {}
    mock_resp.headers = headers or {}
    mock_resp.text = str(json_body or "")
    return mock_resp


BASE_URL = "https://tidepool.atlassian.net"
AUTH = ("user@example.com", "token123")
ENV_VARS = {
    "JIRA_BASE_URL": BASE_URL,
    "JIRA_USERNAME": "user@example.com",
    "JIRA_API_TOKEN": "token123",
}


# ─── normalize_tlr_key ────────────────────────────────────────────────────────

class TestNormalizeTlrKey:
    def test_plain_key_unchanged(self):
        assert normalize_tlr_key("TLR-552") == "TLR-552"

    def test_single_suffix_stripped(self):
        assert normalize_tlr_key("TLR-1117_bike") == "TLR-1117"

    def test_multi_part_suffix_stripped(self):
        assert normalize_tlr_key("TLR-899_01_025") == "TLR-899"

    def test_word_suffix_stripped(self):
        assert normalize_tlr_key("TLR-847_wmeal") == "TLR-847"

    def test_corr_suffix_stripped(self):
        assert normalize_tlr_key("TLR-847_corr") == "TLR-847"

    def test_leading_whitespace_tolerated(self):
        assert normalize_tlr_key("  TLR-1011") == "TLR-1011"

    def test_non_matching_returns_none(self):
        assert normalize_tlr_key("LOOP-123") is None

    def test_empty_string_returns_none(self):
        assert normalize_tlr_key("") is None

    def test_large_numeric_id(self):
        assert normalize_tlr_key("TLR-9999_foo_bar") == "TLR-9999"


# ─── unique_base_keys ─────────────────────────────────────────────────────────

class TestUniqueBaseKeys:
    def test_deduplicates_variants(self):
        raw = ["TLR-1117_bike", "TLR-1117_jog", "TLR-1117_walk", "TLR-1117"]
        assert unique_base_keys(raw) == ["TLR-1117"]

    def test_sorted_output(self):
        raw = ["TLR-900", "TLR-100", "TLR-500"]
        assert unique_base_keys(raw) == ["TLR-100", "TLR-500", "TLR-900"]

    def test_invalid_keys_excluded(self):
        raw = ["TLR-552", "LOOP-99", "", "TLR-553"]
        result = unique_base_keys(raw)
        assert "LOOP-99" not in result
        assert "" not in result
        assert "TLR-552" in result and "TLR-553" in result

    def test_empty_input_returns_empty_list(self):
        assert unique_base_keys([]) == []


# ─── get_nested_property ──────────────────────────────────────────────────────

class TestGetNestedProperty:
    def test_simple_lookup(self):
        assert get_nested_property({"a": 42}, "a") == 42

    def test_two_levels(self):
        assert get_nested_property({"a": {"b": 99}}, "a", "b") == 99

    def test_three_levels(self):
        data = {"x": {"y": {"z": "found"}}}
        assert get_nested_property(data, "x", "y", "z") == "found"

    def test_missing_key_returns_none(self):
        assert get_nested_property({"a": {}}, "a", "missing") is None

    def test_none_input_returns_none(self):
        assert get_nested_property(None, "a") is None

    def test_non_dict_intermediate_returns_none(self):
        assert get_nested_property({"a": "string"}, "a", "b") is None

    def test_zero_keys_returns_data(self):
        data = {"k": "v"}
        assert get_nested_property(data) == data


# ─── verify_connection ────────────────────────────────────────────────────────

class TestVerifyConnection:
    def test_returns_true_on_200(self):
        payload = {"displayName": "Shawn Foster", "emailAddress": "shawn@tidepool.org"}
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            assert verify_connection(BASE_URL, AUTH) is True

        expected_url = f"{BASE_URL}/rest/api/{DEFAULT_API_VERSION}/myself"
        mock_get.assert_called_once_with(expected_url, auth=AUTH, timeout=10)

    def test_returns_false_on_401(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(401)
            assert verify_connection(BASE_URL, AUTH) is False

    def test_returns_false_on_403(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(403)
            assert verify_connection(BASE_URL, AUTH) is False

    def test_returns_false_on_request_exception(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("timeout")
            assert verify_connection(BASE_URL, AUTH) is False

    def test_respects_api_version_parameter(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"displayName": "X"})
            verify_connection(BASE_URL, AUTH, api_version="3")

        called_url = mock_get.call_args[0][0]
        assert "/rest/api/3/myself" in called_url


# ─── fetch_issue_property ─────────────────────────────────────────────────────

class TestFetchIssueProperty:
    ISSUE_KEY = "TLR-552"

    def _expected_url(self, version=DEFAULT_API_VERSION):
        return (
            f"{BASE_URL}/rest/api/{version}"
            f"/issue/{self.ISSUE_KEY}/properties/{PROPERTY_KEY}"
        )

    def test_successful_response_returns_value_dict(self):
        value = {INITIAL_PROB_FIELD: "1B", RESIDUAL_PROB_FIELD: "1A"}
        payload = {"key": PROPERTY_KEY, "value": value}
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY)

        assert result == value
        mock_get.assert_called_once_with(self._expected_url(), auth=AUTH, timeout=10)

    def test_uses_property_endpoint_not_issue_endpoint(self):
        """Confirm URL contains /properties/boja_prop_issue, not ?properties=."""
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"key": PROPERTY_KEY, "value": {}})
            fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY)

        called_url = mock_get.call_args[0][0]
        assert f"/properties/{PROPERTY_KEY}" in called_url
        assert "?properties=" not in called_url

    def test_respects_api_version_parameter(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"key": PROPERTY_KEY, "value": {}})
            fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY, api_version="3")

        called_url = mock_get.call_args[0][0]
        assert "/rest/api/3/" in called_url

    def test_404_returns_none(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404)
            assert fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY) is None

    def test_500_returns_none(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(500)
            assert fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY) is None

    def test_401_returns_none(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(401)
            assert fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY) is None

    def test_429_retries_then_succeeds(self):
        value = {INITIAL_PROB_FIELD: "2C", RESIDUAL_PROB_FIELD: "1B"}
        rate_limited = _mock_response(429, headers={"Retry-After": "0"})
        success = _mock_response(200, {"key": PROPERTY_KEY, "value": value})

        with patch("jira_risk_probabilities.requests.get") as mock_get, \
             patch("jira_risk_probabilities.time.sleep"):
            mock_get.side_effect = [rate_limited, success]
            result = fetch_issue_property(
                BASE_URL, AUTH, self.ISSUE_KEY, max_retries=3, backoff_seconds=0
            )

        assert result == value
        assert mock_get.call_count == 2

    def test_request_exception_retries_then_returns_none(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get, \
             patch("jira_risk_probabilities.time.sleep"):
            mock_get.side_effect = requests.RequestException("connection error")
            result = fetch_issue_property(
                BASE_URL, AUTH, self.ISSUE_KEY, max_retries=2, backoff_seconds=0
            )

        assert result is None
        assert mock_get.call_count == 2

    def test_empty_value_returns_empty_dict(self):
        """Property exists but has no sub-fields – value is empty dict."""
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"key": PROPERTY_KEY, "value": {}})
            result = fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result == {}

    def test_missing_value_key_returns_none(self):
        """Response 200 but 'value' key absent."""
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"key": PROPERTY_KEY})
            result = fetch_issue_property(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result is None


# ─── fetch_probabilities ──────────────────────────────────────────────────────

class TestFetchProbabilities:
    def _prop_value(self, initial, residual):
        return {INITIAL_PROB_FIELD: initial, RESIDUAL_PROB_FIELD: residual}

    def test_returns_dataframe_with_correct_columns(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property") as mock_fetch, \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value={"summary": None, "hazard_category": None, "no_automation": False}):
            mock_fetch.return_value = self._prop_value("1B", "1A")
            df = fetch_probabilities(["TLR-552"])

        assert list(df.columns) == ["TLR Key", "Summary", "Hazard Category", "No Automation", "Initial Probability", "Residual Probability"]

    def test_deduplicates_variant_keys(self):
        """Four scenario variants of TLR-1117 produce a single Jira fetch."""
        raw = ["TLR-1117_bike", "TLR-1117_jog", "TLR-1117_walk", "TLR-1117_str_training"]
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property") as mock_fetch, \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value={"summary": None, "hazard_category": None, "no_automation": False}):
            mock_fetch.return_value = self._prop_value("2A", "1A")
            df = fetch_probabilities(raw)

        assert len(df) == 1
        assert df.iloc[0]["TLR Key"] == "TLR-1117"
        mock_fetch.assert_called_once()

    def test_multiple_keys_returned_sorted(self):
        raw = ["TLR-900", "TLR-100_foo", "TLR-500"]
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property") as mock_fetch, \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value={"summary": None, "hazard_category": None, "no_automation": False}):
            mock_fetch.return_value = self._prop_value("2B", "1B")
            df = fetch_probabilities(raw)

        assert list(df["TLR Key"]) == ["TLR-100", "TLR-500", "TLR-900"]

    def test_missing_property_block_yields_none_values(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property") as mock_fetch, \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value={"summary": None, "hazard_category": None, "no_automation": False}):
            mock_fetch.return_value = {}  # value dict present but fields absent
            df = fetch_probabilities(["TLR-552"])

        assert df.iloc[0]["Initial Probability"] is None
        assert df.iloc[0]["Residual Probability"] is None

    def test_none_response_yields_none_values(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property") as mock_fetch, \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value={"summary": None, "hazard_category": None, "no_automation": False}):
            mock_fetch.return_value = None
            df = fetch_probabilities(["TLR-552"])

        assert df.iloc[0]["Initial Probability"] is None
        assert df.iloc[0]["Residual Probability"] is None

    def test_raises_connection_error_when_verify_fails(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=False):
            with pytest.raises(ConnectionError, match="Could not authenticate"):
                fetch_probabilities(["TLR-552"])

    def test_raises_on_missing_env_vars(self):
        env = {k: v for k, v in ENV_VARS.items() if k != "JIRA_API_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("JIRA_API_TOKEN", None)
            with pytest.raises(EnvironmentError, match="JIRA_API_TOKEN"):
                fetch_probabilities(["TLR-552"])

    def test_empty_key_list_returns_empty_dataframe(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True):
            df = fetch_probabilities([])

        assert len(df) == 0
        assert list(df.columns) == ["TLR Key", "Summary", "Hazard Category", "Initial Probability", "Residual Probability"]

    def test_api_version_passed_through_to_fetch(self):
        """api_version arg is forwarded to fetch_issue_property."""
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property") as mock_fetch, \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value={"summary": None, "hazard_category": None, "no_automation": False}):
            mock_fetch.return_value = self._prop_value("1A", "1A")
            fetch_probabilities(["TLR-552"], api_version="3")

        _, kwargs = mock_fetch.call_args
        assert kwargs.get("api_version") == "3"


# ─── _extract_field_value ─────────────────────────────────────────────────────────

class TestExtractFieldValue:
    def test_none_returns_none(self):
        assert _extract_field_value(None) is None

    def test_bare_string_returned_unchanged(self):
        assert _extract_field_value("Software") == "Software"

    def test_select_dict_returns_value_key(self):
        assert _extract_field_value({"value": "Hardware", "id": "10001"}) == "Hardware"

    def test_dict_without_value_key_returns_none(self):
        assert _extract_field_value({"id": "10001"}) is None

    def test_integer_coerced_to_string(self):
        assert _extract_field_value(42) == "42"

    def test_empty_dict_returns_none(self):
        assert _extract_field_value({}) is None


# ─── list_issue_fields ──────────────────────────────────────────────────────────────

class TestListIssueFields:
    ISSUE_KEY = "TLR-552"

    def test_returns_names_dict_on_200(self):
        names = {"summary": "Summary", "customfield_10200": "Hazard Category"}
        payload = {"fields": {}, "names": names}
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = list_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result == names

    def test_uses_expand_names_param(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"names": {}})
            list_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        _, kwargs = mock_get.call_args
        assert kwargs.get("params", {}).get("expand") == "names"

    def test_404_returns_none(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404)
            assert list_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY) is None

    def test_request_exception_returns_none(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("timeout")
            assert list_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY) is None

    def test_missing_names_key_returns_empty_dict(self):
        """200 response but no 'names' key – treated as empty mapping."""
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"fields": {}})
            result = list_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result == {}


# ─── fetch_issue_fields ──────────────────────────────────────────────────────────────

class TestFetchIssueFields:
    ISSUE_KEY = "TLR-552"

    def _issue_payload(self, summary: str, hazard_category, components=None):
        """Build a minimal Jira issue JSON response."""
        return {"fields": {
            "summary": summary,
            HAZARD_CATEGORY_FIELD: hazard_category,
            "components": components or [],
        }}

    def test_returns_summary_and_hazard_category_as_strings(self):
        payload = self._issue_payload("Some TLR summary", {"value": "Software", "id": "1"})
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["summary"] == "Some TLR summary"
        assert result["hazard_category"] == "Software"

    def test_bare_string_hazard_category_returned_unchanged(self):
        payload = self._issue_payload("Summary text", "Hardware")
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["hazard_category"] == "Hardware"

    def test_no_automation_component_present_returns_true(self):
        components = [
            {"name": "firmware", "id": "1"},
            {"name": "no_automation", "id": "2"},
            {"name": "mobile", "id": "3"},
        ]
        payload = self._issue_payload("Summary", None, components=components)
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["no_automation"] is True

    def test_no_automation_component_absent_returns_false(self):
        components = [
            {"name": "firmware", "id": "1"},
            {"name": "mobile", "id": "3"},
        ]
        payload = self._issue_payload("Summary", None, components=components)
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["no_automation"] is False

    def test_empty_components_list_returns_false(self):
        payload = self._issue_payload("Summary", None, components=[])
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["no_automation"] is False

    def test_null_components_field_returns_false(self):
        """Jira may return null for components if the field is empty."""
        payload = {"fields": {"summary": "x", HAZARD_CATEGORY_FIELD: None, "components": None}}
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["no_automation"] is False

    def test_null_fields_return_none(self):
        payload = {"fields": {"summary": None, HAZARD_CATEGORY_FIELD: None, "components": []}}
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["summary"] is None
        assert result["hazard_category"] is None
        assert result["no_automation"] is False

    def test_missing_fields_return_none(self):
        """Fields dict present but custom field absent."""
        payload = {"fields": {}}
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result["summary"] is None
        assert result["hazard_category"] is None
        assert result["no_automation"] is False

    def test_requests_correct_fields_param(self):
        payload = self._issue_payload("x", None)
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, payload)
            fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        _, kwargs = mock_get.call_args
        params = kwargs.get("params", {})
        assert "summary" in params["fields"]
        assert HAZARD_CATEGORY_FIELD in params["fields"]
        assert "components" in params["fields"]

    def test_404_returns_empty_dict_with_none_values(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404)
            result = fetch_issue_fields(BASE_URL, AUTH, self.ISSUE_KEY)
        assert result == {"summary": None, "hazard_category": None, "no_automation": False}

    def test_429_retries_then_succeeds(self):
        payload = self._issue_payload("Summary", {"value": "Software"})
        rate_limited = _mock_response(429, headers={"Retry-After": "0"})
        success = _mock_response(200, payload)
        with patch("jira_risk_probabilities.requests.get") as mock_get, \
             patch("jira_risk_probabilities.time.sleep"):
            mock_get.side_effect = [rate_limited, success]
            result = fetch_issue_fields(
                BASE_URL, AUTH, self.ISSUE_KEY, max_retries=3, backoff_seconds=0
            )
        assert result["summary"] == "Summary"
        assert mock_get.call_count == 2

    def test_request_exception_returns_none_values(self):
        with patch("jira_risk_probabilities.requests.get") as mock_get, \
             patch("jira_risk_probabilities.time.sleep"):
            mock_get.side_effect = requests.RequestException("connection error")
            result = fetch_issue_fields(
                BASE_URL, AUTH, self.ISSUE_KEY, max_retries=2, backoff_seconds=0
            )
        assert result == {"summary": None, "hazard_category": None, "no_automation": False}


# ─── fetch_probabilities (updated column assertions) ───────────────────────────────────

class TestFetchProbabilitiesNewColumns:
    """Focused tests for Summary and Hazard Category column behaviour."""

    _fields_ok = {"summary": "Insulin overdose risk", "hazard_category": "Software", "no_automation": True}
    _fields_empty = {"summary": None, "hazard_category": None, "no_automation": False}

    def _prop_value(self, initial, residual):
        return {INITIAL_PROB_FIELD: initial, RESIDUAL_PROB_FIELD: residual}

    def test_new_columns_present_in_output(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property", return_value=self._prop_value("1B", "1A")), \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value=self._fields_ok):
            df = fetch_probabilities(["TLR-552"])
        assert "Summary" in df.columns
        assert "Hazard Category" in df.columns
        assert "No Automation" in df.columns

    def test_column_order(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property", return_value=self._prop_value("1B", "1A")), \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value=self._fields_ok):
            df = fetch_probabilities(["TLR-552"])
        assert list(df.columns) == [
            "TLR Key", "Summary", "Hazard Category", "No Automation", "Initial Probability", "Residual Probability"
        ]

    def test_summary_and_hazard_category_values_populated(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property", return_value=self._prop_value("2B", "1B")), \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value=self._fields_ok):
            df = fetch_probabilities(["TLR-552"])
        assert df.iloc[0]["Summary"] == "Insulin overdose risk"
        assert df.iloc[0]["Hazard Category"] == "Software"
        assert df.iloc[0]["No Automation"] is True

    def test_no_automation_false_when_component_absent(self):
        fields = {"summary": "Some summary", "hazard_category": "Hardware", "no_automation": False}
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property", return_value=self._prop_value("1A", "1A")), \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value=fields):
            df = fetch_probabilities(["TLR-552"])
        assert df.iloc[0]["No Automation"] is False

    def test_missing_issue_fields_yields_none_columns(self):
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property", return_value=self._prop_value("1A", "1A")), \
             patch("jira_risk_probabilities.fetch_issue_fields", return_value=self._fields_empty):
            df = fetch_probabilities(["TLR-552"])
        assert df.iloc[0]["Summary"] is None
        assert df.iloc[0]["Hazard Category"] is None
        assert df.iloc[0]["No Automation"] is False

    def test_fetch_issue_fields_called_once_per_unique_key(self):
        raw = ["TLR-552_foo", "TLR-552_bar", "TLR-552"]
        with patch.dict(os.environ, ENV_VARS), \
             patch("jira_risk_probabilities.verify_connection", return_value=True), \
             patch("jira_risk_probabilities.fetch_issue_property", return_value=self._prop_value("1A", "1A")), \
             patch("jira_risk_probabilities.fetch_issue_fields") as mock_fields:
            mock_fields.return_value = self._fields_ok
            fetch_probabilities(raw)
        mock_fields.assert_called_once()
