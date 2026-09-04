#!/usr/bin/env python3
"""
Jira Risk Probability Fetcher

Fetches initial and residual probability values from Jira TLR issues using
the dedicated issue-property endpoint, and returns them as a DataFrame suitable
for appending to the algorithm risk comparison report as a second sheet.

Usage (standalone):
    python jira_risk_probabilities.py --keys TLR-552 TLR-553 TLR-554
    python jira_risk_probabilities.py --table4 path/to/table4_individual_hazardous_situations.csv
    python jira_risk_probabilities.py --list-fields TLR-552  # discover custom field IDs

Credentials are read from environment variables (or a .env file):
    JIRA_BASE_URL   - e.g. https://tidepool.atlassian.net
    JIRA_USERNAME   - Jira account email
    JIRA_API_TOKEN  - Jira API token
"""

__author__ = "Shawn Foster"

import argparse
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

PROPERTY_KEY = "boja_prop_issue"
INITIAL_PROB_FIELD = "risk_probability_value"
RESIDUAL_PROB_FIELD = "risk_residual_probability_value"

# Custom field ID for the "Hazard Category" field on TLR issues.
# Run --list-fields on any TLR issue to discover the actual ID, then update this constant.
HAZARD_CATEGORY_FIELD = "customfield_10095"

# Component name that flags an issue as having no automation.
NO_AUTOMATION_COMPONENT = "no_automation"

# Default API version matches the atlassian-python-api library used in reports/
DEFAULT_API_VERSION = "2"

# Matches the canonical Jira key at the start of a string, e.g. "TLR-1117"
# and ignores any trailing suffix like "_bike" or "_01_025"
_TLR_BASE_RE = re.compile(r"^(TLR-\d+)")


# ─── Key normalisation ────────────────────────────────────────────────────────

def normalize_tlr_key(raw_key: str) -> Optional[str]:
    """
    Extract the base Jira issue key from a scenario-specific TLR key.

    Strips variant suffixes so that simulation scenario keys map back to
    their parent Jira ticket:
        TLR-1117_bike      → TLR-1117
        TLR-899_01_025     → TLR-899
        TLR-552            → TLR-552   (unchanged)

    Args:
        raw_key: Raw TLR key string, e.g. from Table 4 'TLR Key' column.

    Returns:
        Base Jira issue key, or None if the string does not match the
        expected pattern.
    """
    match = _TLR_BASE_RE.match(raw_key.strip())
    return match.group(1) if match else None


def unique_base_keys(raw_keys: List[str]) -> List[str]:
    """
    Return sorted, deduplicated base TLR keys from a list of raw scenario keys.

    Args:
        raw_keys: List of TLR key strings (may include duplicates/suffixes).

    Returns:
        Sorted list of unique base keys (e.g. ['TLR-552', 'TLR-553', ...]).
    """
    seen = set()
    for raw in raw_keys:
        norm = normalize_tlr_key(raw)
        if norm:
            seen.add(norm)
    return sorted(seen)


# ─── Property traversal ───────────────────────────────────────────────────────

def get_nested_property(data: Optional[Dict[str, Any]], *keys: str) -> Optional[Any]:
    """
    Safely traverse a nested dict by successive string keys.

    Returns None (rather than raising) if any key is absent or if an
    intermediate value is not a dict.

    Args:
        data: Top-level dict to traverse (may be None).
        *keys: Key sequence to apply in order.

    Returns:
        Value at the terminal key, or None if unreachable.

    Examples:
        >>> get_nested_property({"a": {"b": 42}}, "a", "b")
        42
        >>> get_nested_property({"a": {}}, "a", "missing") is None
        True
        >>> get_nested_property(None, "a") is None
        True
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


# ─── Jira API ────────────────────────────────────────────────────────────────

def verify_connection(base_url: str, auth: Tuple[str, str], api_version: str = DEFAULT_API_VERSION) -> bool:
    """
    Verify that credentials are valid by calling the /myself endpoint.

    Uses ``GET /rest/api/{version}/myself``, which requires only a valid
    authenticated session and no project-level permissions.  This separates
    authentication failures from issue-not-found failures when debugging 404s.

    Args:
        base_url: Jira instance root URL, e.g. 'https://tidepool.atlassian.net'.
        auth: (username, api_token) tuple for HTTP Basic Auth.
        api_version: Jira REST API version string (default: '2').

    Returns:
        True if credentials are accepted (HTTP 200), False otherwise.
    """
    url = f"{base_url.rstrip('/')}/rest/api/{api_version}/myself"
    logger.info("Verifying Jira connection: GET %s", url)
    try:
        response = requests.get(url, auth=auth, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(
                "Jira connection OK – authenticated as '%s' (%s)",
                data.get("displayName", "?"),
                data.get("emailAddress", "?"),
            )
            return True
        logger.error(
            "Jira connection failed – HTTP %d: %s",
            response.status_code,
            response.text[:500],
        )
        return False
    except requests.RequestException as exc:
        logger.error("Jira connection error: %s", exc)
        return False


def fetch_issue_property(
    base_url: str,
    auth: Tuple[str, str],
    issue_key: str,
    api_version: str = DEFAULT_API_VERSION,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetch the ``boja_prop_issue`` property value for a single Jira issue.

    Uses the dedicated property endpoint rather than the full issue endpoint:
        GET /rest/api/{version}/issue/{key}/properties/boja_prop_issue

    This returns ``{"key": "boja_prop_issue", "value": {...}}`` directly,
    avoiding ambiguity in how the ``?properties=`` query parameter is handled
    across API versions.  The caller receives the ``value`` dict (or None).

    Logs the full request URL and up to 500 characters of the response body
    on any non-200 status to aid debugging.

    Retries up to *max_retries* times on HTTP 429 (rate-limited) responses
    with exponential back-off starting at *backoff_seconds*.

    Args:
        base_url: Jira instance root URL, e.g. 'https://tidepool.atlassian.net'.
        auth: (username, api_token) tuple for HTTP Basic Auth.
        issue_key: Jira issue key, e.g. 'TLR-1117'.
        api_version: Jira REST API version string (default: '2').
        max_retries: Maximum retry attempts on rate-limit errors.
        backoff_seconds: Initial back-off duration (doubles each retry).

    Returns:
        The property ``value`` dict (e.g. ``{'risk_probability_value': '1B', ...}``),
        or None on any failure.
    """
    url = (
        f"{base_url.rstrip('/')}/rest/api/{api_version}"
        f"/issue/{issue_key}/properties/{PROPERTY_KEY}"
    )
    delay = backoff_seconds

    for attempt in range(max_retries):
        logger.debug("%s: GET %s (attempt %d/%d)", issue_key, url, attempt + 1, max_retries)
        try:
            response = requests.get(url, auth=auth, timeout=10)

            if response.status_code == 200:
                return response.json().get("value")

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", delay))
                logger.warning(
                    "%s: rate-limited (attempt %d/%d), retrying in %ds",
                    issue_key, attempt + 1, max_retries, retry_after,
                )
                time.sleep(retry_after)
                delay *= 2
                continue

            # Log URL and full response body for all other non-200 statuses
            logger.error(
                "%s: HTTP %d from %s\n  Response: %s",
                issue_key,
                response.status_code,
                url,
                response.text[:500],
            )
            return None

        except requests.RequestException as exc:
            logger.error("%s: request to %s failed: %s", issue_key, url, exc)
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return None

    logger.error("%s: exhausted %d retries for %s", issue_key, max_retries, url)
    return None


# ─── Issue field helpers ─────────────────────────────────────────────────────

def list_issue_fields(
    base_url: str,
    auth: Tuple[str, str],
    issue_key: str,
    api_version: str = DEFAULT_API_VERSION,
) -> Optional[Dict[str, str]]:
    """
    Return a mapping of Jira field ID → display name for a single issue.

    Uses ``GET /rest/api/{version}/issue/{key}?expand=names`` to retrieve the
    full field name map from Jira.  Intended as a discovery helper so callers
    can identify the custom field ID for *Hazard Category* (and any other
    custom fields) to set in ``HAZARD_CATEGORY_FIELD``.

    Args:
        base_url: Jira instance root URL, e.g. 'https://tidepool.atlassian.net'.
        auth: (username, api_token) tuple for HTTP Basic Auth.
        issue_key: Any valid TLR issue key, e.g. 'TLR-552'.
        api_version: Jira REST API version string (default: '2').

    Returns:
        Dict mapping field ID strings to their display names (e.g.
        ``{'customfield_10200': 'Hazard Category', 'summary': 'Summary', ...}``),
        or None on any HTTP/network failure.
    """
    url = f"{base_url.rstrip('/')}/rest/api/{api_version}/issue/{issue_key}"
    logger.info("%s: GET %s?expand=names", issue_key, url)
    try:
        response = requests.get(url, auth=auth, timeout=10, params={"expand": "names"})
        if response.status_code == 200:
            return response.json().get("names", {})
        logger.error(
            "%s: HTTP %d from %s\n  Response: %s",
            issue_key, response.status_code, url, response.text[:500],
        )
        return None
    except requests.RequestException as exc:
        logger.error("%s: request to %s failed: %s", issue_key, url, exc)
        return None


def _extract_field_value(raw: Any) -> Optional[str]:
    """
    Normalise a Jira field value to a plain string.

    Jira returns select/option custom fields as ``{"value": "...", "id": "..."}``
    rather than bare strings.  This helper unwraps that structure so callers
    always receive a ``str`` (or ``None``).

    Args:
        raw: Raw field value from the Jira issue ``fields`` dict.

    Returns:
        String value, or None if the input is None or unrecognised.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("value")
    return str(raw)


def fetch_issue_fields(
    base_url: str,
    auth: Tuple[str, str],
    issue_key: str,
    api_version: str = DEFAULT_API_VERSION,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Dict[str, Optional[str]]:
    """
    Fetch ``Summary`` and ``Hazard Category`` from the Jira issue endpoint.

    Uses ``GET /rest/api/{version}/issue/{key}?fields=summary,{HAZARD_CATEGORY_FIELD}``.
    The ``Hazard Category`` value is extracted via ``_extract_field_value`` to
    handle both bare-string and select-option (``{"value": ...}``) field types.

    Retries up to *max_retries* times on HTTP 429 (rate-limited) responses
    with exponential back-off starting at *backoff_seconds*.

    Args:
        base_url: Jira instance root URL, e.g. 'https://tidepool.atlassian.net'.
        auth: (username, api_token) tuple for HTTP Basic Auth.
        issue_key: Jira issue key, e.g. 'TLR-1117'.
        api_version: Jira REST API version string (default: '2').
        max_retries: Maximum retry attempts on rate-limit errors.
        backoff_seconds: Initial back-off duration (doubles each retry).

    Returns:
        Dict with keys:
            - ``'summary'``        – issue Summary field value (str or None)
            - ``'hazard_category'``– Hazard Category field value (str or None)
            - ``'no_automation'``  – True if any component name equals
              ``NO_AUTOMATION_COMPONENT``, False otherwise

        ``no_automation`` is False (not None) on any HTTP/network failure.
    """
    url = f"{base_url.rstrip('/')}/rest/api/{api_version}/issue/{issue_key}"
    params = {"fields": f"summary,{HAZARD_CATEGORY_FIELD},components"}
    delay = backoff_seconds
    _empty: Dict[str, Any] = {"summary": None, "hazard_category": None, "no_automation": False}

    for attempt in range(max_retries):
        logger.debug("%s: GET %s (attempt %d/%d)", issue_key, url, attempt + 1, max_retries)
        try:
            response = requests.get(url, auth=auth, timeout=10, params=params)

            if response.status_code == 200:
                fields = response.json().get("fields", {})
                components = fields.get("components") or []
                no_automation = any(
                    c.get("name") == NO_AUTOMATION_COMPONENT for c in components
                )
                return {
                    "summary": _extract_field_value(fields.get("summary")),
                    "hazard_category": _extract_field_value(fields.get(HAZARD_CATEGORY_FIELD)),
                    "no_automation": no_automation,
                }

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", delay))
                logger.warning(
                    "%s: rate-limited (attempt %d/%d), retrying in %ds",
                    issue_key, attempt + 1, max_retries, retry_after,
                )
                time.sleep(retry_after)
                delay *= 2
                continue

            logger.error(
                "%s: HTTP %d from %s\n  Response: %s",
                issue_key, response.status_code, url, response.text[:500],
            )
            return _empty

        except requests.RequestException as exc:
            logger.error("%s: request to %s failed: %s", issue_key, url, exc)
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                return _empty

    logger.error("%s: exhausted %d retries for %s", issue_key, max_retries, url)
    return _empty


# ─── Public API ──────────────────────────────────────────────────────────────

def fetch_probabilities(
    raw_tlr_keys: List[str],
    env_path: Optional[str] = None,
    api_version: str = DEFAULT_API_VERSION,
) -> pd.DataFrame:
    """
    Fetch initial and residual probability values for a collection of TLR keys.

    Normalises raw scenario keys to base Jira issue keys, deduplicates them,
    verifies credentials via ``/myself``, then retrieves
    ``boja_prop_issue.risk_probability_value`` and
    ``boja_prop_issue.risk_residual_probability_value`` for each ticket using
    the dedicated property endpoint.

    Credentials are loaded from environment variables.  If *env_path* is
    provided that file is loaded first; otherwise ``load_dotenv()`` looks for
    a ``.env`` file in the current working directory.

    Required environment variables:
        JIRA_BASE_URL   – Jira instance root URL
        JIRA_USERNAME   – Jira account email address
        JIRA_API_TOKEN  – Jira API token

    Args:
        raw_tlr_keys: List of raw TLR key strings (suffixed or plain).
        env_path: Optional path to a ``.env`` credentials file.
        api_version: Jira REST API version string (default: '2').

    Returns:
        DataFrame with columns:
            - ``TLR Key``              – base Jira issue key
            - ``Summary``              – issue Summary field
            - ``Hazard Category``      – Hazard Category custom field value
            - ``No Automation``        – True if ``no_automation`` is among the issue's components
            - ``Initial Probability``  – boja_prop_issue.risk_probability_value
            - ``Residual Probability`` – boja_prop_issue.risk_residual_probability_value

        Rows are sorted by TLR Key.  Missing values appear as ``None``.

        .. note::
            ``Hazard Category`` will be ``None`` for all rows until
            ``HAZARD_CATEGORY_FIELD`` is updated with the correct custom field ID.
            Run ``--list-fields`` on any TLR issue to discover it.

    Raises:
        EnvironmentError: If any required credential variable is absent.
        ConnectionError: If the /myself connection check fails.
    """
    load_dotenv(dotenv_path=env_path)

    base_url = os.environ.get("JIRA_BASE_URL")
    username = os.environ.get("JIRA_USERNAME")
    api_token = os.environ.get("JIRA_API_TOKEN")

    missing = [
        name
        for name, val in [
            ("JIRA_BASE_URL", base_url),
            ("JIRA_USERNAME", username),
            ("JIRA_API_TOKEN", api_token),
        ]
        if val is None
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    auth = (username, api_token)

    if not verify_connection(base_url, auth, api_version=api_version):
        raise ConnectionError(
            f"Could not authenticate with Jira at {base_url}. "
            "Check JIRA_USERNAME and JIRA_API_TOKEN."
        )

    keys = unique_base_keys(raw_tlr_keys)
    logger.info("Fetching Jira probabilities for %d unique TLR keys (API v%s)", len(keys), api_version)

    rows = []
    for issue_key in keys:
        prop_block = fetch_issue_property(base_url, auth, issue_key, api_version=api_version)
        initial = get_nested_property(prop_block, INITIAL_PROB_FIELD)
        residual = get_nested_property(prop_block, RESIDUAL_PROB_FIELD)

        if initial is None or residual is None:
            logger.warning(
                "%s: probability value(s) missing "
                "(initial=%s, residual=%s)",
                issue_key, initial, residual,
            )

        issue_fields = fetch_issue_fields(base_url, auth, issue_key, api_version=api_version)

        rows.append(
            {
                "TLR Key": issue_key,
                "Summary": issue_fields.get("summary"),
                "Hazard Category": issue_fields.get("hazard_category"),
                "No Automation": issue_fields.get("no_automation"),
                "Initial Probability": initial,
                "Residual Probability": residual,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["TLR Key", "Summary", "Hazard Category", "No Automation", "Initial Probability", "Residual Probability"],
    )


# ─── Standalone entry point ───────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Fetch Jira probability values for TLR issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python jira_risk_probabilities.py --keys TLR-552 TLR-553 TLR-554
    python jira_risk_probabilities.py --table4 table4_individual_hazardous_situations.csv
    python jira_risk_probabilities.py --table4 table4.csv --env /path/to/.env --output probs.csv
    python jira_risk_probabilities.py --verify-only
        """,
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--keys", nargs="+", metavar="KEY",
        help="One or more TLR issue keys (suffixes are stripped automatically)",
    )
    source.add_argument(
        "--table4", metavar="CSV",
        help="Path to Table 4 CSV; 'TLR Key' column is used as input",
    )
    parser.add_argument(
        "--list-fields", metavar="KEY",
        help="Print all Jira field IDs and display names for a TLR issue; use this to discover HAZARD_CATEGORY_FIELD",
    )
    parser.add_argument(
        "--verify-only", action="store_true", default=False,
        help="Only test Jira credentials via /myself; do not fetch issue data",
    )
    parser.add_argument(
        "--api-version", metavar="VERSION", default=DEFAULT_API_VERSION,
        help=f"Jira REST API version to use (default: {DEFAULT_API_VERSION})",
    )
    parser.add_argument(
        "--env", metavar="FILE", default=None,
        help="Path to .env file containing Jira credentials (default: .env in CWD)",
    )
    parser.add_argument(
        "--output", "-o", metavar="FILE", default=None,
        help="Write results to this CSV file (default: print to stdout)",
    )
    args = parser.parse_args()

    if not args.verify_only and args.keys is None and args.table4 is None and args.list_fields is None:
        parser.error("one of --keys, --table4, --list-fields, or --verify-only is required")

    load_dotenv(dotenv_path=args.env)
    base_url = os.environ.get("JIRA_BASE_URL")
    username = os.environ.get("JIRA_USERNAME")
    api_token = os.environ.get("JIRA_API_TOKEN")

    missing = [
        name for name, val in [
            ("JIRA_BASE_URL", base_url),
            ("JIRA_USERNAME", username),
            ("JIRA_API_TOKEN", api_token),
        ] if val is None
    ]
    if missing:
        parser.error(f"Missing environment variables: {', '.join(missing)}")

    auth = (username, api_token)

    # Always verify connection first
    ok = verify_connection(base_url, auth, api_version=args.api_version)
    if not ok:
        print("ERROR: Jira authentication failed. Check credentials and base URL.")
        raise SystemExit(1)

    if args.verify_only:
        print("Jira connection verified successfully.")
        return

    if args.list_fields:
        fields_map = list_issue_fields(base_url, auth, args.list_fields, api_version=args.api_version)
        if fields_map is None:
            print(f"ERROR: Could not fetch fields for {args.list_fields}.")
            raise SystemExit(1)
        print(f"\nField ID → Display Name for {args.list_fields}:")
        print(f"  {'FIELD ID':<35}  DISPLAY NAME")
        print(f"  {'-'*35}  {'-'*35}")
        for field_id, display_name in sorted(fields_map.items(), key=lambda x: x[1].lower()):
            print(f"  {field_id:<35}  {display_name}")
        print(f"\nUpdate HAZARD_CATEGORY_FIELD in jira_risk_probabilities.py with the ID for 'Hazard Category'.")
        return

    if args.table4:
        df_t4 = pd.read_csv(args.table4)
        raw_keys = df_t4["TLR Key"].dropna().tolist()
    else:
        raw_keys = args.keys

    df = fetch_probabilities(raw_keys, env_path=args.env, api_version=args.api_version)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows to {args.output}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
