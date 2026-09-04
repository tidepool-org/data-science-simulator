#!/usr/bin/env python3
"""
Consensus Risk List Generator

Generates the consensus approval list for the Clinical and Senior Management
Risk Review (TWI-0025).  Queries Jira for TLR Risk issues matching the consensus
criteria:

  * project       = TLR
  * issuetype     = Risk
  * fixVersion    = "Tidepool Loop 2.0" (configurable)
  * post-mitigation probability (boja_prop_issue.risk_residual_probability_value) = 1
  * status NOT IN (closed, deprecated)
  * components NOT IN (not_Apex, ExcludeFromReport, backend)

and writes a single-sheet Excel file with hyperlinked TLR keys, colour-coded
risk-score cells, and Risk Control Measure titles resolved from their reference
IDs.

Usage (standalone):
    python consensus_risk_list.py --output consensus_list.xlsx
    python consensus_risk_list.py --output consensus_list.xlsx --env /path/to/.env
    python consensus_risk_list.py --output consensus_list.xlsx --fix-version "Tidepool Loop 2.1"

Credentials are read from environment variables (or a .env file):
    JIRA_BASE_URL   - e.g. https://tidepool.atlassian.net
    JIRA_USERNAME   - Jira account email
    JIRA_API_TOKEN  - Jira API token
"""

__author__ = "Shawn Foster"

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from jira_risk_probabilities import (
    DEFAULT_API_VERSION,
    PROPERTY_KEY,
    fetch_issue_property,
    get_nested_property,
    verify_connection,
)
from algorithm_risk_comparison import (
    _format_risk_score_cell,
    get_acceptability,
    parse_probability,
)

logger = logging.getLogger(__name__)


# ─── Jira field IDs ──────────────────────────────────────────────────────────
# From reports/config/inputs/jira/fields.yml and `--list-fields` output.

FIELD_SUMMARY = "summary"
FIELD_STATUS = "status"
FIELD_COMPONENTS = "components"
FIELD_HARM = "customfield_10739"            # RiskRegister "Harm"
FIELD_HAZARD_CATEGORY = "customfield_10095"  # "Hazard Category"
FIELD_RCM = "customfield_10097"             # "Risk Control Measure"

# ─── boja_prop_issue keys ────────────────────────────────────────────────────
# Note: in Tidepool's Jira schema "impact" == severity and "severity" == risk
# score.  See reports/config/inputs/jira/properties.yml for the full mapping.

PROP_PRE_SEVERITY = "risk_impact_value"
PROP_PRE_PROBABILITY = "risk_probability_value"
PROP_POST_SEVERITY = "risk_residual_impact_value"
PROP_POST_PROBABILITY = "risk_residual_probability_value"

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_FIX_VERSION = "Tidepool Loop 2.0"
DEFAULT_EXCLUDE_COMPONENTS: Tuple[str, ...] = (
    "not_Apex",
    "ExcludeFromReport",
    "backend",
)
DEFAULT_EXCLUDE_STATUS: Tuple[str, ...] = (
    "Closed",
    "Deprecated",
    "Clinical Review"
)
DEFAULT_RESIDUAL_PROBABILITY = "1"  # Negligible
DEFAULT_PAGE_SIZE = 100


# ─── Risk Control Measure reference ID → title mapping ───────────────────────
# Source: "Risk Control Measures Map, Tidepool Loop 2.0" (RCM Map document).

RCM_TITLES: Dict[str, str] = {
    "A":   "Algorithm limits",
    "AC":  "Accessibility",
    "CE":  "CGM error states have alerts and alarms",
    "D":   "Design",
    "DEL": "Tidepool Loop will not deliver automated insulin without user instruction",
    "G":   "Guardrails set by Tidepool limit settings to reduce inappropriate values",
    "GLU": "CGM glucose level alerts & alarms repeat every 5 min until acknowledged",
    "HF":  "Risk-critical UI design is HF-validated",
    "IA":  "iAGC error states have alerts and alarms",
    "IFU": "Instructions for use",
    "LNL": ("Time-sensitive Loop Not Looping alerts at 20, 40 minutes from last "
            "successful cycle; Critical Loop Not Looping alerts at 60, 120 minutes "
            "from last successful cycle"),
    "PE":  "Pump error states have alerts and alarms",
    "PM":  "Partner mitigation",
    "PV":  ("HCP defines settings in the prescription and therapy settings are "
            "pre-filled with the HCP recommended values"),
    "Q":   ("Tidepool's rigorous QMS detects and prevents release of issues "
            "likely to cause hazardous situations to occur"),
    "REV": ("Tidepool Loop reevaluates the user's insulin needs every 5 minutes "
            "& reduces or suspends insulin to prevent hypoglycemia & increase "
            "insulin to treat/prevent hyperglycemia"),
    "RX":  "Prescription only",
    "SB":  "Scheduled basal continues as a failsafe",
    "SEC": "Security protocols with connected devices",
    "UI":  "Prominent and persistent UI for error states",
    "UP":  ("Updates available to installed apps and notification to update if "
            "necessary for safety"),
}


# ─── JQL construction ────────────────────────────────────────────────────────

def build_consensus_jql(
    fix_version: str = DEFAULT_FIX_VERSION,
    exclude_components: Tuple[str, ...] = DEFAULT_EXCLUDE_COMPONENTS,
    exclude_status: Tuple[str, ...] = DEFAULT_EXCLUDE_STATUS,
    residual_probability: str = DEFAULT_RESIDUAL_PROBABILITY,
) -> str:
    """
    Build the JQL query for the consensus approval list.

    Filters to TLR Risk issues in the given fix version whose post-mitigation
    probability (``boja_prop_issue.risk_residual_probability_value``) equals
    *residual_probability*, whose status is not in *exclude_status*, and whose
    components do not include any of *exclude_components*.  Issues with no
    components at all are included (``component IS EMPTY``) so that
    unclassified risks are not silently dropped.  Status is always non-empty
    in Jira, so no corresponding ``status IS EMPTY`` clause is needed.

    Args:
        fix_version: Release version to filter on, e.g. "Tidepool Loop 2.0".
        exclude_components: Component names to exclude.
        exclude_status: Issue status values to exclude (e.g. "Closed",
            "Deprecated").
        residual_probability: Required ``boja_prop_issue`` residual probability
            (stored in Jira as a string, e.g. "1").

    Returns:
        JQL query string.
    """
    components_list = ", ".join(f'"{c}"' for c in exclude_components)
    status_list = ", ".join(f'"{s}"' for s in exclude_status)
    return (
        f'project = TLR '
        f'AND issuetype = Risk '
        f'AND fixVersion = "{fix_version}" '
        f'AND "issue.property[boja_prop_issue].risk_residual_probability_value" = '
        f'"{residual_probability}" '
        f'AND status NOT IN ({status_list}) '
        f'AND (component IS EMPTY OR component NOT IN ({components_list})) '
        f'ORDER BY issuekey ASC'
    )


# ─── JQL search (paginated) ──────────────────────────────────────────────────

#: Jira Cloud search endpoint (post-migration per Atlassian CHANGE-2046).
#: The legacy ``/rest/api/{version}/search`` endpoint was removed in 2025 and
#: now returns HTTP 410.  The replacement is cursor-paginated (no ``startAt``,
#: no ``total``) and lives only under API version 3.
SEARCH_ENDPOINT_PATH = "/rest/api/3/search/jql"


def search_issues(
    base_url: str,
    auth: Tuple[str, str],
    jql: str,
    api_version: str = DEFAULT_API_VERSION,  # retained for interface parity; search always uses v3
    fields: Optional[List[str]] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Execute a cursor-paginated JQL search and return all matching issues.

    Uses ``GET /rest/api/3/search/jql`` with ``nextPageToken`` / ``maxResults``
    pagination, per Atlassian's post-migration API
    (https://developer.atlassian.com/changelog/#CHANGE-2046).  Retries up to
    *max_retries* times on HTTP 429 (rate-limited) responses with exponential
    back-off starting at *backoff_seconds*.

    The new endpoint does NOT return a ``total`` count; pagination terminates
    when the response omits ``nextPageToken`` or includes ``isLast: true``.

    Args:
        base_url: Jira instance root URL.
        auth: (username, api_token) tuple for HTTP Basic Auth.
        jql: JQL query string.
        api_version: Kept for API parity with other functions in this module;
            the search endpoint is hard-coded to v3 because the legacy v2/v3
            ``/search`` routes were removed.
        fields: Optional list of field IDs to request.  Defaults to
            [summary, components, harm, hazard_category, RCM].
        page_size: Issues per page (capped by Jira Cloud; typical max 100).
        max_retries: Maximum retry attempts on rate-limit errors.
        backoff_seconds: Initial back-off duration (doubles each retry).

    Returns:
        List of raw issue dicts as returned by the Jira API.  Empty list if
        no matches.

    Raises:
        requests.HTTPError: On any non-retryable HTTP failure.
    """
    del api_version  # intentionally unused; see docstring
    if fields is None:
        fields = [
            FIELD_SUMMARY, FIELD_STATUS, FIELD_COMPONENTS,
            FIELD_HARM, FIELD_HAZARD_CATEGORY, FIELD_RCM,
        ]
    url = f"{base_url.rstrip('/')}{SEARCH_ENDPOINT_PATH}"
    all_issues: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None
    page_num = 0

    while True:
        page_num += 1
        params: Dict[str, Any] = {
            "jql": jql,
            "maxResults": page_size,
            "fields": ",".join(fields),
        }
        if next_page_token is not None:
            params["nextPageToken"] = next_page_token

        delay = backoff_seconds

        for attempt in range(max_retries):
            logger.debug(
                "Search page %d: maxResults=%d (attempt %d/%d)",
                page_num, page_size, attempt + 1, max_retries,
            )
            try:
                response = requests.get(url, auth=auth, params=params, timeout=30)
            except requests.RequestException as exc:
                logger.error("Search request to %s failed: %s", url, exc)
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

            if response.status_code == 200:
                break

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", delay))
                logger.warning(
                    "Search rate-limited (attempt %d/%d), retrying in %ds",
                    attempt + 1, max_retries, retry_after,
                )
                time.sleep(retry_after)
                delay *= 2
                continue

            logger.error(
                "Search: HTTP %d from %s\n  Response: %s",
                response.status_code, url, response.text[:500],
            )
            response.raise_for_status()
        else:
            raise RuntimeError(
                f"Search exhausted {max_retries} retries at page {page_num}"
            )

        data = response.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)
        next_page_token = data.get("nextPageToken")
        # Atlassian sometimes returns 'isLast'; be defensive and treat absence
        # of nextPageToken as the terminal condition either way.
        is_last = data.get("isLast", next_page_token is None)

        logger.info(
            "Fetched %d issues on page %d (cumulative %d); %s",
            len(issues), page_num, len(all_issues),
            "last page" if is_last else "more pages follow",
        )

        if is_last or not next_page_token or not issues:
            break

    return all_issues


# ─── Field helpers ───────────────────────────────────────────────────────────

def _extract_string_field(raw: Any) -> Optional[str]:
    """
    Normalise a Jira field value to a plain string.

    Handles bare strings, select/option dicts (``{"value": "..."}``),
    user-like dicts (``{"name": "..."}``) and ``None``.

    Args:
        raw: Raw field value from the Jira issue ``fields`` dict.

    Returns:
        String value, or ``None`` if the input is ``None``.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("value") or raw.get("name")
    return str(raw)


def parse_rcm_field(raw_value: Any) -> List[str]:
    """
    Parse a comma-separated Risk Control Measure field into reference IDs.

    Splits on commas, strips whitespace, uppercases each token, and drops
    empty entries.  Non-string inputs are coerced via ``str()``.

    Args:
        raw_value: Raw RCM field value (string, None, or anything stringifiable).

    Returns:
        List of uppercase reference ID strings (possibly empty).

    Examples:
        >>> parse_rcm_field("A, CE, D")
        ['A', 'CE', 'D']
        >>> parse_rcm_field("  a ,  GLU  ")
        ['A', 'GLU']
        >>> parse_rcm_field(None)
        []
    """
    if raw_value is None:
        return []
    if not isinstance(raw_value, str):
        raw_value = str(raw_value)
    return [token.strip().upper() for token in raw_value.split(",") if token.strip()]


def lookup_rcm_titles(ref_ids: List[str]) -> List[str]:
    """
    Map Risk Control Measure reference IDs to their human-readable titles.

    Unknown reference IDs are logged as warnings and returned as
    ``"UNKNOWN ({id})"`` so they remain visible in the output rather than
    being silently dropped.

    Args:
        ref_ids: List of uppercase reference ID strings.

    Returns:
        List of title strings in the same order as the input.
    """
    titles: List[str] = []
    for ref_id in ref_ids:
        title = RCM_TITLES.get(ref_id)
        if title is None:
            logger.warning("Unknown RCM reference ID: %s", ref_id)
            titles.append(f"UNKNOWN ({ref_id})")
        else:
            titles.append(title)
    return titles


# ─── Row assembly ────────────────────────────────────────────────────────────

def _compute_risk_score(
    severity: Optional[float], probability: Optional[float]
) -> Optional[float]:
    """
    Multiply severity × probability, propagating ``None`` on missing inputs.

    Args:
        severity: Integer severity score (as float) or None.
        probability: Integer probability score (as float) or None.

    Returns:
        Float risk score, or ``None`` if either input is ``None``.
    """
    if severity is None or probability is None:
        return None
    return float(severity) * float(probability)


def _as_int_or_none(value: Optional[float]) -> Optional[int]:
    """
    Convert a numeric value to int when safe, otherwise return ``None``.

    Used for severity/probability cells so the Excel shows e.g. ``3`` not ``3.0``.

    Args:
        value: Numeric value or None.

    Returns:
        Integer value, or None if input is None / NaN.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_consensus_row(
    issue: Dict[str, Any],
    prop_block: Optional[Dict[str, Any]],
    base_url: str,
) -> Dict[str, Any]:
    """
    Assemble a single consensus-list row from a Jira issue + its property block.

    Computes pre- and post-mitigation risk scores as severity × probability
    and derives acceptability labels/colours using ``get_acceptability`` so
    the output is consistent with the hazardous situation comparison report.

    The returned dict contains both visible column values and two private
    ``_pre_color`` / ``_post_color`` hex strings used by the Excel writer.
    Unknown/missing property values yield ``None`` cells and an "Unknown"
    acceptability (white fill).

    Args:
        issue: Raw issue dict from the Jira search API.
        prop_block: boja_prop_issue value dict (from ``fetch_issue_property``),
            or ``None`` if the property is missing.
        base_url: Jira root URL, used to build the issue browse URL.

    Returns:
        Dict with the row's column values plus ``URL``, ``_pre_color``, and
        ``_post_color`` keys.
    """
    key = issue.get("key", "")
    fields = issue.get("fields", {}) or {}
    url = f"{base_url.rstrip('/')}/browse/{key}"

    summary = _extract_string_field(fields.get(FIELD_SUMMARY)) or ""
    status = _extract_string_field(fields.get(FIELD_STATUS)) or ""
    harm = _extract_string_field(fields.get(FIELD_HARM)) or ""
    hazard_cat = _extract_string_field(fields.get(FIELD_HAZARD_CATEGORY)) or ""
    rcm_raw = _extract_string_field(fields.get(FIELD_RCM))

    pre_sev = parse_probability(get_nested_property(prop_block, PROP_PRE_SEVERITY))
    pre_prob = parse_probability(get_nested_property(prop_block, PROP_PRE_PROBABILITY))
    post_sev = parse_probability(get_nested_property(prop_block, PROP_POST_SEVERITY))
    post_prob = parse_probability(get_nested_property(prop_block, PROP_POST_PROBABILITY))

    pre_score = _compute_risk_score(pre_sev, pre_prob)
    post_score = _compute_risk_score(post_sev, post_prob)

    pre_sev_for_label = int(pre_sev) if pre_sev is not None else 0
    post_sev_for_label = int(post_sev) if post_sev is not None else 0
    pre_label, pre_color = get_acceptability(pre_score, pre_sev_for_label)
    post_label, post_color = get_acceptability(post_score, post_sev_for_label)

    rcm_ids = parse_rcm_field(rcm_raw)
    rcm_titles = lookup_rcm_titles(rcm_ids)

    return {
        "TLR Key": key,
        "URL": url,
        "Summary": summary,
        "Status": status,
        "Hazard Category": hazard_cat,
        "Harm": harm,
        "Pre-mitigation Severity": _as_int_or_none(pre_sev),
        "Pre-mitigation Probability": _as_int_or_none(pre_prob),
        "Pre-mitigation Risk Score": _format_risk_score_cell(pre_score, pre_label),
        "_pre_color": pre_color,
        "Post-mitigation Severity": _as_int_or_none(post_sev),
        "Post-mitigation Probability": _as_int_or_none(post_prob),
        "Post-mitigation Risk Score": _format_risk_score_cell(post_score, post_label),
        "_post_color": post_color,
        "Risk Control Measures": "\n".join(rcm_titles) if rcm_titles else "",
    }


# ─── Excel output ────────────────────────────────────────────────────────────

HEADERS: List[str] = [
    "TLR Key",
    "Summary",
    "Status",
    "Hazard Category",
    "Harm",
    "Pre-mitigation Severity",
    "Pre-mitigation Probability",
    "Pre-mitigation Risk Score",
    "Post-mitigation Severity",
    "Post-mitigation Probability",
    "Post-mitigation Risk Score",
    "Risk Control Measures",
]

COLUMN_WIDTHS: Dict[str, int] = {
    "TLR Key": 12,
    "Summary": 48,
    "Status": 40,
    "Hazard Category": 22,
    "Harm": 28,
    "Pre-mitigation Severity": 14,
    "Pre-mitigation Probability": 14,
    "Pre-mitigation Risk Score": 28,
    "Post-mitigation Severity": 14,
    "Post-mitigation Probability": 14,
    "Post-mitigation Risk Score": 28,
    "Risk Control Measures": 60,
}


def write_consensus_excel(rows: List[Dict[str, Any]], output_path: str) -> str:
    """
    Write the consensus list rows to an Excel workbook.

    Output has a single sheet ("Consensus List") with:
      * Hyperlinked TLR Key cells (browse URL).
      * Colour-coded Pre-/Post-mitigation Risk Score cells
        (green = Acceptable, yellow = Conditionally acceptable, red =
        Unacceptable, white = Unknown).
      * Wrapped text in the Risk Control Measures column.
      * Frozen header row.

    Requires openpyxl (``pip install openpyxl``).

    Args:
        rows: List of row dicts from ``build_consensus_row``.
        output_path: Destination .xlsx file path.

    Returns:
        output_path (unchanged; returned for chaining / logging).
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Consensus List"

    # Header row
    ws.append(HEADERS)
    bold = Font(bold=True)
    for cell in ws[1]:
        cell.font = bold
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    idx_tlr = HEADERS.index("TLR Key") + 1
    idx_pre_score = HEADERS.index("Pre-mitigation Risk Score") + 1
    idx_post_score = HEADERS.index("Post-mitigation Risk Score") + 1
    idx_rcm = HEADERS.index("Risk Control Measures") + 1
    hyperlink_font = Font(color="0563C1", underline="single")

    for row in rows:
        excel_row = [row[h] for h in HEADERS]
        ws.append(excel_row)
        r = ws.max_row

        tlr_cell = ws.cell(row=r, column=idx_tlr)
        tlr_cell.hyperlink = row["URL"]
        tlr_cell.font = hyperlink_font

        pre_color = row["_pre_color"]
        ws.cell(row=r, column=idx_pre_score).fill = PatternFill(
            start_color=pre_color, end_color=pre_color, fill_type="solid"
        )
        post_color = row["_post_color"]
        ws.cell(row=r, column=idx_post_score).fill = PatternFill(
            start_color=post_color, end_color=post_color, fill_type="solid"
        )

        ws.cell(row=r, column=idx_rcm).alignment = Alignment(
            wrap_text=True, vertical="top"
        )

    for idx, header in enumerate(HEADERS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = COLUMN_WIDTHS.get(header, 18)

    ws.freeze_panes = "A2"
    wb.save(output_path)
    return output_path


# ─── Public entry point ──────────────────────────────────────────────────────

def generate_consensus_list(
    output_path: str,
    env_path: Optional[str] = None,
    fix_version: str = DEFAULT_FIX_VERSION,
    exclude_status: Tuple[str, ...] = DEFAULT_EXCLUDE_STATUS,
    exclude_components: Tuple[str, ...] = DEFAULT_EXCLUDE_COMPONENTS,
    residual_probability: str = DEFAULT_RESIDUAL_PROBABILITY,
    api_version: str = DEFAULT_API_VERSION,
) -> str:
    """
    End-to-end: query Jira, enrich with property values, write Excel.

    Loads credentials from environment variables, verifies the connection,
    runs the consensus JQL, fetches ``boja_prop_issue`` for each matching
    issue, and writes a colour-coded single-sheet Excel workbook.

    Args:
        output_path: Destination .xlsx file path.
        env_path: Optional path to a ``.env`` credentials file.
        fix_version: Jira fix version to filter on.
        exclude_status: List of statuses to exclude from the consensus list.
        exclude_components: Component names to exclude.
        residual_probability: Required post-mitigation probability value
            (default "1" = Negligible).
        api_version: Jira REST API version string.

    Returns:
        Path to the written Excel file.

    Raises:
        EnvironmentError: If any required credential variable is missing.
        ConnectionError: If the Jira /myself credential check fails.
    """
    load_dotenv(dotenv_path=env_path)
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
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    auth = (username, api_token)
    if not verify_connection(base_url, auth, api_version=api_version):
        raise ConnectionError(
            f"Could not authenticate with Jira at {base_url}. "
            "Check JIRA_USERNAME and JIRA_API_TOKEN."
        )

    jql = build_consensus_jql(
        fix_version=fix_version,
        exclude_components=exclude_components,
        exclude_status=exclude_status,
        residual_probability=residual_probability,
    )
    logger.info("JQL: %s", jql)

    issues = search_issues(base_url, auth, jql, api_version=api_version)
    logger.info("Building consensus list for %d issues", len(issues))

    rows: List[Dict[str, Any]] = []
    for issue in issues:
        key = issue.get("key", "")
        prop_block = fetch_issue_property(base_url, auth, key, api_version=api_version)
        if prop_block is None:
            logger.warning("%s: boja_prop_issue missing; severity/probability cells will be empty", key)
        rows.append(build_consensus_row(issue, prop_block, base_url))

    return write_consensus_excel(rows, output_path)


# ─── Standalone entry point ──────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Generate the consensus approval list (Excel) for the Clinical and "
            "Senior Management Risk Review (TWI-0025)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python consensus_risk_list.py --output consensus_list.xlsx
    python consensus_risk_list.py --output consensus_list.xlsx --env /path/to/.env
    python consensus_risk_list.py --output consensus_list.xlsx --fix-version "Tidepool Loop 2.1"
    python consensus_risk_list.py --output consensus_list.xlsx \\
        --exclude-components not_Apex ExcludeFromReport backend
        """,
    )
    parser.add_argument(
        "--output", "-o", required=True, metavar="FILE",
        help="Path to output .xlsx file",
    )
    parser.add_argument(
        "--env", metavar="FILE", default=None,
        help="Path to .env file containing Jira credentials (default: .env in CWD)",
    )
    parser.add_argument(
        "--fix-version", default=DEFAULT_FIX_VERSION,
        help=f"Jira fix version to filter on (default: {DEFAULT_FIX_VERSION!r})",
    )
    parser.add_argument(
        "--exclude-status", nargs="+",
        default=list(DEFAULT_EXCLUDE_STATUS), metavar="STATUS",
        help=(
            "Issue statuses to exclude "
            f"(default: {' '.join(DEFAULT_EXCLUDE_STATUS)})"
        ),
    )
    parser.add_argument(
        "--exclude-components", nargs="+",
        default=list(DEFAULT_EXCLUDE_COMPONENTS), metavar="COMPONENT",
        help=(
            "Component names to exclude "
            f"(default: {' '.join(DEFAULT_EXCLUDE_COMPONENTS)})"
        ),
    )
    parser.add_argument(
        "--residual-probability", default=DEFAULT_RESIDUAL_PROBABILITY,
        help=(
            f"Required boja_prop_issue.risk_residual_probability_value "
            f"(default: {DEFAULT_RESIDUAL_PROBABILITY!r} = Negligible)"
        ),
    )
    parser.add_argument(
        "--api-version", default=DEFAULT_API_VERSION,
        help=f"Jira REST API version (default: {DEFAULT_API_VERSION})",
    )
    args = parser.parse_args()

    try:
        path = generate_consensus_list(
            output_path=args.output,
            env_path=args.env,
            fix_version=args.fix_version,
            exclude_status=tuple(args.exclude_status),
            exclude_components=tuple(args.exclude_components),
            residual_probability=args.residual_probability,
            api_version=args.api_version,
        )
    except EnvironmentError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except ConnectionError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Consensus list written to: {path}")


if __name__ == "__main__":
    main()
