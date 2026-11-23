"""
TRSET-82 unit tests: delivery risk type classification, the shared harm rules, the
untruncated-DKAI stage verdict, outlier detection under each risk type, the RTF note
and the structured-object fields. The end-to-end tests on real results directories
are in test_delivery_risk_type_integration.py.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import severity_model  # noqa: E402
from create_severity_summary import (  # noqa: E402
    render_catastrophic_identifier,
    render_delivery_risk_note,
    render_rtf,
)
from severity_model import (  # noqa: E402
    CatastrophicFinding,
    SeverityAssessment,
    StageResult,
    build_assessment,
    detect_outliers,
    determine_delivery_risk_type,
    determine_harm_and_severity,
)
from test_severity_model import (  # noqa: E402
    _PROFILE_A_ROWS,
    _write_outlier_profile,
    _write_summary_csv,
)

OVER, UNDER, UNC = 'overdelivery', 'underdelivery', 'unconstrained'


# --- harm rules (one row per line of the request's table) ------------------------

@pytest.mark.parametrize('risk_type, lbgi, dka, hyper, dkai, expected', [
    (OVER, 0, 3, 2, 50.0, ('Severity = baseline', '0')),
    (OVER, 1, 0, 2, 0.0, ('Hypoglycemia', '1')),
    (OVER, 4, 4, 2, 30.0, ('Hypoglycemia', '4')),
    (UNDER, 4, 0, 2, 0.0, ('Hyperglycemia', '2')),
    (UNDER, 0, 0, 0, 0.0, ('Hyperglycemia', '0')),          # 0 allowed
    (UNDER, 4, 0, 1, 0.5, ('DKA', '0')),                    # DKAI > 0, score 0 allowed
    (UNDER, 0, 3, 1, 15.0, ('DKA', '3')),
    (UNC, 1, 0, 1, 0.0, ('Hyperglycemia', '1')),
    (UNC, 0, 0, 0, 0.0, ('Hyperglycemia', '0')),
    (UNC, 1, 0, 1, 1.9, ('DKA', '0')),                      # DKAI under 2 h, score 0
    (UNC, 1, 2, 1, 9.0, ('DKA', '2')),
    (UNC, 2, 1, 1, 3.0, ('Hypoglycemia', '2')),
    (UNC, 2, 2, 1, 9.0, ('Hypoglycemia', '2')),             # LBGI wins ties
    (UNC, 2, 3, 1, 15.0, ('DKA', '3')),
])
def test_harm_rule_table(risk_type, lbgi, dka, hyper, dkai, expected):
    assert determine_harm_and_severity(
        lbgi, dka, hyper, risk_type, dka_index=dkai) == expected


def test_three_argument_call_still_works_and_infers_dkai_from_the_score():
    assert determine_harm_and_severity(0, 0, 2) == ('Hyperglycemia', '2')
    assert determine_harm_and_severity(1, 1, 1) == ('DKA', '1')


# --- stage verdict keys on the UNtruncated stage-mean DKAI ----------------------

def test_stage_verdict_uses_untruncated_dkai(tmp_path):
    """Mean DKAI 0.004 displays as '0' (truncated) but is not 0: DKA, not Hyperglycemia."""
    tlr = str(tmp_path)
    rows = [
        ("pre-Loop_NoMitigations_t1_median", 80.0, 1.0, 20.0, 0, 0, 0.0, 0.004),
        ("pre-noLoop_t1_median", 80.0, 1.0, 20.0, 0, 0, 0.0, 0.004),
        ("post-Loop_WithMitigations_t1_median", 80.0, 1.0, 20.0, 0, 0, 0.0, 0.0),
    ]
    _write_summary_csv(tlr, 'median', rows)
    for row in rows:
        with open(os.path.join(tlr, f"{row[0]}.json"), 'w') as f:
            json.dump({'duration_hrs': 23.0}, f)
    stages = build_assessment(tlr, 'ts').stages
    assert stages['pre'].dka_index_value_avg == '0'          # display is truncated...
    assert stages['pre'].harm_type == 'DKA'                  # ...the verdict is not
    assert stages['post'].harm_type == 'Hyperglycemia'


# --- delivery risk type classification ------------------------------------------

def _sims(tmp_path, durations):
    ids = []
    for i, d in enumerate(durations):
        sim_id = f"pre-Loop_NoMitigations_t1_p{i}"
        ids.append(sim_id)
        if d is not None:
            body = {} if d == 'nokey' else {'duration_hrs': d}
            (tmp_path / f"{sim_id}.json").write_text(json.dumps(body))
    return ids


@pytest.mark.parametrize('durations, expected', [
    ([8.0, 8.0], (OVER, 8.0)),
    ([8, 8], (OVER, 8.0)),
    ([23.0, 23.0], (UNDER, 23.0)),
    ([12.0, 12.0], (UNC, 12.0)),
    ([8.5], (UNC, 8.5)),
    ([8.0, 23.0], (UNC, [8.0, 23.0])),
    ([23.0, 8.0, 12.0], (UNC, [8.0, 12.0, 23.0])),
    ([8.0, None], (UNC, None)),                # JSON missing
    ([8.0, 'nokey'], (UNC, None)),             # no duration_hrs
    (['x', 8.0], (UNC, None)),                 # non-numeric
    ([float('nan')], (UNC, None)),             # not finite
])
def test_delivery_risk_type(tmp_path, capsys, durations, expected):
    sim_ids = _sims(tmp_path, durations)
    assert determine_delivery_risk_type(str(tmp_path), sim_ids) == expected
    out = capsys.readouterr().out
    if expected[0] == UNC:
        assert 'Warning' in out and tmp_path.name in out
    else:
        assert out == ''


def test_corrupt_json_is_unconstrained_and_warned(tmp_path, capsys):
    (tmp_path / 'pre-x.json').write_text('{oops')
    assert determine_delivery_risk_type(str(tmp_path), ['pre-x']) == (UNC, None)
    assert 'pre-x.json unreadable' in capsys.readouterr().out


def test_no_sims_is_unconstrained(tmp_path):
    assert determine_delivery_risk_type(str(tmp_path), []) == (UNC, None)


# --- outlier detection ------------------------------------------------------------

def test_overdelivery_skips_the_zero_tar_hyperglycemia_check(tmp_path):
    tlr = str(tmp_path)
    _write_outlier_profile(tlr, 'zero', tar=0.0, lbgi=0, dka=0)
    _write_outlier_profile(tlr, 'highA', tar=20.0, lbgi=0, dka=0)
    _write_outlier_profile(tlr, 'highB', tar=25.0, lbgi=0, dka=0)
    assert any(f.harm_type == 'Hyperglycemia' for f in detect_outliers(str(tmp_path))[0])
    findings, status = detect_outliers(tlr, delivery_risk_type=OVER)
    assert status == 'ok' and findings == []


def test_outlier_profiles_use_the_shared_harm_rules(tmp_path):
    """Overdelivery: a profile with LBGI 1 is Hypoglycemia, so it is compared against
    its Hypoglycemia peers; unconstrained it would be Hyperglycemia (DKAI 0)."""
    tlr = str(tmp_path)
    _write_outlier_profile(tlr, 'a', tar=10.0, lbgi=5, dka=0)
    _write_outlier_profile(tlr, 'b', tar=10.0, lbgi=5, dka=0)
    _write_outlier_profile(tlr, 'c', tar=10.0, lbgi=1, dka=0)
    over = {(f.profile, f.harm_type) for f in detect_outliers(tlr, delivery_risk_type=OVER)[0]}
    unc = {(f.profile, f.harm_type) for f in detect_outliers(tlr)[0]}
    assert ('c', 'Hypoglycemia') in over
    assert ('c', 'Hypoglycemia') not in unc


def test_get_profile_metrics_reads_dka_index(tmp_path):
    _write_outlier_profile(str(tmp_path), 'a', tar=10.0, lbgi=1, dka=2)
    metrics = severity_model.get_profile_metrics(str(tmp_path))
    assert metrics.profiles['a']['pre']['dka_index'] == 10.0


# --- required column ---------------------------------------------------------------

def test_dka_index_is_a_required_summary_column():
    assert 'dka_index' in severity_model.REQUIRED_SUMMARY_COLUMNS


# --- structured object -------------------------------------------------------------

def test_assessment_fields_default_and_serialise():
    a = SeverityAssessment('TLR-1', 'TLR-1', 'ts', 1, {})
    assert a.delivery_risk_type == UNC and a.duration_hours is None
    d = a.to_dict()
    assert d['delivery_risk_type'] == UNC and d['duration_hours'] is None
    a.duration_hours = [8.0, 23.0]
    assert a.to_dict()['duration_hours'] == [8.0, 23.0]
    json.dumps(a.to_dict())


def test_assessment_built_from_a_directory_carries_the_type(tmp_path):
    tlr = str(tmp_path)
    _write_summary_csv(tlr, 'median', _PROFILE_A_ROWS)
    for row in _PROFILE_A_ROWS:
        (tmp_path / f"{row[0]}.json").write_text(json.dumps({'duration_hrs': 8}))
    a = build_assessment(tlr, 'ts')
    assert (a.delivery_risk_type, a.duration_hours) == (OVER, 8.0)


# --- rendering ---------------------------------------------------------------------

def test_note_text():
    p = "Harm not constrained by delivery risk type: "
    assert render_delivery_risk_note(OVER, 8.0) == ''
    assert render_delivery_risk_note(UNDER, 23.0) == ''
    assert render_delivery_risk_note(UNC, None) == p + "simulation duration was not recorded for this risk."
    assert render_delivery_risk_note(UNC, [8.0, 23.0]) == p + "simulations for this risk used mixed durations (8, 23 hours)."
    assert render_delivery_risk_note(UNC, 12.5) == (
        p + "simulation duration was 12.5 hours, which matches neither the 8-hour "
        "overdelivery nor the 23-hour underdelivery duration defined in RMP-0002.")


def test_catastrophic_section_not_relevant_only_for_underdelivery():
    finding = CatastrophicFinding('pre-x', 'pre', 'zero_or_negative', 5)
    not_relevant = "Critical/Catastrophic level determination not relevant to this scenario."
    assert render_catastrophic_identifier([finding], UNDER) == not_relevant
    assert render_catastrophic_identifier([finding], OVER) != not_relevant
    assert render_catastrophic_identifier([finding]) != not_relevant


def test_constrained_rtf_has_no_note_line(tmp_path):
    tlr = str(tmp_path)
    _write_summary_csv(tlr, 'median', _PROFILE_A_ROWS)
    for row in _PROFILE_A_ROWS:
        (tmp_path / f"{row[0]}.json").write_text(json.dumps({'duration_hrs': 23.0}))
    assert 'Harm not constrained' not in render_rtf(build_assessment(tlr, '2026-01-01T00:00:00'))


def test_stage_result_positional_constructor_unchanged():
    StageResult('pre', 'Hypoglycemia', '3', '78.0', '4.5', '17.5', 3, 1, 2, 2)
