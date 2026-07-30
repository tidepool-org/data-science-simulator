"""
Unit tests for severity_model.py

Covers the SOP-facing logic that the RTF/GUI both depend on:
  - determine_harm_and_severity truth table (baseline / hypo / DKA / hyper, tie-breaks)
  - calculate_hyperglycemia_score including the SOP-correct 0-only-if-TAR-truly-0 case
  - catastrophic threshold logic (check_consecutive_low_values boundaries)
  - stage classification via prefixes
  - SeverityAssessment.to_dict() JSON round-trip

round_half_up / calculate_integer_averages already have coverage in
test_create_severity_summary.py (which imports them via the re-export); not
duplicated here.
"""

import json
import os

import pytest

from severity_model import (
    build_assessment,
    determine_harm_and_severity,
    calculate_hyperglycemia_score,
    check_consecutive_low_values,
    classify_sim_id,
    StageResult,
    CatastrophicFinding,
    OutlierFinding,
    SeverityAssessment,
)


class TestDetermineHarmAndSeverity:
    def test_all_zero_is_baseline(self):
        assert determine_harm_and_severity(0, 0, 0) == ("Severity = baseline", "0")

    def test_hyperglycemia_when_lbgi_and_dka_low(self):
        # lbgi<=1, dka==0 -> hyperglycemia, carrying the hyper score
        assert determine_harm_and_severity(0, 0, 2) == ("Hyperglycemia", "2")
        assert determine_harm_and_severity(1, 0, 1) == ("Hyperglycemia", "1")

    def test_hypoglycemia_straight(self):
        assert determine_harm_and_severity(3, 1, 0) == ("Hypoglycemia", "3")

    def test_lbgi_wins_tie_over_dka(self):
        # lbgi == dka -> LBGI (Hypoglycemia) wins
        assert determine_harm_and_severity(2, 2, 0) == ("Hypoglycemia", "2")

    def test_dka_when_greater(self):
        assert determine_harm_and_severity(1, 3, 0) == ("DKA", "3")

    def test_lbgi_1_dka_nonzero_is_hypo_not_hyper(self):
        # lbgi<=1 but dka!=0 -> falls through to the lbgi>=dka / DKA logic,
        # NOT hyperglycemia. Here dka=2 > lbgi=1 -> DKA.
        assert determine_harm_and_severity(1, 2, 1) == ("DKA", "2")


class TestHyperglycemiaScore:
    def test_true_zero_is_zero(self):
        # SOP-correct main path: 0 only if TAR truly 0.
        assert calculate_hyperglycemia_score("0.0") == 0

    def test_below_12_is_one(self):
        assert calculate_hyperglycemia_score("5.5") == 1

    def test_at_12_is_two(self):
        assert calculate_hyperglycemia_score("12.0") == 2

    def test_above_12_is_two(self):
        assert calculate_hyperglycemia_score("30.0") == 2

    def test_na_is_one(self):
        # preserved original behavior: no data -> 1
        assert calculate_hyperglycemia_score("NA") == 1


class TestConsecutiveLowValues:
    def test_exactly_48_triggers(self):
        assert check_consecutive_low_values([40] * 48) is True

    def test_47_does_not_trigger(self):
        assert check_consecutive_low_values([40] * 47) is False

    def test_49_triggers(self):
        assert check_consecutive_low_values([40] * 49) is True

    def test_broken_run_resets(self):
        # 40 lows, one normal, 40 lows -> longest run 40 < 48 -> False
        series = [40] * 40 + [120] + [40] * 40
        assert check_consecutive_low_values(series) is False

    def test_value_above_threshold_not_counted(self):
        assert check_consecutive_low_values([41] * 60) is False

    def test_boundary_40_is_included(self):
        # threshold is <=40, so 40 counts
        assert check_consecutive_low_values([40] * 48) is True


class TestClassifySimId:
    def test_pre_variants(self):
        assert classify_sim_id("pre-Loop_NoMitigations_t1_median") == "pre"
        assert classify_sim_id("pre-LoopNoMitigations_x") == "pre"

    def test_no_loop_variants(self):
        assert classify_sim_id("pre-noLoop_t1") == "no_loop"
        assert classify_sim_id("pre-NoLoop_t1") == "no_loop"

    def test_post_variants(self):
        assert classify_sim_id("post-Loop-WithMitigations_t1") == "post"
        assert classify_sim_id("post-Loop_WithMitigations_t1") == "post"

    def test_unmatched_returns_none(self):
        assert classify_sim_id("something_else") is None


class TestToDictRoundTrip:
    def _make_assessment(self):
        stages = {
            'pre': StageResult('pre', 'Hypoglycemia', '4', '78.0', '4.5', '17.5', 4, 1, 2, 2),
            'no_loop': StageResult('no_loop', 'Hypoglycemia', '3', '69.0', '3.5', '27.5', 3, 2, 2, 2),
            'post': StageResult('post', 'Hyperglycemia', '1', '94.5', '0.0', '5.5', 1, 0, 1, 2),
        }
        return SeverityAssessment(
            simulation_id='TLR-TEST',
            subdirectory_name='TLR-TEST',
            timestamp='2026-06-09T15:23:25.050187',
            profile_count=2,
            stages=stages,
            catastrophic_findings=[
                CatastrophicFinding('pre-Loop_NoMitigations_t1_sensitive', 'pre', 'extended_low', 5),
            ],
            outlier_findings=[
                OutlierFinding('pre', 'sensitive', 'Hypoglycemia', 4.0, 2.0),
            ],
            outlier_status='ok',
        )

    def test_to_dict_is_json_serializable(self):
        d = self._make_assessment().to_dict()
        s = json.dumps(d)              # must not raise
        back = json.loads(s)
        assert back['simulation_id'] == 'TLR-TEST'
        assert back['stages']['pre']['harm_type'] == 'Hypoglycemia'
        assert back['catastrophic_findings'][0]['updated_severity'] == 5
        assert back['outlier_findings'][0]['profile'] == 'sensitive'
        assert back['outlier_status'] == 'ok'

    def test_stage_keys_present(self):
        d = self._make_assessment().to_dict()
        assert set(d['stages'].keys()) == {'pre', 'no_loop', 'post'}


# Columns build_assessment reads; kept together so the fixtures stay valid if the
# extraction set changes. 'lbgi'/'dka_index' are the raw values the new
# *_value_avg fields average; the *_risk_score columns feed the 0-4 scores.
_SUMMARY_COLUMNS = [
    "sim_id", "percent_values_ge_70_le_180", "percent_cgm_lt_54",
    "percent_cgm_gt_180", "lbgi_risk_score", "dka_risk_score", "lbgi", "dka_index",
]
# One row per stage; two profiles (below) so the averages exercise real division.
# lbgi_risk_score kept < 4 so the catastrophic (4->5) path (which reads per-sim
# time-series files this fixture doesn't create) is never entered.
_PROFILE_A_ROWS = [
    # sim_id,                              tir,  tbr, tar, lbgi_s, dka_s, lbgi, dka_index
    ("pre-Loop_NoMitigations_t1_median",  78.0, 4.5, 17.5, 3, 1, 2.0, 20.0),
    ("pre-noLoop_t1_median",              69.0, 3.5, 27.5, 3, 2, 4.0, 30.0),
    ("post-Loop_WithMitigations_t1_median", 94.5, 0.0, 5.5, 1, 0, 1.0, 10.0),
]
_PROFILE_B_ROWS = [
    ("pre-Loop_NoMitigations_t1_median",  80.0, 4.0, 16.0, 3, 1, 3.0, 22.0),
    ("pre-noLoop_t1_median",              70.0, 3.0, 26.0, 3, 2, 5.0, 28.0),
    ("post-Loop_WithMitigations_t1_median", 95.0, 0.0, 5.0, 1, 0, 1.0, 12.0),
]


def _write_summary_csv(directory, profile, rows, columns=_SUMMARY_COLUMNS):
    path = os.path.join(
        directory,
        f"summary_results_Simulation-Configuration-TLR-999-test_{profile}_profile.csv",
    )
    with open(path, "w") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            # Drop trailing columns if a variant fixture omits them (e.g. no lbgi).
            fh.write(",".join(str(v) for v in row[: len(columns)]) + "\n")
    return path


class TestBuildAssessmentValueFields:
    """The new raw-value fields (lbgi_value_avg / dka_index_value_avg) are
    averaged from the summary 'lbgi'/'dka_index' columns, 1 dp as a string,
    mirroring tir/tbr/tar -- and degrade to 'NA' when the column is absent."""

    def test_value_fields_are_averaged_from_raw_columns(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)

        assessment = build_assessment(tlr, "2026-07-29T00:00:00")
        assert assessment is not None

        pre = assessment.stages["pre"]
        # Raw VALUES: (2.0+3.0)/2 = 2.5 ; (20.0+22.0)/2 = 21.0  -- distinct from
        # the risk SCORE (still the integer 3), proving they are separate fields.
        assert pre.lbgi_value_avg == "2.5"
        assert pre.dka_index_value_avg == "21.0"
        assert pre.lbgi_score_avg == 3

        assert assessment.stages["no_loop"].lbgi_value_avg == "4.5"
        assert assessment.stages["post"].dka_index_value_avg == "11.0"

    def test_value_fields_are_na_when_columns_absent(self, tmp_path):
        tlr = str(tmp_path)
        # Same fixtures but without the trailing lbgi/dka_index columns.
        cols = _SUMMARY_COLUMNS[:-2]
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS, columns=cols)

        assessment = build_assessment(tlr, "2026-07-29T00:00:00")
        assert assessment is not None
        for stage in ("pre", "no_loop", "post"):
            assert assessment.stages[stage].lbgi_value_avg == "NA"
            assert assessment.stages[stage].dka_index_value_avg == "NA"

    def test_stageresult_value_fields_default_to_na(self):
        """Positional constructors that predate these fields still work."""
        sr = StageResult("pre", "Hypoglycemia", "3", "78.0", "4.5", "17.5", 3, 1, 2, 2)
        assert sr.lbgi_value_avg == "NA"
        assert sr.dka_index_value_avg == "NA"
