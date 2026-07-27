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
import pytest

from tidepool_data_science_simulator.post_processing.severity_model import (
    determine_harm_and_severity,
    calculate_hyperglycemia_score,
    check_consecutive_low_values,
    classify_sim_id,
    truncate_2dp,
    calculate_truncated_averages,
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


class TestTruncate2dp:
    def test_truncates_toward_zero_not_rounds(self):
        # 21.917 must truncate to 21.91, NOT round to 21.92.
        assert truncate_2dp(21.917) == 21.91

    def test_third_decimal_dropped_even_when_large(self):
        assert truncate_2dp(3.149) == 3.14
        assert truncate_2dp(2.999) == 2.99

    def test_whole_number_unchanged(self):
        assert truncate_2dp(3.0) == 3.0
        assert truncate_2dp(0.0) == 0.0

    def test_already_two_decimals_unchanged(self):
        assert truncate_2dp(2.5) == 2.5
        assert truncate_2dp(3.14) == 3.14

    def test_distinct_from_round_half_up_behavior(self):
        # A value that would round UP but must truncate DOWN.
        assert truncate_2dp(1.999) == 1.99


class TestCalculateTruncatedAverages:
    def test_whole_number_result_has_no_decimal(self):
        # mean 3.0 -> "3", mean 0.0 -> "0"
        data = {'pre': [3.0, 3.0], 'no_loop': [0.0], 'post': [2.0, 4.0]}
        out = calculate_truncated_averages(data)
        assert out['pre'] == "3"
        assert out['no_loop'] == "0"
        assert out['post'] == "3"

    def test_trailing_zero_dropped(self):
        # mean 2.5 -> "2.5" (not "2.50", not "2.5" via str(float))
        data = {'pre': [2.0, 3.0], 'no_loop': [], 'post': []}
        assert calculate_truncated_averages(data)['pre'] == "2.5"

    def test_multi_decimal_truncation(self):
        # three identical 21.917 -> mean 21.917 -> truncate -> "21.91"
        data = {'pre': [21.917, 21.917, 21.917], 'no_loop': [], 'post': []}
        assert calculate_truncated_averages(data)['pre'] == "21.91"

    def test_fractional_mean_truncated_then_stripped(self):
        # mean of 3.14 and 3.16 = 3.15 -> "3.15"; mean 3.1/3.3 = 3.2 -> "3.2"
        data = {'pre': [3.14, 3.16], 'no_loop': [3.1, 3.3], 'post': []}
        out = calculate_truncated_averages(data)
        assert out['pre'] == "3.15"
        assert out['no_loop'] == "3.2"

    def test_empty_stage_is_na(self):
        data = {'pre': [], 'no_loop': [], 'post': []}
        out = calculate_truncated_averages(data)
        assert out == {'pre': "NA", 'no_loop': "NA", 'post': "NA"}

    def test_str_float_pitfall_avoided(self):
        # Guard the exact bug the spec warns about: str(3.0) == "3.0".
        data = {'pre': [3.0], 'no_loop': [], 'post': []}
        assert calculate_truncated_averages(data)['pre'] == "3"


class TestToDictRoundTrip:
    def _make_assessment(self):
        stages = {
            'pre': StageResult('pre', 'Hypoglycemia', '4', '78.0', '4.5', '17.5', 4, 1, 2, 2, '2.5', '0'),
            'no_loop': StageResult('no_loop', 'Hypoglycemia', '3', '69.0', '3.5', '27.5', 3, 2, 2, 2, '1.75', '3.14'),
            'post': StageResult('post', 'Hyperglycemia', '1', '94.5', '0.0', '5.5', 1, 0, 1, 2, '0', 'NA'),
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
