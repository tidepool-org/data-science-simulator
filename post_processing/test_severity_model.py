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

import severity_model
from severity_model import (
    build_assessment,
    build_assessment_result,
    calculate_truncated_averages,
    classify_summary_files,
    count_profiles,
    count_usable_profiles,
    detect_outliers,
    determine_harm_and_severity,
    calculate_hyperglycemia_score,
    check_consecutive_low_values,
    classify_sim_id,
    extract_metric_data,
    find_summary_files,
    get_profile_metrics,
    identify_severity_4_hypoglycemia,
    resolve_simulation_id,
    truncate_2dp,
    REQUIRED_SUMMARY_COLUMNS,
    SUMMARY_RESULTS_GLOB,
    StageResult,
    CatastrophicFinding,
    OutlierFinding,
    SeverityAssessment,
    apply_catastrophic_floor,
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
#
# The raw lbgi/dka_index values are chosen so each stage lands on a different
# formatting branch of calculate_truncated_averages:
#   pre     lbgi (2.0+3.339)/2 = 2.6695 -> '2.66'  (truncated, NOT rounded to 2.67)
#   no_loop lbgi (4.0+5.0)/2   = 4.5    -> '4.5'   (trailing zero dropped)
#   post    lbgi (1.0+1.0)/2   = 1.0    -> '1'     (whole number, no decimal)
#   post    dka  (10.0+12.5)/2 = 11.25  -> '11.25' (both decimals kept)
_PROFILE_A_ROWS = [
    # sim_id,                              tir,  tbr, tar, lbgi_s, dka_s, lbgi, dka_index
    ("pre-Loop_NoMitigations_t1_median",  78.0, 4.5, 17.5, 3, 1, 2.0, 20.0),
    ("pre-noLoop_t1_median",              69.0, 3.5, 27.5, 3, 2, 4.0, 30.0),
    ("post-Loop_WithMitigations_t1_median", 94.5, 0.0, 5.5, 1, 0, 1.0, 10.0),
]
_PROFILE_B_ROWS = [
    ("pre-Loop_NoMitigations_t1_median",  80.0, 4.0, 16.0, 3, 1, 3.339, 22.0),
    ("pre-noLoop_t1_median",              70.0, 3.0, 26.0, 3, 2, 5.0, 28.0),
    ("post-Loop_WithMitigations_t1_median", 95.0, 0.0, 5.0, 1, 0, 1.0, 12.5),
]


_NARROW_STEM = "Simulation-Configuration-TLR-999-test"


def _write_summary_csv(directory, profile, rows, columns=_SUMMARY_COLUMNS,
                       stem=_NARROW_STEM):
    path = os.path.join(
        directory,
        f"summary_results_{stem}_{profile}_profile.csv",
    )
    with open(path, "w") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            # Drop trailing columns if a variant fixture omits them (e.g. no lbgi).
            fh.write(",".join(str(v) for v in row[: len(columns)]) + "\n")
    return path


class TestTruncate2dp:
    """Truncation toward zero at hundredths -- never rounding."""

    def test_truncates_it_does_not_round(self):
        # 2.6695 would ROUND to 2.67; truncation must yield 2.66.
        assert truncate_2dp(2.6695) == 2.66
        assert truncate_2dp(2.999) == 2.99

    def test_exact_values_unchanged(self):
        assert truncate_2dp(3.0) == 3.0
        assert truncate_2dp(2.5) == 2.5
        assert truncate_2dp(0.0) == 0.0

    def test_third_decimal_dropped(self):
        assert truncate_2dp(1.333) == 1.33
        assert truncate_2dp(21.918) == 21.91


class TestCalculateTruncatedAverages:
    """String formatting rules for the raw-value averages."""

    def _avg(self, pre=None, no_loop=None, post=None):
        return calculate_truncated_averages({
            'pre': pre or [], 'no_loop': no_loop or [], 'post': post or [],
        })

    def test_empty_stage_is_na(self):
        assert self._avg()['pre'] == "NA"

    def test_whole_number_has_no_decimal(self):
        assert self._avg(pre=[3.0, 3.0])['pre'] == "3"
        assert self._avg(pre=[0.0, 0.0])['pre'] == "0"
        assert self._avg(pre=[20.0, 22.0])['pre'] == "21"

    def test_trailing_zeros_dropped(self):
        assert self._avg(pre=[2.5, 2.5])['pre'] == "2.5"
        assert self._avg(pre=[3.1, 3.1])['pre'] == "3.1"

    def test_two_decimals_preserved(self):
        assert self._avg(pre=[3.14, 3.14])['pre'] == "3.14"
        assert self._avg(pre=[10.0, 12.5])['pre'] == "11.25"

    def test_multi_decimal_average_is_truncated_not_rounded(self):
        # (2.0 + 3.339)/2 = 2.6695 -> '2.66', never '2.67'.
        assert self._avg(pre=[2.0, 3.339])['pre'] == "2.66"

    def test_stages_are_independent(self):
        result = self._avg(pre=[2.0, 3.339], no_loop=[4.0, 5.0], post=[1.0, 1.0])
        assert result == {'pre': "2.66", 'no_loop': "4.5", 'post': "1"}


class TestBuildAssessmentValueFields:
    """The raw-value fields (lbgi_value_avg / dka_index_value_avg) are averaged
    from the summary 'lbgi'/'dka_index' columns and truncated to 2dp -- and
    degrade to 'NA' when the column is absent."""

    def test_value_fields_are_averaged_from_raw_columns(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)

        assessment = build_assessment(tlr, "2026-07-29T00:00:00")
        assert assessment is not None

        pre = assessment.stages["pre"]
        # (2.0+3.339)/2 = 2.6695 -> truncated '2.66' (rounding would give 2.67).
        # Distinct from the risk SCORE (still the integer 3) -- separate fields.
        assert pre.lbgi_value_avg == "2.66"
        # (20.0+22.0)/2 = 21.0 -> whole number renders without a decimal.
        assert pre.dka_index_value_avg == "21"
        assert pre.lbgi_score_avg == 3

        # Trailing zero dropped, and both decimals kept.
        assert assessment.stages["no_loop"].lbgi_value_avg == "4.5"
        assert assessment.stages["post"].dka_index_value_avg == "11.25"
        assert assessment.stages["post"].lbgi_value_avg == "1"

    def test_raw_values_carry_no_escalation(self, tmp_path):
        """The 4->5 catastrophic escalation applies to the SCORE, never the raw
        value -- so the value fields are extracted without severity_updates."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)

        assessment = build_assessment(tlr, "2026-07-29T00:00:00")
        # Raw averages reflect the CSV columns verbatim, independent of scores.
        assert assessment.stages["no_loop"].lbgi_value_avg == "4.5"
        assert assessment.stages["no_loop"].dka_index_value_avg == "29"

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


# =============================================================================
# TRSET-62 -- catastrophic (4->5) escalation must not be diluted by averaging
# =============================================================================

_POST_SIM = "post-Loop_WithMitigations_t1_{p}"
_TSV_COLUMNS = "time\tbg\n"


def _write_tsv(directory, sim_id, bg_values):
    with open(os.path.join(directory, f"{sim_id}.tsv"), "w") as fh:
        fh.write(_TSV_COLUMNS)
        for i, bg in enumerate(bg_values):
            fh.write(f"{i}\t{bg}\n")


def _post_row(profile, lbgi_score):
    return (_POST_SIM.format(p=profile), 90.0, 0.0, 5.0, lbgi_score, 0, 1.0, 10.0)


class TestApplyCatastrophicFloor:
    """The pure floor helper."""

    def test_escalated_stage_floored_at_5(self):
        results = {"s": {"stage": "post", "updated_severity": 5, "condition": "zero_or_negative"}}
        out = apply_catastrophic_floor({"pre": 3, "no_loop": 2, "post": 4}, results)
        assert out == {"pre": 3, "no_loop": 2, "post": 5}

    def test_unescalated_sim_does_not_floor(self):
        results = {"s": {"stage": "post", "updated_severity": 4, "condition": "none"}}
        scores = {"pre": 3, "no_loop": 2, "post": 4}
        assert apply_catastrophic_floor(scores, results) == scores

    def test_no_results_returns_equal_copy(self):
        scores = {"pre": 1, "no_loop": 2, "post": 3}
        out = apply_catastrophic_floor(scores, {})
        assert out == scores and out is not scores

    def test_mixed_stages_only_escalated_moves(self):
        results = {
            "a": {"stage": "pre", "updated_severity": 5, "condition": "extended_low"},
            "b": {"stage": "post", "updated_severity": 4, "condition": "none"},
        }
        out = apply_catastrophic_floor({"pre": 4, "no_loop": 4, "post": 4}, results)
        assert out == {"pre": 5, "no_loop": 4, "post": 4}


class TestCatastrophicFloorEndToEnd:
    """build_assessment on the TLR-899 shape: one low profile must not dilute."""

    def _build(self, tmp_path, profile_scores, bg_by_profile):
        tlr = str(tmp_path)
        for profile, score in profile_scores.items():
            _write_summary_csv(tlr, profile, [_post_row(profile, score)])
            _write_tsv(tlr, _POST_SIM.format(p=profile), bg_by_profile.get(profile, [120] * 10))
        return build_assessment(tlr, "2026-10-06T00:00:00")

    def test_diluted_case_post_is_5(self, tmp_path):
        # TLR-899: adolescent 2, three profiles 4 with BG <= 0 -> mean(2,5,5,5)=4.25.
        scores = {"adolescent": 2, "median": 4, "resistant": 4, "sensitive": 4}
        bg = {p: [120, 0, 120] for p in ("median", "resistant", "sensitive")}
        post = self._build(tmp_path, scores, bg).stages["post"]
        assert post.lbgi_score_avg == 5
        assert post.severity == "5" and post.harm_type == "Hypoglycemia"

    def test_all_escalated_is_5(self, tmp_path):
        scores = {"median": 4, "resistant": 4}
        bg = {p: [120, -3, 120] for p in scores}
        assert self._build(tmp_path, scores, bg).stages["post"].lbgi_score_avg == 5

    def test_none_escalated_keeps_mean(self, tmp_path):
        scores = {"adolescent": 2, "median": 4, "resistant": 3, "sensitive": 3}
        # mean 3.0 -> 3; the 4 has healthy BG so it is not escalated.
        assert self._build(tmp_path, scores, {}).stages["post"].lbgi_score_avg == 3

    def test_extended_low_only_is_5(self, tmp_path):
        scores = {"adolescent": 2, "median": 4}
        bg = {"median": [35] * 48 + [120] * 10}   # <=40 for 48 readings, never <=0
        post = self._build(tmp_path, scores, bg).stages["post"]
        assert post.lbgi_score_avg == 5   # unfloored: round_half_up(3.5) = 4

    def test_other_stages_unaffected(self, tmp_path):
        tlr = str(tmp_path)
        rows = [_post_row("median", 4),
                ("pre-Loop_NoMitigations_t1_median", 90.0, 0.0, 5.0, 2, 0, 1.0, 10.0)]
        _write_summary_csv(tlr, "median", rows)
        _write_tsv(tlr, _POST_SIM.format(p="median"), [120, 0, 120])
        stages = build_assessment(tlr, "2026-10-06T00:00:00").stages
        assert stages["post"].lbgi_score_avg == 5
        assert stages["pre"].lbgi_score_avg == 2


# =============================================================================
# TRSET-28 -- silent/ambiguous failures in the post-processing layer
# =============================================================================

# Column sets for the degraded fixtures. "Required" is the verdict-input set:
# without it a file cannot contribute to harm/severity at all. Dropping only the
# reported metrics (TIR/TBR) or the raw values (lbgi/dka_index) leaves a file
# perfectly usable -- that distinction is what keeps an older-format directory off
# the malformed path.
_MISSING_REQUIRED_COLUMNS = ["sim_id", "percent_values_ge_70_le_180", "percent_cgm_lt_54"]
_WITHOUT_RAW_VALUE_COLUMNS = _SUMMARY_COLUMNS[:-2]


def _write_unreadable_csv(directory, profile):
    """A file that pandas cannot parse at all (ragged rows), not merely one with
    the wrong columns -- the other half of 'present but unusable'."""
    path = os.path.join(
        directory, f"summary_results_{_NARROW_STEM}_{profile}_profile.csv"
    )
    with open(path, "w") as fh:
        fh.write("a,b\n1,2,3,4,5\n")
    return path


class TestSummaryResultsGlobIsSharedAndLoose:
    """Finding 4: one pattern, defined once, honored by every call site."""

    def test_the_pattern_is_the_loose_one(self):
        assert SUMMARY_RESULTS_GLOB == "summary_results_*.csv"

    def test_glob_is_called_in_exactly_one_place(self):
        """The DRY guard. Five call sites each built their own glob expression,
        and build_assessment's disagreed with the other four."""
        with open(severity_model.__file__) as fh:
            source = fh.read()
        assert source.count("glob.glob(") == 1

    def test_files_are_returned_sorted(self, tmp_path):
        """Unsorted glob order made simulation-ID resolution depend on the
        filesystem once the pattern widened."""
        tlr = str(tmp_path)
        for profile in ("zebra", "alpha", "median"):
            _write_summary_csv(tlr, profile, _PROFILE_A_ROWS)

        found = find_summary_files(tlr)

        assert found == sorted(found)
        assert len(found) == 3

    def test_every_helper_reads_a_loosely_named_file(self, tmp_path):
        """A directory whose CSVs match the loose pattern but not the old narrow
        one: previously build_assessment alone rejected it."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS, stem="Config-TLR-777")
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS, stem="Config-TLR-777")

        assert count_profiles(tlr) == 2
        assert count_usable_profiles(tlr) == 2
        assert extract_metric_data(tlr, "lbgi_risk_score")["pre"] == [3, 3]
        assert set(get_profile_metrics(tlr).profiles) == {"median", "adolescent"}
        assert identify_severity_4_hypoglycemia(tlr) == {}  # no score-4 rows, but it read them

        outcome = build_assessment_result(tlr, "2026-08-06T00:00:00")
        assert outcome.status == "ok"
        assert outcome.assessment.simulation_id == "TLR-777"

    def test_a_narrowly_named_directory_still_works(self, tmp_path):
        """The widening must not cost the directories that already worked."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)

        outcome = build_assessment_result(tlr, "2026-08-06T00:00:00")

        assert outcome.status == "ok"
        assert outcome.assessment.simulation_id == "TLR-999"


class TestResolveSimulationId:
    """Finding 4 fallout: summary_files[0] alone is no longer safe."""

    def test_falls_through_to_the_first_filename_that_yields_an_id(self):
        """Sorted first is a file with no TLR part. Reading [0] blindly would
        return None and skip a directory that renders today."""
        assert resolve_simulation_id([
            "summary_results_AAA-noTLRhere_median_profile.csv",
            "summary_results_Config-TLR-777_adolescent_profile.csv",
        ]) == "TLR-777"

    def test_none_when_no_filename_carries_a_tlr_part(self):
        assert resolve_simulation_id([
            "summary_results_AAA-nope_median_profile.csv",
        ]) is None

    def test_empty_input_is_none(self):
        assert resolve_simulation_id([]) is None

    def test_a_mixed_directory_resolves_rather_than_skipping(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS, stem="AAA-noTLRhere")
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS, stem="Config-TLR-777")

        outcome = build_assessment_result(tlr, "2026-08-06T00:00:00")

        assert outcome.status == "ok"
        assert outcome.assessment.simulation_id == "TLR-777"


class TestAssessmentOutcomeDistinguishesEmptyFromMalformed:
    """Finding 1: the two conditions used to collapse into a bare None."""

    def test_empty_directory_is_empty(self, tmp_path):
        outcome = build_assessment_result(str(tmp_path), "2026-08-06T00:00:00")

        assert outcome.status == "empty"
        assert outcome.assessment is None
        assert outcome.detail

    def test_all_files_unusable_is_malformed_not_empty(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        outcome = build_assessment_result(tlr, "2026-08-06T00:00:00")

        assert outcome.status == "malformed"
        assert outcome.assessment is None

    def test_unreadable_file_is_malformed_not_empty(self, tmp_path):
        tlr = str(tmp_path)
        _write_unreadable_csv(tlr, "median")

        outcome = build_assessment_result(tlr, "2026-08-06T00:00:00")

        assert outcome.status == "malformed"

    def test_unresolvable_simulation_id_is_malformed(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS, stem="AAA-noTLRhere")

        outcome = build_assessment_result(tlr, "2026-08-06T00:00:00")

        assert outcome.status == "malformed"
        assert outcome.assessment is None

    def test_the_two_statuses_carry_different_details(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        broken_dir = tmp_path / "broken"
        broken_dir.mkdir()
        _write_summary_csv(str(broken_dir), "median", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        empty = build_assessment_result(str(empty_dir), "2026-08-06T00:00:00")
        broken = build_assessment_result(str(broken_dir), "2026-08-06T00:00:00")

        assert empty.status != broken.status
        assert empty.detail != broken.detail

    def test_a_malformed_directory_no_longer_renders_an_all_na_assessment(self, tmp_path):
        """The worst of the four findings: every metric unreadable used to still
        produce a complete assessment (and so a complete document)."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        assert build_assessment_result(tlr, "2026-08-06T00:00:00").assessment is None

    def test_good_directory_is_ok(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)

        outcome = build_assessment_result(tlr, "2026-08-06T00:00:00")

        assert outcome.status == "ok"
        assert isinstance(outcome.assessment, SeverityAssessment)

    def test_outcome_to_dict_is_json_serializable(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)

        payload = json.dumps(build_assessment_result(tlr, "2026-08-06T00:00:00").to_dict())

        assert json.loads(payload)["status"] == "ok"

    def test_empty_outcome_to_dict_carries_no_assessment(self, tmp_path):
        payload = build_assessment_result(str(tmp_path), "2026-08-06T00:00:00").to_dict()

        assert payload["assessment"] is None
        assert payload["status"] == "empty"


class TestBuildAssessmentWrapperContract:
    """The Optional[SeverityAssessment] contract the GUI runner is typed on."""

    def test_returns_the_assessment_when_usable(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)

        assert isinstance(build_assessment(tlr, "2026-08-06T00:00:00"), SeverityAssessment)

    def test_returns_none_for_both_failure_conditions(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        broken_dir = tmp_path / "broken"
        broken_dir.mkdir()
        _write_summary_csv(str(broken_dir), "median", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        assert build_assessment(str(empty_dir), "2026-08-06T00:00:00") is None
        assert build_assessment(str(broken_dir), "2026-08-06T00:00:00") is None


class TestUsableProfileCount:
    """Finding 3: the count must reflect contribution, not just file presence."""

    def test_m_equals_n_on_clean_data(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)

        assessment = build_assessment(tlr, "2026-08-06T00:00:00")

        assert assessment.profile_count == 2
        assert assessment.usable_profile_count == 2

    def test_m_is_less_than_n_when_a_file_is_dropped(self, tmp_path):
        """3 files, 1 malformed: extract_metric_data averages 2, so the old
        unqualified count named 3 profiles when 2 contributed."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        assessment = build_assessment(tlr, "2026-08-06T00:00:00")

        assert assessment.profile_count == 3
        assert assessment.usable_profile_count == 2

    def test_an_unreadable_file_also_reduces_m(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_unreadable_csv(tlr, "broken")

        assessment = build_assessment(tlr, "2026-08-06T00:00:00")

        assert (assessment.profile_count, assessment.usable_profile_count) == (2, 1)

    def test_missing_only_the_raw_value_columns_keeps_a_file_usable(self, tmp_path):
        """Older CSVs predate lbgi/dka_index and are DESIGNED to degrade to 'NA'.
        Counting them unusable would drop a clean directory to M == 0."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS,
                           columns=_WITHOUT_RAW_VALUE_COLUMNS)

        assessment = build_assessment(tlr, "2026-08-06T00:00:00")

        assert assessment is not None
        assert assessment.usable_profile_count == assessment.profile_count == 1

    def test_count_is_carried_through_to_dict(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        payload = build_assessment(tlr, "2026-08-06T00:00:00").to_dict()

        assert payload["profile_count"] == 2
        assert payload["usable_profile_count"] == 1

    def test_it_defaults_to_none_for_older_constructors(self):
        """None means 'not measured' and renders as M == N, so an assessment built
        without it is unaffected."""
        assessment = SeverityAssessment(
            simulation_id="TLR-TEST", subdirectory_name="TLR-TEST",
            timestamp="2026-08-06T00:00:00", profile_count=2, stages={},
        )

        assert assessment.usable_profile_count is None


class TestClassifySummaryFiles:
    """The usable/unusable split that defines M."""

    def test_clean_files_are_all_usable(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)

        usable, unusable = classify_summary_files(tlr)

        assert len(usable) == 2
        assert unusable == []

    def test_a_file_missing_a_required_column_is_unusable(self, tmp_path):
        tlr = str(tmp_path)
        broken = _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                                    columns=_MISSING_REQUIRED_COLUMNS)

        usable, unusable = classify_summary_files(tlr)

        assert usable == []
        assert unusable == [broken]

    def test_every_required_column_is_load_bearing(self, tmp_path):
        """Each REQUIRED_SUMMARY_COLUMNS entry, dropped on its own, makes the file
        unusable -- so the constant is not carrying a column that does not matter."""
        for dropped in REQUIRED_SUMMARY_COLUMNS:
            directory = tmp_path / f"without_{dropped}"
            directory.mkdir()
            columns = [c for c in _SUMMARY_COLUMNS if c != dropped]
            _write_summary_csv(str(directory), "median", _PROFILE_A_ROWS, columns=columns)

            usable, unusable = classify_summary_files(str(directory))

            assert usable == [], f"{dropped} should be required"
            assert len(unusable) == 1

    def test_an_empty_directory_splits_to_two_empty_lists(self, tmp_path):
        assert classify_summary_files(str(tmp_path)) == ([], [])


class TestGetProfileMetrics:
    """Finding 2 at the source, plus Decision C: an unreadable file is EXCLUDED and
    named rather than discarding every readable profile with it."""

    def test_no_files_means_no_files_present(self, tmp_path):
        metrics = get_profile_metrics(str(tmp_path))

        assert metrics.files_present is False
        assert metrics.profiles == {}
        assert metrics.excluded == []

    def test_a_file_missing_required_columns_is_excluded_and_named(self, tmp_path):
        tlr = str(tmp_path)
        broken = _write_summary_csv(tlr, "median", _PROFILE_A_ROWS,
                                    columns=_MISSING_REQUIRED_COLUMNS)

        metrics = get_profile_metrics(tlr)

        assert metrics.files_present is True
        assert metrics.profiles == {}
        assert metrics.excluded == [broken]

    def test_an_unreadable_file_is_excluded_and_named(self, tmp_path):
        tlr = str(tmp_path)
        broken = _write_unreadable_csv(tlr, "median")

        assert get_profile_metrics(tlr).excluded == [broken]

    def test_clean_data_yields_its_profiles_and_excludes_nothing(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)

        metrics = get_profile_metrics(tlr)

        assert set(metrics.profiles) == {"median"}
        assert metrics.excluded == []

    def test_one_bad_file_no_longer_discards_the_good_ones(self, tmp_path):
        """The Decision C change. This used to return None for the whole directory,
        so a single malformed profile cost the outlier analysis of every valid one."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)
        broken = _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                                    columns=_MISSING_REQUIRED_COLUMNS)

        metrics = get_profile_metrics(tlr)

        assert set(metrics.profiles) == {"median", "adolescent"}
        assert metrics.excluded == [broken]

    def test_an_excluded_file_leaves_no_partial_profile_behind(self, tmp_path):
        """Profiles are published only once fully built, so an excluded file cannot
        appear in `profiles` with some stages filled in."""
        tlr = str(tmp_path)
        _write_unreadable_csv(tlr, "broken")
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)

        metrics = get_profile_metrics(tlr)

        assert "broken" not in metrics.profiles
        assert set(metrics.profiles) == {"median"}


class TestDetectOutliersStatus:
    """Finding 2 at the boundary the renderer reads."""

    def test_every_profile_file_unreadable_is_malformed_not_no_data(self, tmp_path):
        """'malformed_data' now means NOTHING survived exclusion (Decision C): one
        bad file among good ones is excluded and the good ones analyzed."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "brokenA", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)
        _write_summary_csv(tlr, "brokenB", _PROFILE_B_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        findings, status = detect_outliers(tlr)

        assert status == "malformed_data"
        assert findings == []

    def test_one_bad_file_among_good_ones_no_longer_blocks_the_analysis(self, tmp_path):
        """Was 'malformed_data' with zero findings for the whole directory. The two
        readable profiles are now compared, and the drop is disclosed by the
        renderer from the assessment's usable-vs-total counts."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        findings, status = detect_outliers(tlr)

        assert status == "ok"
        assert findings == []       # these two profiles simply have no outlier

    def test_a_genuinely_absent_directory_is_still_no_data(self, tmp_path):
        assert detect_outliers(str(tmp_path)) == ([], "no_data")

    def test_one_profile_is_still_single_profile(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)

        assert detect_outliers(tlr)[1] == "single_profile"

    def test_clean_multi_profile_data_is_still_ok(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "adolescent", _PROFILE_B_ROWS)

        assert detect_outliers(tlr)[1] == "ok"

    def test_the_status_reaches_the_assessment(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "brokenA", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)
        _write_summary_csv(tlr, "brokenB", _PROFILE_B_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)
        # A usable file with NO profile part in its name, so M >= 1 and
        # build_assessment produces a document at all -- while leaving every
        # per-profile file excluded. Written by hand: _write_summary_csv always
        # appends '_<profile>_profile.csv', which would make it a valid profile.
        aggregate = os.path.join(tlr, f"summary_results_{_NARROW_STEM}.csv")
        with open(aggregate, "w") as fh:
            fh.write(",".join(_SUMMARY_COLUMNS) + "\n")
            for row in _PROFILE_A_ROWS:
                fh.write(",".join(str(v) for v in row) + "\n")

        assert build_assessment(tlr, "2026-08-06T00:00:00").outlier_status == "malformed_data"

    def test_a_surviving_profile_beside_an_excluded_one_is_single_profile(self, tmp_path):
        """One readable profile plus one excluded: there is one analyzable profile,
        which is the single-profile case -- not 'malformed_data' as it used to be."""
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        assert detect_outliers(tlr)[1] == "single_profile"


class TestOptionalColumnIsNotCalledMalformed:
    """An older CSV that legitimately degrades to 'NA' was labeled malformed on
    the console -- valid data reported as broken, the inverse of finding 2."""

    def test_absent_optional_column_is_reported_as_na_not_malformed(self, tmp_path, capsys):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS,
                           columns=_WITHOUT_RAW_VALUE_COLUMNS)

        extract_metric_data(tlr, "lbgi")
        output = capsys.readouterr().out

        assert "malformed" not in output
        assert "will report NA" in output

    def test_absent_required_column_is_still_reported_as_malformed(self, tmp_path, capsys):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        extract_metric_data(tlr, "lbgi_risk_score")

        assert "malformed" in capsys.readouterr().out


# =============================================================================
# Hyperglycemia zero case in the outlier path (was: KNOWN INCONSISTENCY)
# =============================================================================

# detect_outliers computed its own TAR->score mapping without a zero case, so a
# profile with TAR == 0.0 scored 1 where calculate_hyperglycemia_score gives 0.
# Correcting it moves such a profile from the 'Hyperglycemia' harm group to
# baseline -- and the zero-TAR outlier check keyed on that group, so the naive fix
# would have stopped flagging the cleanest profiles. These tests pin BOTH halves:
# the score is now the module's single mapping, AND the check still fires.

_OUTLIER_COLUMNS = [
    "sim_id", "percent_values_ge_70_le_180", "percent_cgm_lt_54",
    "percent_cgm_gt_180", "lbgi_risk_score", "dka_risk_score",
]
_OUTLIER_STAGE_STEMS = (
    "pre-Loop_NoMitigations_t1", "pre-noLoop_t1", "post-Loop_WithMitigations_t1",
)


def _write_outlier_profile(directory, profile, tar, lbgi, dka):
    """A profile complete across all three stages, every stage carrying the same
    (tar, lbgi, dka) -- so a finding appears once per stage and the assertions do
    not depend on which stage is which."""
    rows = [
        (f"{stem}_{profile}", 80.0, 1.0, tar, lbgi, dka)
        for stem in _OUTLIER_STAGE_STEMS
    ]
    return _write_summary_csv(directory, profile, rows, columns=_OUTLIER_COLUMNS)


def _hyper_outliers(tlr_dir):
    findings, status = detect_outliers(tlr_dir)
    assert status == "ok", f"expected usable data, got {status!r}"
    return [f for f in findings if f.harm_type == "Hyperglycemia"]


class TestHyperglycemiaScoreHasOneMapping:
    """The outlier path no longer carries its own TAR->score mapping."""

    def test_the_outlier_path_calls_the_shared_mapping(self):
        """Asserted over the parsed AST, not the source text: the docstring quotes
        the old inline expression verbatim, so a substring check would pass or fail
        on prose rather than on code."""
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(detect_outliers))
        called = {
            node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        assert "calculate_hyperglycemia_score" in called

    def test_baseline_label_is_named_once(self):
        """detect_outliers has to reason about the baseline GROUP, so the label and
        the string determine_harm_and_severity returns must not drift apart."""
        assert determine_harm_and_severity(0, 0, 0) == (severity_model.BASELINE_HARM, "0")

    def test_true_zero_tar_scores_zero_on_the_outlier_path_too(self):
        """The mapping the outlier path now shares: 0 only if TAR is truly 0."""
        assert calculate_hyperglycemia_score(0.0) == 0
        assert calculate_hyperglycemia_score(5.0) == 1
        assert calculate_hyperglycemia_score(20.0) == 2


class TestZeroTarOutlierStillFlagged:
    """The regression the naive score fix would have introduced."""

    def test_flagged_when_the_zero_profile_has_no_hypo_or_dka_risk(self, tmp_path):
        """lbgi=0/dka=0/TAR=0 is now baseline, not Hyperglycemia. Keying the check
        on the Hyperglycemia group alone would silently drop this finding -- the
        cleanest profile among high-TAR peers is exactly the one worth flagging."""
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "zero", tar=0.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "highA", tar=20.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "highB", tar=25.0, lbgi=0, dka=0)

        found = _hyper_outliers(tlr)

        assert {f.profile for f in found} == {"zero"}
        assert sorted(f.stage for f in found) == ["no_loop", "post", "pre"]
        assert all(f.value == 0.0 for f in found)

    def test_flagged_when_the_zero_profile_stays_in_the_hyperglycemia_group(self, tmp_path):
        """lbgi=1 keeps it out of baseline, so this half worked before and after."""
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "zero", tar=0.0, lbgi=1, dka=0)
        _write_outlier_profile(tlr, "highA", tar=20.0, lbgi=1, dka=0)
        _write_outlier_profile(tlr, "highB", tar=25.0, lbgi=1, dka=0)

        assert {f.profile for f in _hyper_outliers(tlr)} == {"zero"}

    def test_the_reported_median_is_of_the_non_zero_profiles(self, tmp_path):
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "zero", tar=0.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "highA", tar=20.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "highB", tar=30.0, lbgi=0, dka=0)

        assert {f.comparison_median for f in _hyper_outliers(tlr)} == {30.0}

    def test_several_zero_profiles_are_each_flagged(self, tmp_path):
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "zeroA", tar=0.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "zeroB", tar=0.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "high", tar=20.0, lbgi=0, dka=0)

        assert {f.profile for f in _hyper_outliers(tlr)} == {"zeroA", "zeroB"}

    def test_not_flagged_when_a_peer_is_not_high(self, tmp_path):
        """Unchanged gate: every non-zero profile must be >= 12.0 TAR."""
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "zero", tar=0.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "mid", tar=5.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "high", tar=20.0, lbgi=0, dka=0)

        assert _hyper_outliers(tlr) == []

    def test_not_flagged_when_no_profile_is_at_zero(self, tmp_path):
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "a", tar=15.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "b", tar=20.0, lbgi=0, dka=0)

        assert _hyper_outliers(tlr) == []

    def test_a_zero_tar_profile_with_hypo_risk_is_not_swept_in(self, tmp_path):
        """The check spans Hyperglycemia + baseline only. A TAR == 0 profile whose
        LBGI puts it in the Hypoglycemia group was never compared on the
        hyperglycemia axis, and still is not."""
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "hypoZero", tar=0.0, lbgi=4, dka=0)
        _write_outlier_profile(tlr, "highA", tar=20.0, lbgi=0, dka=0)
        _write_outlier_profile(tlr, "highB", tar=25.0, lbgi=0, dka=0)

        assert _hyper_outliers(tlr) == []

    def test_hypoglycemia_outliers_are_unaffected(self, tmp_path):
        """The score fix touches only the hyperglycemia axis.

        lbgi=2, not 1, for the peers: determine_harm_and_severity sends lbgi<=1 with
        dka==0 to Hyperglycemia, so lbgi=1 peers would never form a Hypoglycemia
        group to be an outlier within.
        """
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "severe", tar=20.0, lbgi=4, dka=0)
        _write_outlier_profile(tlr, "mildA", tar=20.0, lbgi=2, dka=0)
        _write_outlier_profile(tlr, "mildB", tar=20.0, lbgi=2, dka=0)

        findings, status = detect_outliers(tlr)

        assert status == "ok"
        hypo = [f for f in findings if f.harm_type == "Hypoglycemia"]
        assert {f.profile for f in hypo} == {"severe"}


class TestIncompleteStagesIsNotReportedAsAbsent:
    """The third condition that used to hide inside 'no_data' (TRSET-28 Decision B).

    Profiles that parse but where none carries all three stages is neither absence
    nor corruption -- it is a run that produced only some stages, a recoverable
    configuration problem that was reported with the same sentence as a missing
    directory.
    """

    def _write_partial_stage_profile(self, directory, profile, stems):
        """A profile whose rows cover only `stems`, so it is never stage-complete."""
        rows = [(f"{stem}_{profile}", 80.0, 1.0, 20.0, 1, 0) for stem in stems]
        return _write_summary_csv(directory, profile, rows, columns=_OUTLIER_COLUMNS)

    def test_profiles_missing_a_stage_report_incomplete_not_no_data(self, tmp_path):
        tlr = str(tmp_path)
        for profile in ("median", "adolescent"):
            self._write_partial_stage_profile(
                tlr, profile, ("pre-Loop_NoMitigations_t1", "pre-noLoop_t1"),
            )  # no post- row anywhere

        findings, status = detect_outliers(tlr)

        assert status == "incomplete_stages"
        assert findings == []

    def test_a_single_missing_stage_is_enough(self, tmp_path):
        """Only the post stage is absent; pre and no_loop are fully populated."""
        tlr = str(tmp_path)
        self._write_partial_stage_profile(
            tlr, "median", ("pre-Loop_NoMitigations_t1", "pre-noLoop_t1"),
        )

        assert detect_outliers(tlr)[1] == "incomplete_stages"

    def test_no_profile_names_at_all_is_still_no_data(self, tmp_path):
        """An aggregate-only directory: summary CSVs exist but none is a per-profile
        file, so no per-profile results exist to compare. Genuine absence."""
        tlr = str(tmp_path)
        path = os.path.join(tlr, f"summary_results_{_NARROW_STEM}.csv")
        with open(path, "w") as fh:
            fh.write(",".join(_OUTLIER_COLUMNS) + "\n")
            fh.write("pre-Loop_NoMitigations_t1_x,80.0,1.0,20.0,1,0\n")

        assert detect_outliers(tlr)[1] == "no_data"

    def test_an_empty_directory_is_still_no_data(self, tmp_path):
        assert detect_outliers(str(tmp_path))[1] == "no_data"

    def test_incomplete_now_outranks_an_excluded_file(self, tmp_path):
        """Precedence FLIPPED with Decision C, deliberately. An unreadable file no
        longer poisons the directory -- it is excluded -- so what the reader needs to
        know is the state of the data that REMAINS: readable, but stage-thin.
        'malformed_data' is reserved for nothing surviving at all."""
        tlr = str(tmp_path)
        self._write_partial_stage_profile(
            tlr, "median", ("pre-Loop_NoMitigations_t1",),
        )
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        assert detect_outliers(tlr)[1] == "incomplete_stages"

    def test_one_complete_profile_is_still_single_profile(self, tmp_path):
        """A complete profile alongside incomplete ones is the single-profile case,
        not the incomplete one -- the check is 'none complete', not 'any incomplete'."""
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "complete", tar=20.0, lbgi=1, dka=0)
        self._write_partial_stage_profile(
            tlr, "partial", ("pre-Loop_NoMitigations_t1",),
        )

        assert detect_outliers(tlr)[1] == "single_profile"

    def test_complete_profiles_are_still_ok(self, tmp_path):
        tlr = str(tmp_path)
        _write_outlier_profile(tlr, "median", tar=20.0, lbgi=1, dka=0)
        _write_outlier_profile(tlr, "adolescent", tar=25.0, lbgi=1, dka=0)

        assert detect_outliers(tlr)[1] == "ok"

    def test_the_status_reaches_the_assessment(self, tmp_path):
        tlr = str(tmp_path)
        for profile in ("median", "adolescent"):
            self._write_partial_stage_profile(
                tlr, profile, ("pre-Loop_NoMitigations_t1", "pre-noLoop_t1"),
            )

        assessment = build_assessment(tlr, "2026-08-06T00:00:00")

        assert assessment is not None, "readable data must still produce a document"
        assert assessment.outlier_status == "incomplete_stages"


# --- profile name from summary filename ----------------------------------------

@pytest.mark.parametrize("filename, expected", [
    ("summary_results_Simulation-Configuration-TLR-552_Adolescent_profile.csv", "Adolescent"),
    # run.py names the file after the scenario JSON, so '.json' precedes '.csv'.
    ("summary_results_Simulation-Configuration-TLR-552_Adolescent_profile.json.csv", "Adolescent"),
    ("summary_results_Simulation-Configuration-TLR-000-base_median_profile_v1.csv", "median"),
    ("summary_results_base_median.csv", None),
])
def test_extract_profile_from_filename(filename, expected):
    assert severity_model.extract_profile_from_filename(os.path.join("any", "dir", filename)) == expected


# --- TRSET-59: .json.csv summaries reach the outlier analysis -------------------

def _write_outlier_profile_named(directory, profile, suffix):
    """_write_outlier_profile with a chosen summary-filename suffix."""
    path = _write_outlier_profile(directory, profile, tar=10.0, lbgi=1.0, dka=1.0)
    new = path[: -len("_profile.csv")] + suffix
    os.rename(path, new)
    return new


@pytest.mark.parametrize("suffix", ["_profile.csv", "_profile.json.csv"])
def test_outlier_analysis_runs_for_either_summary_filename_suffix(tmp_path, suffix):
    for profile in ("adolescent", "median", "resistant", "sensitive"):
        _write_outlier_profile_named(str(tmp_path), profile, suffix)

    _, status = detect_outliers(str(tmp_path))

    assert status != "no_data"


def test_reverting_double_extension_handling_is_caught(tmp_path, monkeypatch):
    """Mutation check: the pre-e378b9d parse (drop '.csv' only) must fail the
    json.csv case, proving the test above depends on the fix."""
    def old_extract(csv_path):
        parts = os.path.basename(csv_path).replace('.csv', '').split('_')
        try:
            i = parts.index('profile')
            return parts[i - 1] if i > 0 else None
        except ValueError:
            return None

    for profile in ("adolescent", "median", "resistant", "sensitive"):
        _write_outlier_profile_named(str(tmp_path), profile, "_profile.json.csv")
    monkeypatch.setattr(severity_model, "extract_profile_from_filename", old_extract)

    _, status = detect_outliers(str(tmp_path))

    assert status == "no_data"


# =============================================================================
# TRSET-63 -- a stage with no sims has no verdict
# =============================================================================

class TestEmptyStageNaVerdict:
    """A stage with n_sims == 0 reports NA harm/severity, not Hyperglycemia/1."""

    def _build(self, tmp_path, rows, bg=None):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", rows)
        if bg is not None:
            _write_tsv(tlr, _POST_SIM.format(p="median"), bg)
        return build_assessment(tlr, "2026-10-06T00:00:00")

    def test_empty_stage_is_na_na(self, tmp_path):
        rows = [_PROFILE_A_ROWS[0], _PROFILE_A_ROWS[1]]   # no post-mitigation sims
        stages = self._build(tmp_path, rows).stages
        assert stages["post"].n_sims == 0
        assert stages["post"].harm_type == "NA"
        assert stages["post"].severity == "NA"

    def test_all_stages_populated_unchanged(self, tmp_path):
        stages = self._build(tmp_path, _PROFILE_A_ROWS).stages
        for stage in ("pre", "no_loop", "post"):
            assert stages[stage].n_sims == 1
            assert stages[stage].harm_type != "NA"
            assert stages[stage].severity != "NA"

    def test_empty_stage_beside_floored_stage(self, tmp_path):
        rows = [_post_row("median", 4)]                    # pre / no_loop empty
        stages = self._build(tmp_path, rows, bg=[120, 0, 120]).stages
        assert stages["post"].severity == "5"
        assert stages["post"].harm_type == "Hypoglycemia"
        for stage in ("pre", "no_loop"):
            assert stages[stage].harm_type == "NA"
            assert stages[stage].severity == "NA"
