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
        assert set(get_profile_metrics(tlr)[0]) == {"median", "adolescent"}
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


class TestGetProfileMetricsStatus:
    """Finding 2 at the source: absent and malformed are now separate."""

    def test_no_files_is_no_data(self, tmp_path):
        assert get_profile_metrics(str(tmp_path)) == (None, "no_data")

    def test_missing_required_columns_is_malformed(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        profile_data, status = get_profile_metrics(tlr)

        assert status == "malformed_data"
        assert profile_data is None

    def test_an_unreadable_file_is_malformed(self, tmp_path):
        tlr = str(tmp_path)
        _write_unreadable_csv(tlr, "median")

        assert get_profile_metrics(tlr)[1] == "malformed_data"

    def test_clean_data_is_ok(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)

        profile_data, status = get_profile_metrics(tlr)

        assert status == "ok"
        assert set(profile_data) == {"median"}


class TestDetectOutliersStatus:
    """Finding 2 at the boundary the renderer reads."""

    def test_malformed_data_is_not_reported_as_no_data(self, tmp_path):
        tlr = str(tmp_path)
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        findings, status = detect_outliers(tlr)

        assert status == "malformed_data"
        assert findings == []

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
        _write_summary_csv(tlr, "median", _PROFILE_A_ROWS)
        _write_summary_csv(tlr, "broken", _PROFILE_A_ROWS,
                           columns=_MISSING_REQUIRED_COLUMNS)

        assert build_assessment(tlr, "2026-08-06T00:00:00").outlier_status == "malformed_data"


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
