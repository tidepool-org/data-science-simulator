"""
TRSET-82 integration tests: RTF risk summary constrained by delivery risk type.

End to end through ``process_results_directory`` on COPIES of real TRSET-81 arm A1
TLR directories (never the originals, so the RTFs already sitting in the results
directory are not overwritten), plus synthetic variants made by editing the copies'
per-sim ``{sim_id}.json`` files.

The real A1 results are large and not part of the repository. Set
``TRSET81_A1_RESULTS_DIR`` to the arm-A1 directory (the one that contains
``Risk_Run_*``); when unset, the sibling ``../trset-81-results/A1_old_RAI`` of the
repo's parent directory is tried. The real-data tests skip, naming where they
looked, when it is not found.

Expected outcomes are derived here from the raw summary CSVs by an independent,
deliberately plain calculation -- not by calling the harm rules under test.
"""

import glob
import json
import math
import os
import re
import shutil
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from create_severity_summary import process_results_directory  # noqa: E402
from severity_model import build_assessment, build_assessment_result  # noqa: E402

ENV_VAR = 'TRSET81_A1_RESULTS_DIR'
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_A1_DIR = os.path.join(os.path.dirname(REPO_ROOT), 'trset-81-results', 'A1_old_RAI')
TIMESTAMP = '2026-11-23T00:00:00.000000'

NOT_RELEVANT = "Critical/Catastrophic level determination not relevant to this scenario."
NOTE_PREFIX = "Harm not constrained by delivery risk type: "


def _a1_run_dir():
    base = os.environ.get(ENV_VAR) or DEFAULT_A1_DIR
    runs = sorted(glob.glob(os.path.join(base, 'Risk_Run_*')))
    if not runs:
        pytest.skip(f"TRSET-81 arm A1 results not found under {base} "
                    f"(set {ENV_VAR} to the arm-A1 directory)")
    return runs[0]


def _stage_harms(assessment):
    return {s: (r.harm_type, r.severity) for s, r in assessment.stages.items()}


def make_run(tmp_path, a1_run, tlr_names, subdir='run'):
    """A run directory in tmp_path holding copies of the named A1 TLR directories."""
    run = tmp_path / subdir
    run.mkdir()
    (run / 'metadata.json').write_text(json.dumps({'timestamp': TIMESTAMP}))
    for name in tlr_names:
        shutil.copytree(
            os.path.join(a1_run, name), run / name,
            ignore=shutil.ignore_patterns('*.png', 'loop_algo_io', '*.rtf'),
        )
    return str(run)


def sim_json_paths(tlr_dir):
    return sorted(glob.glob(os.path.join(tlr_dir, 'pre-*.json')) +
                  glob.glob(os.path.join(tlr_dir, 'post-*.json')))


def set_durations(tlr_dir, durations):
    """Rewrite duration_hrs in every per-sim JSON; durations is a value or a list
    cycled over the sorted sims."""
    paths = sim_json_paths(tlr_dir)
    assert paths
    for i, path in enumerate(paths):
        with open(path) as f:
            data = json.load(f)
        data['duration_hrs'] = (durations[i % len(durations)]
                                if isinstance(durations, list) else durations)
        with open(path, 'w') as f:
            json.dump(data, f)


def rtf_text(tlr_dir):
    (path,) = glob.glob(os.path.join(tlr_dir, 'risk_summary_*.rtf'))
    with open(path, encoding='utf-8', newline='') as f:
        return f.read()


def without_timestamp(rtf):
    return re.sub(r'\{\\b Date and time of simulation run:\} [^\n]*', 'TS', rtf)


def run_pipeline(run_dir):
    result = process_results_directory(run_dir)
    assert not result.skipped, result.skipped
    return result


def mean_col(tlr_dir, stage_prefixes, column):
    """Independent stage mean of a raw summary column, across all profile CSVs."""
    vals = []
    for f in sorted(glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            if row['sim_id'].startswith(stage_prefixes):
                vals.append(row[column])
    return sum(vals) / len(vals) if vals else None


STAGE_PREFIX = {
    'pre': ('pre-Loop',),
    'no_loop': ('pre-noLoop_', 'pre-NoLoop_'),
    'post': ('post-Loop',),
}


def expected_underdelivery(tlr_dir, stage):
    dkai = mean_col(tlr_dir, STAGE_PREFIX[stage], 'dka_index')
    if dkai == 0.0:
        tar = mean_col(tlr_dir, STAGE_PREFIX[stage], 'percent_cgm_gt_180')
        tar_str = f"{tar:.1f}"
        score = 0 if float(tar_str) == 0.0 else (1 if float(tar_str) < 12.0 else 2)
        return ('Hyperglycemia', str(score))
    score = mean_col(tlr_dir, STAGE_PREFIX[stage], 'dka_risk_score')
    return ('DKA', str(math.floor(score + 0.5)))


# --------------------------------------------------------------------------- #
# Real A1 directories: expected to change
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('tlr', ['TLR-613', 'TLR-843'])
def test_underdelivery_tlr_reports_only_dka_or_hyperglycemia(tmp_path, capsys, tlr):
    a1 = _a1_run_dir()
    run = make_run(tmp_path, a1, [tlr])
    tlr_dir = os.path.join(run, tlr)
    run_pipeline(run)

    assessment = build_assessment(tlr_dir, TIMESTAMP)
    assert assessment.delivery_risk_type == 'underdelivery'
    assert assessment.duration_hours == 23.0
    for stage, (harm, severity) in _stage_harms(assessment).items():
        assert harm in ('DKA', 'Hyperglycemia'), (stage, harm)
        assert (harm, severity) == expected_underdelivery(tlr_dir, stage), stage

    rtf = rtf_text(tlr_dir)
    assert NOT_RELEVANT in rtf
    assert NOTE_PREFIX not in rtf
    assert 'Hypoglycemia' not in rtf.split('Table of results', 1)[1].split('Critical/Catastrophic Identifier', 1)[0]


def test_tlr613_catastrophic_hits_are_warned_on_the_console(tmp_path, capsys):
    a1 = _a1_run_dir()
    run = make_run(tmp_path, a1, ['TLR-613'])
    tlr_dir = os.path.join(run, 'TLR-613')
    assessment = build_assessment(tlr_dir, TIMESTAMP)
    hits = [f for f in assessment.catastrophic_findings if f.condition != 'none']
    out = capsys.readouterr().out
    # The check still runs for underdelivery: every hit is named in a console warning
    # even though the RTF section reads "not relevant".
    for finding in hits:
        assert finding.sim_id in out
    if hits:
        assert 'atastrophic' in out and 'Warning' in out


# --------------------------------------------------------------------------- #
# Real A1 directories: controls expected not to change
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('tlr, risk_type, harm', [
    ('TLR-552', 'underdelivery', 'DKA'),
    ('TLR-553', 'overdelivery', 'Hypoglycemia'),
])
def test_control_tlrs_are_unchanged_except_the_documented_outlier_sentence(tmp_path, tlr, risk_type, harm):
    a1 = _a1_run_dir()
    stored = glob.glob(os.path.join(a1, tlr, 'risk_summary_*.rtf'))
    run = make_run(tmp_path, a1, [tlr])
    tlr_dir = os.path.join(run, tlr)
    run_pipeline(run)

    assessment = build_assessment(tlr_dir, TIMESTAMP)
    assert assessment.delivery_risk_type == risk_type
    assert all(h == harm and s in ('4', '5') for h, s in _stage_harms(assessment).values())
    rtf = rtf_text(tlr_dir)
    assert NOTE_PREFIX not in rtf
    if not stored:
        pytest.skip("no stored RTF in the A1 results to compare against")
    with open(stored[0], encoding='utf-8', newline='') as f:
        stored_rtf = f.read()
    # Everything above the Outlier Results section -- table, count line, catastrophic
    # section -- is byte-identical apart from the run timestamp.
    head = lambda text: without_timestamp(text).split('{\\b Outlier Results}', 1)[0]
    assert head(rtf) == head(stored_rtf)
    if tlr == 'TLR-552':
        assert without_timestamp(rtf) == without_timestamp(stored_rtf)
    else:
        # TLR-553 (finding, not an expectation of the request): its Adolescent
        # post-mitigation profile has LBGI score 1, DKAI 0, TAR 25.8%. Before, that was
        # classed Hyperglycemia (alone in its group, so no outlier); an overdelivery
        # TLR classes it Hypoglycemia 1 against the other profiles' 5, so the outlier
        # sentence changes. That is an allowed diff class (changed outlier sentence).
        assert ("Outlier profile exists. Adolescent has a Hypoglycemia score of 1 at "
                "Post-mitigation, while other profiles have a Hypoglycemia score of 5."
                ) in rtf


def test_overdelivery_catastrophic_section_is_unchanged(tmp_path):
    a1 = _a1_run_dir()
    run = make_run(tmp_path, a1, ['TLR-553'])
    run_pipeline(run)
    rtf = rtf_text(os.path.join(run, 'TLR-553'))
    assert NOT_RELEVANT not in rtf


# --------------------------------------------------------------------------- #
# Synthetic / derived unconstrained cases (copies of real TLR-553)
# --------------------------------------------------------------------------- #

def _unconstrained_run(tmp_path, mutate):
    a1 = _a1_run_dir()
    run = make_run(tmp_path, a1, ['TLR-553'])
    tlr_dir = os.path.join(run, 'TLR-553')
    mutate(tlr_dir)
    return run, tlr_dir


def test_non_standard_duration_is_unconstrained(tmp_path, capsys):
    run, tlr_dir = _unconstrained_run(tmp_path, lambda d: set_durations(d, 12.0))
    run_pipeline(run)
    assessment = build_assessment(tlr_dir, TIMESTAMP)
    assert assessment.delivery_risk_type == 'unconstrained'
    assert assessment.duration_hours == 12.0
    rtf = rtf_text(tlr_dir)
    assert (NOTE_PREFIX + "simulation duration was 12 hours, which matches neither the "
            "8-hour overdelivery nor the 23-hour underdelivery duration defined in "
            "RMP-0002.") in rtf
    # Placed after the profile-count line, before the Catastrophic Identifier heading.
    count_at = rtf.index('virtual patient profiles aggregated')
    note_at = rtf.index(NOTE_PREFIX)
    assert count_at < note_at < rtf.index('Critical/Catastrophic Identifier')
    out = capsys.readouterr().out
    assert 'TLR-553' in out and '12' in out


def test_mixed_durations_are_unconstrained(tmp_path, capsys):
    run, tlr_dir = _unconstrained_run(tmp_path, lambda d: set_durations(d, [8.0, 23.0]))
    run_pipeline(run)
    assessment = build_assessment(tlr_dir, TIMESTAMP)
    assert assessment.delivery_risk_type == 'unconstrained'
    assert assessment.duration_hours == [8.0, 23.0]
    assert (NOTE_PREFIX + "simulations for this risk used mixed durations "
            "(8, 23 hours).") in rtf_text(tlr_dir)
    out = capsys.readouterr().out
    assert 'TLR-553' in out and '8' in out and '23' in out


def _delete_one_json(tlr_dir):
    os.remove(sim_json_paths(tlr_dir)[0])


def _drop_key_from_one_json(tlr_dir):
    path = sim_json_paths(tlr_dir)[0]
    with open(path) as f:
        data = json.load(f)
    del data['duration_hrs']
    with open(path, 'w') as f:
        json.dump(data, f)


def _corrupt_one_json(tlr_dir):
    with open(sim_json_paths(tlr_dir)[0], 'w') as f:
        f.write('{not json')


@pytest.mark.parametrize('mutate', [_delete_one_json, _drop_key_from_one_json, _corrupt_one_json])
def test_duration_not_recorded_is_unconstrained(tmp_path, capsys, mutate):
    run, tlr_dir = _unconstrained_run(tmp_path, mutate)
    run_pipeline(run)
    assessment = build_assessment(tlr_dir, TIMESTAMP)
    assert assessment.delivery_risk_type == 'unconstrained'
    assert assessment.duration_hours is None
    assert (NOTE_PREFIX + "simulation duration was not recorded for this risk.") in rtf_text(tlr_dir)
    out = capsys.readouterr().out
    assert 'TLR-553' in out and 'duration' in out.lower()


def test_unconstrained_tlr_uses_the_unconstrained_harm_rules(tmp_path):
    run, tlr_dir = _unconstrained_run(tmp_path, lambda d: set_durations(d, 12.0))
    assessment = build_assessment(tlr_dir, TIMESTAMP)
    # TLR-553 is Hypoglycemia 5 throughout: LBGI score >= 2 and >= the DKA score.
    assert all(h == 'Hypoglycemia' and s == '5' for h, s in _stage_harms(assessment).values())


# --------------------------------------------------------------------------- #
# Required column
# --------------------------------------------------------------------------- #

def test_summary_csv_without_dka_index_is_malformed(tmp_path):
    a1 = _a1_run_dir()
    run = make_run(tmp_path, a1, ['TLR-553'])
    tlr_dir = os.path.join(run, 'TLR-553')
    for f in glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv')):
        pd.read_csv(f).drop(columns=['dka_index']).to_csv(f, index=False)
    outcome = build_assessment_result(tlr_dir, TIMESTAMP)
    assert outcome.assessment is None
    assert outcome.status == 'malformed'
    result = process_results_directory(run)
    assert result.written == [] and len(result.skipped) == 1
