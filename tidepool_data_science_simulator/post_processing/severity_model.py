#!/usr/bin/env python3
"""
severity_model.py — Structured severity assessment for Tidepool Risk Severity Evaluation.

This module is the *computation* half of the risk-severity tooling. It produces a
structured `SeverityAssessment` object per TLR results directory, decoupled from any
output format. RTF generation (see create_severity_summary.py) and the forthcoming GUI
are both consumers of this object.

Design principle (Phase 2 refactor): all severity logic lives here and returns data;
create_severity_summary.py became a thin RTF renderer over the object this module builds.
The numeric/SOP semantics are preserved exactly from the original create_severity_summary.py
so that RTF output is byte-for-byte identical after the refactor.

SOP grounding (see 'Using the Tidepool Risk Severity Evaluation Tool' docs):
  - LBGI/DKAI risk scores arrive pre-computed in the summary CSVs (DOC-0015 tables).
  - Three fixed stages: pre-mitigation, No Loop (pump-only), post-mitigation (TWI-0006).
  - Averaging across profiles per stage (TWI-0006 2.g).
  - Catastrophic 4->5 escalation read from the TSV (DOC-0015 Table 1 / TWI-0006 2.f).

Provenance flags on scores:
  - round-half-up on averaged scores: script convention (conservative), NOT SOP-mandated.
  - hyperglycemia TAR->score mapping: script convention; DOC-0015 treats hyperglycemia as
    secondary and defines no TAR->severity map. The main-path mapping
    (calculate_hyperglycemia_score) honors SOP intent that a score of 0 means the index
    is truly 0. See KNOWN INCONSISTENCY note in detect_outliers().
"""

import math
import os
import glob
from dataclasses import dataclass, field, asdict
from typing import Optional


# =============================================================================
# Stage prefix definitions (single source of truth)
# -----------------------------------------------------------------------------
# The original code inlined these prefix tuples in four separate places with
# slightly error-prone repetition. Centralizing them here removes that
# duplication risk. The exact spellings are preserved verbatim from the
# original create_severity_summary.py.
#
# NOTE (carried finding): prefix matching is fragile — a mislabeled sim silently
# drops from a stage. build_assessment() records per-stage counts so a consumer
# can surface "stage X aggregated N sims" and catch silent drops. Hardening the
# matching itself is a separate, deferred decision.
# =============================================================================

STAGE_PREFIXES = {
    'pre': (
        'pre-Loop_NoMitigations_',
        'pre-Loop-NoMitigations_',
        'pre-Loop-noMitigations_',
        'pre-LoopNoMitigations_',
        'pre-LoopNoMitigationss_',
    ),
    'no_loop': (
        'pre-noLoop_',
        'pre-NoLoop_',
    ),
    'post': (
        'post-Loop-WithMitigations_',
        'post-LoopWithMitigations_',
        'post-Loop_WithMitigations_',
    ),
}

STAGE_ORDER = ['pre', 'no_loop', 'post']
STAGE_DISPLAY = {'pre': 'Pre-mitigation', 'no_loop': 'No Loop', 'post': 'Post-mitigation'}


def classify_sim_id(sim_id):
    """Return the stage ('pre'/'no_loop'/'post') for a sim_id, or None if unmatched.

    Preserves the original prefix-matching behavior exactly, just centralized.
    """
    for stage, prefixes in STAGE_PREFIXES.items():
        if sim_id.startswith(prefixes):
            return stage
    return None


# =============================================================================
# Dataclasses — the structured assessment object (the result contract)
# =============================================================================

@dataclass
class StageResult:
    """Per-stage aggregated result.

    harm_type / severity are the SOP-facing verdict. The raw averaged component
    scores that fed determine_harm_and_severity are retained so nothing the RTF
    or GUI might need is lost, and so the derivation is auditable.
    """
    stage: str                      # 'pre' | 'no_loop' | 'post'
    harm_type: str                  # 'Severity = baseline' | 'Hypoglycemia' | 'DKA' | 'Hyperglycemia'
    severity: str                   # severity score as string (matches RTF cell)
    tir: str                        # averaged TIR %, 1 dp, or 'NA'
    tbr: str                        # averaged TBR (<54) %, 1 dp, or 'NA'
    tar: str                        # averaged TAR (>180) %, 1 dp, or 'NA'
    lbgi_score_avg: int             # round-half-up averaged LBGI risk score (post-escalation)
    dka_score_avg: int              # round-half-up averaged DKA risk score
    hyperglycemia_score: int        # derived from averaged TAR (main-path mapping)
    n_sims: int                     # number of sims aggregated into this stage (drop-detection)

    def to_dict(self):
        return asdict(self)


@dataclass
class CatastrophicFinding:
    """A severity-4 Hypoglycemia sim assessed for catastrophic (4->5) escalation."""
    sim_id: str
    stage: str
    condition: str                  # 'zero_or_negative' | 'extended_low' | 'none'
    updated_severity: int           # 5 if escalated, else 4

    def to_dict(self):
        return asdict(self)


@dataclass
class OutlierFinding:
    """A profile flagged as a within-stage outlier (>=2 severity levels from median)."""
    stage: str
    profile: str
    harm_type: str
    value: float                    # the profile's score/value on the relevant axis
    comparison_median: float        # the median the other profiles sit at

    def to_dict(self):
        return asdict(self)


@dataclass
class SeverityAssessment:
    """Top-level structured result for one TLR directory.

    This is the object the RTF renderer and the GUI both consume. `to_dict()`
    yields a JSON-serializable structure (dataclass has no auto-JSON, so this
    helper is what gives the teammate's experiments clean JSON round-tripping).
    """
    simulation_id: str
    subdirectory_name: str
    timestamp: str
    profile_count: int
    stages: dict                    # stage -> StageResult
    catastrophic_findings: list = field(default_factory=list)   # list[CatastrophicFinding]
    outlier_findings: list = field(default_factory=list)        # list[OutlierFinding]
    outlier_status: str = 'ok'      # 'ok' | 'no_data' | 'single_profile'

    def to_dict(self):
        return {
            'simulation_id': self.simulation_id,
            'subdirectory_name': self.subdirectory_name,
            'timestamp': self.timestamp,
            'profile_count': self.profile_count,
            'stages': {stage: sr.to_dict() for stage, sr in self.stages.items()},
            'catastrophic_findings': [c.to_dict() for c in self.catastrophic_findings],
            'outlier_findings': [o.to_dict() for o in self.outlier_findings],
            'outlier_status': self.outlier_status,
        }


# =============================================================================
# Pure compute functions
# (lifted verbatim in behavior from the original create_severity_summary.py)
# =============================================================================

def round_half_up(value):
    """Round to nearest integer, always rounding 0.5 UP (not banker's rounding).

    Conservative (higher) risk estimates when averaging severity scores.
    Script convention, not SOP-mandated.
    """
    return math.floor(value + 0.5)


def check_consecutive_low_values(bg_series, threshold=40, min_consecutive=48):
    """True if >= min_consecutive values <= threshold occur consecutively.

    48 five-minute readings = 4 hours (DOC-0015 Table 1 / TWI-0006 2.f).
    Uses pandas notna to skip missing values (which break a run).
    """
    import pandas as pd
    consecutive_count = 0
    for value in bg_series:
        if pd.notna(value) and value <= threshold:
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                return True
        else:
            consecutive_count = 0
    return False


def check_catastrophic_conditions(tlr_dir, sim_id):
    """Read a sim's TSV and test the two catastrophic conditions.

    Returns (has_zero_or_negative, has_extended_low). (False, False) if unreadable.
    """
    import pandas as pd
    tsv_path = os.path.join(tlr_dir, f"{sim_id}.tsv")
    if not os.path.exists(tsv_path):
        print(f"    Warning: TSV file not found for {sim_id}")
        return (False, False)
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        if 'bg' not in df.columns:
            print(f"    Warning: 'bg' column not found in TSV for {sim_id}")
            return (False, False)
        bg_series = df['bg']
        has_zero_or_negative = (bg_series <= 0).any()
        has_extended_low = check_consecutive_low_values(bg_series, threshold=40, min_consecutive=48)
        return (has_zero_or_negative, has_extended_low)
    except Exception as e:
        print(f"    Error reading TSV for {sim_id}: {e}")
        return (False, False)


def identify_severity_4_hypoglycemia(tlr_dir):
    """Find all sim_ids with lbgi_risk_score == 4, mapped to their stage."""
    import pandas as pd
    severity_4_sim_ids = {}
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'sim_id' not in df.columns or 'lbgi_risk_score' not in df.columns:
                continue
            for _, row in df.iterrows():
                if row['lbgi_risk_score'] == 4:
                    sim_id = row['sim_id']
                    stage = classify_sim_id(sim_id)
                    if stage is None:
                        continue
                    severity_4_sim_ids[sim_id] = stage
        except Exception as e:
            print(f"    Error processing CSV {csv_file}: {e}")
            continue
    return severity_4_sim_ids


def assess_and_update_severity(tlr_dir, severity_4_sim_ids):
    """For each severity-4 hypo sim, test catastrophic conditions and set 4 or 5.

    Returns {sim_id: {'stage', 'updated_severity', 'condition'}}.
    """
    assessment_results = {}
    for sim_id, stage in severity_4_sim_ids.items():
        has_zero_or_negative, has_extended_low = check_catastrophic_conditions(tlr_dir, sim_id)
        if has_zero_or_negative:
            condition = 'zero_or_negative'
            updated_severity = 5
        elif has_extended_low:
            condition = 'extended_low'
            updated_severity = 5
        else:
            condition = 'none'
            updated_severity = 4
        assessment_results[sim_id] = {
            'stage': stage,
            'updated_severity': updated_severity,
            'condition': condition,
        }
    return assessment_results


def extract_metric_data(tlr_dir, column_name, severity_updates=None):
    """Collect a metric column's values per stage across all profile CSVs.

    If column_name == 'lbgi_risk_score' and severity_updates is given, the
    escalated (4->5) values replace the raw ones — matching the original.
    """
    import pandas as pd
    metric_data = {'pre': [], 'no_loop': [], 'post': []}
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    if not csv_files:
        print(f"  Warning: No CSV files found in {tlr_dir}")
        return metric_data
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'sim_id' not in df.columns or column_name not in df.columns:
                print(f"  CSV file malformed; check data configuration: {csv_file}")
                continue
            for _, row in df.iterrows():
                sim_id = row['sim_id']
                metric_value = row[column_name]
                if column_name == 'lbgi_risk_score' and severity_updates and sim_id in severity_updates:
                    metric_value = severity_updates[sim_id]['updated_severity']
                stage = classify_sim_id(sim_id)
                if stage is not None:
                    metric_data[stage].append(metric_value)
        except Exception as e:
            print(f"  CSV file malformed; check data configuration: {csv_file}")
            print(f"  Error details: {e}")
            continue
    return metric_data


def calculate_stage_averages(metric_data):
    """Average each stage to 1 decimal place as a string; 'NA' if no data."""
    averages = {}
    for stage in STAGE_ORDER:
        values = metric_data[stage]
        averages[stage] = f"{sum(values) / len(values):.1f}" if values else "NA"
    return averages


def calculate_integer_averages(metric_data):
    """Average each stage, round-half-up to int; 0 if no data."""
    averages = {}
    for stage in STAGE_ORDER:
        values = metric_data[stage]
        averages[stage] = round_half_up(sum(values) / len(values)) if values else 0
    return averages


def calculate_hyperglycemia_score(tar_value):
    """Main-path TAR->hyperglycemia score. SOP-honoring: 0 only if TAR truly 0.

    'NA' (no data) -> 1 (preserves original behavior).
    0.0 -> 0 ; <12.0 -> 1 ; >=12.0 -> 2.
    """
    if tar_value == "NA":
        return 1
    tar_float = float(tar_value)
    if tar_float == 0.0:
        return 0
    if tar_float < 12.0:
        return 1
    else:
        return 2


def determine_harm_and_severity(lbgi_score, dka_score, hyperglycemia_score):
    """Derive (harm_type, severity_str) from the three component scores.

    SOP logic, preserved exactly:
      - all zero -> baseline
      - lbgi<=1 and dka==0 -> Hyperglycemia (score = hyperglycemia_score)
      - lbgi>=dka (LBGI wins ties) -> Hypoglycemia (score = lbgi)
      - else -> DKA (score = dka)
    """
    if lbgi_score == 0 and dka_score == 0 and hyperglycemia_score == 0:
        return ("Severity = baseline", "0")
    if lbgi_score <= 1 and dka_score == 0:
        return ("Hyperglycemia", str(hyperglycemia_score))
    if lbgi_score >= dka_score:
        return ("Hypoglycemia", str(lbgi_score))
    return ("DKA", str(dka_score))


def count_profiles(tlr_dir):
    """Number of summary_results_*.csv files (== profile count)."""
    return len(glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv')))


def extract_profile_from_filename(csv_path):
    """Parse the profile name out of a summary_results_*_<Profile>_profile.csv filename."""
    filename = os.path.basename(csv_path)
    if '_profile.csv' in filename or '_profile' in filename:
        parts = filename.replace('.csv', '').split('_')
        try:
            profile_index = parts.index('profile')
            if profile_index > 0:
                return parts[profile_index - 1]
        except ValueError:
            pass
    return None


def get_profile_metrics(tlr_dir, severity_updates=None):
    """Per-profile, per-stage {lbgi, dka, tar} for outlier detection.

    Returns None if any CSV is malformed/missing required columns (preserves original).
    """
    import pandas as pd
    profile_data = {}
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    if not csv_files:
        return None
    for csv_file in csv_files:
        profile_name = extract_profile_from_filename(csv_file)
        if not profile_name:
            continue
        try:
            df = pd.read_csv(csv_file)
            required_cols = ['sim_id', 'lbgi_risk_score', 'dka_risk_score', 'percent_cgm_gt_180']
            if not all(col in df.columns for col in required_cols):
                return None
            profile_data[profile_name] = {'pre': {}, 'no_loop': {}, 'post': {}}
            for _, row in df.iterrows():
                sim_id = row['sim_id']
                lbgi_score = row['lbgi_risk_score']
                if severity_updates and sim_id in severity_updates:
                    lbgi_score = severity_updates[sim_id]['updated_severity']
                dka_score = row['dka_risk_score']
                tar_value = row['percent_cgm_gt_180']
                if pd.isna(lbgi_score) or pd.isna(dka_score) or pd.isna(tar_value):
                    continue
                stage = classify_sim_id(sim_id)
                if stage is not None:
                    profile_data[profile_name][stage] = {
                        'lbgi': int(lbgi_score),
                        'dka': int(dka_score),
                        'tar': float(tar_value),
                    }
        except Exception as e:
            print(f"  Error processing CSV for outlier detection: {csv_file}")
            print(f"  Error details: {e}")
            return None
    return profile_data


def extract_simulation_id(summary_file_path):
    """Extract 'TLR-XXXX' from a summary_results filename."""
    filename = os.path.basename(summary_file_path)
    parts = filename.split('-')
    for i, part in enumerate(parts):
        if part == 'TLR':
            return f"TLR-{parts[i + 1].split('.')[0].split('_')[0]}"
    return None


# =============================================================================
# Outlier detection — SPLIT from rendering (returns structured findings)
# =============================================================================

def detect_outliers(tlr_dir, severity_updates=None):
    """Detect within-stage outlier profiles; return (findings, status).

    Returns:
        (list[OutlierFinding], status_str)
        status_str is one of:
          'ok'                -> findings list is authoritative (may be empty)
          'no_data'           -> data missing/malformed
          'single_profile'    -> only one profile, outliers not meaningful

    This is the detection half only. Rendering the RTF/GUI text from these
    findings lives in the consumer (create_severity_summary.render_*).

    ---------------------------------------------------------------------------
    KNOWN INCONSISTENCY (preserved deliberately; flagged for adjudication):
    This function computes each profile's per-stage hyperglycemia score inline as
        hyper_score = 1 if tar < 12.0 else 2
    which has NO zero case. This CONTRADICTS SOP intent (a score of 0 should mean
    the index — TAR% — is truly 0), and it differs from the main path's
    calculate_hyperglycemia_score(), which correctly returns 0 for TAR==0.0.

    The main path is the SOP-correct one; the outlier path is the deviant one.
    It is preserved here EXACTLY so the refactor produces byte-identical output.
    Do NOT "fix" it as part of the refactor — changing it alters results and is a
    separate, deliberate correctness decision for Shawn to make.
    ---------------------------------------------------------------------------
    """
    profile_data = get_profile_metrics(tlr_dir, severity_updates)

    if profile_data is None:
        print("  Necessary data not present; check configurations.")
        return ([], 'no_data')

    complete_profiles = {
        profile: stages
        for profile, stages in profile_data.items()
        if all(stages[stage] for stage in STAGE_ORDER)
    }

    if len(complete_profiles) == 0:
        print("  Necessary data not present; check configurations.")
        return ([], 'no_data')

    if len(complete_profiles) == 1:
        return ([], 'single_profile')

    findings = []

    for stage in STAGE_ORDER:
        # Determine harm type per profile at this stage.
        profile_harms = {}
        for profile, stages in complete_profiles.items():
            lbgi = stages[stage]['lbgi']
            dka = stages[stage]['dka']
            tar = stages[stage]['tar']
            # PRESERVED inconsistency (see docstring): no zero case here.
            hyper_score = 1 if tar < 12.0 else 2
            harm, _severity = determine_harm_and_severity(lbgi, dka, hyper_score)
            profile_harms[profile] = {'harm': harm, 'lbgi': lbgi, 'dka': dka, 'tar': tar}

        # Group by harm type.
        harm_groups = {}
        for profile, data in profile_harms.items():
            harm_groups.setdefault(data['harm'], []).append(profile)

        for harm_type, profiles in harm_groups.items():
            if len(profiles) < 2:
                continue

            if harm_type == 'Hypoglycemia':
                lbgi_scores = [profile_harms[p]['lbgi'] for p in profiles]
                median_lbgi = sorted(lbgi_scores)[len(lbgi_scores) // 2]
                for profile in profiles:
                    lbgi = profile_harms[profile]['lbgi']
                    if abs(lbgi - median_lbgi) >= 2:
                        findings.append(OutlierFinding(
                            stage=stage, profile=profile, harm_type='Hypoglycemia',
                            value=float(lbgi), comparison_median=float(median_lbgi),
                        ))

            if harm_type == 'DKA':
                dka_scores = [profile_harms[p]['dka'] for p in profiles]
                median_dka = sorted(dka_scores)[len(dka_scores) // 2]
                for profile in profiles:
                    dka = profile_harms[profile]['dka']
                    if abs(dka - median_dka) >= 2:
                        findings.append(OutlierFinding(
                            stage=stage, profile=profile, harm_type='DKA',
                            value=float(dka), comparison_median=float(median_dka),
                        ))

            if harm_type == 'Hyperglycemia':
                zero_profiles = [p for p in profiles if profile_harms[p]['tar'] == 0.0]
                non_zero_profiles = [p for p in profiles if profile_harms[p]['tar'] != 0.0]
                if len(zero_profiles) > 0 and len(non_zero_profiles) > 0:
                    all_others_high = all(profile_harms[p]['tar'] >= 12.0 for p in non_zero_profiles)
                    if all_others_high:
                        non_zero_tars = [profile_harms[p]['tar'] for p in non_zero_profiles]
                        median_tar = sorted(non_zero_tars)[len(non_zero_tars) // 2]
                        for zero_profile in zero_profiles:
                            findings.append(OutlierFinding(
                                stage=stage, profile=zero_profile, harm_type='Hyperglycemia',
                                value=0.0, comparison_median=float(median_tar),
                            ))

    return (findings, 'ok')


# =============================================================================
# Orchestrator — build the structured assessment (no I/O side effects beyond reads)
# =============================================================================

def build_assessment(tlr_dir, timestamp):
    """Build a SeverityAssessment for one TLR directory.

    Mirrors the per-TLR computation in the original process_results_directory(),
    but RETURNS the structured object instead of writing RTF. Returns None if the
    directory has no usable summary files (caller decides how to report).
    """
    summary_files = glob.glob(
        os.path.join(tlr_dir, 'summary_results_Simulation-Configuration-TLR*.csv')
    )
    if not summary_files:
        print(f"  Warning: No summary results files found in {tlr_dir}")
        return None

    simulation_id = extract_simulation_id(summary_files[0])
    if not simulation_id:
        print(f"  Error: Could not extract simulation ID from {summary_files[0]}")
        return None

    profile_count = count_profiles(tlr_dir)

    # Catastrophic (4->5) assessment.
    severity_4_sim_ids = identify_severity_4_hypoglycemia(tlr_dir)
    assessment_results = {}
    if severity_4_sim_ids:
        assessment_results = assess_and_update_severity(tlr_dir, severity_4_sim_ids)

    # Metric extraction + averaging (LBGI carries the escalation).
    tir_averages = calculate_stage_averages(extract_metric_data(tlr_dir, 'percent_values_ge_70_le_180'))
    tbr_averages = calculate_stage_averages(extract_metric_data(tlr_dir, 'percent_cgm_lt_54'))
    tar_averages = calculate_stage_averages(extract_metric_data(tlr_dir, 'percent_cgm_gt_180'))
    lbgi_data = extract_metric_data(tlr_dir, 'lbgi_risk_score', assessment_results)
    lbgi_averages = calculate_integer_averages(lbgi_data)
    dka_averages = calculate_integer_averages(extract_metric_data(tlr_dir, 'dka_risk_score'))

    # Per-stage n for silent-drop detection.
    n_by_stage = {stage: len(lbgi_data[stage]) for stage in STAGE_ORDER}

    # Harm/severity per stage (main-path hyperglycemia mapping).
    stages = {}
    for stage in STAGE_ORDER:
        hyper = calculate_hyperglycemia_score(tar_averages[stage])
        harm, severity = determine_harm_and_severity(lbgi_averages[stage], dka_averages[stage], hyper)
        stages[stage] = StageResult(
            stage=stage,
            harm_type=harm,
            severity=severity,
            tir=tir_averages[stage],
            tbr=tbr_averages[stage],
            tar=tar_averages[stage],
            lbgi_score_avg=lbgi_averages[stage],
            dka_score_avg=dka_averages[stage],
            hyperglycemia_score=hyper,
            n_sims=n_by_stage[stage],
        )

    # Catastrophic findings as structured objects.
    catastrophic_findings = [
        CatastrophicFinding(
            sim_id=sim_id,
            stage=result['stage'],
            condition=result['condition'],
            updated_severity=result['updated_severity'],
        )
        for sim_id, result in sorted(assessment_results.items())
    ]

    # Outlier findings (detection only) + status, both carried on the object so
    # renderers never re-read the directory or re-derive the branch.
    outlier_list, outlier_status = detect_outliers(tlr_dir, assessment_results)

    return SeverityAssessment(
        simulation_id=simulation_id,
        subdirectory_name=os.path.basename(tlr_dir),
        timestamp=timestamp,
        profile_count=profile_count,
        stages=stages,
        catastrophic_findings=catastrophic_findings,
        outlier_findings=outlier_list,
        outlier_status=outlier_status,
    )
