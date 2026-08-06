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

Failure reporting (TRSET-28): a directory that cannot be assessed reports WHY.
build_assessment_result returns an AssessmentOutcome whose status separates a
genuinely empty directory from real-but-malformed data; build_assessment remains a
wrapper returning Optional[SeverityAssessment] for consumers typed on it. Malformed
input is never reported as absent input -- see detect_outliers' 'malformed_data'
status and SeverityAssessment.usable_profile_count. See README_severity_model.md.

Provenance flags on scores:
  - round-half-up on averaged scores: script convention (conservative), NOT SOP-mandated.
  - hyperglycemia TAR->score mapping: script convention; DOC-0015 treats hyperglycemia as
    secondary and defines no TAR->severity map. calculate_hyperglycemia_score honors SOP
    intent that a score of 0 means the index is truly 0, and is now the ONLY mapping in
    the module -- detect_outliers used to compute its own without a zero case.
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

# The harm label for "no harm indicated" (all three component scores zero). Named
# because detect_outliers has to reason about this group as well as 'Hyperglycemia'
# -- see the hyperglycemia-axis comment there. The string is the RTF cell text, so
# it is exactly what determine_harm_and_severity returned before it was named.
BASELINE_HARM = 'Severity = baseline'


# =============================================================================
# Summary results discovery (single source of truth)
# -----------------------------------------------------------------------------
# The module previously matched summary CSVs with two different patterns:
# build_assessment required 'summary_results_Simulation-Configuration-TLR*.csv'
# while count_profiles, extract_metric_data, get_profile_metrics and
# identify_severity_4_hypoglycemia all used the looser 'summary_results_*.csv'. A
# directory matching only the loose pattern was therefore reported unusable even
# though every downstream helper would have read it. All five now go through
# find_summary_files().
#
# REQUIRED_SUMMARY_COLUMNS is the set a summary CSV must carry to contribute to
# the severity VERDICT: the sim identity plus the three columns that feed
# determine_harm_and_severity. TIR/TBR ('percent_values_ge_70_le_180',
# 'percent_cgm_lt_54') and the raw values ('lbgi', 'dka_index') are *reported*
# metrics, not verdict inputs -- they degrade to 'NA' by design, so their absence
# must not mark a file malformed.
# =============================================================================

SUMMARY_RESULTS_GLOB = 'summary_results_*.csv'

REQUIRED_SUMMARY_COLUMNS = (
    'sim_id',
    'lbgi_risk_score',
    'dka_risk_score',
    'percent_cgm_gt_180',
)


def find_summary_files(tlr_dir):
    """Every summary results CSV in tlr_dir, sorted.

    Sorted, not raw glob order: simulation-ID resolution reads the filenames in
    order, and glob returns whatever the filesystem hands back. Under the old
    narrow pattern every match carried the same 'Simulation-Configuration-TLR'
    stem so arbitrary order was harmless; under the loose pattern it is not.
    """
    return sorted(glob.glob(os.path.join(tlr_dir, SUMMARY_RESULTS_GLOB)))


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
    # Raw averaged metric VALUES (not the 0-4 risk scores above): the mean raw
    # LBGI and DKA-index across the stage's sims, truncated to 2dp (never
    # rounded, no 4->5 escalation), or 'NA' when there is no data. Whole numbers
    # carry no decimal ('3'); fractional values drop trailing zeros ('2.5',
    # '3.14'). Surfaced alongside the scores so a consumer (the GUI stage table,
    # the RTF) can show the underlying value, not only the escalated score.
    # Defaulted so existing positional constructors keep working;
    # build_assessment always populates them.
    lbgi_value_avg: str = "NA"      # averaged raw LBGI value (summary column 'lbgi')
    dka_index_value_avg: str = "NA"  # averaged raw DKA index (summary column 'dka_index')

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
    profile_count: int              # N: total summary results files present
    stages: dict                    # stage -> StageResult
    catastrophic_findings: list = field(default_factory=list)   # list[CatastrophicFinding]
    outlier_findings: list = field(default_factory=list)        # list[OutlierFinding]
    # 'ok' | 'no_data' | 'malformed_data' | 'incomplete_stages' | 'single_profile'
    outlier_status: str = 'ok'
    # M: how many of the N files could actually contribute a value. None means
    # "not measured" and is treated as M == N by consumers, so an object built by
    # an older positional constructor renders exactly as it used to.
    # build_assessment always populates it.
    usable_profile_count: Optional[int] = None

    def to_dict(self):
        return {
            'simulation_id': self.simulation_id,
            'subdirectory_name': self.subdirectory_name,
            'timestamp': self.timestamp,
            'profile_count': self.profile_count,
            'usable_profile_count': self.usable_profile_count,
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
    csv_files = find_summary_files(tlr_dir)
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
    csv_files = find_summary_files(tlr_dir)
    if not csv_files:
        print(f"  Warning: No CSV files found in {tlr_dir}")
        return metric_data
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'sim_id' not in df.columns or column_name not in df.columns:
                # A REQUIRED column absent means the file is malformed. An
                # optional/reported column absent is the designed 'NA' degrade
                # (older CSVs predate 'lbgi'/'dka_index'), so calling it
                # malformed cried wolf on valid data -- but it is still not
                # silent: say which metric will report NA.
                if 'sim_id' not in df.columns or column_name in REQUIRED_SUMMARY_COLUMNS:
                    print(f"  CSV file malformed; check data configuration: {csv_file}")
                else:
                    print(f"  Column '{column_name}' not present in {csv_file}; "
                          f"this metric will report NA.")
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


def truncate_2dp(value):
    """Truncate toward zero at hundredths (NOT rounding).

    Deliberately not round_half_up / not f-string rounding: the raw LBGI and
    DKA-index averages are reported as truncated values, so a stage never shows a
    value higher than the data supports.
    """
    return math.trunc(value * 100) / 100


def calculate_truncated_averages(metric_data):
    """Average each stage, truncate to 2dp, format as a string; 'NA' if no data.

    Mirrors calculate_stage_averages but for raw metric VALUES rather than the
    1-dp percentages. Formatting: whole numbers carry no decimal (3.0 -> '3'),
    fractional values drop trailing zeros (2.50 -> '2.5', 3.14 -> '3.14'). Bare
    str(float) is unusable here -- str(3.0) == '3.0' violates the whole-number
    rule.
    """
    averages = {}
    for stage in STAGE_ORDER:
        values = metric_data[stage]
        if not values:
            averages[stage] = "NA"
            continue
        truncated = truncate_2dp(sum(values) / len(values))
        if truncated == int(truncated):
            averages[stage] = str(int(truncated))
        else:
            averages[stage] = "{:.2f}".format(truncated).rstrip("0")
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
        return (BASELINE_HARM, "0")
    if lbgi_score <= 1 and dka_score == 0:
        return ("Hyperglycemia", str(hyperglycemia_score))
    if lbgi_score >= dka_score:
        return ("Hypoglycemia", str(lbgi_score))
    return ("DKA", str(dka_score))


def count_profiles(tlr_dir):
    """Number of summary results files (== total profile count, N).

    This is the TOTAL, which is not the same as the number that CONTRIBUTED a
    value -- see count_usable_profiles for M.
    """
    return len(find_summary_files(tlr_dir))


def classify_summary_files(tlr_dir):
    """Split the directory's summary CSVs into (usable, unusable) path lists.

    Usable means: parses as CSV and carries every REQUIRED_SUMMARY_COLUMNS entry,
    i.e. it can contribute to the severity verdict. This is what separates
    "N profiles were aggregated" from "N files were present": extract_metric_data
    skips a malformed file and averages the remainder, so the file count alone
    could name more profiles than contributed a single value.
    """
    import pandas as pd
    usable, unusable = [], []
    for csv_file in find_summary_files(tlr_dir):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  Summary results file unreadable: {csv_file}")
            print(f"  Error details: {e}")
            unusable.append(csv_file)
            continue
        missing = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in df.columns]
        if missing:
            print(f"  Summary results file missing required column(s) {missing}: {csv_file}")
            unusable.append(csv_file)
            continue
        usable.append(csv_file)
    return usable, unusable


def count_usable_profiles(tlr_dir):
    """Number of summary results files that can contribute a value (M)."""
    return len(classify_summary_files(tlr_dir)[0])


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

    Returns ``(profile_data, status)``:
      'ok'              -> profile_data is a dict (possibly empty, if no filename
                           yielded a profile name)
      'no_data'         -> no summary results files at all; profile_data is None
      'malformed_data'  -> files present but unreadable or missing a required
                           column; profile_data is None

    CONTRACT CHANGE: this used to return a bare ``None`` for both the absent and
    the malformed case, collapsing them into detect_outliers' 'no_data' and thence
    into an RTF line asserting the data was *unavailable* rather than *unusable*.
    A malformed file still abandons the whole analysis (unchanged -- partially
    analyzing would change which outliers are FOUND, a results change out of scope
    here); it is only the REPORTING of that refusal that is now accurate.
    """
    import pandas as pd
    profile_data = {}
    csv_files = find_summary_files(tlr_dir)
    if not csv_files:
        return (None, 'no_data')
    for csv_file in csv_files:
        profile_name = extract_profile_from_filename(csv_file)
        if not profile_name:
            continue
        try:
            df = pd.read_csv(csv_file)
            if not all(col in df.columns for col in REQUIRED_SUMMARY_COLUMNS):
                print(f"  Summary results file missing required columns: {csv_file}")
                return (None, 'malformed_data')
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
            return (None, 'malformed_data')
    return (profile_data, 'ok')


def extract_simulation_id(summary_file_path):
    """Extract 'TLR-XXXX' from a summary_results filename."""
    filename = os.path.basename(summary_file_path)
    parts = filename.split('-')
    for i, part in enumerate(parts):
        if part == 'TLR':
            return f"TLR-{parts[i + 1].split('.')[0].split('_')[0]}"
    return None


def resolve_simulation_id(summary_files):
    """The first TLR ID any of these filenames yields, or None if none does.

    build_assessment used to read summary_files[0] alone. Under the old narrow
    glob every candidate carried the 'Simulation-Configuration-TLR' stem, so
    whichever one glob returned first gave the same answer. The loose glob admits
    filenames with no TLR part, so taking [0] blindly would skip a directory that
    renders today purely on filesystem ordering.
    """
    for summary_file in summary_files:
        simulation_id = extract_simulation_id(summary_file)
        if simulation_id:
            return simulation_id
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
          'no_data'           -> no per-profile results exist to compare
          'malformed_data'    -> data IS present but unreadable/missing a required
                                 column, so the analysis was not performed
          'incomplete_stages' -> profiles exist and parse, but none carries results
                                 for all three stages, so there is nothing to
                                 compare like-for-like
          'single_profile'    -> only one profile, outliers not meaningful

    'no_data' used to cover all three of absent, malformed and incomplete input,
    which the renderer then reported as "Data not available" -- a regulatory
    document asserting the data was missing when it was in fact corrupt, or real but
    thin. All three are now separate, and only genuine absence keeps that sentence.

    This is the detection half only. Rendering the RTF/GUI text from these
    findings lives in the consumer (create_severity_summary.render_*).

    ---------------------------------------------------------------------------
    RESOLVED (was: KNOWN INCONSISTENCY). This function used to compute each
    profile's per-stage hyperglycemia score inline as
        hyper_score = 1 if tar < 12.0 else 2
    which has NO zero case, contradicting SOP intent (a score of 0 should mean the
    index -- TAR% -- is truly 0) and disagreeing with calculate_hyperglycemia_score.
    It now calls that function, so the module has ONE such mapping.

    Correcting the score reclassifies a profile with TAR == 0 and no hypo/DKA risk:
    determine_harm_and_severity(0, 0, 0) is baseline, where the old score of 1 made
    it 'Hyperglycemia'. Since the zero-TAR outlier check keyed on that group, the
    naive fix would have silently STOPPED FLAGGING the cleanest profiles. The check
    now spans the Hyperglycemia and baseline groups together -- provably the same
    population as the old Hyperglycemia group -- so no finding is gained or lost.
    See the hyperglycemia-axis block below.
    ---------------------------------------------------------------------------
    """
    profile_data, metrics_status = get_profile_metrics(tlr_dir, severity_updates)

    if metrics_status == 'malformed_data':
        print("  Summary results data present but unusable; check data configuration.")
        return ([], 'malformed_data')

    if not profile_data:
        # Either no summary files at all (get_profile_metrics said 'no_data', and
        # profile_data is None), or files whose names identify no profile -- an
        # aggregate-only directory with no '*_profile.csv'. Either way there are no
        # per-profile results in existence to compare, which is what 'no_data' means.
        print("  Necessary data not present; check configurations.")
        return ([], 'no_data')

    complete_profiles = {
        profile: stages
        for profile, stages in profile_data.items()
        if all(stages[stage] for stage in STAGE_ORDER)
    }

    if len(complete_profiles) == 0:
        # Profiles DO exist and parsed; none carries results for all three stages,
        # so there is no like-for-like comparison to make. Distinct from 'no_data'
        # (nothing to compare) and from 'malformed_data' (unreadable): this data is
        # readable and real, just too thin for a cross-stage comparison. Reporting it
        # as absent hid a recoverable configuration problem -- a run that produced
        # only some stages -- behind the same sentence as a missing directory.
        print(f"  No profile in {tlr_dir} has results for all three evaluation "
              f"stages ({len(profile_data)} profile(s) found); "
              f"outlier comparison needs at least two that do.")
        return ([], 'incomplete_stages')

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
            # The module's single TAR->score mapping (see the RESOLVED note above).
            hyper_score = calculate_hyperglycemia_score(tar)
            harm, _severity = determine_harm_and_severity(lbgi, dka, hyper_score)
            profile_harms[profile] = {'harm': harm, 'lbgi': lbgi, 'dka': dka, 'tar': tar}

        # Group by harm type.
        harm_groups = {}
        for profile, data in profile_harms.items():
            harm_groups.setdefault(data['harm'], []).append(profile)

        # The hyperglycemia-AXIS population: profiles compared on TAR% rather than
        # on a risk score. It spans two harm groups, because a profile with TAR == 0
        # and no hypo/DKA risk is now correctly 'Severity = baseline' rather than
        # 'Hyperglycemia'. Their union is exactly the old Hyperglycemia group
        # ('lbgi <= 1 and dka == 0' -- baseline is the subset of that with TAR == 0),
        # so the compared population is unchanged by the score fix.
        #
        # Built from profile_harms, so profile order matches complete_profiles order
        # as the per-group lists did; and evaluated at whichever of the two groups
        # harm_groups reaches first, which is where the undivided Hyperglycemia group
        # sat. Findings therefore keep both their identity and their emitted order.
        hyper_axis_harms = ('Hyperglycemia', BASELINE_HARM)
        hyper_axis_profiles = [
            profile for profile, data in profile_harms.items()
            if data['harm'] in hyper_axis_harms
        ]
        hyper_axis_at = next(
            (harm_type for harm_type in harm_groups if harm_type in hyper_axis_harms),
            None,
        )

        for harm_type, profiles in harm_groups.items():
            if harm_type == hyper_axis_at and len(hyper_axis_profiles) >= 2:
                zero_profiles = [p for p in hyper_axis_profiles if profile_harms[p]['tar'] == 0.0]
                non_zero_profiles = [p for p in hyper_axis_profiles if profile_harms[p]['tar'] != 0.0]
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

    return (findings, 'ok')


# =============================================================================
# Orchestrator — build the structured assessment (no I/O side effects beyond reads)
# =============================================================================

@dataclass
class AssessmentOutcome:
    """What build_assessment_result found in one TLR directory.

    status:
      'ok'        -> assessment is a SeverityAssessment
      'empty'     -> no summary results files at all (nothing ran here);
                     assessment is None
      'malformed' -> summary files ARE present but none is usable, or none names a
                     TLR simulation; assessment is None

    'empty' and 'malformed' both used to be a bare None, so a caller could not tell
    "nothing ran here" from "the data is broken." Both remain PARTIAL outcomes for
    one directory -- neither is raised, so a run containing one still summarizes
    every other directory (TRSET-27's fatal-vs-partial split).

    detail is a human-readable reason a caller can report verbatim.
    """
    assessment: Optional[SeverityAssessment]
    status: str
    detail: str = ''

    def to_dict(self):
        return {
            'assessment': self.assessment.to_dict() if self.assessment else None,
            'status': self.status,
            'detail': self.detail,
        }


def build_assessment(tlr_dir, timestamp):
    """The SeverityAssessment for one TLR directory, or None if unusable.

    Backwards-compatible wrapper over build_assessment_result(), kept so consumers
    typed on Optional[SeverityAssessment] -- the GUI runner's RiskDirRunResult, and
    the GUI repo that reads it -- keep working unchanged. A caller that needs to
    tell an empty directory from malformed data should call build_assessment_result
    directly.
    """
    return build_assessment_result(tlr_dir, timestamp).assessment


def build_assessment_result(tlr_dir, timestamp):
    """Build an AssessmentOutcome for one TLR directory.

    Mirrors the per-TLR computation in the original process_results_directory(),
    but RETURNS the structured object instead of writing RTF.
    """
    summary_files = find_summary_files(tlr_dir)
    if not summary_files:
        print(f"  Warning: No summary results files found in {tlr_dir}")
        return AssessmentOutcome(
            None, 'empty',
            f"no {SUMMARY_RESULTS_GLOB} files found (nothing ran in this directory)",
        )

    usable_files, unusable_files = classify_summary_files(tlr_dir)
    if not usable_files:
        # Every file unusable used to render a COMPLETE document with every metric
        # 'NA'/0 and "Data not available for outlier analysis." -- a regulatory
        # document asserting near-baseline results from data that could not be
        # read. It is a malformed directory, and produces no document.
        print(f"  Error: {len(unusable_files)} summary results file(s) present in "
              f"{tlr_dir}, none usable")
        return AssessmentOutcome(
            None, 'malformed',
            f"{len(unusable_files)} summary results file(s) present, none usable: "
            f"unreadable or missing required column(s) "
            f"{list(REQUIRED_SUMMARY_COLUMNS)}",
        )

    simulation_id = resolve_simulation_id(summary_files)
    if not simulation_id:
        print(f"  Error: Could not extract simulation ID from any summary results "
              f"file in {tlr_dir}")
        return AssessmentOutcome(
            None, 'malformed',
            "could not extract a TLR simulation ID from any summary results filename",
        )

    profile_count = len(summary_files)
    usable_profile_count = len(usable_files)

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
    # Raw averaged metric values (underlying LBGI / DKA-index, not the risk
    # scores) for consumers that surface the value itself. Truncated to 2dp, NOT
    # rounded, and deliberately extracted WITHOUT severity_updates -- no 4->5
    # escalation applies to a raw value. Degrades to 'NA' if the summary CSVs lack
    # the column (extract_metric_data warns and returns empty).
    lbgi_value_averages = calculate_truncated_averages(extract_metric_data(tlr_dir, 'lbgi'))
    dka_index_value_averages = calculate_truncated_averages(extract_metric_data(tlr_dir, 'dka_index'))

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
            lbgi_value_avg=lbgi_value_averages[stage],
            dka_index_value_avg=dka_index_value_averages[stage],
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

    assessment = SeverityAssessment(
        simulation_id=simulation_id,
        subdirectory_name=os.path.basename(tlr_dir),
        timestamp=timestamp,
        profile_count=profile_count,
        stages=stages,
        catastrophic_findings=catastrophic_findings,
        outlier_findings=outlier_list,
        outlier_status=outlier_status,
        usable_profile_count=usable_profile_count,
    )
    return AssessmentOutcome(assessment, 'ok')
