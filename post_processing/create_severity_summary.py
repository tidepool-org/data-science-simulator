___author___ = "Shawn Foster"
#!/usr/bin/env python3
"""
Create RTF shell documents for Tidepool Risk Severity Evaluation results.

This script generates placeholder RTF documents for each TLR-* subdirectory
in a simulation results directory, ready to be populated with actual analysis results.
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path


def check_consecutive_low_values(bg_series, threshold=40, min_consecutive=48):
    """
    Check if there are at least min_consecutive values ≤ threshold in a row.
    
    Args:
        bg_series: pandas Series containing glucose values
        threshold: glucose threshold value (default: 40)
        min_consecutive: minimum number of consecutive values needed (default: 48)
    
    Returns:
        bool: True if condition is met, False otherwise
    """
    consecutive_count = 0
    max_consecutive = 0
    
    for value in bg_series:
        if pd.notna(value) and value <= threshold:
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
            if consecutive_count >= min_consecutive:
                return True
        else:
            consecutive_count = 0
    
    return False


def check_catastrophic_conditions(tlr_dir, sim_id):
    """
    Check if a specific sim_id meets catastrophic criteria by analyzing its TSV file.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        sim_id: Simulation ID to check
        
    Returns:
        Tuple of (has_zero_or_negative, has_extended_low) booleans
        Returns (False, False) if TSV cannot be read
    """
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
        
        # Condition 1: Check for any values ≤ 0
        has_zero_or_negative = (bg_series <= 0).any()
        
        # Condition 2: Check for ≥ 48 consecutive values ≤ 40
        has_extended_low = check_consecutive_low_values(bg_series, threshold=40, min_consecutive=48)
        
        return (has_zero_or_negative, has_extended_low)
    
    except Exception as e:
        print(f"    Error reading TSV for {sim_id}: {e}")
        return (False, False)


def identify_severity_4_hypoglycemia(tlr_dir):
    """
    Find all sim_ids with Hypoglycemia (LBGI risk score = 4).
    
    Args:
        tlr_dir: Path to TLR subdirectory
        
    Returns:
        Dictionary mapping sim_id to stage ('pre', 'no_loop', or 'post')
    """
    severity_4_sim_ids = {}
    
    # Find all summary results CSV files
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            if 'sim_id' not in df.columns or 'lbgi_risk_score' not in df.columns:
                continue
            
            # Find rows where LBGI risk score is 4
            for _, row in df.iterrows():
                if row['lbgi_risk_score'] == 4:
                    sim_id = row['sim_id']
                    
                    # Determine stage
                    if sim_id.startswith('pre-Loop_NoMitigations_') or sim_id.startswith('pre-Loop-NoMitigations_'):
                        stage = 'pre'
                    elif sim_id.startswith('pre-noLoop_') or sim_id.startswith('pre-NoLoop_'):
                        stage = 'no_loop'
                    elif sim_id.startswith('post-Loop-WithMitigations_') or sim_id.startswith('post-LoopWithMitigations_'):
                        stage = 'post'
                    else:
                        continue  # Skip if doesn't match expected patterns
                    
                    severity_4_sim_ids[sim_id] = stage
        
        except Exception as e:
            print(f"    Error processing CSV {csv_file}: {e}")
            continue
    
    return severity_4_sim_ids


def assess_and_update_severity(tlr_dir, severity_4_sim_ids):
    """
    Check catastrophic conditions for severity 4 sim_ids and prepare updates.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        severity_4_sim_ids: Dictionary mapping sim_id to stage
        
    Returns:
        Dictionary with assessment results:
        {sim_id: {'stage': str, 'updated_severity': int, 'condition': str}}
    """
    assessment_results = {}
    
    for sim_id, stage in severity_4_sim_ids.items():
        has_zero_or_negative, has_extended_low = check_catastrophic_conditions(tlr_dir, sim_id)
        
        # Determine condition and updated severity
        if has_zero_or_negative:
            condition = 'zero_or_negative'
            updated_severity = 5
        elif has_extended_low:
            condition = 'extended_low'
            updated_severity = 5
        else:
            condition = 'none'
            updated_severity = 4  # Remains Critical
        
        assessment_results[sim_id] = {
            'stage': stage,
            'updated_severity': updated_severity,
            'condition': condition
        }
    
    return assessment_results


def generate_catastrophic_identifier_section(assessment_results):
    """
    Generate RTF content for Critical/Catastrophic Identifier section.
    
    Args:
        assessment_results: Dictionary from assess_and_update_severity()
        
    Returns:
        String containing RTF-formatted content
    """
    if not assessment_results:
        return "Critical/Catastrophic level determination not relevant to this scenario."
    
    # Sort sim_ids for consistent output
    sorted_sim_ids = sorted(assessment_results.keys())
    
    rtf_lines = []
    
    for sim_id in sorted_sim_ids:
        result = assessment_results[sim_id]
        condition = result['condition']
        
        # Generate appropriate message based on condition
        if condition == 'zero_or_negative':
            message = f"Glucose trace for {sim_id} includes values \\u8804? 0 mg/dL."
        elif condition == 'extended_low':
            message = f"Glucose trace for {sim_id} includes values \\u8804? 40 mg/dL for 4 or more hours."
        else:  # condition == 'none'
            message = f"Glucose trace for {sim_id} includes no values \\u8804? 0 mg/dL or \\u8804? 40 mg/dL for \\u8805? 4 hours."
        
        # Add bullet point
        rtf_lines.append(f"\\bullet  {message}")
        rtf_lines.append("\\par")
    
    return "\n".join(rtf_lines)


def extract_metric_data(tlr_dir, column_name, severity_updates=None):
    """
    Extract metric data from all CSV files in a TLR directory.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        column_name: Name of the column to extract (e.g., 'percent_values_ge_70_le_180')
        severity_updates: Optional dictionary from assess_and_update_severity() for updating LBGI scores
        
    Returns:
        Dictionary with keys 'pre', 'no_loop', 'post', each containing a list of metric values
    """
    metric_data = {
        'pre': [],
        'no_loop': [],
        'post': []
    }
    
    # Find all summary results CSV files
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    
    if not csv_files:
        print(f"  Warning: No CSV files found in {tlr_dir}")
        return metric_data
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'sim_id' not in df.columns or column_name not in df.columns:
                print(f"  CSV file malformed; check data configuration: {csv_file}")
                continue
            
            # Extract data for each stage
            for _, row in df.iterrows():
                sim_id = row['sim_id']
                metric_value = row[column_name]
                
                # If we're extracting LBGI risk scores and have severity updates, apply them
                if column_name == 'lbgi_risk_score' and severity_updates and sim_id in severity_updates:
                    metric_value = severity_updates[sim_id]['updated_severity']
                
                # Pre-mitigation
                if sim_id.startswith('pre-Loop_NoMitigations_') or sim_id.startswith('pre-Loop-NoMitigations_'):
                    metric_data['pre'].append(metric_value)
                
                # No Loop
                elif sim_id.startswith('pre-noLoop_') or sim_id.startswith('pre-NoLoop_'):
                    metric_data['no_loop'].append(metric_value)
                
                # Post-mitigation
                elif sim_id.startswith('post-Loop-WithMitigations_') or sim_id.startswith('post-LoopWithMitigations_'):
                    metric_data['post'].append(metric_value)
        
        except Exception as e:
            print(f"  CSV file malformed; check data configuration: {csv_file}")
            print(f"  Error details: {e}")
            continue
    
    return metric_data


def calculate_stage_averages(metric_data):
    """
    Calculate average metric values for each evaluation stage.
    
    Args:
        metric_data: Dictionary with metric values for each stage
        
    Returns:
        Dictionary with formatted average strings for each stage (1 decimal place)
    """
    averages = {}
    
    for stage in ['pre', 'no_loop', 'post']:
        values = metric_data[stage]
        if values:
            avg = sum(values) / len(values)
            averages[stage] = f"{avg:.1f}"
        else:
            averages[stage] = "NA"
    
    return averages


def calculate_integer_averages(metric_data):
    """
    Calculate average metric values for each evaluation stage, rounded to integers.
    
    Args:
        metric_data: Dictionary with metric values for each stage
        
    Returns:
        Dictionary with integer average values for each stage
    """
    averages = {}
    
    for stage in ['pre', 'no_loop', 'post']:
        values = metric_data[stage]
        if values:
            avg = sum(values) / len(values)
            averages[stage] = round(avg)
        else:
            averages[stage] = 0
    
    return averages


def calculate_hyperglycemia_score(tar_value):
    """
    Calculate hyperglycemia score based on TAR percentage.
    
    Args:
        tar_value: TAR percentage as string (e.g., "15.3") or "NA"
        
    Returns:
        Integer score (1 or 2)
    """
    if tar_value == "NA":
        return 1
    
    tar_float = float(tar_value)
    if tar_float < 12.0:
        return 1
    else:
        return 2


def determine_harm_and_severity(lbgi_score, dka_score, hyperglycemia_score):
    """
    Determine harm type and severity score based on risk scores.
    
    Args:
        lbgi_score: Integer LBGI risk score
        dka_score: Integer DKA risk score
        hyperglycemia_score: Integer hyperglycemia score (1 or 2)
        
    Returns:
        Tuple of (harm_type, severity_score) as strings
    """
    # If both lbgi and dka are 0, use hyperglycemia
    if lbgi_score == 0 and dka_score == 0:
        return ("Hyperglycemia", str(hyperglycemia_score))
    
    # If lbgi >= dka (lbgi takes priority in ties), use hypoglycemia
    if lbgi_score >= dka_score:
        return ("Hypoglycemia", str(lbgi_score))
    
    # Otherwise dka > lbgi, use DKA
    return ("DKA", str(dka_score))


def count_profiles(tlr_dir):
    """
    Count the number of profile CSV files in a TLR directory.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        
    Returns:
        Integer count of CSV files
    """
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    return len(csv_files)


def extract_profile_from_filename(csv_path):
    """
    Extract profile name from summary results filename.
    
    Args:
        csv_path: Path to summary results file
        
    Returns:
        Profile name (e.g., "Sensitive") or None if pattern doesn't match
    """
    filename = os.path.basename(csv_path)
    
    # Pattern: summary_results_Simulation-Configuration-TLR-XXXX_ProfileName_profile.csv
    if '_profile.csv' in filename or '_profile' in filename:
        # Split by underscores and find the part before '_profile'
        parts = filename.replace('.csv', '').split('_')
        
        # Find index of 'profile' and get the part before it
        try:
            profile_index = parts.index('profile')
            if profile_index > 0:
                return parts[profile_index - 1]
        except ValueError:
            pass
    
    return None


def get_profile_metrics(tlr_dir, severity_updates=None):
    """
    Extract metrics for each profile at each stage.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        severity_updates: Optional dictionary from assess_and_update_severity() for updating LBGI scores
        
    Returns:
        Dictionary mapping profile names to stage data:
        {
            'ProfileName': {
                'pre': {'lbgi': int, 'dka': int, 'tar': float},
                'no_loop': {...},
                'post': {...}
            }
        }
        Returns None if data is missing or malformed
    """
    profile_data = {}
    
    # Find all summary results CSV files
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    
    if not csv_files:
        return None
    
    for csv_file in csv_files:
        profile_name = extract_profile_from_filename(csv_file)
        
        if not profile_name:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            required_cols = ['sim_id', 'lbgi_risk_score', 'dka_risk_score', 'percent_cgm_gt_180']
            if not all(col in df.columns for col in required_cols):
                return None
            
            profile_data[profile_name] = {
                'pre': {},
                'no_loop': {},
                'post': {}
            }
            
            # Extract data for each stage
            for _, row in df.iterrows():
                sim_id = row['sim_id']
                
                lbgi_score = row['lbgi_risk_score']
                # Apply severity updates if applicable
                if severity_updates and sim_id in severity_updates:
                    lbgi_score = severity_updates[sim_id]['updated_severity']
                
                dka_score = row['dka_risk_score']
                tar_value = row['percent_cgm_gt_180']
                
                # Skip if any value is NA/NaN
                if pd.isna(lbgi_score) or pd.isna(dka_score) or pd.isna(tar_value):
                    continue
                
                # Pre-mitigation
                if sim_id.startswith('pre-Loop_NoMitigations_') or sim_id.startswith('pre-Loop-NoMitigations_'):
                    profile_data[profile_name]['pre'] = {
                        'lbgi': int(lbgi_score),
                        'dka': int(dka_score),
                        'tar': float(tar_value)
                    }
                
                # No Loop
                elif sim_id.startswith('pre-noLoop_') or sim_id.startswith('pre-NoLoop_'):
                    profile_data[profile_name]['no_loop'] = {
                        'lbgi': int(lbgi_score),
                        'dka': int(dka_score),
                        'tar': float(tar_value)
                    }
                
                # Post-mitigation
                elif sim_id.startswith('post-Loop-WithMitigations_') or sim_id.startswith('post-LoopWithMitigations_'):
                    profile_data[profile_name]['post'] = {
                        'lbgi': int(lbgi_score),
                        'dka': int(dka_score),
                        'tar': float(tar_value)
                    }
        
        except Exception as e:
            print(f"  Error processing CSV for outlier detection: {csv_file}")
            print(f"  Error details: {e}")
            return None
    
    return profile_data


def generate_outlier_results_section(tlr_dir, severity_updates=None):
    """
    Generate outlier results section content.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        severity_updates: Optional dictionary from assess_and_update_severity() for updating LBGI scores
        
    Returns:
        String containing outlier results text
    """
    # Get profile metrics
    profile_data = get_profile_metrics(tlr_dir, severity_updates)
    
    if profile_data is None:
        print("  Necessary data not present; check configurations.")
        return "Data not available for outlier analysis."
    
    # Filter out profiles with incomplete data
    complete_profiles = {}
    for profile, stages in profile_data.items():
        if all(stages[stage] for stage in ['pre', 'no_loop', 'post']):
            complete_profiles[profile] = stages
    
    if len(complete_profiles) == 0:
        print("  Necessary data not present; check configurations.")
        return "Data not available for outlier analysis."
    
    if len(complete_profiles) == 1:
        return "Only one profile present, so outliers are not relevant."
    
    # Check for outliers in each stage
    outlier_messages = []
    
    for stage in ['pre', 'no_loop', 'post']:
        stage_name = {'pre': 'Pre-mitigation', 'no_loop': 'No Loop', 'post': 'Post-mitigation'}[stage]
        
        # Determine harm type for each profile at this stage
        profile_harms = {}
        for profile, stages in complete_profiles.items():
            lbgi = stages[stage]['lbgi']
            dka = stages[stage]['dka']
            tar = stages[stage]['tar']
            
            # Calculate hyperglycemia score
            hyper_score = 1 if tar < 12.0 else 2
            
            # Determine harm
            harm, severity = determine_harm_and_severity(lbgi, dka, hyper_score)
            profile_harms[profile] = {
                'harm': harm,
                'lbgi': lbgi,
                'dka': dka,
                'tar': tar
            }
        
        # Group profiles by harm type
        harm_groups = {}
        for profile, data in profile_harms.items():
            harm = data['harm']
            if harm not in harm_groups:
                harm_groups[harm] = []
            harm_groups[harm].append(profile)
        
        # Check each harm group for outliers
        for harm_type, profiles in harm_groups.items():
            if len(profiles) < 2:
                continue  # Need at least 2 profiles to detect outliers
            
            # Check LBGI outliers (for Hypoglycemia harm)
            if harm_type == 'Hypoglycemia':
                lbgi_scores = [profile_harms[p]['lbgi'] for p in profiles]
                median_lbgi = sorted(lbgi_scores)[len(lbgi_scores) // 2]
                
                for profile in profiles:
                    lbgi = profile_harms[profile]['lbgi']
                    if abs(lbgi - median_lbgi) >= 2:
                        outlier_messages.append(
                            f"Outlier profile exists. {profile} has a Hypoglycemia score of {lbgi} at {stage_name}, "
                            f"while other profiles have a Hypoglycemia score of {median_lbgi}."
                        )
            
            # Check DKA outliers (for DKA harm)
            if harm_type == 'DKA':
                dka_scores = [profile_harms[p]['dka'] for p in profiles]
                median_dka = sorted(dka_scores)[len(dka_scores) // 2]
                
                for profile in profiles:
                    dka = profile_harms[profile]['dka']
                    if abs(dka - median_dka) >= 2:
                        outlier_messages.append(
                            f"Outlier profile exists. {profile} has a DKA score of {dka} at {stage_name}, "
                            f"while other profiles have a DKA score of {median_dka}."
                        )
            
            # Check TAR outliers (for Hyperglycemia harm)
            if harm_type == 'Hyperglycemia':
                tar_values = [profile_harms[p]['tar'] for p in profiles]
                
                # Check if one profile is 0.0 and all others are >= 12.0
                zero_profiles = [p for p in profiles if profile_harms[p]['tar'] == 0.0]
                non_zero_profiles = [p for p in profiles if profile_harms[p]['tar'] != 0.0]
                
                if len(zero_profiles) > 0 and len(non_zero_profiles) > 0:
                    all_others_high = all(profile_harms[p]['tar'] >= 12.0 for p in non_zero_profiles)
                    
                    if all_others_high:
                        for zero_profile in zero_profiles:
                            # Get the median TAR of non-zero profiles
                            non_zero_tars = [profile_harms[p]['tar'] for p in non_zero_profiles]
                            median_tar = sorted(non_zero_tars)[len(non_zero_tars) // 2]
                            
                            outlier_messages.append(
                                f"Outlier profile exists. {zero_profile} has a Hyperglycemia percent_cgm_gt_180 of 0.0 at {stage_name}, "
                                f"while other profiles have a Hyperglycemia percent_cgm_gt_180 of {median_tar:.1f}."
                            )
    
    # Return appropriate message
    if outlier_messages:
        return " ".join(outlier_messages)
    else:
        return "No outlier profiles exist. All results are within 1 severity level of one another."


def create_rtf_shell(subdirectory_name, timestamp, tir_averages, tbr_averages, tar_averages,
                     harm_severity_data, profile_count, catastrophic_identifier_content, 
                     outlier_results_content, output_path):
    """
    Create an RTF shell document with metric data populated.
    
    Args:
        subdirectory_name: The full subdirectory name (e.g., "TLR-1119_bike" or "TLR-1119")
        timestamp: Simulation run timestamp
        tir_averages: Dictionary with TIR average values for each stage
        tbr_averages: Dictionary with TBR average values for each stage
        tar_averages: Dictionary with TAR average values for each stage
        harm_severity_data: Dictionary with (harm, severity) tuples for each stage
        profile_count: Number of virtual patient profiles
        catastrophic_identifier_content: RTF content for Critical/Catastrophic Identifier section
        outlier_results_content: Text content for Outlier Results section
        output_path: Full path where the RTF file should be saved
    """
    
    # Format timestamp: YYYY-MM-DD HH:MM:SS
    formatted_timestamp = timestamp.replace('T', ' ').split('.')[0]
    
    rtf_content = r"""{\rtf1\ansi\deff0
{\fonttbl{\f0 Arial;}}
\f0\fs24

{\b\fs28 Risk severity summary for """ + subdirectory_name + r"""}
\par\par

{\b Date and time of simulation run:} """ + formatted_timestamp + r"""
\par\par

Auto-generated output from Tidepool Risk Severity Evaluation Simulator Tool
\par\par

{\b Table of results}
\par\par

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl {\b Evaluation stage}\cell
\pard\intbl {\b Harm}\cell
\pard\intbl {\b Severity}\cell
\pard\intbl {\b TIR % (70 - 180 mg/dL)}\cell
\pard\intbl {\b TBR % (<54 mg/dL)}\cell
\pard\intbl {\b TAR % (>180 mg/dL)}\cell
\row

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl Pre-mitigation\cell
\pard\intbl """ + harm_severity_data['pre'][0] + r"""\cell
\pard\intbl """ + harm_severity_data['pre'][1] + r"""\cell
\pard\intbl """ + tir_averages['pre'] + r"""\cell
\pard\intbl """ + tbr_averages['pre'] + r"""\cell
\pard\intbl """ + tar_averages['pre'] + r"""\cell
\row

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl No Loop\cell
\pard\intbl """ + harm_severity_data['no_loop'][0] + r"""\cell
\pard\intbl """ + harm_severity_data['no_loop'][1] + r"""\cell
\pard\intbl """ + tir_averages['no_loop'] + r"""\cell
\pard\intbl """ + tbr_averages['no_loop'] + r"""\cell
\pard\intbl """ + tar_averages['no_loop'] + r"""\cell
\row

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl Post-mitigation\cell
\pard\intbl """ + harm_severity_data['post'][0] + r"""\cell
\pard\intbl """ + harm_severity_data['post'][1] + r"""\cell
\pard\intbl """ + tir_averages['post'] + r"""\cell
\pard\intbl """ + tbr_averages['post'] + r"""\cell
\pard\intbl """ + tar_averages['post'] + r"""\cell
\row

\pard
\par\par

""" + str(profile_count) + r""" virtual patient profiles aggregated for this summary.
\par\par

{\b Critical/Catastrophic Identifier}
\par\par

""" + catastrophic_identifier_content + r"""
\par\par

{\b Outlier Results}
\par\par

""" + outlier_results_content + r"""

\pard
}"""
    
    with open(output_path, 'w') as f:
        f.write(rtf_content)
    
    print(f"Created shell document: {output_path}")


def extract_simulation_id(summary_file_path):
    """
    Extract simulation ID from summary results filename.
    
    Args:
        summary_file_path: Path to summary results file
        
    Returns:
        Simulation ID (e.g., "TLR-1119")
    """
    filename = os.path.basename(summary_file_path)
    # Extract from pattern: summary_results_Simulation-Configuration-TLR-XXXX
    parts = filename.split('-')
    # Find index of 'TLR' and take TLR-XXXX
    for i, part in enumerate(parts):
        if part == 'TLR':
            # Get TLR and the number after it
            sim_id = f"TLR-{parts[i+1].split('.')[0].split('_')[0]}"
            return sim_id
    return None


def process_results_directory(results_dir):
    """
    Process a results directory and create shell RTF documents for each TLR subdirectory.
    
    Args:
        results_dir: Path to the results directory
    """
    
    # Read metadata.json for timestamp
    metadata_path = os.path.join(results_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {results_dir}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract timestamp - adjust key name as needed based on actual metadata structure
    timestamp = metadata.get('timestamp', metadata.get('run_timestamp', 'Unknown'))
    
    # Find all TLR-* subdirectories
    tlr_dirs = glob.glob(os.path.join(results_dir, 'TLR-*'))
    
    if not tlr_dirs:
        print(f"No TLR-* subdirectories found in {results_dir}")
        return
    
    print(f"Found {len(tlr_dirs)} TLR subdirectories")
    
    for tlr_dir in tlr_dirs:
        print(f"\nProcessing: {tlr_dir}")
        
        # Find summary results files
        summary_files = glob.glob(
            os.path.join(tlr_dir, 'summary_results_Simulation-Configuration-TLR*.csv')
        )
        
        if not summary_files:
            print(f"  Warning: No summary results files found in {tlr_dir}")
            continue
        
        # Extract simulation ID from first summary file
        simulation_id = extract_simulation_id(summary_files[0])
        
        if not simulation_id:
            print(f"  Error: Could not extract simulation ID from {summary_files[0]}")
            continue
        
        print(f"  Simulation ID: {simulation_id}")
        
        # Count profiles
        profile_count = count_profiles(tlr_dir)
        print(f"  Profile count: {profile_count}")
        
        # Step 1: Identify sim_ids with Hypoglycemia severity 4
        print("  Checking for severity 4 hypoglycemia...")
        severity_4_sim_ids = identify_severity_4_hypoglycemia(tlr_dir)
        
        if severity_4_sim_ids:
            print(f"  Found {len(severity_4_sim_ids)} sim_id(s) with severity 4 hypoglycemia")
        
        # Step 2: Assess catastrophic conditions and prepare updates
        assessment_results = {}
        if severity_4_sim_ids:
            print("  Assessing catastrophic conditions...")
            assessment_results = assess_and_update_severity(tlr_dir, severity_4_sim_ids)
            
            # Report assessment results
            for sim_id, result in assessment_results.items():
                sev = result['updated_severity']
                cond = result['condition']
                print(f"    {sim_id}: Severity {sev} ({cond})")
        
        # Step 3: Extract metric data (with updated LBGI scores if applicable)
        tir_data = extract_metric_data(tlr_dir, 'percent_values_ge_70_le_180')
        tir_averages = calculate_stage_averages(tir_data)
        
        tbr_data = extract_metric_data(tlr_dir, 'percent_cgm_lt_54')
        tbr_averages = calculate_stage_averages(tbr_data)
        
        tar_data = extract_metric_data(tlr_dir, 'percent_cgm_gt_180')
        tar_averages = calculate_stage_averages(tar_data)
        
        # Extract LBGI risk scores with severity updates
        lbgi_data = extract_metric_data(tlr_dir, 'lbgi_risk_score', assessment_results)
        lbgi_averages = calculate_integer_averages(lbgi_data)
        
        # Extract DKA risk scores
        dka_data = extract_metric_data(tlr_dir, 'dka_risk_score')
        dka_averages = calculate_integer_averages(dka_data)
        
        # Calculate hyperglycemia scores for each stage
        hyperglycemia_scores = {}
        for stage in ['pre', 'no_loop', 'post']:
            hyperglycemia_scores[stage] = calculate_hyperglycemia_score(tar_averages[stage])
        
        # Determine harm and severity for each stage (with updated LBGI scores)
        harm_severity_data = {}
        for stage in ['pre', 'no_loop', 'post']:
            harm_severity_data[stage] = determine_harm_and_severity(
                lbgi_averages[stage],
                dka_averages[stage],
                hyperglycemia_scores[stage]
            )
        
        # Console output
        print(f"  TIR averages: Pre={tir_averages['pre']}, No Loop={tir_averages['no_loop']}, Post={tir_averages['post']}")
        print(f"  TBR averages: Pre={tbr_averages['pre']}, No Loop={tbr_averages['no_loop']}, Post={tbr_averages['post']}")
        print(f"  TAR averages: Pre={tar_averages['pre']}, No Loop={tar_averages['no_loop']}, Post={tar_averages['post']}")
        print(f"  LBGI averages (updated): Pre={lbgi_averages['pre']}, No Loop={lbgi_averages['no_loop']}, Post={lbgi_averages['post']}")
        print(f"  DKA averages: Pre={dka_averages['pre']}, No Loop={dka_averages['no_loop']}, Post={dka_averages['post']}")
        
        # Output hyperglycemia scores
        for stage, stage_name in [('pre', 'Pre'), ('no_loop', 'No Loop'), ('post', 'Post')]:
            hyper_score = hyperglycemia_scores[stage]
            print(f"  Hyperglycemia score {hyper_score} ({stage_name}-mitigation)")
        
        # Step 4: Generate catastrophic identifier content
        catastrophic_identifier_content = generate_catastrophic_identifier_section(assessment_results)
        
        # Step 5: Generate outlier results content
        print("  Checking for outlier profiles...")
        outlier_results_content = generate_outlier_results_section(tlr_dir, assessment_results)
        
        # Get subdirectory name for the header
        subdirectory_name = os.path.basename(tlr_dir)
        
        # Create RTF shell document with all data including catastrophic identifier and outlier results
        output_path = os.path.join(tlr_dir, f"risk_summary_{simulation_id}.rtf")
        create_rtf_shell(subdirectory_name, timestamp, tir_averages, tbr_averages, tar_averages,
                        harm_severity_data, profile_count, catastrophic_identifier_content, 
                        outlier_results_content, output_path)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create RTF documents for risk severity evaluation results'
    )
    parser.add_argument(
        'results_dir',
        help='Path to results directory (e.g., Risk_Run_2025-10-10T14:30:55.391233)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Directory not found: {args.results_dir}")
        return 1
    
    process_results_directory(args.results_dir)
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
