#!/usr/bin/env python3
"""
Compare JSON scenarios between loop_risk_v2_1-2_comparison_base and loop_risk_v2_2_0_full
Ignores expected base_config version differences (1dotx vs 2_0_v1)
"""
import json
import os
from pathlib import Path
import re

def normalize_base_config(value):
    """Normalize base_config to ignore expected version differences"""
    if isinstance(value, str):
        # Normalize the path patterns
        # 1dotx version: reusable.simulations.1xComparator.*_1dotx
        # 2_0 version: reusable.simulations.activity_presets.*_2_0_v1
        
        # Replace path prefixes
        normalized = re.sub(r'reusable\.simulations\.1xComparator\.', 
                           'reusable.simulations.NORMALIZED.', value)
        normalized = re.sub(r'reusable\.simulations\.activity_presets\.', 
                           'reusable.simulations.NORMALIZED.', normalized)
        
        # Replace version suffixes
        normalized = re.sub(r'_1dotx$', '_VERSION', normalized)
        normalized = re.sub(r'_2_0_v1$', '_VERSION', normalized)
        
        return normalized
    return value

def normalize_for_comparison(obj):
    """Recursively process JSON, normalizing base_config for comparison"""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "base_config":
                result[k] = normalize_base_config(v)
            else:
                result[k] = normalize_for_comparison(v)
        return result
    elif isinstance(obj, list):
        return [normalize_for_comparison(item) for item in obj]
    else:
        return obj

def get_json_diff(data1, data2, path=""):
    """Get detailed differences between two JSON structures"""
    diffs = []
    
    if type(data1) != type(data2):
        diffs.append(f"{path}: type mismatch ({type(data1).__name__} vs {type(data2).__name__})")
        return diffs
    
    if isinstance(data1, dict):
        all_keys = set(data1.keys()) | set(data2.keys())
        for key in sorted(all_keys):
            new_path = f"{path}.{key}" if path else key
            if key not in data1:
                diffs.append(f"{new_path}: only in 2_0_full = {json.dumps(data2[key])[:100]}")
            elif key not in data2:
                diffs.append(f"{new_path}: only in 1-2_comparison = {json.dumps(data1[key])[:100]}")
            else:
                diffs.extend(get_json_diff(data1[key], data2[key], new_path))
    elif isinstance(data1, list):
        if len(data1) != len(data2):
            diffs.append(f"{path}: list length differs ({len(data1)} vs {len(data2)})")
        for i, (item1, item2) in enumerate(zip(data1, data2)):
            diffs.extend(get_json_diff(item1, item2, f"{path}[{i}]"))
        # Handle items beyond the shorter list
        for i in range(len(data1), len(data2)):
            diffs.append(f"{path}[{i}]: only in 2_0_full")
        for i in range(len(data2), len(data1)):
            diffs.append(f"{path}[{i}]: only in 1-2_comparison")
    else:
        if data1 != data2:
            diffs.append(f"{path}: value differs ('{data1}' vs '{data2}')")
    
    return diffs

def compare_json_files(file1, file2):
    """Compare two JSON files and return differences"""
    try:
        with open(file1) as f:
            data1 = json.load(f)
        with open(file2) as f:
            data2 = json.load(f)
        
        # Normalize to ignore expected base_config differences
        norm1 = normalize_for_comparison(data1)
        norm2 = normalize_for_comparison(data2)
        
        if norm1 == norm2:
            return None
        
        # Get detailed differences on original data
        return get_json_diff(data1, data2)
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"]
    except Exception as e:
        return [f"Error comparing: {e}"]

def main():
    base_path = Path(__file__).parent
    dir1 = base_path / "loop_risk_v2_1-2_comparison_base"
    dir2 = base_path / "loop_risk_v2_2_0_full"
    
    # Get all subdirectories
    dirs1 = set(d.name for d in dir1.iterdir() if d.is_dir())
    dirs2 = set(d.name for d in dir2.iterdir() if d.is_dir())
    
    # Find directories only in one set
    only_in_1 = dirs1 - dirs2
    only_in_2 = dirs2 - dirs1
    common = dirs1 & dirs2
    
    print("=" * 70)
    print("SCENARIO COMPARISON REPORT")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("DIRECTORY DIFFERENCES")
    print("-" * 70)
    if only_in_1:
        print(f"\nScenarios only in 1-2_comparison_base ({len(only_in_1)}):")
        for d in sorted(only_in_1):
            print(f"  - {d}")
    else:
        print("\nNo scenarios only in 1-2_comparison_base")
        
    if only_in_2:
        print(f"\nScenarios only in 2_0_full ({len(only_in_2)}):")
        for d in sorted(only_in_2):
            print(f"  - {d}")
    else:
        print("\nNo scenarios only in 2_0_full")
    
    print("\n" + "-" * 70)
    print("JSON CONTENT DIFFERENCES IN COMMON SCENARIOS")
    print("(excluding expected base_config version differences)")
    print("-" * 70)
    
    differences_found = []
    scenarios_identical = 0
    
    for dirname in sorted(common):
        path1 = dir1 / dirname
        path2 = dir2 / dirname
        
        # Get JSON files in each directory
        json1 = {f.name: f for f in path1.glob("*.json")}
        json2 = {f.name: f for f in path2.glob("*.json")}
        
        all_json = set(json1.keys()) | set(json2.keys())
        
        dir_diffs = []
        
        # Check for file presence differences
        files_only_in_1 = set(json1.keys()) - set(json2.keys())
        files_only_in_2 = set(json2.keys()) - set(json1.keys())
        
        for f in sorted(files_only_in_1):
            dir_diffs.append(f"File only in 1-2_comparison: {f}")
        for f in sorted(files_only_in_2):
            dir_diffs.append(f"File only in 2_0_full: {f}")
        
        # Compare common files
        common_files = set(json1.keys()) & set(json2.keys())
        for json_name in sorted(common_files):
            content_diffs = compare_json_files(json1[json_name], json2[json_name])
            if content_diffs:
                dir_diffs.append(f"File '{json_name}' differs:")
                for d in content_diffs:
                    dir_diffs.append(f"  {d}")
        
        if dir_diffs:
            differences_found.append((dirname, dir_diffs))
        else:
            scenarios_identical += 1
    
    if differences_found:
        print(f"\nFound differences in {len(differences_found)} scenarios:")
        for dirname, diffs in differences_found:
            print(f"\n{dirname}:")
            for d in diffs:
                print(f"  {d}")
    else:
        print("\nNo content differences found!")
    
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Total scenarios in 1-2_comparison_base: {len(dirs1)}")
    print(f"Total scenarios in 2_0_full: {len(dirs2)}")
    print(f"Scenarios only in 1-2_comparison_base: {len(only_in_1)}")
    print(f"Scenarios only in 2_0_full: {len(only_in_2)}")
    print(f"Common scenarios: {len(common)}")
    print(f"Common scenarios that are identical: {scenarios_identical}")
    print(f"Common scenarios with differences: {len(differences_found)}")

if __name__ == "__main__":
    main()
