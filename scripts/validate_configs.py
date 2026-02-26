#!/usr/bin/env python3
"""
Configuration validation script for Tidepool Data Science Simulator.

This script validates simulation configuration files before execution,
catching errors early and providing detailed diagnostics.

Usage:
    python validate_configs.py --directory <config_dir> [options]
    python validate_configs.py --file <config_file> [options]
    
Examples:
    # Validate all configs in a directory
    python validate_configs.py --directory ./scenario_configs/tidepool_risk_v2/loop_risk_v2_0
    
    # Validate a single file
    python validate_configs.py --file ./scenario_configs/my_config.json
    
    # Save report to file
    python validate_configs.py --directory ./configs --output validation_report.txt
    
    # Validate recursively (default)
    python validate_configs.py --directory ./configs --recursive
    
    # Non-recursive validation
    python validate_configs.py --directory ./configs --no-recursive
    
    # Show valid configurations too
    python validate_configs.py --directory ./configs --show-valid
    
    # Quick validation (summary only)
    python validate_configs.py --directory ./configs --quiet
"""

import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add parent directory to path to import validation modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidepool_data_science_simulator.validation.config_validator import ConfigValidator
from tidepool_data_science_simulator.validation.value_validators import ValidationError


def print_summary(results):
    """Print summary of validation results"""
    total_files = len(results)
    valid_files = sum(1 for is_valid, _ in results.values() if is_valid)
    invalid_files = total_files - valid_files
    total_errors = sum(len(errors) for _, errors in results.values())
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"📁 Total files checked: {total_files}")
    print(f"✅ Valid files: {valid_files}")
    print(f"❌ Invalid files: {invalid_files}")
    print(f"🔍 Total errors found: {total_errors}")
    print("=" * 80 + "\n")


def print_error_statistics(results):
    """Print statistics about types of errors found"""
    
    error_type_counts = defaultdict(int)
    field_error_counts = defaultdict(int)
    
    for _, (is_valid, errors) in results.items():
        if not is_valid:
            for error in errors:
                # Count by field path root
                field_root = error.field_path.split('.')[0] if '.' in error.field_path else error.field_path
                field_error_counts[field_root] += 1
                
                # Categorize error types
                error_msg_lower = error.error_message.lower()
                if "not found" in error_msg_lower or "missing" in error_msg_lower:
                    error_type_counts["Missing/Not Found"] += 1
                elif "outside" in error_msg_lower or "range" in error_msg_lower:
                    error_type_counts["Out of Range"] += 1
                elif "type" in error_msg_lower or "must be" in error_msg_lower:
                    error_type_counts["Type Mismatch"] += 1
                elif "invalid" in error_msg_lower:
                    error_type_counts["Invalid Value"] += 1
                else:
                    error_type_counts["Other"] += 1
    
    if error_type_counts:
        print("\n" + "=" * 80)
        print("ERROR STATISTICS")
        print("=" * 80 + "\n")
        
        print("By Error Type:")
        for error_type, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")
        
        print("\nBy Configuration Section:")
        for field, count in sorted(field_error_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {field}: {count}")
        
        print()


def print_detailed_results(results, show_valid=False):
    """Print detailed validation results"""
    
    # Group by status
    valid_files = []
    invalid_files = []
    
    for file_path, (is_valid, errors) in results.items():
        if is_valid:
            valid_files.append(file_path)
        else:
            invalid_files.append((file_path, errors))
    
    # Print invalid files first
    if invalid_files:
        print("\n" + "🔴 " + "=" * 78)
        print("INVALID CONFIGURATIONS")
        print("=" * 80 + "\n")
        
        for file_path, errors in invalid_files:
            print(f"\n📄 {file_path}")
            print("-" * 80)
            
            # Group errors by type
            error_groups = defaultdict(list)
            for error in errors:
                error_type = error.field_path.split('.')[0] if '.' in error.field_path else "general"
                error_groups[error_type].append(error)
            
            for error_type, error_list in error_groups.items():
                print(f"\n  [{error_type}] - {len(error_list)} error(s):")
                for error in error_list:
                    print(f"    {error}")
            
            print()
    
    # Print valid files if requested
    if show_valid and valid_files:
        print("\n" + "🟢 " + "=" * 78)
        print("VALID CONFIGURATIONS")
        print("=" * 80 + "\n")
        
        for file_path in valid_files:
            print(f"  ✓ {file_path}")
        print()


def write_to_file(results, output_path, show_valid=False):
    """Write validation results to a file"""
    
    with open(output_path, 'w') as f:
        # Redirect stdout to file temporarily
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            print_summary(results)
            print_error_statistics(results)
            print_detailed_results(results, show_valid=True)
        finally:
            sys.stdout = original_stdout
    
    print(f"\n📝 Detailed report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Tidepool simulator configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Directory containing configuration files to validate'
    )
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Single configuration file to validate'
    )
    
    # Validation options
    parser.add_argument(
        '--pointer-dir',
        type=str,
        default=None,
        help='Directory containing reusable configuration files (auto-detected if not specified)'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='Recursively validate subdirectories (default: True)'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_false',
        dest='recursive',
        help='Do not recursively validate subdirectories'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Write detailed report to file'
    )
    parser.add_argument(
        '--show-valid',
        action='store_true',
        default=False,
        help='Show valid configurations in output (default: only show invalid)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Show detailed error messages'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        default=False,
        help='Only show summary (no detailed errors)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect pointer directory if not specified
    if args.pointer_dir is None:
        if args.directory:
            # Try to find tidepool_risk_v2 directory
            search_path = Path(args.directory).resolve()
            while search_path != search_path.parent:
                candidate = search_path / "tidepool_risk_v2"
                if candidate.exists():
                    args.pointer_dir = str(candidate)
                    break
                search_path = search_path.parent
        
        if args.pointer_dir is None:
            # Default fallback - try relative to script location
            script_dir = Path(__file__).parent.parent
            default_path = script_dir / "scenario_configs" / "tidepool_risk_v2"
            if default_path.exists():
                args.pointer_dir = str(default_path)
            else:
                print("⚠️  Warning: Could not auto-detect pointer directory.")
                print("   Reference validation will be skipped.")
                print("   Use --pointer-dir to specify the location of reusable configs.")
                args.pointer_dir = None
    
    if args.pointer_dir:
        print(f"Using pointer directory: {args.pointer_dir}")
    
    # Create validator
    validator = ConfigValidator(args.pointer_dir)
    
    # Validate
    if args.file:
        print(f"\n🔍 Validating single file: {args.file}\n")
        is_valid, errors = validator.validate_config_file(args.file)
        results = {args.file: (is_valid, errors)}
    else:
        print(f"\n🔍 Validating directory: {args.directory}")
        print(f"   Recursive: {args.recursive}\n")
        results = validator.validate_directory(args.directory, recursive=args.recursive)
    
    # Generate report
    if not args.quiet:
        print_summary(results)
        
        if args.verbose or not args.quiet:
            print_error_statistics(results)
            print_detailed_results(results, show_valid=args.show_valid)
    else:
        print_summary(results)
    
    # Write to file if requested
    if args.output:
        write_to_file(results, args.output, show_valid=True)
    
    # Exit with appropriate code
    all_valid = all(is_valid for is_valid, _ in results.values())
    
    if all_valid:
        print("\n✅ All configurations are valid!\n")
        sys.exit(0)
    else:
        print("\n❌ Some configurations have errors. Please fix them before running simulations.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
