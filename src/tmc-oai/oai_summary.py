#!/usr/bin/env python3
"""
OAI Imaging Data Summary Tool

Generate summaries of OAI imaging data from:
1. Manifest files (image03.txt) - what's supposed to be there
2. Metadata files (_xrmeta01.txt) - clinical metadata and visit info
3. Actual files on disk - what's actually there

Usage examples:
    # Scan all packages for file counts
    python oai_summary.py scan-files --base-dir X:/OAI --output summary.csv
    
    # Analyze a manifest file
    python oai_summary.py analyze-manifest --manifest X:/OAI/Package_1243841_BASELINE/image03.txt
    
    # Analyze metadata
    python oai_summary.py analyze-metadata --metadata X:/OAI/Package_1243845_METADATA/_xrmeta01.txt --type hand
    
    # Full analysis for a specific imaging type
    python oai_summary.py full-analysis --base-dir X:/OAI --type hand --output-dir ./hand_analysis
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from oai_config import list_imaging_types, get_imaging_type
from manifest_analyzer import ManifestAnalyzer, analyze_manifest
from metadata_analyzer import MetadataAnalyzer, analyze_metadata
from file_counter import PackageScanner, scan_all_packages


def cmd_scan_files(args):
    """Scan all packages for actual file counts."""
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return 1
    
    print(f"Scanning packages in: {base_dir}")
    print(f"Recursive search: {args.recursive}")
    print()
    
    df = scan_all_packages(
        base_dir,
        recursive=args.recursive,
        save_path=Path(args.output) if args.output else None
    )
    
    print()
    print("=" * 80)
    print("FILE COUNT SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    # Summary statistics
    print("TOTALS:")
    print(f"  Total packages scanned: {len(df)}")
    print(f"  Total JPG files: {df['total_jpg_files'].sum():,}")
    print(f"  Total archive files: {df['total_archive_files'].sum():,}")
    
    return 0


def cmd_analyze_manifest(args):
    """Analyze manifest file for imaging type counts."""
    manifest_path = Path(args.manifest)
    
    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}")
        return 1
    
    imaging_types = args.types if args.types else list_imaging_types()
    
    print(f"Analyzing manifest: {manifest_path}")
    print(f"Imaging types: {', '.join(imaging_types)}")
    print()
    
    df = analyze_manifest(
        manifest_path,
        imaging_types=imaging_types,
        save_path=Path(args.output) if args.output else None
    )
    
    print()
    print("=" * 80)
    print("MANIFEST SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    
    return 0


def cmd_analyze_metadata(args):
    """Analyze metadata file for visit-based statistics."""
    metadata_path = Path(args.metadata)
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1
    
    imaging_type = args.type
    if not get_imaging_type(imaging_type):
        print(f"Error: Unknown imaging type: {imaging_type}")
        print(f"Available types: {', '.join(list_imaging_types())}")
        return 1
    
    print(f"Analyzing metadata: {metadata_path}")
    print(f"Imaging type: {imaging_type}")
    print()
    
    results = analyze_metadata(
        metadata_path,
        imaging_type,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    print()
    print("=" * 80)
    print(f"METADATA SUMMARY - {imaging_type.upper()}")
    print("=" * 80)
    
    print()
    print("Counts by Visit:")
    print(results["by_visit"].to_string(index=False))
    
    print()
    print("Retention Analysis:")
    print(results["retention"].to_string(index=False))
    
    return 0


def cmd_full_analysis(args):
    """Run full analysis for a specific imaging type."""
    base_dir = Path(args.base_dir)
    imaging_type = args.type
    output_dir = Path(args.output_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return 1
    
    if not get_imaging_type(imaging_type):
        print(f"Error: Unknown imaging type: {imaging_type}")
        print(f"Available types: {', '.join(list_imaging_types())}")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"FULL ANALYSIS: {imaging_type.upper()}")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # 1. Scan for actual files
    print("1. Scanning for actual files...")
    scanner = PackageScanner(base_dir, recursive=args.recursive)
    file_df = scanner.get_summary_df()
    file_df.to_csv(output_dir / f"{imaging_type}_file_counts.csv", index=False)
    print(f"   Found {len(file_df)} packages")
    print(f"   Total JPG files: {file_df['total_jpg_files'].sum():,}")
    print(f"   Total archive files: {file_df['total_archive_files'].sum():,}")
    print()
    
    # 2. Analyze manifests from each package
    print("2. Analyzing manifests...")
    manifest_summaries = []
    for package_dir in scanner.find_packages():
        manifest_path = package_dir / "image03.txt"
        if manifest_path.exists():
            try:
                analyzer = ManifestAnalyzer(manifest_path)
                counts = analyzer.count_files(imaging_type, file_type="both")
                counts["package_name"] = package_dir.name
                manifest_summaries.append(counts)
            except Exception as e:
                print(f"   Warning: Could not analyze {package_dir.name}: {e}")
    
    if manifest_summaries:
        manifest_df = pd.DataFrame(manifest_summaries)
        manifest_df.to_csv(output_dir / f"{imaging_type}_manifest_summary.csv", index=False)
        print(f"   Analyzed {len(manifest_df)} manifests")
        print(f"   Total {imaging_type} entries: {manifest_df['total_entries'].sum():,}")
    else:
        print("   No manifests found")
    print()
    
    # 3. Analyze metadata if available
    print("3. Analyzing metadata...")
    metadata_found = False
    for package_dir in scanner.find_packages():
        if "METADATA" in package_dir.name:
            metadata_path = package_dir / "_xrmeta01.txt"
            if metadata_path.exists():
                try:
                    results = analyze_metadata(
                        metadata_path,
                        imaging_type,
                        output_dir=output_dir
                    )
                    metadata_found = True
                    print(f"   Analyzed metadata from {package_dir.name}")
                    break
                except Exception as e:
                    print(f"   Warning: Could not analyze metadata: {e}")
    
    if not metadata_found:
        print("   No metadata files found")
    print()
    
    print("=" * 80)
    print(f"Analysis complete! Results saved to: {output_dir}")
    print("=" * 80)
    
    return 0


def cmd_list_types(args):
    """List available imaging types."""
    print("Available imaging types:")
    print()
    
    for name in list_imaging_types():
        config = get_imaging_type(name)
        print(f"  {name}:")
        print(f"    Exam types: {', '.join(config.exam_types)}")
        print(f"    Has sides: {config.has_sides}")
        if config.modality:
            print(f"    Modality: {config.modality}")
        print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="OAI Imaging Data Summary Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # scan-files command
    scan_parser = subparsers.add_parser(
        "scan-files",
        help="Scan packages for actual file counts"
    )
    scan_parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing Package_* folders"
    )
    scan_parser.add_argument(
        "--output",
        help="Output CSV file path"
    )
    scan_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search subdirectories (default: True)"
    )
    scan_parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Don't search subdirectories"
    )
    
    # analyze-manifest command
    manifest_parser = subparsers.add_parser(
        "analyze-manifest",
        help="Analyze manifest file (image03.txt)"
    )
    manifest_parser.add_argument(
        "--manifest",
        required=True,
        help="Path to image03.txt file"
    )
    manifest_parser.add_argument(
        "--types",
        nargs="+",
        help="Imaging types to analyze (default: all)"
    )
    manifest_parser.add_argument(
        "--output",
        help="Output CSV file path"
    )
    
    # analyze-metadata command
    metadata_parser = subparsers.add_parser(
        "analyze-metadata",
        help="Analyze metadata file (_xrmeta01.txt)"
    )
    metadata_parser.add_argument(
        "--metadata",
        required=True,
        help="Path to _xrmeta01.txt file"
    )
    metadata_parser.add_argument(
        "--type",
        required=True,
        help="Imaging type to analyze"
    )
    metadata_parser.add_argument(
        "--output-dir",
        help="Output directory for results"
    )
    
    # full-analysis command
    full_parser = subparsers.add_parser(
        "full-analysis",
        help="Run complete analysis for an imaging type"
    )
    full_parser.add_argument(
        "--base-dir",
        required=True,
        help="Base directory containing Package_* folders"
    )
    full_parser.add_argument(
        "--type",
        required=True,
        help="Imaging type to analyze"
    )
    full_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for all results"
    )
    full_parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search subdirectories (default: True)"
    )
    full_parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Don't search subdirectories"
    )
    
    # list-types command
    list_parser = subparsers.add_parser(
        "list-types",
        help="List available imaging types"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    commands = {
        "scan-files": cmd_scan_files,
        "analyze-manifest": cmd_analyze_manifest,
        "analyze-metadata": cmd_analyze_metadata,
        "full-analysis": cmd_full_analysis,
        "list-types": cmd_list_types,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
