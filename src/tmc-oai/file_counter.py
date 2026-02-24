"""
Count actual image files on disk for OAI packages.
Handles both JPG thumbnails and tar.gz archives.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re

from oai_config import ImagingType, get_imaging_type


class FileCounter:
    """Count actual image files in OAI package directories."""
    
    def __init__(self, package_dir: Path, recursive: bool = True):
        """
        Initialize file counter.
        
        Args:
            package_dir: Path to Package_* directory
            recursive: Whether to search subdirectories
        """
        self.package_dir = package_dir
        self.recursive = recursive
        self._file_index: Optional[Dict[str, List[Path]]] = None
    
    def _build_file_index(self, patterns: Tuple[str, ...]) -> Dict[str, List[Path]]:
        """
        Build index of files by basename.
        
        Args:
            patterns: File patterns to search for (e.g., '*.jpg', '*.tar.gz')
            
        Returns:
            Dictionary mapping lowercase basename to list of matching paths
        """
        index = defaultdict(list)
        
        for pattern in patterns:
            if self.recursive:
                iterator = self.package_dir.rglob(pattern)
            else:
                iterator = self.package_dir.glob(pattern)
            
            for path in iterator:
                basename_lower = path.name.lower()
                index[basename_lower].append(path)
        
        return dict(index)
    
    def count_jpg_thumbnails(self) -> Dict[str, int]:
        """
        Count JPG thumbnail files.
        
        Returns:
            Dictionary with thumbnail counts
        """
        jpg_index = self._build_file_index(("*.jpg",))
        
        total = len(jpg_index)
        
        # Count by specific patterns
        _1x1_count = sum(
            1 for name in jpg_index.keys() 
            if "_1x1.jpg" in name
        )
        
        return {
            "total_jpg_files": total,
            "1x1_jpg_files": _1x1_count,
            "other_jpg_files": total - _1x1_count,
        }
    
    def count_tar_archives(self) -> Dict[str, int]:
        """
        Count tar archive files.
        
        Returns:
            Dictionary with archive counts
        """
        tar_index = self._build_file_index(("*.tar.gz", "*.tgz", "*.tar"))
        
        total = len(tar_index)
        
        # Count by extension
        tar_gz_count = sum(
            1 for name in tar_index.keys() 
            if name.endswith(".tar.gz")
        )
        tgz_count = sum(
            1 for name in tar_index.keys() 
            if name.endswith(".tgz")
        )
        tar_count = sum(
            1 for name in tar_index.keys() 
            if name.endswith(".tar") and not name.endswith(".tar.gz")
        )
        
        return {
            "total_archive_files": total,
            "tar_gz_files": tar_gz_count,
            "tgz_files": tgz_count,
            "tar_files": tar_count,
        }
    
    def count_all(self) -> Dict[str, int]:
        """
        Count all image files.
        
        Returns:
            Combined dictionary with all counts
        """
        jpg_counts = self.count_jpg_thumbnails()
        tar_counts = self.count_tar_archives()
        
        return {**jpg_counts, **tar_counts}


class PackageScanner:
    """
    Scan multiple OAI packages and summarize file counts.
    """
    
    def __init__(self, base_dir: Path, recursive: bool = True):
        """
        Initialize package scanner.
        
        Args:
            base_dir: Base directory containing Package_* folders
            recursive: Whether to search subdirectories within packages
        """
        self.base_dir = Path(base_dir)
        self.recursive = recursive
    
    def find_packages(self, pattern: str = "Package_*") -> List[Path]:
        """
        Find all package directories.
        
        Args:
            pattern: Glob pattern for package directories
            
        Returns:
            List of package directory paths
        """
        packages = [p for p in self.base_dir.glob(pattern) if p.is_dir()]
        return sorted(packages)
    
    def scan_package(self, package_dir: Path) -> Dict[str, any]:
        """
        Scan a single package directory.
        
        Args:
            package_dir: Package directory path
            
        Returns:
            Dictionary with package info and counts
        """
        counter = FileCounter(package_dir, self.recursive)
        counts = counter.count_all()
        
        # Parse package name for metadata
        package_name = package_dir.name
        parts = package_name.split("_")
        
        return {
            "package_name": package_name,
            "package_dir": str(package_dir),
            "package_id": parts[1] if len(parts) > 1 else "",
            "timepoint": parts[2] if len(parts) > 2 else "",
            **counts
        }
    
    def scan_all(self) -> List[Dict[str, any]]:
        """
        Scan all packages and return summary.
        
        Returns:
            List of dictionaries with package information
        """
        packages = self.find_packages()
        results = []
        
        for package_dir in packages:
            try:
                result = self.scan_package(package_dir)
                results.append(result)
                print(f"Scanned: {package_dir.name}")
            except Exception as e:
                print(f"Error scanning {package_dir.name}: {e}")
                continue
        
        return results
    
    def get_summary_df(self):
        """
        Get summary as DataFrame.
        
        Returns:
            pandas DataFrame with package summaries
        """
        import pandas as pd
        
        results = self.scan_all()
        return pd.DataFrame(results)


def count_package_files(
    package_dir: Path,
    recursive: bool = True,
    save_path: Optional[Path] = None
) -> Dict[str, int]:
    """
    Convenience function to count files in a package.
    
    Args:
        package_dir: Package directory path
        recursive: Whether to search subdirectories
        save_path: Optional path to save results as JSON
        
    Returns:
        Dictionary with file counts
    """
    counter = FileCounter(package_dir, recursive)
    counts = counter.count_all()
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(counts, f, indent=2)
        print(f"Saved counts to: {save_path}")
    
    return counts


def scan_all_packages(
    base_dir: Path,
    recursive: bool = True,
    save_path: Optional[Path] = None
):
    """
    Convenience function to scan all packages.
    
    Args:
        base_dir: Base directory with Package_* folders
        recursive: Whether to search subdirectories
        save_path: Optional path to save summary CSV
        
    Returns:
        pandas DataFrame with results
    """
    scanner = PackageScanner(base_dir, recursive)
    df = scanner.get_summary_df()
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved package summary to: {save_path}")
    
    return df
