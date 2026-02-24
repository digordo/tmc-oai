"""
Analyze OAI image manifest files (image03.txt).
Extract counts and summaries for different imaging types.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from oai_config import ImagingType, get_imaging_type


class ManifestAnalyzer:
    """Analyze OAI manifest files for imaging data."""
    
    def __init__(self, manifest_path: Path):
        """
        Initialize analyzer with manifest file.
        
        Args:
            manifest_path: Path to image03.txt manifest file
        """
        self.manifest_path = manifest_path
        self.df = self._load_manifest()
        
    def _load_manifest(self) -> pd.DataFrame:
        """Load manifest file."""
        return pd.read_csv(self.manifest_path, sep="\t", low_memory=False, dtype=str)
    
    def get_imaging_type_data(self, imaging_type: str) -> pd.DataFrame:
        """
        Filter manifest for specific imaging type.
        
        Args:
            imaging_type: Name of imaging type (e.g., 'hand', 'knee', 'hip')
            
        Returns:
            Filtered DataFrame
        """
        config = get_imaging_type(imaging_type)
        if not config:
            raise ValueError(f"Unknown imaging type: {imaging_type}")
        
        if "comments_misc" not in self.df.columns:
            raise ValueError("Manifest missing 'comments_misc' column")
        
        # Filter by pattern
        mask = self.df["comments_misc"].astype(str).str.contains(
            config.compiled_pattern, na=False
        )
        
        return self.df[mask].copy()
    
    def count_files(
        self, 
        imaging_type: str,
        file_type: str = "both"
    ) -> Dict[str, int]:
        """
        Count files for an imaging type.
        
        Args:
            imaging_type: Name of imaging type
            file_type: 'thumbnails', 'archives', or 'both'
            
        Returns:
            Dictionary with counts
        """
        df = self.get_imaging_type_data(imaging_type)
        config = get_imaging_type(imaging_type)
        
        result = {
            "total_entries": len(df),
        }
        
        if file_type in ("thumbnails", "both"):
            # Count thumbnail entries
            thumb_col = "image_thumbnail_file"
            if thumb_col in df.columns:
                has_thumb = df[thumb_col].notna() & (df[thumb_col].astype(str).str.strip() != "")
                result["thumbnail_entries"] = has_thumb.sum()
                
                # Count by extension
                thumb_1x1 = df[thumb_col].astype(str).str.contains("_1x1.jpg", na=False)
                result["thumbnail_1x1_jpg"] = thumb_1x1.sum()
        
        if file_type in ("archives", "both"):
            # Count archive entries
            archive_col = "image_file"
            if archive_col in df.columns:
                has_archive = df[archive_col].notna() & (df[archive_col].astype(str).str.strip() != "")
                result["archive_entries"] = has_archive.sum()
                
                # Count by extension
                tar_gz = df[archive_col].astype(str).str.contains(r"\.tar\.gz$", na=False, regex=True)
                result["archive_tar_gz"] = tar_gz.sum()
        
        # Count by side if applicable
        if config.has_sides:
            left_mask = df["comments_misc"].astype(str).str.contains("LEFT", case=False, na=False)
            right_mask = df["comments_misc"].astype(str).str.contains("RIGHT", case=False, na=False)
            
            result["left_entries"] = left_mask.sum()
            result["right_entries"] = right_mask.sum()
        
        return result
    
    def get_summary(self, imaging_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get summary table for multiple imaging types.
        
        Args:
            imaging_types: List of imaging types to include. If None, uses all.
            
        Returns:
            Summary DataFrame
        """
        from oai_config import list_imaging_types
        
        if imaging_types is None:
            imaging_types = list_imaging_types()
        
        summaries = []
        for img_type in imaging_types:
            try:
                counts = self.count_files(img_type, file_type="both")
                config = get_imaging_type(img_type)
                
                summary = {
                    "imaging_type": img_type,
                    "total_entries": counts["total_entries"],
                    "thumbnail_entries": counts.get("thumbnail_entries", 0),
                    "archive_entries": counts.get("archive_entries", 0),
                }
                
                if config.has_sides:
                    summary["left_entries"] = counts.get("left_entries", 0)
                    summary["right_entries"] = counts.get("right_entries", 0)
                
                summaries.append(summary)
            except Exception as e:
                print(f"Warning: Could not process {img_type}: {e}")
                continue
        
        return pd.DataFrame(summaries)


def analyze_manifest(
    manifest_path: Path,
    imaging_types: Optional[List[str]] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convenience function to analyze a manifest file.
    
    Args:
        manifest_path: Path to image03.txt
        imaging_types: List of imaging types to analyze
        save_path: Optional path to save summary CSV
        
    Returns:
        Summary DataFrame
    """
    analyzer = ManifestAnalyzer(manifest_path)
    summary = analyzer.get_summary(imaging_types)
    
    if save_path:
        summary.to_csv(save_path, index=False)
        print(f"Saved summary to: {save_path}")
    
    return summary
