"""
Analyze OAI metadata files (_xrmeta01.txt, _mrimeta01.txt).
Extract detailed statistics including visit-based counts and retention analysis.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from oai_config import ImagingType, get_imaging_type, VISIT_MONTHS_MAP


class MetadataAnalyzer:
    """Analyze OAI metadata files for imaging data."""
    
    def __init__(self, metadata_path: Path):
        """
        Initialize analyzer with metadata file.
        
        Args:
            metadata_path: Path to _xrmeta01.txt or _mrimeta01.txt file
        """
        self.metadata_path = metadata_path
        self.df = self._load_metadata()
        
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata file with proper formatting."""
        df = pd.read_csv(
            self.metadata_path,
            sep="\t",
            header=0,
            skiprows=[1],  # Skip units row
            quotechar='"',
            dtype=str,
            low_memory=False,
        )
        
        required_cols = {"examtype", "visit", "subjectkey"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")
        
        return df
    
    def get_imaging_type_data(self, imaging_type: str) -> pd.DataFrame:
        """
        Filter metadata for specific imaging type.
        
        Args:
            imaging_type: Name of imaging type (e.g., 'hand', 'knee', 'hip')
            
        Returns:
            Filtered DataFrame
        """
        config = get_imaging_type(imaging_type)
        if not config:
            raise ValueError(f"Unknown imaging type: {imaging_type}")
        
        # Normalize exam types for matching
        examtype_norm = self.df["examtype"].astype(str).str.strip().str.lower()
        
        # Create mask for matching exam types
        exam_types_lower = {et.lower() for et in config.exam_types}
        mask = examtype_norm.isin(exam_types_lower)
        
        return self.df[mask].copy()
    
    def _add_visit_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add visit code and months columns to dataframe."""
        df = df.copy()
        
        # Extract visit code (e.g., V00, V01)
        visit_code = df["visit"].astype(str).str.extract(r"(V\d+)", expand=False)
        df["visit_code"] = visit_code
        
        # Map to months
        visit_num = df["visit"].astype(str).str.extract(r"V(\d+)", expand=False)
        df["months"] = visit_num.map(VISIT_MONTHS_MAP).astype("Int64")
        
        return df
    
    def _add_side_info(self, df: pd.DataFrame, config: ImagingType) -> pd.DataFrame:
        """Add normalized side column if applicable."""
        if not config.has_sides:
            return df
        
        df = df.copy()
        examtype_norm = df["examtype"].astype(str).str.strip().str.lower()
        
        side_norm = pd.Series(pd.NA, index=df.index, dtype="object")
        
        # Detect side from exam type
        for exam_type in config.exam_types:
            exam_lower = exam_type.lower()
            if "right" in exam_lower:
                side_norm.loc[examtype_norm == exam_lower] = "right"
            elif "left" in exam_lower:
                side_norm.loc[examtype_norm == exam_lower] = "left"
        
        df["side_norm"] = side_norm
        
        return df
    
    def count_by_visit(self, imaging_type: str) -> pd.DataFrame:
        """
        Count imaging records by visit/timepoint.
        
        Args:
            imaging_type: Name of imaging type
            
        Returns:
            DataFrame with counts by visit
        """
        config = get_imaging_type(imaging_type)
        df = self.get_imaging_type_data(imaging_type)
        df = self._add_visit_info(df)
        df = self._add_side_info(df, config)
        
        # Basic counts by visit
        counts = (
            df.groupby(["visit", "months"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        
        # Sort by months
        counts["months_sort"] = pd.to_numeric(counts["months"], errors="coerce")
        counts = counts.sort_values(["months_sort", "visit"], kind="mergesort")
        counts = counts.drop(columns=["months_sort"])
        
        # Add side breakdown if applicable
        if config.has_sides:
            counts_by_side = (
                df.dropna(subset=["side_norm"])
                .groupby(["visit", "months", "side_norm"], dropna=False)
                .size()
                .reset_index(name="count_side")
            )
            
            # Pivot to get left/right columns
            pivot = (
                counts_by_side.pivot_table(
                    index=["visit", "months"],
                    columns="side_norm",
                    values="count_side",
                    fill_value=0,
                    aggfunc="sum",
                )
                .reset_index()
                .rename_axis(None, axis=1)
            )
            
            # Merge back
            counts = counts.merge(pivot, on=["visit", "months"], how="left")
            
            # Fill missing side columns
            for col in ["right", "left"]:
                if col not in counts.columns:
                    counts[col] = 0
            counts[["right", "left"]] = counts[["right", "left"]].fillna(0).astype(int)
            
            # Count pairs (subjects with both sides at same visit)
            pairs_by_subject = (
                df.dropna(subset=["side_norm", "subjectkey"])
                .groupby(["visit", "months", "subjectkey", "side_norm"], dropna=False)
                .size()
                .unstack(fill_value=0)
            )
            
            if not pairs_by_subject.empty:
                left_counts = pairs_by_subject.get("left", 0)
                right_counts = pairs_by_subject.get("right", 0)
                pair_flags = (left_counts > 0) & (right_counts > 0)
                
                pairs = (
                    pair_flags.groupby(level=["visit", "months"])
                    .sum()
                    .reset_index(name="pairs")
                )
                
                counts = counts.merge(pairs, on=["visit", "months"], how="left")
                counts["pairs"] = counts["pairs"].fillna(0).astype(int)
        
        return counts
    
    def analyze_retention(
        self,
        imaging_type: str,
        retention_visits: List[str] = ["V00", "V01", "V06"]
    ) -> pd.DataFrame:
        """
        Analyze subject retention across key visits.
        
        Args:
            imaging_type: Name of imaging type
            retention_visits: Visit codes to analyze (default: baseline, 12mo, 48mo)
            
        Returns:
            DataFrame with retention statistics
        """
        config = get_imaging_type(imaging_type)
        df = self.get_imaging_type_data(imaging_type)
        df = self._add_visit_info(df)
        df = self._add_side_info(df, config)
        
        # Filter to retention visits
        retention = df.dropna(subset=["subjectkey", "visit_code"])
        retention = retention[retention["visit_code"].isin(retention_visits)]
        
        # Group by subject (and side if applicable)
        if config.has_sides:
            group_cols = ["subjectkey", "side_norm"]
            retention = retention.dropna(subset=["side_norm"])
        else:
            group_cols = ["subjectkey"]
        
        # Remove duplicates
        retention = retention.drop_duplicates(subset=group_cols + ["visit_code"])
        
        # Create pivot showing which visits each subject has
        retention_pivot = (
            retention.assign(has=1)
            .pivot_table(
                index=group_cols,
                columns="visit_code",
                values="has",
                fill_value=0,
                aggfunc="max",
            )
            .reset_index()
            .rename_axis(None, axis=1)
        )
        
        # Ensure all retention visits are present as columns
        for visit in retention_visits:
            if visit not in retention_pivot.columns:
                retention_pivot[visit] = 0
        
        # Define retention patterns (assuming V00, V01, V06)
        if len(retention_visits) >= 3:
            v0, v1, v2 = retention_visits[0], retention_visits[1], retention_visits[2]
            
            retention_pivot["full_time"] = (
                (retention_pivot[v0] > 0) & 
                (retention_pivot[v1] > 0) & 
                (retention_pivot[v2] > 0)
            )
            retention_pivot["end_time"] = (
                (retention_pivot[v0] == 0) & 
                (retention_pivot[v1] > 0) & 
                (retention_pivot[v2] > 0)
            )
            retention_pivot["start_time"] = (
                (retention_pivot[v0] > 0) & 
                (retention_pivot[v1] > 0) & 
                (retention_pivot[v2] == 0)
            )
            retention_pivot["split_time"] = (
                (retention_pivot[v0] > 0) & 
                (retention_pivot[v1] == 0) & 
                (retention_pivot[v2] > 0)
            )
            retention_pivot["other"] = ~(
                retention_pivot["full_time"] |
                retention_pivot["end_time"] |
                retention_pivot["start_time"] |
                retention_pivot["split_time"]
            )
        
        # Summarize by side if applicable
        if config.has_sides:
            summary = (
                retention_pivot.groupby("side_norm")[
                    ["full_time", "end_time", "start_time", "split_time", "other"]
                ]
                .sum()
                .reset_index()
            )
            summary["total_subjects"] = summary[
                ["full_time", "end_time", "start_time", "split_time", "other"]
            ].sum(axis=1)
        else:
            summary = pd.DataFrame([{
                "full_time": retention_pivot["full_time"].sum(),
                "end_time": retention_pivot["end_time"].sum(),
                "start_time": retention_pivot["start_time"].sum(),
                "split_time": retention_pivot["split_time"].sum(),
                "other": retention_pivot["other"].sum(),
            }])
            summary["total_subjects"] = summary[
                ["full_time", "end_time", "start_time", "split_time", "other"]
            ].sum(axis=1)
        
        return summary


def analyze_metadata(
    metadata_path: Path,
    imaging_type: str,
    output_dir: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to analyze metadata file.
    
    Args:
        metadata_path: Path to _xrmeta01.txt or _mrimeta01.txt
        imaging_type: Imaging type to analyze
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = MetadataAnalyzer(metadata_path)
    
    results = {
        "by_visit": analyzer.count_by_visit(imaging_type),
        "retention": analyzer.analyze_retention(imaging_type),
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{imaging_type}_metadata"
        results["by_visit"].to_csv(output_dir / f"{prefix}_by_visit.csv", index=False)
        results["retention"].to_csv(output_dir / f"{prefix}_retention.csv", index=False)
        
        print(f"Saved {imaging_type} metadata analysis to: {output_dir}")
    
    return results
