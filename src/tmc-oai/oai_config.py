"""
Configuration for OAI imaging analysis.
Defines patterns and metadata for different imaging types.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ImagingType:
    """Configuration for a specific imaging type."""
    
    name: str
    # Regex pattern to match in comments_misc field
    manifest_pattern: str
    # Expected exam types in metadata
    exam_types: List[str]
    # Whether this imaging type has left/right sides
    has_sides: bool = False
    # Optional additional filters
    modality: Optional[str] = None
    
    @property
    def compiled_pattern(self) -> re.Pattern:
        """Get compiled regex pattern."""
        return re.compile(self.manifest_pattern, re.IGNORECASE)


# Define all imaging types
IMAGING_TYPES = {
    "hand": ImagingType(
        name="hand",
        manifest_pattern=r"\bOAI XRAY\b\s+\S+(?:\s+\S+)*\s+HAND(?:\s+(?:LEFT|RIGHT))?\b",
        exam_types=["PA Right Hand", "PA Left Hand"],
        has_sides=True,
        modality="XRAY"
    ),
    "knee": ImagingType(
        name="knee",
        manifest_pattern=r"\bOAI\s+(?:XRAY|MR)\b.*\bKNEE\b",
        exam_types=[
            # X-ray knee exams
            "Bilateral PA Fixed Flexion Knee",
            "PA Fixed Flexion Right Knee",
            "PA Fixed Flexion Left Knee",
            "Lateral Right Knee",
            "Lateral Left Knee",
            # MRI knee exams
            "MRI Right Knee",
            "MRI Left Knee",
        ],
        has_sides=True,
        modality=None  # Can be XRAY or MR
    ),
    "hip": ImagingType(
        name="hip",
        manifest_pattern=r"\bOAI XRAY\b.*\bHIP\b",
        exam_types=[
            "AP Pelvis",
            "Supine AP Pelvis",
        ],
        has_sides=False,  # Pelvis captures both hips
        modality="XRAY"
    ),
    "spine": ImagingType(
        name="spine",
        manifest_pattern=r"\bOAI XRAY\b.*\b(?:SPINE|LUMBAR)\b",
        exam_types=[
            "Lateral Lumbar Spine",
            "PA Lumbar Spine",
        ],
        has_sides=False,
        modality="XRAY"
    ),
}


# Visit code to months mapping
VISIT_MONTHS_MAP = {
    "00": 0,    # Baseline
    "01": 12,   # 12 months
    "02": 18,   # 18 months
    "03": 24,   # 24 months
    "04": 30,   # 30 months
    "05": 36,   # 36 months
    "06": 48,   # 48 months
    "07": 72,   # 72 months
    "08": 96,   # 96 months
    "10": 120,  # 120 months (10 years)
}


# Standard timepoint labels found in package names
TIMEPOINT_LABELS = {
    "BASELINE": "V00",
    "12MONTHS": "V01",
    "18MONTHS": "V02",
    "24MONTHS": "V03",
    "30MONTHS": "V04",
    "36MONTHS": "V05",
    "48MONTHS": "V06",
    "72MONTHS": "V07",
    "96MONTHS": "V08",
}


def get_imaging_type(name: str) -> Optional[ImagingType]:
    """Get imaging type configuration by name."""
    return IMAGING_TYPES.get(name.lower())


def list_imaging_types() -> List[str]:
    """Get list of all available imaging types."""
    return list(IMAGING_TYPES.keys())
