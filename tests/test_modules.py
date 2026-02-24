"""
Simple test script to verify the OAI analysis modules work.
"""
from pathlib import Path
import sys

def test_config():
    """Test configuration module."""
    print("Testing oai_config...")
    from oai_config import list_imaging_types, get_imaging_type, VISIT_MONTHS_MAP
    
    types = list_imaging_types()
    assert len(types) > 0, "No imaging types defined!"
    assert "hand" in types, "Hand imaging type missing!"
    
    hand_config = get_imaging_type("hand")
    assert hand_config is not None, "Cannot get hand config!"
    assert hand_config.has_sides == True, "Hand should have sides!"
    assert len(hand_config.exam_types) > 0, "No exam types defined!"
    
    assert "00" in VISIT_MONTHS_MAP, "Visit mapping missing!"
    
    print("  ✓ Configuration module OK")
    return True


def test_manifest_analyzer():
    """Test manifest analyzer with dummy data."""
    print("Testing manifest_analyzer...")
    import pandas as pd
    from manifest_analyzer import ManifestAnalyzer
    from oai_config import get_imaging_type
    import tempfile
    
    # Create dummy manifest data
    hand_config = get_imaging_type("hand")
    pattern_text = "OAI XRAY PA HAND LEFT"
    
    dummy_data = pd.DataFrame({
        "comments_misc": [
            pattern_text,
            "OAI XRAY PA HAND RIGHT",
            "OAI XRAY PA KNEE",
            pattern_text,
        ],
        "image_thumbnail_file": [
            "file1_1x1.jpg",
            "file2_1x1.jpg",
            "file3.jpg",
            "file4_1x1.jpg",
        ],
        "image_file": [
            "file1.tar.gz",
            "file2.tar.gz",
            "file3.tar.gz",
            "file4.tar.gz",
        ]
    })
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        dummy_data.to_csv(f, sep='\t', index=False)
        temp_path = f.name
    
    try:
        # Test analyzer
        analyzer = ManifestAnalyzer(Path(temp_path))
        
        # Test filtering (3 hand entries: 2 left, 1 right, 1 knee excluded)
        hand_df = analyzer.get_imaging_type_data("hand")
        assert len(hand_df) == 3, f"Expected 3 hand entries, got {len(hand_df)}"
        
        # Test counting
        counts = analyzer.count_files("hand")
        assert counts["total_entries"] == 3, "Wrong total count"
        assert counts["left_entries"] == 2, "Wrong left count"
        assert counts["right_entries"] == 1, "Wrong right count"
        
        print("  ✓ Manifest analyzer OK")
        return True
    finally:
        Path(temp_path).unlink()


def test_metadata_analyzer():
    """Test metadata analyzer with dummy data."""
    print("Testing metadata_analyzer...")
    import pandas as pd
    from metadata_analyzer import MetadataAnalyzer
    import tempfile
    
    # Create dummy metadata
    dummy_data = pd.DataFrame({
        "examtype": [
            "PA Right Hand",
            "PA Left Hand",
            "PA Right Hand",
            "PA Left Hand",
        ],
        "visit": ["V00", "V00", "V01", "V01"],
        "subjectkey": ["SUBJ001", "SUBJ001", "SUBJ001", "SUBJ001"],
    })
    
    # Save with header row + units row
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Write header
        f.write('\t'.join(dummy_data.columns) + '\n')
        # Write units row (required by loader)
        f.write('\t'.join([''] * len(dummy_data.columns)) + '\n')
        # Write data
        dummy_data.to_csv(f, sep='\t', index=False, header=False)
        temp_path = f.name
    
    try:
        # Test analyzer
        analyzer = MetadataAnalyzer(Path(temp_path))
        
        # Test filtering
        hand_df = analyzer.get_imaging_type_data("hand")
        assert len(hand_df) == 4, f"Expected 4 hand entries, got {len(hand_df)}"
        
        # Test visit counts
        by_visit = analyzer.count_by_visit("hand")
        assert len(by_visit) == 2, "Should have 2 visits"
        
        print("  ✓ Metadata analyzer OK")
        return True
    finally:
        Path(temp_path).unlink()


def test_file_counter():
    """Test file counter with temp directory."""
    print("Testing file_counter...")
    from file_counter import FileCounter
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy files
        (tmpdir / "file1_1x1.jpg").touch()
        (tmpdir / "file2_1x1.jpg").touch()
        (tmpdir / "file3.jpg").touch()
        (tmpdir / "archive1.tar.gz").touch()
        (tmpdir / "archive2.tar.gz").touch()
        
        # Test counter
        counter = FileCounter(tmpdir, recursive=False)
        
        jpg_counts = counter.count_jpg_thumbnails()
        assert jpg_counts["total_jpg_files"] == 3, "Wrong JPG count"
        assert jpg_counts["1x1_jpg_files"] == 2, "Wrong 1x1 count"
        
        tar_counts = counter.count_tar_archives()
        assert tar_counts["total_archive_files"] == 2, "Wrong archive count"
        
        print("  ✓ File counter OK")
        return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("OAI IMAGING ANALYSIS - MODULE TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_config,
        test_manifest_analyzer,
        test_metadata_analyzer,
        test_file_counter,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        print("\n⚠️  Some tests failed!")
        return 1
    else:
        print("\n✓ All tests passed!")
        print("\nYou can now use the tools:")
        print("  python oai_summary.py --help")
        return 0


if __name__ == "__main__":
    sys.exit(main())
