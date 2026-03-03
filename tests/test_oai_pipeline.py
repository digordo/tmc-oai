from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tmc_oai import (
    build_schema_comparison,
    build_schema_explorer,
    build_semiquant_join,
    build_venn_payload,
    load_oai_env,
)


def test_load_oai_env_missing_config(tmp_path: Path) -> None:
    missing_config = tmp_path / ".oai_env.json"
    with pytest.raises(FileNotFoundError, match=r"\.oai_env\.json"):
        load_oai_env(missing_config)


def test_load_oai_env_valid_defaults(tmp_path: Path) -> None:
    repo_root = tmp_path / "oai_repo"
    repo_root.mkdir()
    dataset_root = repo_root / "dataset"
    dataset_root.mkdir()
    reference_dir = repo_root / "reference"
    reference_dir.mkdir()
    map_csv = reference_dir / "oai_package_timepoint_map.csv"
    map_csv.write_text("package_number,timepoint_label\n1243845,XRAYMETA\n", encoding="utf-8")

    config_path = repo_root / ".oai_env.json"
    config_path.write_text(
        json.dumps({"oai_dataset_root": "dataset"}, indent=2) + "\n",
        encoding="utf-8",
    )

    config = load_oai_env(config_path)
    assert config.repo_root == repo_root.resolve()
    assert config.oai_dataset_root == dataset_root.resolve()
    assert config.timepoint_map_csv == map_csv.resolve()


def test_build_semiquant_join_and_dropdown_rules(tmp_path: Path) -> None:
    dataset_root, map_csv = _build_mock_oai_dataset(tmp_path)
    result = build_semiquant_join(dataset_root, map_csv)

    knee = result.joined_by_region["Knee"]
    assert not knee.empty

    knee_by_asset = knee.set_index("asset_id")
    assert bool(knee_by_asset.loc["A1", "has_jpg"]) is True
    assert bool(knee_by_asset.loc["A1", "has_dicom"]) is True
    assert int(knee_by_asset.loc["A1", "join_match_count"]) == 2
    assert bool(knee_by_asset.loc["A3", "missing_both"]) is True

    summary = result.region_summary.set_index("region")
    assert int(summary.loc["Knee", "rows_missing_both"]) == 1
    assert int(summary.loc["Knee", "collision_assets"]) == 1

    knee_columns = result.category_columns_by_region["Knee"]
    hip_columns = result.category_columns_by_region["Hip"]
    assert knee_columns == ["knee_grade"]
    assert hip_columns == ["hip_grade"]
    assert all("semiquant01_id" not in column.lower() for column in knee_columns + hip_columns)


def test_schema_comparison_shared_dedupe_and_unique_descriptions(tmp_path: Path) -> None:
    dataset_root, _map_csv = _build_mock_oai_dataset(tmp_path)
    package_dir = dataset_root / "Package_1243845"

    explorer = build_schema_explorer(package_dir)
    comparison = build_schema_comparison(
        explorer,
        "oai_kxrsemiquant01.txt",
        "oai_hxrsemiquant01.txt",
    )

    shared_columns = comparison.shared_summary_df.columns.tolist()
    assert "description" in shared_columns
    assert "value_type" in shared_columns
    assert not any(column.endswith(" description") for column in shared_columns)
    assert not any(column.endswith(" value_type") for column in shared_columns)

    assert "description" in comparison.left_unique_summary_df.columns
    assert "value_summary" in comparison.left_unique_summary_df.columns
    assert "description" in comparison.right_unique_summary_df.columns
    assert "value_summary" in comparison.right_unique_summary_df.columns
    assert "source_file" not in comparison.left_unique_summary_df.columns
    assert "source_file" not in comparison.right_unique_summary_df.columns


def test_build_venn_payload_missing_counts(tmp_path: Path) -> None:
    dataset_root, map_csv = _build_mock_oai_dataset(tmp_path)
    result = build_semiquant_join(dataset_root, map_csv)
    knee = result.joined_by_region["Knee"]

    payload = build_venn_payload(knee, category_column="knee_grade", top_n=5)
    assert payload.overall.only_jpg == 0
    assert payload.overall.only_dicom == 1
    assert payload.overall.both == 1
    assert payload.overall.missing_both == 1
    assert "missing_both" in payload.by_value.columns
    assert int(payload.by_value["row_count"].sum()) == len(knee)


def _build_mock_oai_dataset(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()

    map_csv = reference_dir / "oai_package_timepoint_map.csv"
    map_csv.write_text(
        "\n".join(
            [
                "package_number,timepoint_label",
                "1243845,XRAYMETA",
                "999,BASELINE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    package_xraymeta = dataset_root / "Package_1243845"
    package_other = dataset_root / "Package_999"
    package_xraymeta.mkdir()
    package_other.mkdir()

    image_columns = [
        "src_subject_id",
        "visit",
        "accession_number",
        "image_file",
        "image_thumbnail_file",
        "image_modality",
        "image_description",
    ]
    image_descriptions = [
        "Subject ID",
        "Visit label",
        "Accession ID",
        "DICOM archive",
        "JPG thumbnail",
        "Modality",
        "Description",
    ]
    _write_oai_tabular_txt(
        package_xraymeta / "image03.txt",
        columns=image_columns,
        description_row=image_descriptions,
        rows=[
            {
                "src_subject_id": "1001",
                "visit": "V00",
                "accession_number": "A1",
                "image_file": "foo/A1.tar.gz",
                "image_thumbnail_file": "foo/A1_1x1.jpg",
                "image_modality": "X-RAY",
                "image_description": "Knee AP view",
            },
            {
                "src_subject_id": "1002",
                "visit": "V00",
                "accession_number": "A2",
                "image_file": "foo/A2.tar.gz",
                "image_thumbnail_file": "",
                "image_modality": "X-RAY",
                "image_description": "Knee lateral view",
            },
            {
                "src_subject_id": "1003",
                "visit": "V00",
                "accession_number": "H1",
                "image_file": "foo/H1.tar.gz",
                "image_thumbnail_file": "foo/H1_1x1.jpg",
                "image_modality": "X-RAY",
                "image_description": "Pelvis standing view",
            },
            {
                "src_subject_id": "1004",
                "visit": "V00",
                "accession_number": "H2",
                "image_file": "foo/H2.tar.gz",
                "image_thumbnail_file": "",
                "image_modality": "X-RAY",
                "image_description": "Hip AP view",
            },
        ],
    )
    _write_oai_tabular_txt(
        package_other / "image03.txt",
        columns=image_columns,
        description_row=image_descriptions,
        rows=[
            {
                "src_subject_id": "2001",
                "visit": "V01",
                "accession_number": "A1",
                "image_file": "bar/A1.tar.gz",
                "image_thumbnail_file": "",
                "image_modality": "X-RAY",
                "image_description": "Knee AP repeat",
            }
        ],
    )

    _write_oai_tabular_txt(
        package_xraymeta / "oai_kxrsemiquant01.txt",
        columns=["kxrsemiquant01_id", "barcode", "shared_col", "knee_grade"],
        description_row=[
            "Row id",
            "Image barcode",
            "Shared descriptor",
            "Knee category",
        ],
        rows=[
            {
                "kxrsemiquant01_id": "K1",
                "barcode": "A1.tar.gz",
                "shared_col": "SHARED_A",
                "knee_grade": "MILD",
            },
            {
                "kxrsemiquant01_id": "K2",
                "barcode": "A2.tar.gz",
                "shared_col": "SHARED_B",
                "knee_grade": "SEVERE",
            },
            {
                "kxrsemiquant01_id": "K3",
                "barcode": "A3.tar.gz",
                "shared_col": "SHARED_C",
                "knee_grade": "",
            },
        ],
    )
    _write_oai_tabular_txt(
        package_xraymeta / "oai_hxrsemiquant01.txt",
        columns=["hxrsemiquant01_id", "barcode", "shared_col", "hip_grade"],
        description_row=[
            "Row id",
            "Image barcode",
            "Shared descriptor",
            "Hip category",
        ],
        rows=[
            {
                "hxrsemiquant01_id": "H1",
                "barcode": "H1.tar.gz",
                "shared_col": "SHARED_A",
                "hip_grade": "LOW",
            },
            {
                "hxrsemiquant01_id": "H2",
                "barcode": "H2.tar.gz",
                "shared_col": "SHARED_B",
                "hip_grade": "HIGH",
            },
            {
                "hxrsemiquant01_id": "H3",
                "barcode": "H3.tar.gz",
                "shared_col": "SHARED_C",
                "hip_grade": "",
            },
        ],
    )

    for file_name in ["A1.tar.gz", "A1_1x1.jpg", "A2.tar.gz", "H1.tar.gz", "H1_1x1.jpg"]:
        (package_xraymeta / file_name).write_text("", encoding="utf-8")
    (package_other / "A1.tar.gz").write_text("", encoding="utf-8")

    return dataset_root, map_csv


def _write_oai_tabular_txt(
    path: Path,
    *,
    columns: list[str],
    description_row: list[str],
    rows: list[dict[str, str]],
) -> None:
    frame = pd.DataFrame(rows, columns=columns).fillna("")
    lines = [
        "\t".join(columns),
        "\t".join(description_row),
    ]
    for row in frame.itertuples(index=False):
        lines.append("\t".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
