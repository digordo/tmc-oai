from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .inventory import build_package_disk_index, load_package_timepoint_map
from .io import read_oai_txt

SEMIQ_FILES: dict[str, str] = {
    "Knee": "oai_kxrsemiquant01.txt",
    "Hip": "oai_hxrsemiquant01.txt",
}


@dataclass(frozen=True, slots=True)
class SemiquantJoinResult:
    dataset_root: Path
    map_csv: Path
    xraymeta_package_number: str
    xraymeta_package_dir: Path
    image_xray_df: pd.DataFrame
    asset_coverage: pd.DataFrame
    collision_assets: pd.DataFrame
    joined_by_region: dict[str, pd.DataFrame]
    category_columns_raw: dict[str, list[str]]
    category_columns_by_region: dict[str, list[str]]
    region_summary: pd.DataFrame


def build_semiquant_join(dataset_root: Path, map_csv: Path) -> SemiquantJoinResult:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    map_csv_path = Path(map_csv).expanduser().resolve()
    map_df = load_package_timepoint_map(map_csv_path)

    xraymeta = map_df.loc[
        map_df["timepoint_label"].astype(str).str.upper().eq("XRAYMETA"),
        "package_number",
    ]
    if xraymeta.empty:
        raise ValueError("Could not resolve XRAYMETA package number from package map CSV.")

    xraymeta_package_number = str(xraymeta.iloc[-1])
    xraymeta_package_dir = dataset_root_path / f"Package_{xraymeta_package_number}"
    if not xraymeta_package_dir.exists():
        raise FileNotFoundError(f"XRAYMETA package not found: {xraymeta_package_dir}")

    image_rows: list[pd.DataFrame] = []
    for package_dir in sorted(
        [path for path in dataset_root_path.iterdir() if path.is_dir() and path.name.startswith("Package_")],
        key=lambda path: path.name,
    ):
        image03_path = package_dir / "image03.txt"
        if not image03_path.exists():
            continue

        try:
            image_df = read_oai_txt(image03_path, skip_dictionary_row=True)
        except Exception:
            continue

        modality = image_df.get("image_modality", pd.Series("", index=image_df.index)).astype(str)
        description = image_df.get("image_description", pd.Series("", index=image_df.index)).astype(str)
        xray_mask = modality.str.lower().eq("x-ray") | description.str.contains(
            "knee|hip|pelvis|hand",
            case=False,
            regex=True,
        )
        xray_df = image_df.loc[xray_mask].copy()
        if xray_df.empty:
            continue

        xray_df["package_number"] = package_dir.name.replace("Package_", "")
        xray_df["body_region"] = _body_region_from_description(
            xray_df.get("image_description", pd.Series("", index=xray_df.index))
        )
        xray_df["accession_id"] = _normalize_asset_id(
            xray_df.get("accession_number", pd.Series("", index=xray_df.index))
        )
        xray_df["dicom_id"] = _normalize_asset_id(
            xray_df.get("image_file", pd.Series("", index=xray_df.index))
        )
        xray_df["jpg_id"] = _normalize_asset_id(
            xray_df.get("image_thumbnail_file", pd.Series("", index=xray_df.index))
        )

        xray_df["asset_id"] = xray_df["accession_id"].where(
            xray_df["accession_id"].ne(""),
            xray_df["dicom_id"],
        )
        xray_df["asset_id"] = xray_df["asset_id"].where(
            xray_df["asset_id"].ne(""),
            xray_df["jpg_id"],
        )
        xray_df = xray_df.loc[xray_df["asset_id"].ne("")].copy()
        if xray_df.empty:
            continue

        jpg_ids, dicom_ids = build_package_disk_index(package_dir)
        xray_df["has_jpg"] = xray_df["asset_id"].isin(jpg_ids) | xray_df["jpg_id"].isin(jpg_ids)
        xray_df["has_dicom"] = xray_df["asset_id"].isin(dicom_ids) | xray_df["dicom_id"].isin(dicom_ids)

        image_rows.append(
            xray_df[
                [
                    "asset_id",
                    "src_subject_id",
                    "visit",
                    "body_region",
                    "package_number",
                    "has_jpg",
                    "has_dicom",
                ]
            ].copy()
        )

    if image_rows:
        image_xray_df = pd.concat(image_rows, ignore_index=True)
    else:
        image_xray_df = pd.DataFrame(
            columns=[
                "asset_id",
                "src_subject_id",
                "visit",
                "body_region",
                "package_number",
                "has_jpg",
                "has_dicom",
            ]
        )

    asset_coverage = (
        image_xray_df.groupby(["asset_id", "body_region"], as_index=False).agg(
            has_jpg=("has_jpg", "max"),
            has_dicom=("has_dicom", "max"),
            join_match_count=("asset_id", "size"),
        )
        if not image_xray_df.empty
        else pd.DataFrame(
            columns=["asset_id", "body_region", "has_jpg", "has_dicom", "join_match_count"]
        )
    )
    collision_assets = (
        asset_coverage.loc[asset_coverage["join_match_count"] > 1]
        .sort_values(["join_match_count", "body_region", "asset_id"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    joined_by_region: dict[str, pd.DataFrame] = {}
    raw_columns_by_region: dict[str, list[str]] = {}
    unique_columns_by_region: dict[str, list[str]] = {}
    region_rows: list[dict[str, int | str]] = []

    for region, filename in SEMIQ_FILES.items():
        semiquant_path = xraymeta_package_dir / filename
        if not semiquant_path.exists():
            joined_by_region[region] = pd.DataFrame()
            raw_columns_by_region[region] = []
            unique_columns_by_region[region] = []
            continue

        semiquant_df = read_oai_txt(semiquant_path, skip_dictionary_row=True).fillna("")
        semiquant_df["asset_id"] = _normalize_asset_id(
            semiquant_df.get("barcode", pd.Series("", index=semiquant_df.index))
        )
        semiquant_df = semiquant_df.loc[semiquant_df["asset_id"].ne("")].copy()

        region_coverage = asset_coverage.loc[
            asset_coverage["body_region"].eq(region),
            ["asset_id", "has_jpg", "has_dicom", "join_match_count"],
        ].copy()
        joined = semiquant_df.merge(region_coverage, on="asset_id", how="left")
        joined["has_jpg"] = pd.Series(joined["has_jpg"], dtype="boolean").fillna(False).astype(bool)
        joined["has_dicom"] = pd.Series(joined["has_dicom"], dtype="boolean").fillna(False).astype(bool)
        joined["join_match_count"] = joined["join_match_count"].fillna(0).astype(int)
        joined["has_any_image"] = joined["has_jpg"] | joined["has_dicom"]
        joined["missing_both"] = ~(joined["has_jpg"] | joined["has_dicom"])
        joined["body_region"] = region

        joined_by_region[region] = joined
        raw_columns_by_region[region] = _category_columns(joined)
        region_rows.append(
            {
                "region": region,
                "semiquant_rows": int(len(joined)),
                "rows_with_jpeg": int(joined["has_jpg"].sum()),
                "rows_with_dicom": int(joined["has_dicom"].sum()),
                "rows_with_both": int((joined["has_jpg"] & joined["has_dicom"]).sum()),
                "rows_missing_both": int(joined["missing_both"].sum()),
                "distinct_assets": int(joined["asset_id"].nunique()),
                "collision_assets": int(
                    region_coverage.loc[region_coverage["join_match_count"] > 1, "asset_id"].nunique()
                ),
            }
        )

    for region in SEMIQ_FILES:
        current = set(raw_columns_by_region.get(region, []))
        other_regions = [other for other in SEMIQ_FILES if other != region]
        other_columns: set[str] = set()
        for other in other_regions:
            other_columns.update(raw_columns_by_region.get(other, []))
        unique_columns_by_region[region] = sorted(current - other_columns)

    region_summary = pd.DataFrame(region_rows)
    if region_summary.empty:
        region_summary = pd.DataFrame(
            columns=[
                "region",
                "semiquant_rows",
                "rows_with_jpeg",
                "rows_with_dicom",
                "rows_with_both",
                "rows_missing_both",
                "distinct_assets",
                "collision_assets",
            ]
        )

    return SemiquantJoinResult(
        dataset_root=dataset_root_path,
        map_csv=map_csv_path,
        xraymeta_package_number=xraymeta_package_number,
        xraymeta_package_dir=xraymeta_package_dir,
        image_xray_df=image_xray_df,
        asset_coverage=asset_coverage,
        collision_assets=collision_assets,
        joined_by_region=joined_by_region,
        category_columns_raw=raw_columns_by_region,
        category_columns_by_region=unique_columns_by_region,
        region_summary=region_summary,
    )


def _normalize_asset_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.rsplit("/", n=1)
        .str[-1]
        .str.removesuffix(".tar.gz")
        .str.removesuffix(".jpg")
        .str.removesuffix("_1x1")
        .str.removesuffix("_2x2")
    )


def _body_region_from_description(description: pd.Series) -> pd.Series:
    desc = description.astype(str).str.lower()
    return pd.Series(
        np.select(
            [
                desc.str.contains("knee", regex=False),
                desc.str.contains("hip", regex=False),
                desc.str.contains("pelvis", regex=False),
                desc.str.contains("hand", regex=False),
            ],
            ["Knee", "Hip", "Hip", "Hand"],
            default="Other",
        ),
        index=description.index,
    )


def _category_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "collection_id",
        "dataset_id",
        "subjectkey",
        "src_subject_id",
        "interview_date",
        "interview_age",
        "barcode",
        "collection_title",
        "asset_id",
        "has_jpg",
        "has_dicom",
        "has_any_image",
        "missing_both",
        "join_match_count",
        "body_region",
    }

    columns: list[str] = []
    for column_name in df.columns:
        if column_name in excluded:
            continue
        normalized = str(column_name).strip().lower()
        if normalized.endswith("_semiquant01_id") or "semiquant01_id" in normalized:
            continue

        values = df[column_name].astype(str).str.strip()
        if values.eq("").all():
            continue
        columns.append(str(column_name))

    return sorted(columns)


__all__ = [
    "SEMIQ_FILES",
    "SemiquantJoinResult",
    "build_semiquant_join",
]
