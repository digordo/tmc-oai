from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .io import read_oai_txt

XRAY_CATEGORIES: tuple[str, ...] = ("Hand", "Knee", "Pelvis", "Other")


@dataclass(frozen=True, slots=True)
class PackageInventoryResult:
    dataset_root: Path
    map_csv: Path
    package_map: pd.DataFrame
    category_counts: pd.DataFrame
    coverage_summary: pd.DataFrame
    orphan_summary: pd.DataFrame
    venn_rows: pd.DataFrame
    missing_packages: pd.DataFrame


def load_package_timepoint_map(map_csv_path: Path | str) -> pd.DataFrame:
    map_csv = Path(map_csv_path).expanduser().resolve()
    if not map_csv.exists():
        raise FileNotFoundError(f"Package map CSV not found: {map_csv}")

    map_df = pd.read_csv(map_csv, dtype=str).fillna("")
    required = {"package_number", "timepoint_label"}
    missing_required = required - set(map_df.columns)
    if missing_required:
        raise ValueError(
            f"Package map missing required columns {sorted(missing_required)}: {map_csv}"
        )

    normalized = map_df.copy()
    normalized["package_number"] = (
        normalized["package_number"].astype(str).map(_canonical_package_number)
    )
    normalized["timepoint_label"] = (
        normalized["timepoint_label"].astype(str).str.strip().str.upper()
    )
    normalized = normalized.loc[
        normalized["package_number"].ne("") & normalized["timepoint_label"].ne(""),
        ["package_number", "timepoint_label"],
    ].drop_duplicates(subset=["package_number"], keep="last")

    return normalized.sort_values("package_number").reset_index(drop=True)


def resolve_package_selection(
    package_map: pd.DataFrame,
    *,
    package_number: str = "",
    timepoint_label: str = "",
) -> tuple[str, str]:
    if package_map.empty:
        raise ValueError("Package map is empty and cannot resolve package selection.")

    requested_package = _canonical_package_number(str(package_number))
    requested_label = str(timepoint_label).strip().upper()

    if requested_package:
        match = package_map.loc[
            package_map["package_number"].eq(requested_package),
            "timepoint_label",
        ]
        if match.empty:
            known_numbers = sorted(package_map["package_number"].unique().tolist())
            raise ValueError(
                f"Unknown package_number={requested_package}. "
                f"Available package numbers: {known_numbers}"
            )
        return requested_package, str(match.iloc[-1]).strip().upper()

    if not requested_label:
        labels = sorted(package_map["timepoint_label"].unique().tolist())
        raise ValueError(
            "timepoint_label is empty and package_number was not provided. "
            f"Available labels: {labels}"
        )

    match = package_map.loc[
        package_map["timepoint_label"].eq(requested_label),
        "package_number",
    ]
    if match.empty:
        labels = sorted(package_map["timepoint_label"].unique().tolist())
        raise ValueError(
            f"Could not resolve timepoint_label={requested_label!r}. "
            f"Available labels: {labels}"
        )
    return str(match.iloc[-1]), requested_label


def build_package_disk_index(package_dir: Path) -> tuple[set[str], set[str]]:
    if not package_dir.exists():
        return set(), set()

    files = pd.Index([path.name for path in package_dir.iterdir() if path.is_file()])
    leaf = files.str.rsplit("%2F", n=1).str[-1]

    jpg_ids = set(
        leaf[leaf.str.endswith(".jpg")]
        .str.removesuffix(".jpg")
        .str.removesuffix("_1x1")
        .str.removesuffix("_2x2")
    )
    dicom_ids = set(
        leaf[leaf.str.endswith(".tar.gz")]
        .str.removesuffix(".tar.gz")
    )
    return jpg_ids, dicom_ids


def build_package_inventory(dataset_root: Path, map_csv: Path) -> PackageInventoryResult:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    map_csv_path = Path(map_csv).expanduser().resolve()
    package_map = load_package_timepoint_map(map_csv_path)

    category_rows: list[dict[str, int | str]] = []
    venn_frames: list[pd.DataFrame] = []
    orphan_rows: list[dict[str, int | str]] = []
    missing_rows: list[dict[str, str]] = []

    for row in package_map.itertuples(index=False):
        package_number = str(row.package_number)
        timepoint_label = str(row.timepoint_label)
        package_dir = dataset_root_path / f"Package_{package_number}"
        image03_path = package_dir / "image03.txt"

        if not package_dir.exists():
            missing_rows.append(
                {
                    "package_number": package_number,
                    "timepoint_label": timepoint_label,
                    "reason": "missing_package_dir",
                    "path": str(package_dir),
                }
            )
            continue
        if not image03_path.exists():
            missing_rows.append(
                {
                    "package_number": package_number,
                    "timepoint_label": timepoint_label,
                    "reason": "missing_image03",
                    "path": str(image03_path),
                }
            )
            continue

        try:
            image_meta = read_oai_txt(image03_path, skip_dictionary_row=True)
        except Exception as exc:  # pragma: no cover - surfaced as tabular parse failure
            missing_rows.append(
                {
                    "package_number": package_number,
                    "timepoint_label": timepoint_label,
                    "reason": "image03_parse_error",
                    "path": str(image03_path),
                    "detail": str(exc),
                }
            )
            continue

        xray_df = _xray_rows(image_meta)
        if xray_df.empty:
            for category in XRAY_CATEGORIES:
                category_rows.append(
                    {
                        "package_number": package_number,
                        "timepoint": timepoint_label,
                        "category": category,
                        "meta": 0,
                        "jpg": 0,
                        "dicom": 0,
                    }
                )
            continue

        jpg_ids, dicom_ids = build_package_disk_index(package_dir)
        xray_df = _add_disk_presence(xray_df, jpg_ids=jpg_ids, dicom_ids=dicom_ids)

        category_meta = (
            xray_df["body_category"].value_counts().reindex(XRAY_CATEGORIES, fill_value=0)
        )
        category_presence = (
            xray_df.groupby("body_category")
            .agg(jpg=("has_jpg", "sum"), dicom=("has_dicom", "sum"))
            .reindex(XRAY_CATEGORIES, fill_value=0)
            .astype("int64")
        )

        for category in XRAY_CATEGORIES:
            category_rows.append(
                {
                    "package_number": package_number,
                    "timepoint": timepoint_label,
                    "category": category,
                    "meta": int(category_meta.get(category, 0)),
                    "jpg": int(category_presence.loc[category, "jpg"]),
                    "dicom": int(category_presence.loc[category, "dicom"]),
                }
            )

        venn_frame = xray_df[["body_category", "row_id", "has_jpg", "has_dicom"]].copy()
        venn_frame = venn_frame.rename(
            columns={"body_category": "category", "row_id": "row_key"}
        )
        venn_frame["row_key"] = package_number + ":" + venn_frame["row_key"].astype(str)
        venn_frame["timepoint"] = timepoint_label
        venn_frames.append(
            venn_frame[["timepoint", "category", "row_key", "has_jpg", "has_dicom"]]
        )

        metadata_ids = _metadata_asset_ids(image_meta)
        orphan_jpg_ids = jpg_ids - metadata_ids
        orphan_dicom_ids = dicom_ids - metadata_ids
        orphan_rows.append(
            {
                "package_number": package_number,
                "timepoint": timepoint_label,
                "orphan_jpg": int(len(orphan_jpg_ids)),
                "orphan_dicom": int(len(orphan_dicom_ids)),
                "orphan_total": int(len(orphan_jpg_ids | orphan_dicom_ids)),
            }
        )

    category_counts = pd.DataFrame(category_rows)
    if category_counts.empty:
        category_counts = pd.DataFrame(
            columns=["package_number", "timepoint", "category", "meta", "jpg", "dicom"]
        )
    else:
        category_counts = category_counts.sort_values(
            ["timepoint", "package_number", "category"]
        ).reset_index(drop=True)

    coverage_summary = (
        category_counts.groupby("timepoint", as_index=False)[["meta", "jpg", "dicom"]].sum()
        if not category_counts.empty
        else pd.DataFrame(columns=["timepoint", "meta", "jpg", "dicom"])
    )

    orphan_summary = pd.DataFrame(orphan_rows)
    if orphan_summary.empty:
        orphan_summary = pd.DataFrame(
            columns=["timepoint", "orphan_jpg", "orphan_dicom", "orphan_total"]
        )
    else:
        orphan_summary = (
            orphan_summary.groupby("timepoint", as_index=False)[
                ["orphan_jpg", "orphan_dicom", "orphan_total"]
            ]
            .sum()
            .sort_values("timepoint")
            .reset_index(drop=True)
        )

    coverage_summary = coverage_summary.merge(orphan_summary, on="timepoint", how="left")
    for column in ("orphan_jpg", "orphan_dicom", "orphan_total"):
        if column not in coverage_summary.columns:
            coverage_summary[column] = 0
        coverage_summary[column] = coverage_summary[column].fillna(0).astype(int)

    venn_rows = (
        pd.concat(venn_frames, ignore_index=True)
        if venn_frames
        else pd.DataFrame(columns=["timepoint", "category", "row_key", "has_jpg", "has_dicom"])
    )
    missing_packages = pd.DataFrame(missing_rows)
    if missing_packages.empty:
        missing_packages = pd.DataFrame(
            columns=["package_number", "timepoint_label", "reason", "path", "detail"]
        )

    return PackageInventoryResult(
        dataset_root=dataset_root_path,
        map_csv=map_csv_path,
        package_map=package_map,
        category_counts=category_counts,
        coverage_summary=coverage_summary,
        orphan_summary=orphan_summary,
        venn_rows=venn_rows,
        missing_packages=missing_packages,
    )


def _add_disk_presence(
    metadata_rows: pd.DataFrame,
    *,
    jpg_ids: set[str],
    dicom_ids: set[str],
) -> pd.DataFrame:
    frame = metadata_rows.copy()
    frame["dicom_id"] = (
        frame.get("image_file", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.removesuffix(".tar.gz")
    )
    frame["jpg_id"] = (
        frame.get("image_thumbnail_file", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.removesuffix(".jpg")
        .str.removesuffix("_1x1")
        .str.removesuffix("_2x2")
    )

    row_id = frame["dicom_id"].where(frame["dicom_id"].ne(""), frame["jpg_id"])
    row_id = row_id.where(row_id.ne(""), frame.index.astype(str))
    frame["row_id"] = row_id
    frame["has_jpg"] = row_id.isin(jpg_ids)
    frame["has_dicom"] = row_id.isin(dicom_ids)
    return frame


def _xray_rows(image_df: pd.DataFrame) -> pd.DataFrame:
    modality = image_df.get("image_modality", pd.Series("", index=image_df.index))
    description = image_df.get("image_description", pd.Series("", index=image_df.index))
    mask = modality.fillna("").astype(str).str.lower().eq("x-ray")
    frame = image_df.loc[mask].copy()
    if frame.empty:
        return frame

    desc = description.loc[frame.index].fillna("").astype(str).str.lower()
    frame["body_category"] = np.select(
        [
            desc.str.contains("hand", regex=False),
            desc.str.contains("knee", regex=False),
            desc.str.contains("pelvis", regex=False) | desc.str.contains("hip", regex=False),
        ],
        ["Hand", "Knee", "Pelvis"],
        default="Other",
    )
    return frame


def _metadata_asset_ids(image_df: pd.DataFrame) -> set[str]:
    dicom_ref = (
        image_df.get("image_file", pd.Series("", index=image_df.index))
        .fillna("")
        .astype(str)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.removesuffix(".tar.gz")
    )
    jpg_ref = (
        image_df.get("image_thumbnail_file", pd.Series("", index=image_df.index))
        .fillna("")
        .astype(str)
        .str.rsplit("/", n=1)
        .str[-1]
        .str.removesuffix(".jpg")
        .str.removesuffix("_1x1")
        .str.removesuffix("_2x2")
    )
    return set(dicom_ref[dicom_ref.ne("")]) | set(jpg_ref[jpg_ref.ne("")])


def _canonical_package_number(raw_value: str) -> str:
    text = str(raw_value).strip()
    if not text:
        return ""
    digits = re.sub(r"\D+", "", text)
    if not digits:
        return ""
    return str(int(digits))


__all__ = [
    "PackageInventoryResult",
    "XRAY_CATEGORIES",
    "build_package_disk_index",
    "build_package_inventory",
    "load_package_timepoint_map",
    "resolve_package_selection",
]
