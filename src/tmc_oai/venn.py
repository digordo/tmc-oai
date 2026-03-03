from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class VennCounts:
    only_jpg: int
    only_dicom: int
    both: int
    missing_both: int


@dataclass(frozen=True, slots=True)
class VennPayload:
    category_column: str | None
    total_rows: int
    overall: VennCounts
    by_value: pd.DataFrame


def build_venn_payload(
    joined_df: pd.DataFrame,
    category_column: str | None = None,
    top_n: int = 6,
) -> VennPayload:
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    required = {"has_jpg", "has_dicom"}
    missing = required - set(joined_df.columns)
    if missing:
        raise ValueError(f"joined_df is missing required columns: {sorted(missing)}")

    if joined_df.empty:
        empty_by_value = pd.DataFrame(
            columns=[
                "category_value",
                "row_count",
                "only_jpg",
                "only_dicom",
                "both",
                "missing_both",
            ]
        )
        return VennPayload(
            category_column=category_column,
            total_rows=0,
            overall=VennCounts(0, 0, 0, 0),
            by_value=empty_by_value,
        )

    working = joined_df.copy()
    if category_column and category_column in working.columns:
        working["_cat_value"] = (
            working[category_column].astype(str).str.strip().replace("", "<EMPTY>")
        )
    else:
        category_column = None
        working["_cat_value"] = "ALL"

    overall = _venn_counts(working)
    value_rows: list[dict[str, int | str]] = []
    top_values = working["_cat_value"].value_counts(dropna=False).head(int(top_n)).index.tolist()
    for value in top_values:
        subset = working.loc[working["_cat_value"].eq(value)]
        counts = _venn_counts(subset)
        value_rows.append(
            {
                "category_value": str(value),
                "row_count": int(len(subset)),
                "only_jpg": counts.only_jpg,
                "only_dicom": counts.only_dicom,
                "both": counts.both,
                "missing_both": counts.missing_both,
            }
        )

    by_value = pd.DataFrame(
        value_rows,
        columns=[
            "category_value",
            "row_count",
            "only_jpg",
            "only_dicom",
            "both",
            "missing_both",
        ],
    )
    return VennPayload(
        category_column=category_column,
        total_rows=int(len(working)),
        overall=overall,
        by_value=by_value,
    )


def _venn_counts(df: pd.DataFrame) -> VennCounts:
    if df.empty:
        return VennCounts(0, 0, 0, 0)

    jpg = df["has_jpg"].fillna(False).astype(bool)
    dicom = df["has_dicom"].fillna(False).astype(bool)
    only_jpg = int((jpg & ~dicom).sum())
    only_dicom = int((dicom & ~jpg).sum())
    both = int((jpg & dicom).sum())
    missing = int((~jpg & ~dicom).sum())
    return VennCounts(only_jpg=only_jpg, only_dicom=only_dicom, both=both, missing_both=missing)


__all__ = ["VennCounts", "VennPayload", "build_venn_payload"]
