from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path

import pandas as pd

from .io import DELIMITER_CANDIDATES, detect_delimiter, detect_encoding, read_text_preview


@dataclass(frozen=True, slots=True)
class ParsedTxtFile:
    file_name: str
    path: Path
    parse_type: str
    encoding: str
    delimiter: str
    delimiter_scores: dict[str, int]
    column_count: int
    description_row_detected: bool
    preview_row_count: int
    preview_df: pd.DataFrame
    data_df: pd.DataFrame
    dictionary_df: pd.DataFrame
    text_preview_lines: list[str]


@dataclass(frozen=True, slots=True)
class SchemaExplorerResult:
    package_dir: Path
    summary_df: pd.DataFrame
    files: dict[str, ParsedTxtFile]


@dataclass(frozen=True, slots=True)
class SchemaComparisonResult:
    left_file: str
    right_file: str
    shared_summary_df: pd.DataFrame
    left_unique_summary_df: pd.DataFrame
    right_unique_summary_df: pd.DataFrame


def build_schema_explorer(package_dir: Path) -> SchemaExplorerResult:
    package_path = Path(package_dir).expanduser().resolve()
    if not package_path.exists():
        raise FileNotFoundError(f"Package directory not found: {package_path}")
    if not package_path.is_dir():
        raise NotADirectoryError(f"Package path is not a directory: {package_path}")

    txt_files = sorted(package_path.glob("*.txt"), key=lambda path: path.name.lower())
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {package_path}")

    parsed_files: dict[str, ParsedTxtFile] = {}
    summary_rows: list[dict[str, int | str | bool]] = []
    for txt_file in txt_files:
        parsed = _parse_txt_file(txt_file)
        parsed_files[parsed.file_name.lower()] = parsed
        summary_rows.append(
            {
                "file_name": parsed.file_name,
                "parse_type": parsed.parse_type,
                "delimiter": parsed.delimiter,
                "column_count": parsed.column_count,
                "description_row_detected": parsed.description_row_detected,
                "preview_row_count": parsed.preview_row_count,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "file_name",
            "parse_type",
            "delimiter",
            "column_count",
            "description_row_detected",
            "preview_row_count",
        ],
    )

    return SchemaExplorerResult(
        package_dir=package_path,
        summary_df=summary_df,
        files=parsed_files,
    )


def build_schema_comparison(
    result: SchemaExplorerResult,
    left_file: str,
    right_file: str,
) -> SchemaComparisonResult:
    left_key = str(left_file).strip().lower()
    right_key = str(right_file).strip().lower()
    if left_key not in result.files:
        raise FileNotFoundError(f"Missing compare file in package: {left_file}")
    if right_key not in result.files:
        raise FileNotFoundError(f"Missing compare file in package: {right_file}")

    left = result.files[left_key]
    right = result.files[right_key]
    if left.parse_type != "tabular":
        raise ValueError(f"Expected tabular file for comparison: {left.file_name}")
    if right.parse_type != "tabular":
        raise ValueError(f"Expected tabular file for comparison: {right.file_name}")

    left_df = left.data_df.copy()
    right_df = right.data_df.copy()
    left_desc = _description_lookup(left.dictionary_df)
    right_desc = _description_lookup(right.dictionary_df)

    shared_columns = sorted(set(left_df.columns).intersection(set(right_df.columns)))
    left_unique_columns = sorted(set(left_df.columns) - set(right_df.columns))
    right_unique_columns = sorted(set(right_df.columns) - set(left_df.columns))

    left_name = left.file_name
    right_name = right.file_name
    left_desc_col = f"{left_name} description"
    right_desc_col = f"{right_name} description"
    left_type_col = f"{left_name} value_type"
    right_type_col = f"{right_name} value_type"

    shared_rows: list[dict[str, str]] = []
    for column_name in shared_columns:
        left_profile = summarize_column_values(left_df[column_name])
        right_profile = summarize_column_values(right_df[column_name])
        shared_rows.append(
            {
                "column_name": column_name,
                left_desc_col: left_desc.get(column_name, ""),
                left_type_col: left_profile["value_type"],
                f"{left_name} summary": left_profile["summary"],
                right_desc_col: right_desc.get(column_name, ""),
                right_type_col: right_profile["value_type"],
                f"{right_name} summary": right_profile["summary"],
            }
        )

    shared_summary_df = pd.DataFrame(shared_rows)
    if not shared_summary_df.empty:
        if shared_summary_df[left_desc_col].fillna("").astype(str).eq(
            shared_summary_df[right_desc_col].fillna("").astype(str)
        ).all():
            shared_summary_df["description"] = shared_summary_df[left_desc_col]
            shared_summary_df = shared_summary_df.drop(columns=[left_desc_col, right_desc_col])
        if shared_summary_df[left_type_col].fillna("").astype(str).eq(
            shared_summary_df[right_type_col].fillna("").astype(str)
        ).all():
            shared_summary_df["value_type"] = shared_summary_df[left_type_col]
            shared_summary_df = shared_summary_df.drop(columns=[left_type_col, right_type_col])

    expected_shared_columns = ["column_name"]
    if "description" in shared_summary_df.columns:
        expected_shared_columns.append("description")
    else:
        expected_shared_columns.extend([left_desc_col, right_desc_col])
    if "value_type" in shared_summary_df.columns:
        expected_shared_columns.append("value_type")
    else:
        expected_shared_columns.extend([left_type_col, right_type_col])
    expected_shared_columns.extend([f"{left_name} summary", f"{right_name} summary"])
    if shared_summary_df.empty:
        shared_summary_df = pd.DataFrame(columns=expected_shared_columns)
    else:
        shared_summary_df = shared_summary_df.loc[:, expected_shared_columns]

    left_unique_summary_df = build_unique_column_summary(
        left_df,
        left.dictionary_df,
        left_unique_columns,
    )
    right_unique_summary_df = build_unique_column_summary(
        right_df,
        right.dictionary_df,
        right_unique_columns,
    )

    return SchemaComparisonResult(
        left_file=left.file_name,
        right_file=right.file_name,
        shared_summary_df=shared_summary_df,
        left_unique_summary_df=left_unique_summary_df,
        right_unique_summary_df=right_unique_summary_df,
    )


def build_hover_table_html(data_preview: pd.DataFrame, dictionary_df: pd.DataFrame) -> str:
    if data_preview.empty:
        return "<em>No preview rows available.</em>"

    descriptions = _description_lookup(dictionary_df)
    header_cells: list[str] = []
    for column in data_preview.columns:
        description = html.escape(descriptions.get(str(column), ""), quote=True)
        label = html.escape(str(column))
        header_cells.append(f'<th title="{description}">{label}</th>')

    body_rows: list[str] = []
    for _, row in data_preview.iterrows():
        value_cells: list[str] = []
        for column in data_preview.columns:
            value = "" if pd.isna(row[column]) else str(row[column])
            value_cells.append(f"<td>{html.escape(value)}</td>")
        body_rows.append("<tr>" + "".join(value_cells) + "</tr>")

    return (
        '<table border="1" class="dataframe">'
        + "<thead><tr>"
        + "".join(header_cells)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def summarize_column_values(
    series: pd.Series,
    *,
    max_categories: int = 12,
    numeric_threshold: float = 0.90,
) -> dict[str, str]:
    clean = series.astype(str).str.strip()
    non_empty = clean[clean.ne("")]
    missing_count = int(clean.eq("").sum())

    if non_empty.empty:
        return {"value_type": "empty", "summary": f"all empty (missing={missing_count})"}

    numeric = pd.to_numeric(non_empty, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio >= numeric_threshold:
        numeric_values = numeric.dropna()
        summary = (
            f"range=[{_format_number(numeric_values.min())}, {_format_number(numeric_values.max())}] "
            f"| mean={_format_number(numeric_values.mean())} "
            f"| unique={int(numeric_values.nunique())} "
            f"| missing={missing_count}"
        )
        return {"value_type": "numeric", "summary": summary}

    unique_values = sorted(non_empty.unique().tolist(), key=lambda value: str(value))
    preview_values = [str(value) for value in unique_values[:max_categories]]
    preview_text = ", ".join(preview_values)
    if len(unique_values) > max_categories:
        preview_text += f", ... (+{len(unique_values) - max_categories} more)"
    return {
        "value_type": "categorical",
        "summary": f"values=[{preview_text}] | unique={len(unique_values)} | missing={missing_count}",
    }


def build_unique_column_summary(
    data_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    descriptions = _description_lookup(dictionary_df)
    rows: list[dict[str, str]] = []
    for column_name in columns:
        profile = summarize_column_values(data_df[column_name])
        rows.append(
            {
                "column_name": column_name,
                "description": descriptions.get(column_name, ""),
                "value_type": profile["value_type"],
                "value_summary": profile["summary"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=["column_name", "description", "value_type", "value_summary"],
    )


def _parse_txt_file(path: Path) -> ParsedTxtFile:
    encoding = detect_encoding(path)
    delimiter, delimiter_scores = detect_delimiter(path, encoding=encoding)

    if max(delimiter_scores.values()) == 0:
        return ParsedTxtFile(
            file_name=path.name,
            path=path,
            parse_type="text_blob",
            encoding=encoding,
            delimiter="",
            delimiter_scores=delimiter_scores,
            column_count=0,
            description_row_detected=False,
            preview_row_count=0,
            preview_df=pd.DataFrame(),
            data_df=pd.DataFrame(),
            dictionary_df=pd.DataFrame(columns=["field_name", "description"]),
            text_preview_lines=read_text_preview(path, n=12, encoding=encoding),
        )

    frame = pd.read_csv(
        path,
        sep=delimiter,
        dtype=str,
        keep_default_na=False,
        encoding=encoding,
        engine="python",
    ).fillna("")
    frame.columns = [str(column).strip() for column in frame.columns]

    if frame.empty:
        description_row = pd.Series([""] * len(frame.columns), index=frame.columns)
        data_df = frame.copy()
    else:
        description_row = frame.iloc[0]
        data_df = frame.iloc[1:].reset_index(drop=True).copy()
    preview_df = data_df.head(5).copy()

    dictionary_df = pd.DataFrame(
        {
            "field_name": frame.columns,
            "description": [
                str(description_row.get(column, "")).strip() for column in frame.columns
            ],
        }
    )
    description_detected = bool(
        not dictionary_df.empty
        and dictionary_df["description"].astype(str).str.strip().ne("").any()
    )

    return ParsedTxtFile(
        file_name=path.name,
        path=path,
        parse_type="tabular",
        encoding=encoding,
        delimiter=delimiter,
        delimiter_scores=delimiter_scores,
        column_count=int(len(frame.columns)),
        description_row_detected=description_detected,
        preview_row_count=int(len(preview_df)),
        preview_df=preview_df,
        data_df=data_df,
        dictionary_df=dictionary_df,
        text_preview_lines=[],
    )


def _description_lookup(dictionary_df: pd.DataFrame) -> dict[str, str]:
    if dictionary_df.empty:
        return {}
    return {
        str(row["field_name"]): str(row["description"]).strip()
        for _, row in dictionary_df.iterrows()
    }


def _format_number(value: float) -> str:
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


__all__ = [
    "DELIMITER_CANDIDATES",
    "ParsedTxtFile",
    "SchemaComparisonResult",
    "SchemaExplorerResult",
    "build_hover_table_html",
    "build_schema_comparison",
    "build_schema_explorer",
    "build_unique_column_summary",
    "summarize_column_values",
]
