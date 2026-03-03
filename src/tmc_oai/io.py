from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "latin1",
)
DELIMITER_CANDIDATES: tuple[str, ...] = ("\t", ",", "|")


def detect_encoding(path: Path) -> str:
    for encoding in SUPPORTED_ENCODINGS:
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeError:
            continue
    raise UnicodeError(f"Could not decode file with supported encodings: {path}")


def read_text_preview(
    path: Path,
    *,
    n: int = 12,
    encoding: str | None = None,
) -> list[str]:
    effective_encoding = encoding or detect_encoding(path)
    return path.read_text(encoding=effective_encoding).splitlines()[:n]


def detect_delimiter(path: Path, encoding: str) -> tuple[str, dict[str, int]]:
    lines = [line for line in read_text_preview(path, n=12, encoding=encoding) if line.strip()]
    if not lines:
        return "\t", {candidate: 0 for candidate in DELIMITER_CANDIDATES}

    scores = {
        candidate: sum(line.count(candidate) for line in lines)
        for candidate in DELIMITER_CANDIDATES
    }
    return max(scores, key=scores.get), scores


def read_oai_txt(path: Path, *, skip_dictionary_row: bool = True) -> pd.DataFrame:
    encoding = detect_encoding(path)
    delimiter, scores = detect_delimiter(path, encoding=encoding)
    if max(scores.values()) == 0:
        raise ValueError(f"File does not appear tabular: {path}")

    read_kwargs = {
        "sep": delimiter,
        "dtype": str,
        "keep_default_na": False,
        "encoding": encoding,
        "engine": "python",
    }
    try:
        frame = pd.read_csv(path, **read_kwargs)
    except pd.errors.ParserError:
        # Some OAI image03 exports contain malformed quoting on a small number of lines.
        # Fallback to skipping only the malformed lines so the rest of the table is still usable.
        frame = pd.read_csv(path, on_bad_lines="skip", **read_kwargs)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.fillna("")

    if skip_dictionary_row and not frame.empty:
        frame = frame.iloc[1:].reset_index(drop=True)

    return frame


__all__ = [
    "DELIMITER_CANDIDATES",
    "SUPPORTED_ENCODINGS",
    "detect_delimiter",
    "detect_encoding",
    "read_oai_txt",
    "read_text_preview",
]
