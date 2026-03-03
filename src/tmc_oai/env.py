from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class OAIEnv:
    config_path: Path
    repo_root: Path
    oai_dataset_root: Path
    timepoint_map_csv: Path
    output_root: Path


def load_oai_env(config_path: Path | None = None) -> OAIEnv:
    repo_root = _resolve_repo_root(config_path)
    config_file = (
        config_path.expanduser().resolve()
        if config_path is not None
        else (repo_root / ".oai_env.json").resolve()
    )
    if not config_file.exists():
        raise FileNotFoundError(
            f"OAI config not found: {config_file}. "
            "Create it from OAI/.oai_env.template.json."
        )

    try:
        payload: Any = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in OAI config: {config_file}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"OAI config must be a JSON object: {config_file}")

    dataset_root_raw = str(payload.get("oai_dataset_root", "")).strip()
    if not dataset_root_raw:
        raise ValueError(
            "Missing required config field 'oai_dataset_root' in "
            f"{config_file}."
        )
    dataset_root = _resolve_path(dataset_root_raw, base_dir=config_file.parent)
    if not dataset_root.exists():
        raise FileNotFoundError(f"oai_dataset_root does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"oai_dataset_root is not a directory: {dataset_root}")

    map_csv_raw = str(payload.get("timepoint_map_csv", "")).strip()
    if map_csv_raw:
        map_csv = _resolve_path(map_csv_raw, base_dir=config_file.parent)
    else:
        map_csv = (repo_root / "reference" / "oai_package_timepoint_map.csv").resolve()
    if not map_csv.exists():
        raise FileNotFoundError(f"timepoint_map_csv not found: {map_csv}")
    if not map_csv.is_file():
        raise FileNotFoundError(f"timepoint_map_csv is not a file: {map_csv}")

    output_root_raw = str(payload.get("output_root", "")).strip()
    output_root = (
        _resolve_path(output_root_raw, base_dir=config_file.parent)
        if output_root_raw
        else (repo_root / "outputs").resolve()
    )

    return OAIEnv(
        config_path=config_file,
        repo_root=repo_root,
        oai_dataset_root=dataset_root,
        timepoint_map_csv=map_csv,
        output_root=output_root,
    )


def _resolve_repo_root(config_path: Path | None) -> Path:
    if config_path is not None:
        return config_path.expanduser().resolve().parent
    return Path(__file__).resolve().parents[2]


def _resolve_path(path_text: str, *, base_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


__all__ = ["OAIEnv", "load_oai_env"]
