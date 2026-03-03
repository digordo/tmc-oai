# OAI Module Contract

## Purpose

`tmc_oai` is the production module for OAI notebook-pipeline workflows:

- package inventory from `image03.txt` and on-disk JPEG/DICOM files
- XRAYMETA semiquant joins and venn payloads
- OAI TXT schema exploration

## Public API

### Environment

- `load_oai_env(config_path: Path | None = None) -> OAIEnv`

### IO

- `read_oai_txt(path: Path, *, skip_dictionary_row: bool = True) -> pd.DataFrame`

### Inventory

- `build_package_inventory(dataset_root: Path, map_csv: Path) -> PackageInventoryResult`

### Schema Explorer

- `build_schema_explorer(package_dir: Path) -> SchemaExplorerResult`

### Semiquant

- `build_semiquant_join(dataset_root: Path, map_csv: Path) -> SemiquantJoinResult`

### Venn

- `build_venn_payload(
  joined_df: pd.DataFrame,
  category_column: str | None,
  top_n: int,
) -> VennPayload`

## Data Ownership

- `oai_package_timepoint_map.csv` is owned by this module at:
  - `OAI/reference/oai_package_timepoint_map.csv`

## Removed Cross-Module Dependencies

- `tmc_common` removed from this pipeline.
- `tmc_data` not required by this pipeline.

## Repository Boundary

- `OAI` is designed to be extracted and maintained as its own repository (or submodule boundary in the monorepo).
- Code in `OAI` must only reference:
  - `tmc_oai` (this module),
  - Python standard library,
  - third-party dependencies declared in `OAI/pyproject.toml`.
- Code in `OAI` must not import from sibling repositories/modules in `TMCProject_V2` (`Common`, `Data`, `Analysis`, `MyPyMujoco`, `MyMLClassification`, `_development`).
- Other modules in `TMCProject_V2` may import and depend on `tmc_oai`.

## Legacy Code

- `OAI/src/tmc-oai/*` is legacy and out of scope for this notebook-pipeline migration.
- New production path is `OAI/src/tmc_oai/*`.
