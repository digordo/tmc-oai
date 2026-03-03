# OAI Module

This module contains the productionized OAI notebook pipeline previously developed under `_development/experiments`:

- `_development/experiments/oai_package_inventory.ipynb`
- `_development/experiments/oai_xray_txt_schema_explorer.ipynb`

## Scope

This migration includes:

- OAI dataset environment loading via local `.oai_env.json`.
- Package inventory building for `image03.txt` metadata vs on-disk JPEG/DICOM assets.
- XRAYMETA semiquant joins and venn payload generation.
- TXT schema exploration and shared/unique column summaries.
- Thin notebooks under `OAI/notebooks/` that call production APIs in `tmc_oai`.

This migration does not include legacy code under `OAI/src/tmc-oai/` (hyphenated path). That code is left unchanged and documented as legacy in `MODULE.md`.

## Configuration

Copy `OAI/.oai_env.template.json` to `OAI/.oai_env.json` and set:

- `oai_dataset_root` (required)
- `timepoint_map_csv` (optional; defaults to `OAI/reference/oai_package_timepoint_map.csv`)
- `output_root` (optional; defaults to `OAI/outputs`)

`OAI/.oai_env.json` is gitignored.

## Dependency Notes

The notebook pipeline no longer depends on:

- `tmc_common`
- `tmc_data`

The package-timepoint lookup CSV has moved from:

- `MyMLClassification/reference/oai_package_timepoint_map.csv`

to:

- `OAI/reference/oai_package_timepoint_map.csv`

## Standalone Repo Extraction Notes

When extracting `OAI/` into its own git repository:

1. Preserve history with `git subtree split` or `git filter-repo`.
2. Keep `OAI/.oai_env.template.json` tracked.
3. Keep `OAI/.oai_env.json` untracked (already in `OAI/.gitignore`).
4. Treat `OAI/src/tmc-oai` as legacy and out-of-scope unless you choose to migrate it later.

## Quick API Example

```python
from tmc_oai import (
    load_oai_env,
    build_package_inventory,
    build_semiquant_join,
    build_venn_payload,
)

cfg = load_oai_env()
inventory = build_package_inventory(cfg.oai_dataset_root, cfg.timepoint_map_csv)
semiquant = build_semiquant_join(cfg.oai_dataset_root, cfg.timepoint_map_csv)
knee_payload = build_venn_payload(semiquant.joined_by_region["Knee"], "knee_grade", top_n=6)
```
