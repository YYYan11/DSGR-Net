# Data interface and availability

The paper training archive and field-validation workbooks are not redistributed. They contain WL/DSSAT-derived simulation records or local field data whose redistribution rights are separate from the code license.

The repository provides:

- `schema.json`: the exact sequence/static feature schema used by the formal experiment;
- `demo_synthetic.npz`: generated values with the same cache interface, used only for tests;
- `src/experiment_data.py`: streaming conversion from a user-supplied source JSON to the NumPy cache.

## Cache arrays

| Key | Shape | Meaning |
|---|---|---|
| `sequence` | `[N, 211, 9]` | daily SRAD, TMAX, TMIN, RAIN, WIND, IRR, N, P, K |
| `static` | `[N, D]` | initial soil, soil-profile, planting, and weather descriptors |
| `labels` | `[N]` | terminal yield in kg/ha |
| `planting_day` | `[N]` | planting day of year |
| `groups` | `[N]` | environment group used for leakage-safe splitting |
| `metadata` | scalar JSON string | ordered feature names and provenance |

All samples belonging to one environment group must remain in a single split. The formal seeds are 42, 43, and 44.

## Excluded material

The release deliberately excludes `wl-dssat/`, DSSAT executables and input/output files, `output_all_12_22.json`, 2024 field workbooks/caches, water-allocation workbooks, and serialized checkpoints.
