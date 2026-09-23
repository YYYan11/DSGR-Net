# Open-source package manifest

This package contains the minimum source, configuration, derived results, and
documentation needed to inspect DSGR-Net and rerun the published protocol with
appropriately licensed input data.

## Included

- `src/dsgr_net.py`: exact formal V16 source snapshot;
- `src/experiment_data.py`: leakage-safe group splitting and cache construction;
- `src/calibrate_fixed_process.py` and `src/build_dssat_reference.py`: process
  calibration and DSSAT-reference conversion;
- `src/ablation/`: exact source snapshots for the controlled comparisons;
- `configs/`: formal seeds, hyperparameters, and frozen process settings;
- `scripts/`: cache preparation, training, ablation, gradient audit, testing,
  release audit, and table-reproduction commands;
- `data/schema.json` and `data/demo_synthetic.npz`: interface definition and a
  non-field synthetic smoke-test cache;
- `results/`: three-seed metrics and raw formal gradient-audit JSON files;
- `figures/`: plotting code, derived plotting inputs, and the paper figure;
- `paper/`: the current ICASSP LaTeX draft and included figures;
- `tests/`: forward, daily water-closure, and AD/finite-difference checks.

## Deliberately excluded

- WL-DNN implementation, WL/DSSAT project folders, and WL model data;
- DSSAT executables, weather/soil/cultivar input files, and raw output files;
- the multi-gigabyte simulation source JSON and derived training cache;
- 2024 field workbooks, sensor data, and field-validation cache;
- irrigation-allocation workbooks and historical experiment directories;
- trained checkpoints and temporary build products.

WL-DNN is retained only as a cited comparison with archived aggregate metrics.
Its implementation and data remain governed by their original source and are
not relicensed here.

## Release decisions

- Code license: MIT.
- Paper source: included.
- Checkpoints: excluded.
- Real and DSSAT-derived data: excluded pending separate redistribution rights.
