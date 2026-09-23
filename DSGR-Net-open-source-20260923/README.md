# DSGR-Net

Official code package for **DSGR-Net: A Differentiable Process-Informed Neural Network for Cotton Yield Response Modeling under Irrigation Management**.

DSGR-Net couples a strong environmental encoder with a differentiable daily crop-water-growth recurrence. Irrigation affects root-zone water, water stress, growth, and terminal yield in one computation graph. A temporal process encoder and bounded residual fusion combine the process trajectory with the environmental representation.

## What is included

- formal V16 implementation used for the paper;
- exact controlled-ablation source files;
- group-wise data preparation and process calibration code;
- paper metrics for seeds 42, 43, and 44;
- raw AD/finite-difference gradient audits;
- a synthetic cache for installation and smoke tests;
- scripts for training, ablation, gradient checks, and table reproduction.

The WL/DSSAT project data, DSSAT executables, field workbooks, and model checkpoints are not distributed. See [data/README.md](data/README.md).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is optional for the synthetic smoke test and required only for practical full-data training.

## Quick verification

```bash
python3 scripts/audit_release.py
python3 scripts/run_tests.py
```

The same tests are also compatible with `pytest` when it is available.

To run one small end-to-end training smoke test:

```bash
PYTHONPATH=src python3 src/dsgr_net.py train \
  --cache data/demo_synthetic.npz \
  --output-dir results/demo_run \
  --device cpu \
  --backbone dnn \
  --seed 42 \
  --hidden 16 \
  --batch-size 8 \
  --epochs 1 \
  --patience 1 \
  --learning-rate 5e-4 \
  --train-model process_nn \
  --use-nutrients \
  --fixed-yield-coefficient 0.42305775 \
  --skip-gradient-scan
```

The synthetic sample checks software behavior only and cannot reproduce the paper metrics.

## Reproducing the paper protocol

Prepare a compatible cache from data that you are licensed to use:

```bash
bash scripts/prepare_cache.sh /path/to/source.json /path/to/training_cache.npz 51480
```

Train formal V16 for the three paper seeds:

```bash
bash scripts/train_dsgr.sh /path/to/training_cache.npz cuda
```

Run the controlled ablation:

```bash
bash scripts/run_ablation.sh /path/to/training_cache.npz cuda
```

Audit irrigation gradients for trained checkpoints:

```bash
bash scripts/audit_gradients.sh /path/to/training_cache.npz /path/to/checkpoints cpu
```

Recreate the aggregate metrics table from the archived JSON files:

```bash
bash scripts/reproduce_tables.sh
```

The formal hyperparameters are recorded in `configs/dsgr_v16.json`; model-specific ablation settings are in `configs/experiments.json`.

## Repository structure

```text
configs/     formal hyperparameters and frozen process settings
src/         DSGR-Net, data pipeline, calibration, and exact ablations
scripts/     reproducible command-line workflows and release audit
data/        schema, data-availability note, and synthetic smoke-test cache
results/     paper metrics and gradient-audit JSON files
figures/     irrigation-sensitivity plotting inputs and output
paper/       ICASSP LaTeX draft
tests/       forward, water-balance, and gradient checks
```

## Formal experiment

The paper uses 51,282 cleaned DSSAT simulation records from 518 environment groups. Each seed uses a group-wise 363/78/77 train/validation/test split, preventing management scenarios from the same environment from crossing splits. The complete dataset is not part of this repository.

The exact A8 V16 source used to produce the archived results had SHA256:

```text
945ed86a070a7c8de638fcf01e03f746bf2e50a71083c8e848769ddb273687a7
```

`src/dsgr_net.py` is that source snapshot. Release metadata and later documentation files do not change the model computation.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The WL-DNN comparison should also cite Wang et al., *Agricultural Water Management*, 2025, doi: `10.1016/j.agwat.2025.109624`.

## License

Code is released under the MIT License. Third-party data and DSSAT components are not covered by this license and are not included.
