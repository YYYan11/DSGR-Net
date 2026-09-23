# Release audit

Audit date: 2026-09-23

## Integrity and privacy

- The formal V16 source hash is
  `945ed86a070a7c8de638fcf01e03f746bf2e50a71083c8e848769ddb273687a7`.
- The package contains no local/server absolute paths, host addresses,
  credential assignments, DSSAT/WL project directories, serialized
  checkpoints, or restricted agricultural workbooks.
- File types associated with DSSAT executables and raw DSSAT projects are
  rejected by `scripts/audit_release.py`.
- Model and command identifiers are in English. The exact source snapshot keeps
  its original explanatory comments so its formal hash remains auditable.

## Verification performed

The following checks passed in the existing Linux deep-learning environment:

1. static release audit and exact V16 SHA256 check;
2. byte-code compilation for `src/`, `scripts/`, and `tests/`;
3. finite forward outputs and expected trajectory shapes;
4. daily water-balance closure below `1e-5` mm;
5. autograd irrigation sensitivity consistent with central differences;
6. one-epoch end-to-end training on the synthetic cache, with a saved
   checkpoint and zero reported water-closure RMSE;
7. regeneration of the irrigation-response figure from packaged inputs;
8. regeneration of the nine-model three-seed summary table from packaged JSON.

The one-epoch synthetic run is a software smoke test. Its accuracy statistics
are not scientific results and are not used in the paper.

## Scientific-result boundary

The archived three-seed JSON files are the formal results produced with the
cleaned 51,282-record, 518-environment-group cache. Full training was not rerun
as part of the public-package audit because that cache is intentionally not
redistributed. Users with licensed source data can reconstruct the cache and
run the commands documented in `README.md`.
