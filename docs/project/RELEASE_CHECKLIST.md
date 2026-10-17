# Public release checklist

Paths below are relative to `benchmark/`.

## Ownership and legal review

- [ ] Confirm project copyright holders and the Apache-2.0 licence choice.
- [ ] Complete every row of `datasets/mi32/source_redistribution_review.csv`.
- [ ] Verify third-party licences at the pinned commits.
- [ ] Determine whether each checkpoint may be redistributed or only linked.
- [ ] Complete model and dataset citations.

## Reproducibility

- [ ] Run `python scripts/verify_repository.py --release` with no failures.
- [ ] Rebuild the environment on a clean Linux/CUDA host.
- [ ] Fetch every upstream repository and checkpoint from documented public locations.
- [ ] Verify the full dataset against its SHA-256 manifest.
- [ ] Pass CPU structural tests and the real CUDA/data gate.
- [ ] Complete at least one clean smoke run from a fresh clone.
- [ ] Confirm that every formal result includes run configuration, run manifest, adapter manifest, source snapshot, environment identity, dataset identity, and output hashes.

## Scientific claims

- [ ] State exact test-fold coverage next to every result summary.
- [ ] Keep Uni-NTFM protocol results outside the formal-reproduction table.
- [ ] Run every predeclared fold before publishing a final model ranking.
- [ ] Report source-dataset sensitivity and uncertainty.
- [ ] Record artifact-exclusion rules before examining test outcomes.
- [ ] Keep traditional report transcriptions distinct from prediction-derived result packages.

## GitHub

- [ ] Enable branch protection and required CI.
- [ ] Enable Dependabot and dependency review.
- [ ] Enable secret scanning and push protection where available.
- [ ] Create a signed semantic tag and GitHub release.
- [ ] Keep large data and checkpoints outside Git history with SHA-256 manifests.
- [ ] Archive a citable version, such as Zenodo, only after licence review passes.
