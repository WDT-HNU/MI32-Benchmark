# Public release checklist

## Ownership and legal

- [ ] Replace `OWNER` placeholders in README and CITATION.cff.
- [ ] Confirm the project copyright holder and Apache-2.0 choice.
- [ ] Complete every row of `datasets/mi32/source_redistribution_review.csv`.
- [ ] Verify third-party licenses at the pinned commits.
- [ ] Confirm checkpoint redistribution versus download-only handling.
- [ ] Add complete model and dataset citations.

## Reproducibility

- [ ] Run `python scripts/verify_repository.py --release` with no failures.
- [ ] Recreate the environment on a clean Linux/CUDA host.
- [ ] Fetch all upstream repositories and checkpoints from documented public URLs.
- [ ] Verify the complete dataset against its SHA-256 manifest.
- [ ] Pass CPU structure tests and the real CUDA/data gate.
- [ ] Produce at least one clean smoke run from a fresh clone.
- [ ] Ensure all formal results contain run config, manifest, adapter manifest, source snapshot,
      environment identity, dataset identity, and output hashes.

## Scientific claims

- [ ] Publish the exact fold coverage beside every result summary.
- [ ] Keep Uni-NTFM protocol results outside the formal-reproduction table.
- [ ] Run all predeclared folds before claiming a final model ranking.
- [ ] Report source-dataset sensitivity and uncertainty.
- [ ] Document any artifact exclusion before looking at test outcomes.

## GitHub

- [ ] Enable branch protection and required CI.
- [ ] Enable Dependabot and dependency review.
- [ ] Enable secret scanning/push protection where available.
- [ ] Create a signed semantic tag and GitHub release.
- [ ] Publish large data/checkpoints outside Git history and attach SHA-256 manifests.
- [ ] Archive a citable release (for example Zenodo) only after the licensing review passes.
