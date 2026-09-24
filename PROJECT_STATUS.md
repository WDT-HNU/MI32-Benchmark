# Project status

This is the short version of what is already in the repository.

## Data

- MI32 common-32 v4 metadata, split assignments, channel masks, and SHA-256 manifest.
- Dataset card, loader, 32-electrode RGNN adjacency matrix, and full-release verifier.
- Fetch and rebuild entry points for the 8.7 GB signal package.

## Models

All eight selected models have a runner, an explicit input adapter, a model card, and a structure
test: EEGNet, TSception, RGNN, EEG-Conformer, LaBraM, EEGMamba, CodeBrain, and Uni-NTFM. Upstream
commits are fixed in `configs/models.json`; the three pretrained models also record checkpoint
hashes.

## Runs on record

| Model | Folds in the current result directory |
|---|---|
| EEGNet, TSception, RGNN, EEG-Conformer | 0 |
| LaBraM | 0, 2 |
| EEGMamba, CodeBrain | 0, 2, 3 |
| Uni-NTFM | separate protocol-benchmark track |

The matching CSV files and run manifests live in
`results/mi32-common32-v4/current-evidence-20260911/`. Earlier fold-0 files remain under the
historical directory so old runs can still be traced without entering the current comparison.

## Repository layout

Git holds code, metadata, manifests, tests, and compact result files. EEG signals, upstream source
trees, and pretrained weights are fetched separately and checked against the recorded identities.
