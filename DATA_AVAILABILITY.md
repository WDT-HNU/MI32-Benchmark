# Data availability

## What is in Git

The repository includes the complete data contract and non-signal release metadata:

- version, shape, channel order, label ontology, and preprocessing contract;
- subject provenance and subject-level split assignments;
- interpolation coverage, quality-review flags, and independent release audit;
- the SHA-256 manifest for all 230 subject NPZ files;
- the 32-channel RGNN adjacency matrix and loader.

## What is not in Git

The 230 signal archives occupy approximately 8.7 GB. They are excluded from Git history and
Git LFS because the benchmark combines multiple source datasets with independent access and
redistribution terms. A public release must not be uploaded until the maintainer records a
redistribution decision for every source dataset.

## Supported access modes

1. **Verified release archive** — set `MI32_DATA_URL` to an approved institutional, Zenodo, or
   Hugging Face dataset artifact and run `scripts/fetch_dataset.py`.
2. **Bring your own sealed copy** — copy `MI32_COMMON32_V4` locally and run
   `scripts/verify_dataset.py PATH --full`.
3. **Rebuild from the curated precursor** — use `dataset_tools/build_mi3_32_interp.py`; this
   reproduces the common-32 transformation but does not by itself reconstruct the precursor
   corpus from each raw source.

The current alpha release therefore provides exact downstream reproducibility for holders of
the sealed dataset and transparent metadata for everyone else. Raw-to-curated dataset adapters
remain a release blocker for fully public end-to-end reconstruction.
