# Data availability

## Repository contents

The repository includes the complete data contract and non-signal release metadata:

- version, shape, channel order, label ontology, and preprocessing contract;
- subject provenance and subject-level split assignments;
- interpolation coverage, quality-review flags, and independent release audit;
- the SHA-256 manifest for all 230 subject NPZ files;
- the 32-channel RGNN adjacency matrix and loader.

## Signal artifact management

The 230 signal archives occupy approximately 8.7 GB and are managed outside Git history. This
keeps the repository lightweight while allowing each source dataset's access and distribution terms
to remain attached to the signal artifact. Every authorized copy is verified against the published
SHA-256 manifest.

## Supported access modes

1. **Verified release archive** — set `MI32_DATA_URL` to an approved institutional, Zenodo, or
   Hugging Face dataset artifact and run `scripts/fetch_dataset.py`.
2. **Bring your own sealed copy** — place `MI32_COMMON32_V4` locally and run
   `scripts/verify_dataset.py PATH --full`.
3. **Rebuild from the curated precursor** — use `dataset_tools/build_mi3_32_interp.py` to reproduce
   the common-32 transformation.

These modes share the same downstream data contract, subject splits, manifests, and verification
commands. The source review table records the access and distribution terms used when attaching a
public signal endpoint.
