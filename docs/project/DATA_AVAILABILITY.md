# Accessing MI32 data

The signal package is approximately 8.7 GB and is not stored in Git. The repository provides:

- shapes, channel order, labels, and preprocessing contract;
- subject provenance and subject-level splits;
- interpolation coverage and quality flags;
- SHA-256 values for 230 subject files;
- a loader and the 32-channel RGNN adjacency matrix.

Commands and relative paths below use `benchmark/` as the working directory.

## Three preparation routes

### 1. Download an approved package

Point `MI32_DATA_URL` to an approved institutional, Zenodo, or Hugging Face artifact:

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
```

### 2. Use an existing authorized copy

```bash
python scripts/verify_dataset.py /path/to/MI32_COMMON32_V4 --full
```

### 3. Rebuild common-32

Starting from the curated precursor data, run `dataset_tools/build_mi3_32_interp.py`.

Whichever route is used, finish with the full verifier. It checks the data contract used by the runners, the split file, and hashes. Source access and redistribution notes are tracked in `datasets/mi32/source_redistribution_review.csv`.
