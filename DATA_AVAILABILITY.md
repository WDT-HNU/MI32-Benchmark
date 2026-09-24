# Getting the MI32 data

The signal files are about 8.7 GB, so they are not stored in Git. What *is* in the repository:

- the data shape, channel order, labels, and preprocessing recipe;
- subject provenance and subject-level splits;
- interpolation coverage and QC flags;
- SHA-256 entries for all 230 subject files;
- the loader and the 32-channel RGNN adjacency matrix.

## Three ways to put the data in place

### 1. Download a published package

Point `MI32_DATA_URL` at the approved institutional, Zenodo, or Hugging Face artifact:

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
```

### 2. Use a copy you already have

```bash
python scripts/verify_dataset.py /path/to/MI32_COMMON32_V4 --full
```

### 3. Rebuild the common-32 version

Start from the curated precursor and use `dataset_tools/build_mi3_32_interp.py`.

Whichever route you take, finish with the full verifier. It checks the same data contract, split
files, and hashes used by the runners. Source-specific access and redistribution notes are tracked
in `datasets/mi32/source_redistribution_review.csv`.
