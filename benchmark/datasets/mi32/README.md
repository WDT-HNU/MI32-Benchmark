# MI32 dataset directory

- `DATASET_CARD.md` defines the scientific and legal boundaries.
- `metadata/` is the signal-free metadata snapshot from release `4.0-common32`.
- `full/` is Git-ignored and is the default location for the approximately 8.7 GB sealed dataset.

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
```

Do not commit subject NPZ files, credentials, temporary AutoDL URLs, or source archives.
