# MI32 data directory

- `DATASET_CARD.md` documents the scientific and legal boundary.
- `metadata/` is the exact non-signal metadata snapshot of release 4.0-common32.
- `full/` is ignored by Git and is the default location for the 8.7 GB sealed dataset.

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
```

Do not commit subject NPZ files, credentials, temporary AutoDL URLs, or source archives.
