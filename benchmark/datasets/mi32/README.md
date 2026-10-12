# MI32 数据目录

- `DATASET_CARD.md` 说明科学和法律边界。
- `metadata/` 是 4.0-common32 版本中不含信号的完整元数据快照。
- `full/` 被 Git 忽略，是约 8.7 GB 封存数据集的默认位置。

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
```

不要提交受试者 NPZ 文件、凭据、临时 AutoDL URL 或来源压缩包。
