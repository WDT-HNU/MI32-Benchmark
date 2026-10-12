# 获取 MI32 数据

信号文件约 8.7 GB，因此不放进 Git。仓库中提供的是：

- 数据形状、通道顺序、标签和预处理方法；
- 受试者来源和受试者级数据划分；
- 插值覆盖情况和质量检查标记；
- 230 个受试者文件的 SHA-256；
- 数据加载器和 32 通道 RGNN 邻接矩阵。

## 三种准备数据的方式

### 1. 下载已发布的数据包

让 `MI32_DATA_URL` 指向经过批准的机构、Zenodo 或 Hugging Face 文件：

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
```

### 2. 使用已有副本

```bash
python scripts/verify_dataset.py /path/to/MI32_COMMON32_V4 --full
```

### 3. 重建 common-32 版本

从整理后的前置数据开始，运行 `dataset_tools/build_mi3_32_interp.py`。

无论采用哪种方式，最后都要运行完整检查器。它会核对运行器使用的数据约定、划分文件和哈希。
各来源数据集的访问与再分发情况记录在 `datasets/mi32/source_redistribution_review.csv`。
