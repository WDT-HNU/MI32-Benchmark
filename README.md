# MI32 Benchmark

把 8 个 EEG 模型放到同一套 32 通道运动想象数据上，按同一套规矩比较。

这个项目最关心的不是“模型跑通了没有”，而是结果还能不能顺着数据版本、受试者划分、
上游代码、checkpoint 和实际运行参数一路查回去。数据整理、模型适配、训练入口和现有结果
都放在这里。

当前版本是 `0.1.0-alpha`。

## 第一次来，先看这里

| 想做什么 | 从哪里开始 |
|---|---|
| 看数据怎么整理 | [数据集卡](datasets/mi32/DATASET_CARD.md) |
| 看八个模型到底怎么接入 | [模型总览](docs/MODELS.md) |
| 重跑实验 | [复现指南](REPRODUCIBILITY.md) |
| 在 AutoDL 上训练 | [AutoDL 指南](docs/AUTODL.md) |
| 看已有分数 | [结果页](docs/RESULTS.md) |
| 核对实验规矩 | [Benchmark 协议](docs/BENCHMARK_PROTOCOL.md) |

## 数据长什么样

`MI32 common32 v4` 来自 8 个源数据集，共 230 名受试者、97,608 个 trial。

- 每个 trial：`float32 [32, 750]`，250 Hz，3 秒，单位为 Volt；
- 标签：`0=left_upper`、`1=right_upper`、`2=non_upper`；
- 类别数：`24,402 / 24,402 / 48,804`，即 `1:1:2`；
- 缺失的目标电极用 MNE `standard_1005` 坐标上的 Perrin 球面样条插值；
- `measured_mask` 记录哪些通道来自原始测量，哪些来自插值。

完整信号约 8.7 GB，不直接塞进 Git。仓库里保留元数据、划分、校验脚本和 SHA-256
清单。数据文件放好后运行：

```bash
python scripts/verify_dataset.py datasets/mi32/full --full
```

## 目前接了哪些模型

| 模型 | 这里运行的版本 |
|---|---|
| EEGNet | EEGNet-8,2，三分类监督训练 |
| TSception | V2 时间/空间分支，按上游算法选取左右配对通道 |
| RGNN | RGNN backbone，五频带节点特征 |
| EEG-Conformer | 官方 patch embedding 和六层 encoder |
| LaBraM | 官方权重，非 CLS token 平均池化 |
| EEGMamba | 官方 12 层 Mamba2 backbone，直接使用 32 通道 |
| CodeBrain | 官方 EEGSSM backbone，展开全部 patch 后分类 |
| Uni-NTFM | 官方 backbone 加本项目的监督 head，单独列为 protocol benchmark |

每个模型都有一张[模型卡](docs/model_cards/README.md)，里面写明上游 commit、checkpoint
哈希、输入变换、分类头和对应测试。这里的模型名不是模糊标签；表里写的实现才是本项目实际
比较的对象。

## 跑起来

推荐环境是 Linux、Python 3.12 和 CUDA 12.8。基础模型需要真实 CUDA 依赖和 checkpoint；
CPU 更适合先跑结构测试。

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0
pip install -r requirements/core.txt
pip install -r requirements/foundation.txt

python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
python scripts/benchmark.py doctor --data datasets/mi32/full
```

先用 EEGNet 做一次预检：

```bash
python scripts/benchmark.py run \
  --model eegnet \
  --data datasets/mi32/full \
  --output outputs/eegnet_preflight \
  --preflight
```

只想检查正式命令、不想马上占 GPU：

```bash
python scripts/benchmark.py run \
  --model eegmamba \
  --data datasets/mi32/full \
  --output outputs/eegmamba_fold0 \
  --fold 0 \
  --final-test \
  --dry-run
```

## 实验规矩

这里有几条不能临时改：

- 受试者级划分；小数据集使用 LOSO，其余使用 5-fold GroupKFold；
- 学习率只看 validation macro-F1；
- 配置冻结后才运行 test；
- 主指标为 macro-F1 和 balanced accuracy；
- 看过测试结果后，不再回头改筛选、结构或超参数。

模型适配器里需要学习的统计量也只从训练受试者估计。更完整的定义在
[Benchmark 协议](docs/BENCHMARK_PROTOCOL.md)。

## 已有结果

当前结果放在
[`results/mi32-common32-v4/current-evidence-20260911`](results/mi32-common32-v4/current-evidence-20260911)。
每个结果 CSV 都配有 run manifest。较早的 fold-0 运行另存为历史审计记录，不与当前表混用。

目前的 fold 覆盖并不相同：四个传统模型完成 fold 0，LaBraM 完成 fold 0/2，EEGMamba
和 CodeBrain 完成 fold 0/2/3。比较模型时请只使用共同完成的 fold。具体数值见
[结果页](docs/RESULTS.md)。

## 目录

```text
MI32-Benchmark/
├── datasets/mi32/          数据集卡、元数据和数据校验入口
├── model_adapters/         八个模型的输入适配器
├── tests/                  结构、输入边界和数据契约测试
├── configs/                模型来源与默认配置
├── scripts/                下载、检查、运行和打包脚本
├── results/                指标 CSV、run manifest 和历史记录
├── docs/                   协议、模型卡、结果说明和 AutoDL 指南
├── mi3_eegnet.py           EEGNet / TSception / RGNN / EEG-Conformer
├── mi3_foundation.py       LaBraM / EEGMamba / CodeBrain
└── uni_mi3_supervised.py   Uni-NTFM 的 protocol benchmark runner
```

## 许可与引用

本项目原创代码使用 Apache-2.0。源 EEG 数据、上游模型和预训练权重仍受各自条款约束；
使用时请同时引用原论文。第三方来源和固定版本列在 [THIRD_PARTY.md](THIRD_PARTY.md)。

如果这个仓库进入论文或公开报告，请使用 [CITATION.cff](CITATION.cff) 中的项目信息，
并补充所用数据集与模型的原始引用。

---

**English:** MI32 Benchmark compares eight EEG models on one common 32-channel motor-imagery
dataset, with subject-level splits, explicit input adapters, training scripts, and traceable run
records.
