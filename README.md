# MI32 Benchmark

一个面向运动想象 EEG 的、可审计且尽可能可复现的统一 benchmark。项目把数据契约、
八个模型的输入适配、训练/验证/测试协议、结构门禁、上游代码与 checkpoint 身份、运行清单
和结果证据放在同一仓库中。

> 当前发布状态：`0.1.0-alpha`。代码与已封存的部分 fold 结果可公开审查；完整 8.7 GB 数据包
> 不进入 Git 历史，公共下载端点与源数据再分发许可仍需在正式公开发布前确认。

## 这个仓库回答什么问题

它不是“八个名字相似的网络脚本合集”。每次正式实验都要回答：

1. 使用的是哪个数据版本、哪些受试者和哪个划分？
2. 上游模型源码、checkpoint 和本项目适配器分别是什么身份？
3. 超参数是否只由验证集选择，测试集是否保持封存？
4. 输入单位、采样率、通道顺序、插值与 patch 形状如何变化？
5. 结果 CSV 能否追溯到运行配置、源码快照与 SHA-256？

## 数据集

`MI32 common32 v4` 汇总 8 个来源数据集，共 230 名受试者、97,608 个三分类 trial：

- 输入：`float32 [N, 32, 750]`，250 Hz，3 秒，单位 Volt；
- 标签：`0=left_upper`、`1=right_upper`、`2=non_upper`；
- 每名受试者保持 `1:1:2`，全局为 `24,402:24,402:48,804`；
- 缺失目标通道使用 MNE `standard_1005` 坐标上的 Perrin 球面样条插值；
- `measured_mask` 区分直接测量与插值通道，插值信号不会再次补零；
- 数据发布由 `SHA256SUMS` 封存，清单自身 SHA-256 为
  `1017a3dd50d11ff9d92d987a40996fc2fd61ace99d465213bdb0316ff52aad06`。

完整说明见 [数据集卡](datasets/mi32/DATASET_CARD.md) 和
[数据发布策略](DATA_AVAILABILITY.md)。

## 模型

| 模型 | 轨道 | 当前身份 |
|---|---|---|
| EEGNet | 传统深度学习 | official-aligned EEGNet-8,2 |
| TSception | 传统深度学习 | 官方 V2 拓扑与官方通道配对算法 |
| RGNN | 传统深度学习 | RGNN backbone，不含 NodeDAT/EmotionDL |
| EEG-Conformer | 传统深度学习 | 官方图的 MI32 任务适配 |
| LaBraM | 基础模型 | 官方权重的 sealed Braindecode 1.7.0 等价适配 |
| EEGMamba | 基础模型 | 冻结官方提交与 checkpoint，`V→µV/100`、直接 32 通道、3 patch |
| CodeBrain | 基础模型 | 冻结官方提交与 checkpoint，`V→µV/100`、flatten-all-patches head |
| Uni-NTFM | 协议适配 | 官方 backbone + 本项目监督 head；不宣称论文复现 |

详细边界见 [模型与适配器](docs/MODELS.md) 以及 [八份逐模型复现卡](docs/model_cards/README.md)。

## 五分钟开始

环境建议为 Linux、Python 3.12、NVIDIA CUDA 12.8。CPU 可完成轻量结构测试，基础模型的
最终验收必须使用真实 CUDA 依赖与真实 checkpoint。

```bash
git clone https://github.com/OWNER/MI32-Benchmark.git
cd MI32-Benchmark
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchaudio==2.8.0
pip install -r requirements/core.txt
pip install -r requirements/foundation.txt

python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
python scripts/benchmark.py doctor --data datasets/mi32/full
pytest -q tests/test_eegnet_structure.py tests/test_tsception_structure.py \
  tests/test_eegconformer_structure.py model_adapters/tests/test_adapters_contract.py
```

单模型预检：

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_preflight --preflight
```

生成正式命令但暂不运行：

```bash
python scripts/benchmark.py run --model eegmamba --data datasets/mi32/full \
  --output outputs/eegmamba_fold0 --fold 0 --final-test --dry-run
```

完整流程见 [复现指南](REPRODUCIBILITY.md) 和 [AutoDL 指南](docs/AUTODL.md)。

## Benchmark 协议

- 小数据集（不超过 10 名受试者）采用 LOSO；其余采用 subject-level 5-fold GroupKFold。
- fold `f` 为测试集，相邻预定义 fold 为验证集，其余为训练集。
- 模型适配器中的可学习统计量只允许从训练受试者拟合。
- 学习率由 validation macro-F1 选择；只有冻结配置后才能打开 test。
- 主指标为 macro-F1 与 balanced accuracy，同时报告 accuracy、逐类 recall、二阶段上肢指标和混淆矩阵。
- 不根据测试表现改数据筛选、模型结构或超参数。

详见 [benchmark 协议](docs/BENCHMARK_PROTOCOL.md)。

## 当前结果边界

仓库包含与当前单位合同一致的部分 fold 证据快照，位于
[`results/mi32-common32-v4/current-partial-20260911`](results/mi32-common32-v4/current-partial-20260911)。
各模型 fold 覆盖并不相等，因此不能冒充完整多 fold 均值或最终排名。旧 fold-0 包仅作为
历史审计证据保留。Uni-NTFM 没有混入正式复现表。查看
[结果说明](docs/RESULTS.md) 与 [证据矩阵](docs/audits/EVIDENCE_MATRIX.md)。

## 仓库结构

```text
MI32-Benchmark/
├── datasets/mi32/          # 数据集卡、公开元数据和完整数据占位目录
├── model_adapters/         # 八个模型的显式输入适配器
├── tests/                  # 结构、边界和数据契约测试
├── configs/                # 模型身份和 benchmark 默认配置
├── scripts/                # 下载、校验、预检、运行和封存工具
├── results/                # 结果表、run manifest 与执行源码快照
├── docs/                   # 协议、模型卡、AutoDL 和审计说明
├── mi3_eegnet.py           # EEGNet/TSception/RGNN/EEG-Conformer 正式 runner
├── mi3_foundation.py       # LaBraM/EEGMamba/CodeBrain 正式 runner
└── uni_mi3_supervised.py   # Uni-NTFM 协议适配 runner
```

## 许可与引用

本项目原创代码使用 Apache-2.0。源 EEG 数据、上游模型代码和 checkpoint 不因本仓库而
改变许可；请同时遵守各自条款并引用原论文。Uni-NTFM 上游仓库当前没有显式 LICENSE，
因此不在本仓库重新分发其源码。详见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## English summary

MI32 Benchmark is an auditable common-32-channel motor-imagery EEG benchmark. It ships the
dataset contract and public metadata, eight explicit model-input adapters, leakage-resistant
subject splits, pinned upstream/checkpoint identities, structure/CUDA gates, run manifests,
and a contract-matched partial result snapshot. Large or restricted data, third-party
repositories, and checkpoints are fetched separately and verified by SHA-256.
