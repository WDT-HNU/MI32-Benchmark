# MI32 Benchmark

统一 32 通道运动想象 EEG 的数据、深度学习模型与黎曼几何方法整理。

当前仓库汇总了两条已经完成并交付的工作线：一条是 MI32 三分类数据上的 8 种深度模型评测，
另一条是二分类 MI、05 三分类 MI 和 P300 上的黎曼几何方法整理。两条工作线共享“冻结数据与
划分、再做方法专用适配”的原则，但使用不同的标签、受试者集合和评测协议。

## 从这棵树开始

```text
MI32 Benchmark
├── 1. 数据
│   ├── MI32 三分类：8 个来源 / 230 人 / 97,608 trials
│   └── Binary MI：18 个来源 / 389 人 / 59,216 trials
├── 2. 方法
│   ├── 深度学习：8 种模型
│   └── 黎曼几何：11 种方法
├── 3. 实验规则
│   ├── 深度学习：train / validation / test 受试者独立
│   └── 黎曼几何：source-only / target adaptation / target calibration
└── 4. 结果与证据
    ├── 深度学习：CSV、run manifest、模型卡片、结构测试
    └── 黎曼几何：任务状态、fold coverage、方法与输入契约摘要
```

| 想先看什么 | 入口 |
|---|---|
| 两套数据的区别 | [数据索引](benchmark/datasets/README.md) |
| 8 种深度模型 | [深度学习分支](docs/deep_learning/README.md) |
| 11 种黎曼方法 | [黎曼几何分支](docs/riemannian/README.md) |
| 仓库所有文档 | [文档导航](docs/README.md) |
| 当前收录范围 | [项目状态](docs/project/PROJECT_STATUS.md) |

## 两条工作线不能混在一起

| 项目 | 深度学习分支 | 黎曼几何分支的 Binary MI |
|---|---|---|
| 输入形状 | `[32,750] @ 250 Hz` | `[32,750] @ 250 Hz` |
| 任务 | 三分类 | 左手/右手二分类 |
| 数据规模 | 8 datasets / 230 subjects / 97,608 trials | 18 datasets / 389 subjects / 59,216 trials |
| 标签 | `left_upper` / `right_upper` / `non_upper` | `Left` / `Right` |
| 主要方法 | CNN、GNN、Transformer、SSM、基础模型 | SPD/AIRM、切空间、对齐与迁移 |
| 评测核心 | 验证集选参，测试集最终评估 | 冻结 fold，并披露目标域信息使用方式 |

相同的 32 通道形状只说明模型接口接近，不代表样本、标签和结果可以合并。所有结果都必须回到自己
所属的数据契约和 split 解释。

## 深度学习分支

当前整理了 EEGNet、TSception、RGNN、EEG-Conformer、LaBraM、EEGMamba、CodeBrain 和
Uni-NTFM。仓库中包含运行器、输入适配器、模型卡片、结构测试以及现有结果 CSV 与运行清单。

缺失电极采用 Perrin 球面样条法插值，实测与插值通道由 `measured_mask` 区分。测试集不参与
学习率等超参数选择；需要学习的均值、方差或校准量只从训练受试者得到。完整说明见
[深度学习分支](docs/deep_learning/README.md)。

快速检查：

```bash
cd benchmark
python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

运行示例：

```bash
cd benchmark
python scripts/benchmark.py run \
  --model eegnet \
  --data datasets/mi32/full \
  --output outputs/eegnet_preflight \
  --preflight
```

## 黎曼几何分支

当前整理了 MDRM、TSLDA、KernelSVM、FGMDRM、RA、SPDNet、MFACC、RPA、TSA、
Manifold-SimCLR 和 MEKT-R。方法摘要分别记录 covariance 构造、SPD 数值边界、分类器、目标域
信息使用方式、适用任务和有效 fold 数。

```text
冻结 trial
└── method-specific adapter
    └── official input contract
        └── frozen method core
            └── evaluation + coverage reporting
```

Binary MI 的冻结数据说明见[数据契约](benchmark/datasets/binary_mi/DATASET_CARD.md)，完整方法树见
[黎曼几何方法汇总](docs/riemannian/METHODS.md)。

## 根目录现在怎样分工

```text
MI32-Benchmark/
├── benchmark/        可运行工作区：数据、模型、脚本、测试和结果
├── docs/             深度学习、黎曼几何和项目维护文档
└── .github/          GitHub 持续集成和协作规范
```

首次进入只需要在“运行内容”和“说明文档”之间选择。进入 [`benchmark/`](benchmark/README.md)
以后，再按数据、模型、实验和结果展开；GitHub 协作文件统一收在 `.github/`。

## 复现与引用

MI32 深度学习分支的环境、数据获取和完整运行步骤见[复现说明](docs/project/REPRODUCIBILITY.md)；
AutoDL 路径和关机注意事项见[运行说明](docs/deep_learning/AUTODL.md)。EEG 信号、上游源码和预训练
权重不写入 Git 历史，按清单单独获取并核验身份。

原创代码采用 Apache-2.0 许可证。数据、上游模型和权重遵循各自授权，详见
[第三方来源](docs/project/THIRD_PARTY.md)。机器可读引用信息见 [CITATION.cff](CITATION.cff)。
