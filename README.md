# MI32 Benchmark

这个仓库整理三个 EEG 任务的数据、方法适配、评测协议和结果证据：运动想象三分类、运动想象
左手/右手二分类，以及 P300 二分类。

项目不是按“深度学习分支”和“黎曼几何分支”拆成两项互不相干的工作。正确的阅读方式是先选任务，
再看同一任务下的三类方法。

```text
MI32 Benchmark
├── 1. 任务
│   ├── MI3：运动想象三分类
│   ├── MI2：左手/右手二分类
│   └── P300：Target/NonTarget 二分类
├── 2. 方法
│   ├── 任务专用深度模型：EEGNet、TSception、RGNN、EEG-Conformer
│   ├── EEG 基础与长序列模型：LaBraM、EEGMamba、CodeBrain、Uni-NTFM
│   └── 黎曼几何方法：MDRM、TSLDA、KernelSVM 等 11 种方法
└── 3. 每个任务内完成适配、训练、评测和证据保存
```

## 实验矩阵

| 任务 | 任务专用深度模型 | EEG 基础与长序列模型 | 黎曼几何方法 |
|---|---|---|---|
| MI 三分类 | 已运行；当前公开证据的 fold 覆盖不完全相同 | LaBraM、EEGMamba、CodeBrain 已有结果；Uni-NTFM 保留协议路线 | 已完成 05 三分类代表协议，按方法报告 16-fold coverage |
| MI 二分类 | 7 个深度模型已有 fold 0 测试结果 | 同一 fold 0 内已有 LaBraM、EEGMamba、CodeBrain 结果 | 已完成 Binary MI 代表性 36-fold 协议，保留真实失败和方法边界 |
| P300 | 7 个深度模型完成 102 个 sampled LOGO folds | LaBraM、EEGMamba、CodeBrain 包含在同一 102-fold 协议中 | 已完成 reduced frozen LOGO 协议，按方法报告 85-fold coverage |

这里的“已运行”表示相应任务和方法组合已经有运行产物或正式状态记录，不表示所有格子都采用同一
数据 release、同一折数或同一指标。严格比较必须继续核对每个格子的冻结数据身份和 split。

## 从哪里开始

| 想先看什么 | 入口 |
|---|---|
| 第一次了解整个项目 | [项目导读](docs/TECHNICAL_GUIDE.md) |
| 三个任务和各自数据 | [数据索引](benchmark/datasets/README.md) |
| 深度模型与基础模型 | [深度学习方法](docs/deep_learning/README.md) |
| 黎曼几何方法 | [黎曼几何方法](docs/riemannian/README.md) |
| 当前跨任务结果 | [深度模型结果](docs/deep_learning/RESULTS.md) |
| 仓库所有文档 | [文档导航](docs/README.md) |
| 当前收录范围 | [项目状态](docs/project/PROJECT_STATUS.md) |

## 为什么按任务组织

MI3、MI2 和 P300 的标签、事件语义、类别平衡和评测指标不同。P300 是 time-locked ERP 任务，
不能机械套用 MI 的时间窗或协方差表示；MI2 和 MI3 虽可共享一部分信号处理，却仍有不同标签空间。

因此每次实验都沿同一条证据链记录：

```text
任务与冻结数据
└── 冻结 split
    └── 方法专用 adapter
        └── 模型或方法核心
            └── prediction + metrics + manifest
```

“深度学习”“基础模型”“黎曼几何”是方法类别，不是数据集类别。一个方法类别可以在三个任务上分别
适配和评测。

## 仓库结构

```text
MI32-Benchmark/
├── benchmark/        可运行工作区：数据契约、适配器、运行器、测试和结果
├── docs/             任务说明、方法说明、结果边界和复现文档
└── .github/          持续集成与协作规范
```

快速检查：

```bash
cd benchmark
python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

EEG 信号、上游源码和预训练权重不写入 Git 历史，按清单单独获取并核验身份。原创代码采用
Apache-2.0 许可证；数据、上游模型和权重遵循各自授权，详见
[第三方来源](docs/project/THIRD_PARTY.md)。
