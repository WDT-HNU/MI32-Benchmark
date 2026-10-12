# 深度学习分支

这一分支在 MI32 common-32 v4 三分类数据上整理 8 种 EEG 模型的输入适配、训练、结构审计和
现有运行结果。

```text
MI32 common-32 v4
└── 230 subjects / 97,608 trials / 3 classes
    └── subject-level train / validation / test
        ├── 传统与任务模型：EEGNet、TSception、RGNN、EEG-Conformer
        ├── 预训练模型：LaBraM、EEGMamba、CodeBrain
        └── 独立协议模型：Uni-NTFM
            └── CSV + run manifest + structure tests
```

## 数据输入

所有运行器从 `[B, 32, 750] @ 250 Hz`、单位 V 的试次开始，标签为 `left_upper`、
`right_upper`、`non_upper`。当前数据统计为 8 个来源数据集、230 名受试者、97,608 个试次，
三类计数为 24,402 / 24,402 / 48,804。

完整数据来源、通道插值、标签映射和划分见
[MI32 数据集卡片](../../datasets/mi32/DATASET_CARD.md)。

## 方法与证据

| 内容 | 位置 |
|---|---|
| 8 种模型的统一说明 | [MODELS.md](MODELS.md) |
| 每个模型的实现身份与边界 | [model_cards/](model_cards/README.md) |
| 受试者划分、选参与指标 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) |
| 当前收录结果 | [RESULTS.md](RESULTS.md) |
| 结构审计证据矩阵 | [audits/EVIDENCE_MATRIX.md](audits/EVIDENCE_MATRIX.md) |
| AutoDL 运行说明 | [AUTODL.md](AUTODL.md) |

模型输入转换集中在 [`model_adapters/`](../../model_adapters/)，实际训练程序集中在
[`runners/`](../../runners/)，统一调度入口是 [`scripts/benchmark.py`](../../scripts/benchmark.py)。

## 运行入口

准备数据后，先检查仓库和数据身份：

```bash
python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/full --full
python scripts/benchmark.py doctor --data datasets/mi32/full
```

以 EEGNet 短预检为例：

```bash
python scripts/benchmark.py run \
  --model eegnet \
  --data datasets/mi32/full \
  --output outputs/eegnet_preflight \
  --preflight
```

测试集不参与选参；学习率等超参数只由验证集确定。上游源码、权重、数据清单、划分、随机种子和
实际执行源码均写入运行清单。

## 当前结果边界

仓库已收录 EEGNet、TSception、RGNN、EEG-Conformer 的 fold 0，LaBraM 的 fold 0/2，以及
EEGMamba、CodeBrain 的 fold 0/2/3 结果。各模型覆盖 fold 不完全相同，因此当前表只支持同折
比较，不作为最终模型排名。具体数值和清单位置见[结果说明](RESULTS.md)。

返回[项目总览](../../README.md)或查看并列的[黎曼几何分支](../riemannian/README.md)。
