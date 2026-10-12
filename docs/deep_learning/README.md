# 深度学习方法

这里保存两类深度方法的结构、输入适配、训练协议和结果：

- 任务专用深度模型：EEGNet、TSception、RGNN、EEG-Conformer；
- EEG 基础与长序列模型：LaBraM、EEGMamba、CodeBrain、Uni-NTFM。

这些方法不是 MI3 专属。现有工作已经覆盖 MI 三分类、MI 左右手二分类和 P300 三个任务。

## 跨任务状态

| 任务 | 已运行模型 | 当前证据范围 |
|---|---|---|
| MI3 | EEGNet、TSception、RGNN、EEG-Conformer、LaBraM、EEGMamba、CodeBrain | fold 覆盖不齐；详见结果表 |
| MI2 | 同上 7 个模型 | fold 0；测试集 45 人、9,462 trials |
| P300 | 同上 7 个模型 | 102 sampled LOGO folds，共 714 runs |
| Uni-NTFM | 结构与协议路线 | 不把缺少可接受下游权重和分类头契约的运行写成正式结果 |

## 三个任务的输入

| 任务 | 公共输入 | 标签 | 评测边界 |
|---|---|---|---|
| MI3 | `[B,32,750] @ 250 Hz`、V | left_upper / right_upper / non_upper | 受试者独立 fold；validation 选参 |
| MI2 | 同一 MI release 的 left/right 派生视图 | Left / Right | 当前正式证据为 fold 0 |
| P300 | `[B,16,121] @ 200 Hz`、V | NonTarget / Target | 102 个冻结 sampled LOGO folds；固定 8 epochs |

模型 adapter 只把冻结数据转换成目标模型需要的单位、采样率、patch、图特征或标准化形式，不得
改变标签、样本归属和 split。所有需要学习的统计只从当前训练部分得到。

## 文档与代码

| 内容 | 位置 |
|---|---|
| 模型结构与分类路径 | [MODELS.md](MODELS.md) |
| 每个模型的实现身份与边界 | [model_cards/](model_cards/README.md) |
| 受试者划分、选参与指标 | [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md) |
| 三个任务的现有结果 | [RESULTS.md](RESULTS.md) |
| 结构审计证据矩阵 | [audits/EVIDENCE_MATRIX.md](audits/EVIDENCE_MATRIX.md) |
| AutoDL 运行说明 | [AUTODL.md](AUTODL.md) |

当前公开 `benchmark/` 工作区重点保留 MI3 的可运行适配器、运行器、结构测试和逐运行证据；MI2 与
P300 的跨任务结果摘要来自已完成的独立运行包，并在结果文档中按原协议标明范围。

返回[项目总览](../../README.md)，或查看同样覆盖三个任务的
[黎曼几何方法](../riemannian/README.md)。
