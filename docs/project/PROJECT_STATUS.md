# 项目状态

这里按三个任务汇总仓库已经归档的内容。

## 任务与方法矩阵

| 任务 | 任务专用深度模型 | EEG 基础与长序列模型 | 黎曼几何方法 |
|---|---|---|---|
| MI 三分类 | 4 个模型已有当前结果 | LaBraM、EEGMamba、CodeBrain 已有结果；Uni-NTFM 为协议路线 | 05 代表协议已归档 |
| MI 二分类 | 4 个模型完成 fold 0 | 3 个模型完成 fold 0 | Binary MI 代表 36-fold 状态已归档 |
| P300 | 4 个模型完成 102 sampled LOGO folds | 3 个模型完成同一 102 folds | reduced 85-fold 状态已归档 |

## 数据

- MI32 common-32 v4：8 个来源、230 名受试者、97,608 个三分类 trials；
- MI2 派生视图：同一批受试者，保留左右手后为 48,804 trials；
- P300：15 个来源、158,667 trials，深度模型协议使用 102 sampled LOGO folds；
- Binary MI 黎曼数据契约：18 个来源、389 名受试者、59,216 trials，完整 84 folds、代表 36 folds。

## 深度方法

EEGNet、TSception、RGNN、EEG-Conformer、LaBraM、EEGMamba、CodeBrain 和 Uni-NTFM 均有明确的
输入适配、模型说明或结构测试。现有正式结果覆盖前 7 个模型；Uni-NTFM 保留独立协议路线。

跨任务结果见[深度模型结果](../deep_learning/RESULTS.md)。MI3 的逐运行 CSV 和 manifest 位于
`benchmark/results/mi32-common32-v4/current-evidence-20260911/`。

## 黎曼几何方法

已归档 MDRM、TSLDA、KernelSVM、FGMDRM、RA、SPDNet、MFACC、RPA、TSA、
Manifold-SimCLR 和 MEKT-R 的输入、信息条件、适用任务和 coverage。状态覆盖 MI2、MI3、P300，
并保留 partial、binary-only 和 implementation-blocked 等真实边界。

## 归档原则

Git 保存代码、元数据、清单、测试和体积较小的结果。EEG 信号、上游源码树和预训练权重单独获取，
并按记录的哈希和提交身份核验。任何结果都要带任务、数据 release、split、adapter 和 coverage，
不能只用模型名称代替运行证据。
