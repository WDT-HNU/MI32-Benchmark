# 黎曼几何方法

这里保存 11 种黎曼几何方法的输入表示、信息条件、任务适用范围和 coverage。它不是一个只针对
Binary MI 的独立分支；现有方法状态同时覆盖 MI 二分类、MI 三分类和 P300。

```text
每个任务的冻结数据与 split
└── method-specific adapter
    └── official input contract
        └── frozen method core
            └── evaluation + coverage reporting
```

## 跨任务状态

| 任务 | 协议 | 当前覆盖 |
|---|---|---|
| MI 二分类 | Binary MI representative 36 folds | 多数方法 formal；TSLDA partial；Manifold-SimCLR 为非正式重建 |
| MI 三分类 | 05 representative 16 folds | 多数方法 16/16；MFACC 13/16；MEKT-R 保持 binary-only |
| P300 | reduced frozen LOGO 85 folds | 多数方法 85/85；TSLDA 保留 zero-feature failures |

当前归档的方法包括 MDRM、TSLDA、KernelSVM、FGMDRM、RA、SPDNet、MFACC、RPA、TSA、
Manifold-SimCLR 和 MEKT-R。完整定义见[方法汇总](METHODS.md)。

Binary MI 的 18-source 数据契约见
[数据卡片](../../benchmark/datasets/binary_mi/DATASET_CARD.md)。它和深度模型使用的 MI2 派生视图
属于同一个二分类任务，但冻结 release 不同；两边结果不能省略数据身份后直接排名。

返回[项目总览](../../README.md)，或查看同样覆盖三个任务的
[深度学习方法](../deep_learning/README.md)。
