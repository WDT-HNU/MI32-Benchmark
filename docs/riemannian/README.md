# 黎曼几何分支

这一分支整理左手/右手二分类 MI 的数据契约和 11 种黎曼几何方法。阅读顺序只有两步：

```text
Binary MI 冻结数据
└── benchmark/datasets/binary_mi/DATASET_CARD.md
    └── 方法输入、信息边界与任务状态
        └── docs/riemannian/METHODS.md
```

当前归档内容包括：

- 18 个 component datasets、389 名受试者、59,216 个试次的二分类数据口径；
- 完整 84-fold 与代表性 36-fold 受试者独立评测口径；
- MDRM、TSLDA、KernelSVM、FGMDRM、RA、SPDNet、MFACC、RPA、TSA、Manifold-SimCLR、
  MEKT-R 的输入、训练信息和状态整理；
- Binary MI、05 三分类 MI、P300 三类任务之间不能机械共用表示的边界；
- source-only、无标签目标域适配、带标签目标域校准和自监督四类信息使用方式。

这条分支与[深度学习分支](../deep_learning/README.md)共用“先冻结数据与划分，再做方法专用适配”的
原则，但二者不是同一套标签或结果表。
