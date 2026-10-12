# 文档导航

文档按“任务是主轴，方法是横向比较维度”组织。

```text
三个任务
├── MI 三分类
├── MI 二分类
└── P300
    ×
三类方法
├── 任务专用深度模型
├── EEG 基础与长序列模型
└── 黎曼几何方法
```

现有文件夹仍按方法保存详细材料，避免把模型卡片、结构测试和黎曼方法契约混在一起；这只是文件
归档方式，不代表深度模型只跑 MI3，也不代表黎曼方法只跑 MI2。

```text
docs/
├── TECHNICAL_GUIDE.md  三任务 × 三类方法的总导读
├── deep_learning/     深度模型和基础模型的结构、适配、协议与跨任务结果
├── riemannian/        黎曼方法在 MI3、MI2、P300 上的输入与状态
└── project/           获取、复现、状态、发布和第三方来源
```

| 想了解的问题 | 阅读入口 |
|---|---|
| 整个项目怎样组成 | [项目导读](TECHNICAL_GUIDE.md) |
| 三个任务使用什么数据 | [数据索引](../benchmark/datasets/README.md) |
| 深度模型在三个任务上的情况 | [深度学习方法](deep_learning/README.md) |
| 黎曼方法在三个任务上的情况 | [黎曼几何方法](riemannian/README.md) |
| 如何从头复现实验 | [复现说明](project/REPRODUCIBILITY.md) |
| 仓库目前收录了什么 | [项目状态](project/PROJECT_STATUS.md) |
