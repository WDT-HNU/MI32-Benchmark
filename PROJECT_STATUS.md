# 项目状态

这里只汇总仓库中已经存在的内容。

## 数据

- MI32 common-32 v4 元数据、划分结果、通道掩码和 SHA-256 清单。
- 数据集卡片、加载器、32 电极 RGNN 邻接矩阵和完整版本检查器。
- 约 8.7 GB 信号包的拉取与重建入口。

## 模型

8 种模型均已提供运行器、明确的输入适配器、模型卡片和结构测试：EEGNet、TSception、RGNN、
EEG-Conformer、LaBraM、EEGMamba、CodeBrain 和 Uni-NTFM。上游提交固定在
`configs/models.json`；3 种预训练模型还记录了权重哈希。

## 已收录的运行

| 模型 | 当前结果目录包含的折 |
|---|---|
| EEGNet, TSception, RGNN, EEG-Conformer | 0 |
| LaBraM | 0, 2 |
| EEGMamba, CodeBrain | 0, 2, 3 |
| Uni-NTFM | 单独的协议评测路线 |

对应的 CSV 和运行清单位于 `results/mi32-common32-v4/current-evidence-20260911/`。早期 fold-0
文件保留在历史目录中，便于追溯，但不进入当前比较。

## 仓库边界

Git 保存代码、元数据、清单、测试和体积较小的结果文件。EEG 信号、上游源码树和预训练权重
单独获取，并按仓库记录的身份信息进行核验。
