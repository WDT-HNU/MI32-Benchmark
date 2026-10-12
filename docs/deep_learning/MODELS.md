# 模型

每个运行器都从相同的 `[B, 32, 750]`、250 Hz、伏特单位张量开始。各模型旁的适配器完成
模型特定处理；任何需要学习的校准量都只用训练受试者拟合。

| 模型 | 本仓库实现 | 适配器输出 | 分类路径 |
|---|---|---|---|
| EEGNet | EEGNet-8,2；64 点时间卷积核；覆盖全部通道的 depthwise 空间卷积 | 固定缩放和仅训练集校准 | 单个三分类线性层 |
| TSception | V2 时间分支、非对称空间分支和融合模块 | 官方 26 通道左右配对顺序 | V2 分类器 |
| RGNN | 带符号图、二阶 SGC、求和池化 | `[B,32,5]` 五频带对数功率 | RGNN 主干分类头 |
| EEG-Conformer | 官方 patch embedding 和六层编码器 | 仅训练集全局标准化 | 展平 MLP |
| LaBraM | 与 Braindecode 1.7.0 等价的主干和官方权重 | µV、250→200 Hz、`[B,32,3,200]` | 非 CLS token 均值后接 `Linear(200,3)` |
| EEGMamba | 官方 12 层 Mamba2 主干和权重 | V→µV/100、200 Hz、3 个 patch | 全 patch 分类头 |
| CodeBrain | 官方 EEGSSM 主干和权重 | V→µV/100、200 Hz、3 个 patch | 展平全部 patch 的 MLP |
| Uni-NTFM | 公开主干加本项目监督封装 | 确定性的五脑区表示 | 协议评测分类头 |

适配器提供 `none`、`naive` 和 `official` 三种模式用于消融；benchmark 正式运行使用
`official`。

## 固定实现身份

每个模型的上游 URL 和准确提交记录在 `configs/models.json`。LaBraM、EEGMamba 和 CodeBrain
还记录权重 URL、版本、大小和 SHA-256。运行清单另外保存数据清单、划分受试者、命令行参数、
适配器设置和实际执行源码。

如果上游源码或必要权重缺失，运行器会停止，不会换成相似实现。各模型的详细定义和已知不等价
替代见[模型卡片](model_cards/README.md)。
