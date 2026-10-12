# 第三方来源与授权边界

本仓库不直接收录上游模型仓库或预训练权重。拉取工具会把固定身份的源码和权重下载到 Git
忽略的本地目录。

审计时能从上游源码中取得的许可证，以及 LaBraM 源码快照要求保留的 Braindecode 1.7.0
许可证和声明，均保存在 `THIRD_PARTY_LICENSES/`。缺少许可证文件（特别是已审计的 Uni-NTFM
源码和 RGNN 快照）应理解为“尚未确认再分发许可”，不能当作公有领域作品。

| 组件 | 上游地址 | 固定身份 | 已观察到的上游许可证 |
|---|---|---|---|
| EEGNet | https://github.com/vlawhern/arl-eegmodels | `4a512e503198db2010848813ead9afbf8cd54c97` | 见上游仓库 |
| TSception | https://github.com/yi-ding-cs/TSception | `9efd666b618d006e32e6da1d30dbc79b1d190604` | 自定义，见上游仓库 |
| RGNN | https://github.com/zhongpeixiang/RGNN | `685b1a2645185bd8129c04a789dbfde8ad896a59` | 见上游仓库 |
| EEG-Conformer | https://github.com/eeyhsong/EEG-Conformer | `9ae149ba62487ceae723277d13adac27837113d2` | 见上游仓库 |
| LaBraM | https://github.com/935963004/LaBraM | `c431221e6cfd23dbfa9950e0180682fb322b0548` | MIT |
| EEGMamba | https://github.com/wjq-learning/EEGMamba | `dbc83fa072744201e8897aeb9f65007b952ad323` | MIT |
| CodeBrain | https://github.com/jingyingma01/CodeBrain | `22d350caf68246d2fda4f630ef837420db3fb130` | Apache-2.0 |
| Uni-NTFM | https://github.com/Zhisheng-researcher/Uni-NTFM | `c0ce0152f94366e59b31b3fb2c108ce909bcd95c` | 未观察到明确的 LICENSE |

权重 URL、版本、大小和哈希记录在 `configs/models.json`。下载文件不会带来超出上游条款的权利。

MI32 信号版本来源于 BNCI2014_001、PhysionetMI、Schirrmeister2017、Stieger2021、
Wairagkar2018、Weibo2014、Zhou2016 和 Zhou2020。使用者必须引用实际使用的每个来源。
发布合并信号包之前，维护者必须完成 `datasets/mi32/DATASET_CARD.md` 中所述的逐来源再分发审查。
