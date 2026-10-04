# MI32 Benchmark

把 8 种 EEG 模型放到同一套 32 通道运动想象数据上，认真比一次。

这个仓库是我在做运动想象 EEG 对比实验时一点点整理出来的。最开始我只是想回答一个很直接的
问题：如果数据、划分和评价方法都相同，EEGNet、图网络、Transformer、Mamba 和 EEG 基础模型
到底会表现得怎么样？

真正动手以后，我发现麻烦往往不在 `model.fit()`。不同数据集的通道数、采样率和标签不一样；
同一段信号用伏特还是微伏，足以让预训练模型得到完全不同的输入；有些代码看起来像论文里的模型，
逐层对照后却会少一层卷积，或者换了一种池化。如果这些地方说不清，最后那张成绩表再漂亮也没有
太大意义。

所以我把数据整理、模型审计、训练协议和运行证据放进了同一个仓库。我希望别人拿到它时，不仅能
看到一个分数，还能继续追问：这份数据从哪里来？模型到底改了什么？测试集有没有参与调参？这个
结果对应的是哪份源码和哪一个权重？

## 现在仓库里有什么

- 一套统一的 MI32 common-32 v4 数据约定：230 名受试者，97,608 个试次；
- 8 个来源数据集，每个试次都整理为 `[32, 750] @ 250 Hz`；
- 8 种模型的运行器、输入适配器、模型卡片和结构测试；
- 受试者级训练、验证、测试划分，验证集负责选参，测试集最后才打开；
- 当前结果 CSV，以及与每次运行一一对应的源码、数据、权重和参数清单。

信号文件大约 8.7 GB，不放进 Git 历史。仓库里保留的是代码、元数据、哈希、划分和结果证据。

## 我比较较真的几件事

**受试者不能泄露。** 同一个人不会同时出现在训练集、验证集和测试集。适配器里需要学习的均值、
方差或校准量，也只能从训练受试者得到。

**测试集不负责调参。** 每个模型只在验证集上比较学习率。学习率和训练设置写下来以后，才允许
执行最终测试。

**“模型名相同”不等于“模型相同”。** 每张模型卡片都记录上游提交、输入转换、分类头和权重
哈希。找不到指定源码或权重时，程序会停下来，而不是临时换一个相似实现。

**适配必须明说。** 32 通道、3 分类、重采样、单位转换、patch 划分和分类头都属于实验定义，
不能藏在一句“按照官方模型实现”后面。

## 先跑通一个 EEGNet

环境以 Python 3.12、PyTorch 2.8.0 和 CUDA 12.8 为准：

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0
pip install -r requirements/core.txt
pip install -r requirements/foundation.txt
```

准备好 MI32 数据以后，先检查数据和运行环境：

```bash
python scripts/verify_dataset.py datasets/mi32/full --full
python scripts/benchmark.py doctor --data datasets/mi32/full
```

然后做一次短预检：

```bash
python scripts/benchmark.py run \
  --model eegnet \
  --data datasets/mi32/full \
  --output outputs/eegnet_preflight \
  --preflight
```

如果只是想确认完整命令，不准备立刻占用 GPU，可以加 `--dry-run`：

```bash
python scripts/benchmark.py run \
  --model eegmamba \
  --data datasets/mi32/full \
  --output outputs/eegmamba_fold0 \
  --fold 0 \
  --final-test \
  --dry-run
```

上游源码和预训练权重由下面两个脚本按固定版本拉取：

```bash
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

## 数据到底是什么样

所有模型拿到的原始试次都遵守同一个约定：

| 项目 | 取值 |
|---|---|
| 张量形状 | `[batch, 32, 750]` |
| 采样率 | 250 Hz |
| 时间窗 | 3 秒 |
| 数据类型 / 单位 | `float32` / 伏特 |
| 标签 | `left_upper`、`right_upper`、`non_upper` |
| 类别数量 | 24,402 / 24,402 / 48,804 |
| 通道来源 | `measured_mask` 区分实测电极和插值电极 |

缺失电极不是补零，而是使用 Perrin 球面样条法，在 MNE 的 `standard_1005` 模板坐标上插值。
实测通道保持原值，插值通道也会正常送入模型。

| 来源数据集 | 受试者数 | 32 个目标通道中实测 | 需要插值 |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 11 | 21 |
| PhysionetMI | 109 | 32 | 0 |
| Schirrmeister2017 | 14 | 32 | 0 |
| Stieger2021 | 62 | 32 | 0 |
| Wairagkar2018 | 14 | 19 | 13 |
| Weibo2014 | 10 | 32 | 0 |
| Zhou2016 | 4 | 14 | 18 |
| Zhou2020 | 8 | 16 | 16 |

更细的数据来源、标签映射、质量检查和划分方法写在[数据集卡片](datasets/mi32/DATASET_CARD.md)里。
数据包的获取和本地重建方式见[获取 MI32 数据](DATA_AVAILABILITY.md)。

## 这 8 个模型分别代表什么

我没有只挑同一类网络。这里既有体量很小的 CNN，也有显式使用电极拓扑的图网络，还有
Transformer、状态空间模型和预训练基础模型。

| 模型 | 在这次比较中的位置 | 本仓库采用的定义 |
|---|---|---|
| EEGNet | 轻量级 CNN 基线 | EEGNet-8,2，三分类线性头 |
| TSception | 多尺度时空 CNN | V2 时间、非对称空间和融合模块 |
| RGNN | 图神经网络 | 带符号图、二阶 SGC、求和池化 |
| EEG-Conformer | CNN + Transformer | 卷积 patch embedding、六层编码器 |
| LaBraM | EEG 基础模型 | 官方权重、非 CLS token 均值池化 |
| EEGMamba | 状态空间基础模型 | 官方 12 层 Mamba2、全 patch 分类头 |
| CodeBrain | EEG 基础模型 | 官方 EEGSSM、展平全部 patch 的 MLP |
| Uni-NTFM | 独立协议评测 | 公开主干、本仓库明确声明的监督分类头 |

这里最容易被忽略的是最后一列。比如 EEGMamba 不能偷偷把 32 通道插值到 60 通道，LaBraM
不能把 mean pooling 换成 CLS pooling，CodeBrain 也不能把全部 patch 的分类头简化为一个
`Linear(200,3)`。这些边界都写在[模型卡片](docs/model_cards/README.md)里，并由结构测试守住。

## 当前跑出来的结果

先别急着从这张表里选“第一名”。目前不同模型覆盖的测试折并不完全相同，只有相同测试折上的
结果才适合直接比较。

| 模型 | 测试折 | Macro-F1 | 平衡准确率 | 准确率 |
|---|---:|---:|---:|---:|
| EEG-Conformer | 0 | 0.579492 | 0.599415 | 0.588935 |
| EEGNet | 0 | 0.555890 | 0.568942 | 0.570968 |
| TSception | 0 | 0.510936 | 0.541781 | 0.516698 |
| RGNN backbone | 0 | 0.408277 | 0.418587 | 0.421105 |
| CodeBrain | 0 | 0.543431 | 0.550130 | 0.564627 |
| CodeBrain | 2 | 0.513453 | 0.518010 | 0.537174 |
| CodeBrain | 3 | 0.544182 | 0.538975 | 0.592015 |
| EEGMamba | 0 | 0.516615 | 0.516276 | 0.549144 |
| EEGMamba | 2 | 0.516945 | 0.519419 | 0.542929 |
| EEGMamba | 3 | 0.526835 | 0.519032 | 0.580357 |
| LaBraM | 0 | 0.544850 | 0.547383 | 0.574403 |
| LaBraM | 2 | 0.466047 | 0.468288 | 0.494127 |

这张表现在能证明的是：当前几条 pipeline 已经按照各自记录的输入和模型定义完成运行，并且同折
结果可以复核。它还不是一个已经完成所有折的最终排行榜。

CSV 和对应的运行清单位于
[`results/mi32-common32-v4/current-evidence-20260911`](results/mi32-common32-v4/current-evidence-20260911)。
每份清单都会记录数据身份、源码提交、权重、划分、随机种子和选定轮次。详细解释见
[结果说明](docs/RESULTS.md)。

## 仓库怎么找东西

```text
MI32-Benchmark/
├── datasets/          数据约定、元数据、划分和加载器
├── dataset_tools/     32 通道数据构建与检查
├── model_adapters/    8 种模型各自的输入适配
├── docs/model_cards/  每个模型的实现边界和审计记录
├── scripts/           拉取、检查、训练和打包入口
├── tests/             结构、数据语义和泄露回归测试
└── results/           结果 CSV 与对应运行证据
```

如果准备完整复现实验，建议按[复现说明](REPRODUCIBILITY.md)从头走一遍；在 AutoDL 上运行时，
目录和关机注意事项见 [AutoDL 说明](docs/AUTODL.md)。正式的划分、调参和指标定义放在
[评测协议](docs/BENCHMARK_PROTOCOL.md)里。

## 最后说明

本仓库原创代码采用 Apache-2.0 许可证。原始 EEG 数据、上游模型代码和预训练权重各自遵循原
授权条款，详见[第三方来源](THIRD_PARTY.md)。如果使用本仓库，请同时引用实际用到的数据集和
模型；机器可读的引用信息在 [CITATION.cff](CITATION.cff)。

如果你发现模型结构对照、单位换算、数据划分或结果记录有问题，欢迎直接指出。对这个仓库来说，
找出一个会让结果失效的细节，比把表格里的数字再抬高一点更有价值。
