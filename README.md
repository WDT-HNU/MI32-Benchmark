# 🧠 MI32 Benchmark

在同一套 32 通道运动想象数据上，公平比较 8 种 EEG 模型。

## ✨ 这个仓库有什么？

- 🎛️ **统一信号格式**：每个试次都是 3 秒、32 通道、250 Hz 的 EEG 张量
- 📦 **8 个来源数据集**：共 230 名受试者、97,608 个试次，统一为三分类任务
- 🔌 **8 条模型路线**：覆盖 CNN、GNN、Transformer、状态空间模型和 EEG 基础模型
- 🔒 **受试者级评估**：只用验证集选择学习率；配置冻结后才运行测试集
- 🧾 **完整运行记录**：保留数据清单、上游提交、权重哈希、划分、参数和源码快照

## 🚀 安装

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

拉取已经固定版本的上游代码和预训练权重：

```bash
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

## 🏁 跑一个模型

把 MI32 数据放到 `datasets/mi32/full`，先检查数据，再运行一次预检：

```bash
python scripts/verify_dataset.py datasets/mi32/full --full
python scripts/benchmark.py doctor --data datasets/mi32/full

python scripts/benchmark.py run \
  --model eegnet \
  --data datasets/mi32/full \
  --output outputs/eegnet_preflight \
  --preflight
```

如果只想检查完整命令，不启动 GPU 训练：

```bash
python scripts/benchmark.py run \
  --model eegmamba \
  --data datasets/mi32/full \
  --output outputs/eegmamba_fold0 \
  --fold 0 \
  --final-test \
  --dry-run
```

运行器会把指标、训练历史、最佳轮次、适配器配置、运行环境和源码身份写入输出目录。

## 📐 数据约定

所有模型都从同一种试次表示开始：

| 项目 | 取值 |
|---|---|
| 张量形状 | `[batch, 32, 750]` |
| 采样率 | 250 Hz |
| 时间窗 | 3 秒 |
| 数据类型 / 单位 | `float32` / 伏特 |
| 标签 | `left_upper`、`right_upper`、`non_upper` |
| 类别数量 | 24,402 / 24,402 / 48,804 |
| 通道来源 | `measured_mask` 标记实测电极和插值电极 |

缺少的目标电极使用 Perrin 球面样条法，在 MNE 的 `standard_1005` 电极模板上插值。通道顺序、
各来源数据集的组成、质量检查和划分规则见[数据集卡片](datasets/mi32/DATASET_CARD.md)。

## ⚙️ 运行参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--model` | 是 | `eegnet`、`tsception`、`rgnn`、`eegconformer`、`labram`、`eegmamba`、`codebrain` 或 `uni_ntfm` |
| `--data` | 是 | 已通过检查的 MI32 数据目录 |
| `--output` | 是 | 指标和运行记录的输出目录 |
| `--fold` | 否 | 测试折，默认 `0` |
| `--epochs` | 否 | 训练轮数 |
| `--batch-size` | 否 | 覆盖模型默认批量大小 |
| `--lr` | 否 | 在验证集上选择的学习率 |
| `--balanced-loss` | 否 | 使用类别平衡损失 |
| `--preflight` | 否 | 运行简短的环境、模型和数据预检 |
| `--final-test` | 否 | 配置冻结后评估测试折 |
| `--dry-run` | 否 | 只打印解析后的命令，不实际执行 |

## 📊 来源数据集

| 数据集 | 受试者数 | 实测目标通道 | 插值目标通道 |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 11 | 21 |
| PhysionetMI | 109 | 32 | 0 |
| Schirrmeister2017 | 14 | 32 | 0 |
| Stieger2021 | 62 | 32 | 0 |
| Wairagkar2018 | 14 | 19 | 13 |
| Weibo2014 | 10 | 32 | 0 |
| Zhou2016 | 4 | 14 | 18 |
| Zhou2020 | 8 | 16 | 16 |

信号文件约 8.7 GB，不放进 Git 历史。下载、复制已有数据和从头重建的方法见
[获取 MI32 数据](DATA_AVAILABILITY.md)。

## 🧩 已接入模型

| 模型 | 类型 | 本仓库采用的定义 |
|---|---|---|
| EEGNet | 轻量级 CNN | EEGNet-8,2，分类头改为三分类线性层 |
| TSception | 多尺度 CNN | V2 时间卷积、非对称空间卷积和融合模块 |
| RGNN | 图神经网络 | 带符号图、二阶 SGC 和求和池化 |
| EEG-Conformer | CNN + Transformer | 卷积式 patch embedding 和六层编码器 |
| LaBraM | EEG 基础模型 | 官方权重，非 CLS token 均值池化 |
| EEGMamba | 状态空间基础模型 | 官方 12 层 Mamba2 主干和全 patch 分类头 |
| CodeBrain | EEG 基础模型 | 官方 EEGSSM 主干和展平全部 patch 的 MLP |
| Uni-NTFM | EEG 基础模型 | 公开主干和独立的协议评测分类头 |

每张[模型卡片](docs/model_cards/README.md)都记录上游提交、输入转换、权重哈希（如适用）、
分类头和结构测试。

## 📈 当前结果

仓库目前收录以下测试结果：

| 模型 | 折 | Macro-F1 | 平衡准确率 | 准确率 |
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

只有使用同一测试折的结果才适合直接比较。CSV 和对应的运行清单保存在
[`results/mi32-common32-v4/current-evidence-20260911`](results/mi32-common32-v4/current-evidence-20260911)，
具体解释见[结果说明](docs/RESULTS.md)。

## 📄 许可证

本仓库原创代码采用 Apache-2.0 许可证。原始 EEG 数据、上游模型代码和预训练权重仍遵循
各自的授权条款，详见[第三方来源](THIRD_PARTY.md)。

## 📣 引用

如果你的工作使用了 MI32 Benchmark，请同时引用本仓库以及实验中实际使用的原始数据集和模型。
机器可读的项目信息见 [CITATION.cff](CITATION.cff)。

## 📚 继续阅读

- [复现一次实验](REPRODUCIBILITY.md)：环境、身份校验、CUDA 门禁、调参和结果打包
- [评测协议](docs/BENCHMARK_PROTOCOL.md)：数据划分、超参数、指标和结果标签
- [模型卡片](docs/model_cards/README.md)：准确的模型定义和不等价替代
- [在 AutoDL 上运行](docs/AUTODL.md)：目录、安装、检查和关机流程
- [项目状态](PROJECT_STATUS.md)：仓库现有的数据、模型和测试折覆盖情况
- [参与贡献](CONTRIBUTING.md)：怎样修改 benchmark 而不悄悄改变它的含义
