# 复现一次运行

简要流程：核验数据，获取指定版本的上游代码和权重，通过轻量测试和一个真实 CUDA 批次，
只在验证集上调参，最后才打开测试折。

## 什么程度算复现？

整个流程分为四个可核验层级：

| 层级 | 实际检查内容 |
|---|---|
| 1. 身份 | 仓库提交、源码哈希、权重大小和 SHA-256 |
| 2. CPU 结构 | 模型图、参数量、张量形状和合成数据前向传播 |
| 3. CUDA 预检 | 真实依赖、真实权重、真实 MI32 批次，以及 CUDA 前向和反向传播 |
| 4. 完整运行 | 仅验证集选参、冻结配置、一次测试评估和完整结果包 |

Mamba stub 可以用于第 2 层结构检查，但不能证明第 3 或第 4 层成立。

## 1. 配置环境

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark
conda env create -f requirements/environment.yml
conda activate mi32-benchmark
```

如果 Conda 无法解析 CUDA 包，请使用 Python 3.12，先安装 PyTorch 2.8.0 CUDA 12.8 wheel，
再安装 `requirements/core.txt` 和 `requirements/foundation.txt`。

## 2. 获取上游代码和权重

```bash
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

第一条命令把 `configs/models.json` 中固定的提交克隆到 Git 忽略的 `third_party/` 目录；第二条
命令先检查文件大小和 SHA-256，再把权重放到目标位置。

## 3. 准备数据集

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
```

也可以使用有权访问的本地副本。把它放在 `datasets/mi32/full`，或者直接把路径传给检查器。
凡是需要报告的运行，都不能跳过完整哈希检查。

## 4. 运行轻量测试

```bash
python scripts/benchmark.py doctor --data datasets/mi32/full
python -m pytest -q \
  tests/test_eegnet_structure.py \
  tests/test_tsception_structure.py \
  tests/test_rgnn_structure.py \
  tests/test_eegconformer_structure.py \
  tests/test_mi3rawtrials_semantics.py \
  tests/test_model_adapter_boundary.py \
  model_adapters/tests/test_adapters_contract.py
```

这些测试耗时较短，应在租用 GPU 之前完成。

## 5. 检查一个真实 CUDA 批次

```bash
bash scripts/run_full_gate.sh datasets/mi32/full
```

这一步使用真实上游依赖、权重和 MI32 样本，并记录是否确实使用了 CUDA。任何文件缺失或
身份变化都会使门禁停止。

## 6. 不加载测试集进行调参

alpha 协议为每个模型比较两个学习率，并按验证集 macro-F1 选择。调参时不要添加
`--final-test`。

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr1e3 --epochs 15 --lr 0.001 --limit 64
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr3e4 --epochs 15 --lr 0.0003 --limit 64
```

进入下一步之前，先记录选定的学习率。

## 7. 测试集只运行一次

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_final_fold0 --epochs 50 --lr 0.001 \
  --balanced-loss --final-test
```

对计划中的各折重复运行冻结后的配置。Uni-NTFM 还需要 `--allow-protocol-benchmark`，其输出
保存在单独的协议评测路线中。

## 8. 收集结果文件

```bash
python collect_remote_results.py --root outputs
python scripts/package_release.py --input outputs --output release/mi32-run
```

结果包包括指标、训练历史、运行配置、适配器清单、源码快照、环境信息、数据集身份和新生成的
`SHA256SUMS`。

## 关于数值完全一致

运行器会固定 Python、NumPy 和 PyTorch 的随机种子，但不同 GPU、驱动、CUDA 内核和依赖构建
仍可能造成末位差异。这里的复现单位是相同的数据、划分、源码、权重、适配器和超参数，并得到
统计上相符的指标；不要求不同机器上的浮点结果逐位一致。
