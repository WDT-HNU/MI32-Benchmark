# 训练入口

这里集中存放深度模型在各任务中调用的训练程序。当前公开运行器以 MI3 为主，通常不需要直接执行
这些文件，统一从
[`scripts/benchmark.py`](../scripts/benchmark.py) 进入，可以避免遗漏数据身份、权重和运行清单检查。

| 文件 | 负责内容 |
|---|---|
| `mi3_eegnet.py` | EEGNet、TSception、RGNN、EEG-Conformer |
| `mi3_foundation.py` | LaBraM、EEGMamba、CodeBrain |
| `uni_mi3_supervised.py` | Uni-NTFM 独立协议评测 |
| `mi3_deep_models.py` | 传统深度模型兼容导出 |
| `remote_mi3_deep_models.py` | 远程运行所需的冻结兼容实现 |

例如，EEGNet 的统一入口是：

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full --output outputs/eegnet
```

模型输入适配放在 [`model_adapters/`](../model_adapters/)，结构与数据语义检查放在
[`tests/`](../tests/)。
