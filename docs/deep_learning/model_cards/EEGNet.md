# EEGNet

- **路线：** 从头监督训练的正式复现。
- **上游：** `vlawhern/arl-eegmodels`，提交 `4a512e503198db2010848813ead9afbf8cd54c97`。
- **Benchmark 输入：** `[B,32,750]`、250 Hz、伏特。
- **定义：** EEGNet-8,2；64 点时间卷积核；覆盖全部通道的分组 depthwise 空间卷积；
  separable 时间卷积；单个三分类线性层。
- **必要适配：** 32 个输入通道、750 个采样点和 3 个输出类别。
- **禁止替代：** 普通空间卷积、伪 separable 卷积、额外隐藏/attention 层，或用测试集选择超参数。
- **门禁：** `python -m pytest -q tests/test_eegnet_structure.py`。
- **运行：** `python scripts/benchmark.py run --model eegnet --data DATA --output OUT --fold 0`。

64 点时间卷积核在 250 Hz 下覆盖 256 ms。这是明确声明的 benchmark 选择，不表示其物理时间
跨度与所有历史 EEGNet 配置相同。
