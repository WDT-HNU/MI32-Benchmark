# EEG-Conformer

- **路线：** 从头监督训练的正式复现。
- **上游：** `eeyhsong/EEG-Conformer`，提交 `9ae149ba62487ceae723277d13adac27837113d2`。
- **Benchmark 输入：** `[B,32,750]`、250 Hz、伏特；标准化统计量仅从训练集得到。
- **定义：** 官方卷积 patch embedding、多头 attention 编码器和实际启用的展平 MLP 分类器
  （`1760 -> 256 -> 32 -> 3`）。
- **禁止替代：** 缺失投影卷积、自行添加 CLS token、自行添加位置嵌入，或使用未启用的备用分类头。
- **门禁：** `python -m pytest -q tests/test_eegconformer_structure.py`。
- **运行：** `python scripts/benchmark.py run --model eegconformer --data DATA --output OUT --fold 0`。
