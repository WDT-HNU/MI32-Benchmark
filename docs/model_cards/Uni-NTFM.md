# Uni-NTFM

- **路线：** 仅协议评测。
- **上游：** `Zhisheng-researcher/Uni-NTFM`，提交
  `c0ce0152f94366e59b31b3fb2c108ce909bcd95c`.
- **权重：** 已审计公开版本中没有可用权重。
- **适配器：** 从封存的 32 通道输入确定性生成五脑区表示。
- **边界：** 公开上游没有定义官方 MI32 下游权重或分类头约定，因此本仓库提供明确声明的监督
  封装。该结果不是论文复现，不能混入正式复现排行榜。
- **门禁：** `python -m pytest -q tests/test_uni_ntfm_structure.py`。
- **运行：** `python scripts/benchmark.py run --model uni_ntfm --allow-protocol-benchmark --data DATA --output OUT --fold 0`。

已审计上游快照没有明确许可证，因此其源码单独拉取，不在本仓库再分发。
