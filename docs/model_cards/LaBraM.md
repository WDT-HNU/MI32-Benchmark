# LaBraM

- **路线：** 正式权重适配。
- **上游：** `935963004/LaBraM`，提交 `c431221e6cfd23dbfa9950e0180682fb322b0548`。
- **权重：** 转换后的 Braindecode 文件 SHA-256 为
  `86a40de11088b85291eb47b788b049d784e619a6b3f8e84accbf640b2c59eec3`；作者原始文件 SHA-256 为
  `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`。
- **适配器：** 伏特转微伏，带限 250→200 Hz 重采样，输出 `[B,32,3,200]` patch。
- **分类头约定：** 非 CLS patch token 均值后接 `Linear(200,3)`。
- **禁止替代：** CLS pooling、线性插值或逐试次 z-score。
- **门禁：** `python -m pytest -q tests/test_labram_structure.py model_adapters/tests/test_adapters_contract.py`。
- **运行：** 先获取固定身份的源码和权重，再运行
  `python scripts/benchmark.py run --model labram --data DATA --output OUT --fold 0`.
