# CodeBrain

- **路线：** 正式权重适配。
- **上游：** `jingyingma01/CodeBrain`，提交 `22d350caf68246d2fda4f630ef837420db3fb130`。
- **权重 SHA-256：** `d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352`。
- **适配器：** 伏特转 `µV/100`（`×10,000`），带限 250→200 Hz 重采样，然后输出
  `[B,32,3,200]`.
- **分类头约定：** 展平全部 patch 表示，并使用已声明的数据集特定 MLP
  （`19200 -> 600 -> 200 -> 3`）。
- **禁止替代：** 均值池化、`Linear(200,3)` 捷径、正式路线中的线性重采样，或只转微伏的
  `×1,000,000` 输入尺度。
- **门禁：** `python -m pytest -q tests/test_codebrain_structure.py model_adapters/tests/test_adapters_contract.py`。
- **运行：** 先获取固定身份的源码和权重，再运行
  `python scripts/benchmark.py run --model codebrain --data DATA --output OUT --fold 0`.
