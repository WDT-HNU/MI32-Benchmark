# EEGMamba

- **路线：** 正式权重适配。
- **上游：** `wjq-learning/EEGMamba`，提交 `dbc83fa072744201e8897aeb9f65007b952ad323`。
- **权重 SHA-256：** `b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe`。
- **适配器：** 伏特转 `µV/100`（`×10,000`），带限 250→200 Hz 重采样，然后输出
  `[B,32,3,200]`.
- **分类头约定：** 使用全部 patch 表示；直接使用 32 通道拓扑。
- **禁止替代：** 32→60 通道插值、750→800 补零、自定义均值池化、把未验证的 Mamba stub
  当作 CUDA 证据，或使用伏特尺度输入。
- **门禁：** `python -m pytest -q tests/test_eegmamba_structure.py model_adapters/tests/test_adapters_contract.py`；
  正式结果还要求在 CUDA 上使用真实 Mamba2。
- **运行：** 先获取固定身份的源码和权重，再运行
  `python scripts/benchmark.py run --model eegmamba --data DATA --output OUT --fold 0`.
