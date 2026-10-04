# RGNN 主干

- **路线：** 从头监督训练的正式主干 benchmark。
- **上游：** `zhongpeixiang/RGNN`，提交 `685b1a2645185bd8129c04a789dbfde8ad896a59`。
- **Benchmark 输入：** `[B,32,750]`，转换为 `[B,32,5]` 五频带对数功率节点特征。
- **图：** 封存的 32 电极距离邻接矩阵、适用的负全局连接对、使用 `abs(A)` 的带符号度、
  二阶 SGC 和求和池化。
- **范围边界：** 不包含 NodeDAT 和 EmotionDL 的目标域流程，因此结果必须命名为
  `RGNN backbone`，不能称为完整域适配系统。
- **门禁：** `python -m pytest -q tests/test_rgnn_structure.py` 和
  `python audit_rgnn_training_features.py --help`.
- **运行：** `python scripts/benchmark.py run --model rgnn --data DATA --output OUT --fold 0`。
