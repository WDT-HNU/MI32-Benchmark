# TSception

- **路线：** 从头监督训练的正式复现。
- **上游：** `yi-ding-cs/TSception`，提交 `9efd666b618d006e32e6da1d30dbc79b1d190604`。
- **Benchmark 输入：** `[B,32,750]`、250 Hz、伏特。
- **定义：** TSception V2 时间分支、非对称空间分支和融合模块。
- **通道规则：** 上游配对算法选择 26 个左右配对通道；6 个中线通道由该算法排除，并非在
  数据创建时被静默丢弃。
- **门禁：** `python -m pytest -q tests/test_tsception_structure.py`。
- **运行：** `python scripts/benchmark.py run --model tsception --data DATA --output OUT --fold 0`。

结果应表述为“在已声明 MI32 通道顺序适配下的官方 V2 拓扑”，不能表述为原作者数据脚本的
原样执行。
