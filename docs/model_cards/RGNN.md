# RGNN backbone

- **Track:** formal backbone benchmark, supervised from scratch.
- **Upstream:** `zhongpeixiang/RGNN` at `685b1a2645185bd8129c04a789dbfde8ad896a59`.
- **Benchmark input:** `[B,32,750]`, converted to five-band log-power node features `[B,32,5]`.
- **Graph:** sealed 32-electrode distance adjacency, applicable negative global pairs, signed
  degree using `abs(A)`, SGC order 2, and sum pooling.
- **Scope boundary:** NodeDAT and EmotionDL target-domain procedures are not included. The result
  must therefore be named `RGNN backbone`, never the full domain-adaptation system.
- **Gates:** `pytest -q tests/test_rgnn_structure.py` and
  `python audit_rgnn_training_features.py --help`.
- **Run:** `python scripts/benchmark.py run --model rgnn --data DATA --output OUT --fold 0`.
