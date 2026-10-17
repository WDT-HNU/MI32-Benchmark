# RGNN backbone

- **Route:** formal supervised backbone benchmark trained from scratch.
- **Upstream:** `zhongpeixiang/RGNN`, commit `685b1a2645185bd8129c04a789dbfde8ad896a59`.
- **Benchmark input:** `[B,32,750]`, converted to `[B,32,5]` five-band log-power node features.
- **Graph:** sealed 32-electrode distance adjacency; applicable negative global connection pairs; signed degree computed with `abs(A)`; second-order SGC; sum pooling.
- **Scope boundary:** NodeDAT and EmotionDL target-domain pipelines are not included. Results are named `RGNN backbone`, not the complete domain-adaptation system.
- **Gates:** `python -m pytest -q tests/test_rgnn_structure.py` and `python -m scripts.audit_rgnn_training_features --help`.
- **Run:** `python scripts/benchmark.py run --model rgnn --data DATA --output OUT --fold 0`.
