# CodeBrain

- **Route:** formal checkpoint adaptation.
- **Upstream:** `jingyingma01/CodeBrain`, commit `22d350caf68246d2fda4f630ef837420db3fb130`.
- **Checkpoint SHA-256:** `d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352`.
- **Adapter:** volts to `µV/100` (`×10,000`), band-limited 250→200 Hz resampling, then `[B,32,3,200]`.
- **Head contract:** flatten every patch representation and use the declared dataset-specific MLP (`19200 -> 600 -> 200 -> 3`).
- **Prohibited substitute:** mean pooling, a `Linear(200,3)` shortcut, linear resampling in the formal route, or `×1,000,000` microvolt-scale input.
- **Gate:** `python -m pytest -q tests/test_codebrain_structure.py model_adapters/tests/test_adapters_contract.py`.
- **Run:** fetch the pinned source and checkpoint, then execute `python scripts/benchmark.py run --model codebrain --data DATA --output OUT --fold 0`.
