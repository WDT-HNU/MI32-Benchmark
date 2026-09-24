# CodeBrain

- **Track:** formal checkpoint adaptation.
- **Upstream:** `jingyingma01/CodeBrain` at `22d350caf68246d2fda4f630ef837420db3fb130`.
- **Checkpoint SHA-256:** `d9714b8732c9883a04d022ee66254cd578ae1fa27f5458e6ab7f1aa96e9a7352`.
- **Adapter:** Volt to `µV/100` (`×10,000`), bandlimited 250→200 Hz resampling, then
  `[B,32,3,200]`.
- **Head contract:** flatten all patch representations and use the declared dataset-specific MLP
  (`19200 -> 600 -> 200 -> 3`).
- **Forbidden substitutions:** mean pooling, a `Linear(200,3)` shortcut, linear resampling in the
  formal track, or a microvolt-only `×1,000,000` input scale.
- **Gates:** `pytest -q tests/test_codebrain_structure.py model_adapters/tests/test_adapters_contract.py`.
- **Run:** fetch identities first, then
  `python scripts/benchmark.py run --model codebrain --data DATA --output OUT --fold 0`.
