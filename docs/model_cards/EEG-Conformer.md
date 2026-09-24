# EEG-Conformer

- **Track:** formal reproduction, supervised from scratch.
- **Upstream:** `eeyhsong/EEG-Conformer` at `9ae149ba62487ceae723277d13adac27837113d2`.
- **Benchmark input:** `[B,32,750]`, 250 Hz, Volt; standardization statistics are training-only.
- **Definition:** official convolutional patch embedding, multi-head attention encoder, and active
  flatten MLP classifier (`1760 -> 256 -> 32 -> 3`).
- **Forbidden substitutions:** missing projection convolution, an invented CLS token, an invented
  positional embedding, or routing through an inactive alternate head.
- **Gate:** `pytest -q tests/test_eegconformer_structure.py`.
- **Run:** `python scripts/benchmark.py run --model eegconformer --data DATA --output OUT --fold 0`.
