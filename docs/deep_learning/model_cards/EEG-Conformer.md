# EEG-Conformer

- **Route:** formal supervised reproduction trained from scratch.
- **Upstream:** `eeyhsong/EEG-Conformer`, commit `9ae149ba62487ceae723277d13adac27837113d2`.
- **Benchmark input:** `[B,32,750]`, 250 Hz, volts; standardization statistics come from training data only.
- **Definition:** official convolutional patch embedding, multi-head attention encoder, and the active flattened MLP classifier (`1760 -> 256 -> 32 -> 3`).
- **Prohibited substitute:** missing projection convolution, added CLS token, added positional embedding, or the inactive alternative classifier head.
- **Gate:** `python -m pytest -q tests/test_eegconformer_structure.py`.
- **Run:** `python scripts/benchmark.py run --model eegconformer --data DATA --output OUT --fold 0`.
