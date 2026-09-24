# EEGNet

- **Track:** formal reproduction, supervised from scratch.
- **Upstream:** `vlawhern/arl-eegmodels` at `4a512e503198db2010848813ead9afbf8cd54c97`.
- **Benchmark input:** `[B,32,750]`, 250 Hz, Volt.
- **Definition:** EEGNet-8,2; 64-sample temporal kernel; full-channel grouped depthwise spatial
  convolution; separable temporal convolution; one three-class linear head.
- **Necessary adaptations:** 32 input channels, 750 samples, and 3 output classes.
- **Forbidden substitutions:** ordinary spatial convolution, fake separable convolution, added
  hidden/attention layers, or test-selected hyperparameters.
- **Gate:** `pytest -q tests/test_eegnet_structure.py`.
- **Run:** `python scripts/benchmark.py run --model eegnet --data DATA --output OUT --fold 0`.

The 64-point temporal kernel covers 256 ms at 250 Hz. This is a declared benchmark choice rather
than a claim that the physical span equals every historical EEGNet configuration.
