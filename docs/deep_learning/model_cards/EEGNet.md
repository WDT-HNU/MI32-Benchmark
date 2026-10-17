# EEGNet

- **Route:** formal supervised reproduction trained from scratch.
- **Upstream:** `vlawhern/arl-eegmodels`, commit `4a512e503198db2010848813ead9afbf8cd54c97`.
- **Benchmark input:** `[B,32,750]`, 250 Hz, volts.
- **Definition:** EEGNet-8,2; 64-sample temporal kernel; grouped depthwise spatial convolution across all channels; separable temporal convolution; one three-class linear layer.
- **Necessary adaptation:** 32 input channels, 750 samples, and 3 output classes.
- **Prohibited substitute:** ordinary spatial convolution, pseudo-separable convolution, extra hidden or attention layers, or test-driven hyperparameter selection.
- **Gate:** `python -m pytest -q tests/test_eegnet_structure.py`.
- **Run:** `python scripts/benchmark.py run --model eegnet --data DATA --output OUT --fold 0`.

At 250 Hz, the 64-sample temporal kernel spans 256 ms. This is an explicit benchmark choice; it does not claim the same physical duration as every historical EEGNet configuration.
