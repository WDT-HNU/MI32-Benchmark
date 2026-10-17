# Environment

Recorded formal fold-0 runs used an RTX 4090 D, Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8, MNE 1.12.1, NumPy 2.3.2, SciPy 1.18.0, and pandas 3.0.5.

`core.txt` contains the data tools and dependencies for the four supervised architectures. `foundation.txt` adds LaBraM and Mamba dependencies required by the sealed configuration. CUDA wheels are platform-specific, so install PyTorch from the CUDA 12.8 index before these requirement files.

This environment records successful execution conditions. It does not imply that a CPU-only structural stub is numerically equivalent to a CUDA kernel.
