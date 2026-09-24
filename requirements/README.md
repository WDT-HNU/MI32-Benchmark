# Environment profiles

The formal fold-0 run recorded Python 3.12.3, PyTorch 2.8.0+cu128, CUDA 12.8,
MNE 1.12.1, NumPy 2.3.2, SciPy 1.18.0, and pandas 3.0.5 on an RTX 4090 D.

`core.txt` supports dataset tools and the four supervised models. `foundation.txt`
adds the sealed LaBraM and Mamba dependencies. CUDA wheels are platform-specific,
so install PyTorch first from its CUDA 12.8 index and then install these files.

The environment specification records the successful run; it does not imply that
CPU-only structure stubs are numerically equivalent to the CUDA kernels.
