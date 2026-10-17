# EEGMamba

- **Route:** formal checkpoint adaptation.
- **Upstream:** `wjq-learning/EEGMamba`, commit `dbc83fa072744201e8897aeb9f65007b952ad323`.
- **Checkpoint SHA-256:** `b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe`.
- **Adapter:** volts to `µV/100` (`×10,000`), band-limited 250→200 Hz resampling, then `[B,32,3,200]`.
- **Head contract:** use all patch representations and the native 32-channel topology.
- **Prohibited substitute:** 32→60 channel interpolation, 750→800 zero padding, custom mean pooling, a Mamba stub presented as CUDA evidence, or volt-scale model input.
- **Gate:** `python -m pytest -q tests/test_eegmamba_structure.py model_adapters/tests/test_adapters_contract.py`; a formal result also requires real Mamba2 on CUDA.
- **Run:** fetch the pinned source and checkpoint, then execute `python scripts/benchmark.py run --model eegmamba --data DATA --output OUT --fold 0`.
