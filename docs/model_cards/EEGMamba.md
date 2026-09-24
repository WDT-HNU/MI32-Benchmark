# EEGMamba

- **Track:** formal checkpoint adaptation.
- **Upstream:** `wjq-learning/EEGMamba` at `dbc83fa072744201e8897aeb9f65007b952ad323`.
- **Checkpoint SHA-256:** `b452bb29ecf1d6131ba82a50c6e13823ec1d660d9009d013e691d19b2916f4fe`.
- **Adapter:** Volt to `µV/100` (`×10,000`), bandlimited 250→200 Hz resampling, then
  `[B,32,3,200]`.
- **Head contract:** all patch representations; direct 32-channel topology.
- **Forbidden substitutions:** 32→60 channel interpolation, 750→800 zero padding, custom mean
  pooling, unverified Mamba stubs as CUDA evidence, or a Volt-scale input.
- **Gates:** `pytest -q tests/test_eegmamba_structure.py model_adapters/tests/test_adapters_contract.py`;
  a formal result additionally requires real Mamba2 on CUDA.
- **Run:** fetch identities first, then
  `python scripts/benchmark.py run --model eegmamba --data DATA --output OUT --fold 0`.
