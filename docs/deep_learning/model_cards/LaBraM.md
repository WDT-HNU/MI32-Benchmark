# LaBraM

- **Route:** formal checkpoint adaptation.
- **Upstream:** `935963004/LaBraM`, commit `c431221e6cfd23dbfa9950e0180682fb322b0548`.
- **Checkpoint:** converted Braindecode file SHA-256 `86a40de11088b85291eb47b788b049d784e619a6b3f8e84accbf640b2c59eec3`; original author file SHA-256 `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`.
- **Adapter:** volts to microvolts, band-limited 250→200 Hz resampling, output `[B,32,3,200]` patches.
- **Head contract:** mean of non-CLS patch tokens followed by `Linear(200,3)`.
- **Prohibited substitute:** CLS pooling, linear interpolation, or per-trial z-scoring.
- **Gate:** `python -m pytest -q tests/test_labram_structure.py model_adapters/tests/test_adapters_contract.py`.
- **Run:** fetch the pinned source and checkpoint, then execute `python scripts/benchmark.py run --model labram --data DATA --output OUT --fold 0`.
