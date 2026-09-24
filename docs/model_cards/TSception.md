# TSception

- **Track:** formal reproduction, supervised from scratch.
- **Upstream:** `yi-ding-cs/TSception` at `9efd666b618d006e32e6da1d30dbc79b1d190604`.
- **Benchmark input:** `[B,32,750]`, 250 Hz, Volt.
- **Definition:** TSception V2 temporal branches, asymmetric spatial branches, and fusion block.
- **Channel rule:** the upstream pairing algorithm selects 26 left/right-paired channels; six
  midline channels are excluded by that algorithm, not silently dropped during data creation.
- **Gate:** `pytest -q tests/test_tsception_structure.py`.
- **Run:** `python scripts/benchmark.py run --model tsception --data DATA --output OUT --fold 0`.

Results must be described as the official V2 topology under the declared MI32 channel-order
adaptation, not as an unchanged execution of the authors' original dataset script.
