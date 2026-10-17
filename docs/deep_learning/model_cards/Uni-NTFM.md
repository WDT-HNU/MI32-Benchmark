# Uni-NTFM

- **Route:** protocol benchmark only.
- **Upstream:** `Zhisheng-researcher/Uni-NTFM`, commit `c0ce0152f94366e59b31b3fb2c108ce909bcd95c`.
- **Checkpoint:** no usable checkpoint was present in the audited public release.
- **Adapter:** deterministic five-region representation generated from the sealed 32-channel input.
- **Boundary:** the public upstream does not define an official MI32 downstream checkpoint or head contract. This repository therefore provides an explicitly labelled supervised wrapper. It is not a paper reproduction and must not enter a formal-reproduction leaderboard.
- **Gate:** `python -m pytest -q tests/test_uni_ntfm_structure.py`.
- **Run:** `python scripts/benchmark.py run --model uni_ntfm --allow-protocol-benchmark --data DATA --output OUT --fold 0`.

No explicit licence was found in the audited upstream snapshot, so its source is fetched separately and is not redistributed here.
