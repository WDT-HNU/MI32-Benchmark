# Uni-NTFM

- **Track:** protocol benchmark only.
- **Upstream:** `Zhisheng-researcher/Uni-NTFM` at
  `c0ce0152f94366e59b31b3fb2c108ce909bcd95c`.
- **Checkpoint:** none available in the audited public release.
- **Adapter:** deterministic five-region representation produced from the sealed 32-channel input.
- **Boundary:** the repository supplies an explicit supervised wrapper because the public upstream
  does not define an official MI32 downstream checkpoint/head contract. A result is not a paper
  reproduction and must not be mixed into the formal-reproduction leaderboard.
- **Gate:** `pytest -q tests/test_uni_ntfm_structure.py`.
- **Run:** `python scripts/benchmark.py run --model uni_ntfm --allow-protocol-benchmark --data DATA --output OUT --fold 0`.

The upstream repository has no explicit license in the audited snapshot, so its source is fetched
separately and is not redistributed here.
