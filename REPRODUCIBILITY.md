# Reproducibility guide

## Reproduction levels

The project distinguishes four evidence levels. Do not report a higher level than was executed.

1. **Static identity** — pinned repository commit, checkpoint size/hash, and source hashes.
2. **CPU structure** — model graph, parameter counts, tensor contracts, and synthetic forward.
   Dependency stubs are allowed only when the test explicitly says so.
3. **CUDA preflight** — real dependencies, real checkpoint, real MI32 samples, one forward/backward
   batch on CUDA, without opening the test split.
4. **Formal run** — validation-only tuning, frozen hyperparameters, one final test evaluation, and
   a sealed result/evidence package.

A Mamba stub passing level 2 is not evidence for level 3 or 4.

## 1. Clone and create the environment

```bash
git clone https://github.com/OWNER/MI32-Benchmark.git
cd MI32-Benchmark
conda env create -f requirements/environment.yml
conda activate mi32-benchmark
```

If Conda cannot resolve CUDA packages, create Python 3.12 manually, install the PyTorch 2.8.0
CUDA 12.8 wheel, then install `requirements/core.txt` and `requirements/foundation.txt`.

## 2. Fetch exact upstream repositories

```bash
python scripts/fetch_upstreams.py
```

The script clones into ignored `third_party/` directories, checks out detached pinned commits,
and fails if the working tree differs. It does not copy third-party source into this repository.

## 3. Fetch exact checkpoints

```bash
python scripts/fetch_checkpoints.py
```

Every download is checked for byte length and SHA-256 before atomic placement under
`checkpoints/`. LaBraM's author checkpoint is copied from the pinned upstream checkout and is
used to audit the converted Braindecode checkpoint.

## 4. Obtain and verify data

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
```

If the combined archive is not publicly distributable, place an authorized sealed copy at that
path. Never bypass the manifest check for a formal run.

## 5. Run lightweight gates

```bash
python scripts/benchmark.py doctor --data datasets/mi32/full
pytest -q \
  tests/test_eegnet_structure.py \
  tests/test_tsception_structure.py \
  tests/test_rgnn_structure.py \
  tests/test_eegconformer_structure.py \
  tests/test_mi3rawtrials_semantics.py \
  tests/test_model_adapter_boundary.py \
  model_adapters/tests/test_adapters_contract.py
```

## 6. Run full identity and CUDA gates

```bash
bash scripts/run_full_gate.sh datasets/mi32/full
```

This gate requires all upstream checkouts and checkpoints and records whether a real CUDA batch
was executed. It fails closed on missing or changed identities.

## 7. Tune using validation only

For each formal model, compare only the predeclared learning rates and use validation macro-F1.
Run without `--final-test`; the test dataset is not instantiated.

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr1e3 --epochs 15 --lr 0.001 --limit 64
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr3e4 --epochs 15 --lr 0.0003 --limit 64
```

Freeze the chosen learning rate in a run note before proceeding.

## 8. Run the formal test once

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_final_fold0 --epochs 50 --lr 0.001 \
  --balanced-loss --final-test
```

Repeat for each model/fold with the frozen configuration. Uni-NTFM requires the explicit
`--allow-protocol-benchmark` flag and must remain outside the formal-reproduction leaderboard.

## 9. Collect and seal

```bash
python collect_remote_results.py --root outputs
python scripts/package_release.py --input outputs --output release/mi32-run
```

The release must contain results, histories, run configs, adapter manifests, source snapshots,
environment audit, dataset manifest identity, and a new `SHA256SUMS`.

## Determinism boundary

The runners seed Python, NumPy, and PyTorch. Exact floating-point equality across GPU models,
drivers, CUDA kernels, and library builds is not promised. Reproduction means the same artifact
identities, split, code path, hyperparameters, and statistically consistent metrics. Record GPU,
driver, CUDA, PyTorch, and package versions in every formal release.
