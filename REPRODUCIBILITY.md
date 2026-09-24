# Reproducing a run

The short version: verify the data, fetch the exact upstream code and weights, pass the lightweight
tests, pass one real CUDA batch, tune on validation, and only then open the test fold.

## What counts as a reproduction?

There are four useful checkpoints along the way:

| Level | What has actually been checked |
|---|---|
| 1. Identity | repository commit, source hashes, checkpoint size and SHA-256 |
| 2. CPU structure | model graph, parameter count, tensor shapes, synthetic forward pass |
| 3. CUDA preflight | real dependency, real checkpoint, real MI32 batch, forward and backward on CUDA |
| 4. Full run | validation-only selection, frozen settings, one test evaluation, packaged outputs |

A stubbed Mamba layer can help with level 2. It says nothing about levels 3 or 4.

## 1. Set up the environment

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark
conda env create -f requirements/environment.yml
conda activate mi32-benchmark
```

If Conda cannot resolve the CUDA packages, use Python 3.12, install the PyTorch 2.8.0 CUDA 12.8
wheel first, and then install `requirements/core.txt` and `requirements/foundation.txt`.

## 2. Fetch upstream code and checkpoints

```bash
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

The first command clones the commits listed in `configs/models.json` into ignored `third_party/`
directories. The second checks file size and SHA-256 before moving a checkpoint into place.

## 3. Put the dataset in place

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
```

An authorized local copy works too. Put it at `datasets/mi32/full` or pass its path directly to the
verifier. Do not skip the full hash check for a reported run.

## 4. Run the small tests

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

These tests are quick enough to run before renting a GPU.

## 5. Check one real CUDA batch

```bash
bash scripts/run_full_gate.sh datasets/mi32/full
```

This step uses the real upstream packages, checkpoints, and MI32 samples. It also records whether
CUDA was genuinely used. A missing or changed artifact stops the gate.

## 6. Tune without loading the test set

The alpha protocol compares two learning rates per model and selects by validation macro-F1. Leave
off `--final-test` while tuning.

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr1e3 --epochs 15 --lr 0.001 --limit 64
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr3e4 --epochs 15 --lr 0.0003 --limit 64
```

Write down the chosen learning rate before moving on.

## 7. Run the test once

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_final_fold0 --epochs 50 --lr 0.001 \
  --balanced-loss --final-test
```

Repeat with the frozen settings for the planned folds. Uni-NTFM also needs
`--allow-protocol-benchmark`; its output stays in the separate protocol track.

## 8. Collect the files

```bash
python collect_remote_results.py --root outputs
python scripts/package_release.py --input outputs --output release/mi32-run
```

The package includes results, histories, run configs, adapter manifests, source snapshots,
environment information, dataset identity, and a fresh `SHA256SUMS`.

## About exact numerical matches

The runners seed Python, NumPy, and PyTorch, but different GPUs, drivers, CUDA kernels, and library
builds can still move the last digits. The reproducible unit here is the same data, split, source,
checkpoint, adapter, and hyperparameters, with statistically consistent metrics—not bit-for-bit
floating-point equality across machines.
