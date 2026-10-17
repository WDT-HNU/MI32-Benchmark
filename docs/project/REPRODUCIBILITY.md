# Reproducing a run

The short version is: verify the dataset, fetch pinned upstream source and checkpoints, pass lightweight tests and one real CUDA batch, tune on validation data only, and open the test fold last.

## Four evidence levels

| Level | What is actually verified |
|---|---|
| 1. Identity | repository commit, source hashes, checkpoint size, and SHA-256 |
| 2. CPU structure | model graph, parameter count, tensor shapes, and synthetic forward pass |
| 3. CUDA preflight | real dependencies, real checkpoint, real MI32 batch, CUDA forward and backward pass |
| 4. Full run | validation-only selection, frozen configuration, one test evaluation, and complete result package |

A Mamba stub can support level 2. It cannot establish level 3 or 4.

## 1. Create the environment

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark/benchmark
conda env create -f requirements/environment.yml
conda activate mi32-benchmark
```

If Conda cannot resolve CUDA packages, use Python 3.12, install the PyTorch 2.8.0 CUDA 12.8 wheel first, then install `requirements/core.txt` and `requirements/foundation.txt`.

## 2. Fetch pinned source and checkpoints

```bash
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

The first command clones exact commits from `configs/models.json` into the Git-ignored `third_party/` directory. The second verifies file size and SHA-256 before placing each checkpoint.

## 3. Prepare the dataset

```bash
python scripts/fetch_dataset.py --url "$MI32_DATA_URL" --out datasets/mi32/full
python scripts/verify_dataset.py datasets/mi32/full --full
```

You may instead use a local copy you are entitled to access. Place it under `datasets/mi32/full` or pass its path directly to the verifier. A reportable run must not skip the full hash check.

## 4. Run lightweight checks

```bash
python scripts/benchmark.py doctor --data datasets/mi32/full
python -m pytest -q \
  tests/test_eegnet_structure.py \
  tests/test_tsception_structure.py \
  tests/test_rgnn_structure.py \
  tests/test_eegconformer_structure.py \
  tests/test_mi3rawtrials_semantics.py \
  tests/test_model_adapter_boundary.py \
  model_adapters/tests/test_adapters_contract.py
```

Run these inexpensive checks before renting a GPU.

## 5. Verify one real CUDA batch

```bash
bash scripts/run_full_gate.sh datasets/mi32/full
```

This gate uses real upstream dependencies, checkpoints, and MI32 samples and records whether CUDA was actually used. Missing or changed identities stop the gate.

## 6. Tune without loading the test set

The alpha protocol compares two learning rates per model and selects by validation Macro-F1. Do not add `--final-test` while tuning.

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr1e3 --epochs 15 --lr 0.001 --limit 64
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_tune_lr3e4 --epochs 15 --lr 0.0003 --limit 64
```

Record the selected learning rate before continuing.

## 7. Evaluate the test set once

```bash
python scripts/benchmark.py run --model eegnet --data datasets/mi32/full \
  --output outputs/eegnet_final_fold0 --epochs 50 --lr 0.001 \
  --balanced-loss --final-test
```

Repeat the frozen configuration for every planned fold. Uni-NTFM additionally requires `--allow-protocol-benchmark` and remains in a separate reporting route.

## 8. Collect the result package

```bash
python scripts/collect_remote_results.py --root outputs
python scripts/package_release.py --input outputs --output release/mi32-run
```

The package contains metrics, training history, run configuration, adapter manifest, source snapshot, environment identity, dataset identity, and a newly generated `SHA256SUMS`.

## Numerical reproducibility

The runners seed Python, NumPy, and PyTorch. Different GPUs, drivers, CUDA kernels, and dependency builds may still change least-significant digits. The reproducibility unit is the same data, split, source, checkpoint, adapter, and hyperparameters with statistically consistent metrics—not bitwise-identical floating-point output across machines.
