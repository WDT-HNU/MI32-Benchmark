# 🧠 MI32 Benchmark

Run eight EEG models on one common 32-channel motor-imagery benchmark.

## ✨ Why MI32?

- 🎛️ **One signal format** — Every trial is a 3-second, 32-channel tensor sampled at 250 Hz
- 📦 **Eight source datasets** — 230 subjects and 97,608 trials in a shared three-class task
- 🔌 **Eight model paths** — CNN, GNN, Transformer, state-space, and EEG foundation models
- 🔒 **Subject-level evaluation** — Validation selects the learning rate; the test fold stays closed until the run is frozen
- 🧾 **Traceable runs** — Results keep the data manifest, upstream commit, checkpoint hash, split, arguments, and source snapshot

## 🚀 Installation

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.8.0 torchaudio==2.8.0
pip install -r requirements/core.txt
pip install -r requirements/foundation.txt
```

Fetch the pinned upstream code and pretrained weights:

```bash
python scripts/fetch_upstreams.py
python scripts/fetch_checkpoints.py
```

## 🏁 Run a model

Put the MI32 release under `datasets/mi32/full`, verify it, and run a preflight:

```bash
python scripts/verify_dataset.py datasets/mi32/full --full
python scripts/benchmark.py doctor --data datasets/mi32/full

python scripts/benchmark.py run \
  --model eegnet \
  --data datasets/mi32/full \
  --output outputs/eegnet_preflight \
  --preflight
```

To inspect a full command without starting a GPU run:

```bash
python scripts/benchmark.py run \
  --model eegmamba \
  --data datasets/mi32/full \
  --output outputs/eegmamba_fold0 \
  --fold 0 \
  --final-test \
  --dry-run
```

The runner writes metrics, training history, the selected epoch, adapter settings, environment
information, and source identities to the output directory.

## 📐 Data contract

Every model starts from the same trial representation:

| Field | Value |
|---|---|
| Shape | `[batch, 32, 750]` |
| Sampling rate | 250 Hz |
| Window | 3 seconds |
| Dtype / unit | `float32` / Volt |
| Labels | `left_upper`, `right_upper`, `non_upper` |
| Class counts | 24,402 / 24,402 / 48,804 |
| Channel provenance | `measured_mask` marks measured and interpolated electrodes |

Missing target electrodes are estimated with Perrin spherical-spline interpolation on MNE's
`standard_1005` montage. See the [dataset card](datasets/mi32/DATASET_CARD.md) for the channel order,
source composition, QC, and split rules.

## ⚙️ Run parameters

| Parameter | Required | Description |
|---|---|---|
| `--model` | Yes | `eegnet`, `tsception`, `rgnn`, `eegconformer`, `labram`, `eegmamba`, `codebrain`, or `uni_ntfm` |
| `--data` | Yes | Path to the verified MI32 release |
| `--output` | Yes | Directory for metrics and run records |
| `--fold` | No | Test fold, default `0` |
| `--epochs` | No | Training epochs |
| `--batch-size` | No | Override the model default |
| `--lr` | No | Learning rate selected on validation |
| `--balanced-loss` | No | Use class-balanced training loss |
| `--preflight` | No | Run the short environment/model/data check |
| `--final-test` | No | Evaluate the test fold after settings are frozen |
| `--dry-run` | No | Print the resolved command without executing it |

## 📊 Source datasets

| Dataset | Subjects | Measured target channels | Interpolated target channels |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 11 | 21 |
| PhysionetMI | 109 | 32 | 0 |
| Schirrmeister2017 | 14 | 32 | 0 |
| Stieger2021 | 62 | 32 | 0 |
| Wairagkar2018 | 14 | 19 | 13 |
| Weibo2014 | 10 | 32 | 0 |
| Zhou2016 | 4 | 14 | 18 |
| Zhou2020 | 8 | 16 | 16 |

The signal package is about 8.7 GB and stays outside Git history. Download, local-copy, and rebuild
options are documented in [Getting the MI32 data](DATA_AVAILABILITY.md).

## 🧩 Available models

| Model | Family | Definition used here |
|---|---|---|
| EEGNet | compact CNN | EEGNet-8,2 with a three-class linear head |
| TSception | multi-scale CNN | V2 temporal, asymmetric spatial, and fusion blocks |
| RGNN | graph neural network | signed graph, second-order SGC, and sum pooling |
| EEG-Conformer | CNN + Transformer | convolutional patch embedding and six-layer encoder |
| LaBraM | EEG foundation model | official weights and non-CLS token mean pooling |
| EEGMamba | state-space foundation model | official 12-layer Mamba2 backbone and all-patch head |
| CodeBrain | EEG foundation model | official EEGSSM backbone and flatten-all-patches MLP |
| Uni-NTFM | EEG foundation model | public backbone with a separate protocol-benchmark head |

Each [model card](docs/model_cards/README.md) records the upstream commit, input conversion,
checkpoint hash where applicable, classification head, and structure test.

## 📈 Results

The repository currently contains these test rows:

| Model | Fold | Macro-F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|
| EEG-Conformer | 0 | 0.579492 | 0.599415 | 0.588935 |
| EEGNet | 0 | 0.555890 | 0.568942 | 0.570968 |
| TSception | 0 | 0.510936 | 0.541781 | 0.516698 |
| RGNN backbone | 0 | 0.408277 | 0.418587 | 0.421105 |
| CodeBrain | 0 | 0.543431 | 0.550130 | 0.564627 |
| CodeBrain | 2 | 0.513453 | 0.518010 | 0.537174 |
| CodeBrain | 3 | 0.544182 | 0.538975 | 0.592015 |
| EEGMamba | 0 | 0.516615 | 0.516276 | 0.549144 |
| EEGMamba | 2 | 0.516945 | 0.519419 | 0.542929 |
| EEGMamba | 3 | 0.526835 | 0.519032 | 0.580357 |
| LaBraM | 0 | 0.544850 | 0.547383 | 0.574403 |
| LaBraM | 2 | 0.466047 | 0.468288 | 0.494127 |

Matched-fold comparisons use rows that share the same fold. The CSV files and paired run manifests
are under
[`results/mi32-common32-v4/current-evidence-20260911`](results/mi32-common32-v4/current-evidence-20260911);
see [Results](docs/RESULTS.md) for interpretation.

## 📄 License

Original code in this repository is Apache-2.0 licensed. Source EEG datasets, upstream model code,
and pretrained weights keep their own terms. See [Third-party sources](THIRD_PARTY.md).

## 📣 Citation

If you use MI32 Benchmark, cite this repository together with the original datasets and models used
in your experiment. Machine-readable project metadata is available in [CITATION.cff](CITATION.cff).

## 📚 Further reading

- [Reproducing a run](REPRODUCIBILITY.md) — environment, identity checks, CUDA gate, tuning, and packaging
- [Benchmark protocol](docs/BENCHMARK_PROTOCOL.md) — splits, hyperparameters, metrics, and result labels
- [Model cards](docs/model_cards/README.md) — exact model definitions and non-equivalent substitutions
- [Running on AutoDL](docs/AUTODL.md) — directory layout, installation, validation, and shutdown
- [Project status](PROJECT_STATUS.md) — data, model, and fold coverage already in the repository
- [Contributing](CONTRIBUTING.md) — how to change a benchmark without silently changing its meaning
