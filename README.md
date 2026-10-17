# MI32 Benchmark

MI32 Benchmark collects data contracts, method adaptations, evaluation protocols, and result evidence for three EEG classification tasks: three-class motor imagery (MI3), left-versus-right motor imagery (MI2), and P300 Target/NonTarget classification.

The repository is task-first. Choose a task, then compare the three method families evaluated for that task.

```text
MI32 Benchmark
├── Tasks
│   ├── MI3: left-upper / right-upper / non-upper
│   ├── MI2: left / right
│   └── P300: Target / NonTarget
├── Method families
│   ├── Traditional signal-processing and machine-learning baselines
│   ├── Deep models, including EEG foundation models
│   └── Riemannian geometry methods
└── Evidence
    ├── frozen data and split identity
    ├── method-specific adapter and implementation identity
    └── predictions, metrics, manifests, and coverage
```

## Experiment matrix

| Task | Traditional methods | Deep models | Riemannian methods |
|---|---|---|---|
| MI3 | 7-method, five-fold cross-subject snapshot | 7 models have current results with unequal fold coverage; Uni-NTFM remains a protocol route | representative 16-fold protocol with method-level coverage |
| MI2 | 7-method, five-fold cross-subject snapshot on the reported 18-dataset release | 7 models have fold-0 results on the MI32-derived view | representative 36-fold protocol on the 18-source Binary MI release |
| P300 | 16 methods across the reported full 873-fold LOGO protocol | 7 models across a fixed 102-fold LOGO sample | reduced frozen 85-fold LOGO protocol |

“Available” means that a result artifact or formal method status is recorded. It does not imply that every cell uses the same data release, fold set, metric, or evidence level. A valid comparison must retain those identities.

## Start here

| Question | Entry point |
|---|---|
| How is the project organized? | [Technical guide](docs/TECHNICAL_GUIDE.md) |
| What data contract belongs to each task? | [Dataset index](benchmark/datasets/README.md) |
| What are the conventional baselines? | [Traditional methods](docs/traditional/README.md) |
| Which deep and foundation models are included? | [Deep learning](docs/deep_learning/README.md) |
| Which manifold methods are included? | [Riemannian methods](docs/riemannian/README.md) |
| What is currently archived? | [Project status](docs/project/PROJECT_STATUS.md) |
| Where are all documents? | [Documentation index](docs/README.md) |

## Why task-first organization matters

MI3, MI2, and P300 differ in label semantics, class balance, trial construction, and evaluation metrics. P300 is a stimulus-locked ERP task and cannot inherit an MI time window or covariance representation by default. MI2 and MI3 share parts of the signal-processing pipeline but have different label spaces.

Every formal result should therefore follow the same evidence chain:

```text
task and frozen release
└── frozen split
    └── method-specific adapter
        └── method or model core
            └── predictions, metrics, manifest, and coverage
```

Traditional, deep, and Riemannian are method families, not dataset branches. Deep foundation models are a subgroup of the deep-model family rather than a fourth top-level family.

## Repository layout

```text
MI32-Benchmark/
├── benchmark/        runnable workspace, data contracts, tests, and result evidence
├── docs/             task, method, protocol, and reproducibility documentation
└── .github/          continuous integration and collaboration settings
```

Quick checks:

```bash
cd benchmark
python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

EEG signals, upstream source trees, and pretrained weights are not committed to Git history. Obtain them under their original terms and verify the recorded identities. Original code in this repository is licensed under Apache-2.0; datasets, upstream models, and weights retain their own licenses. See [third-party sources](docs/project/THIRD_PARTY.md).
