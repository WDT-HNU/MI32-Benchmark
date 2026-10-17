# MI32 Benchmark technical guide

MI32 Benchmark is organized around three EEG classification tasks: MI3, MI2, and P300. Each task is studied with three method families: traditional baselines, deep models, and Riemannian geometry methods.

## Project logic

```text
EEG benchmark
├── MI3
│   ├── traditional methods
│   ├── deep models, including foundation models
│   └── Riemannian methods
├── MI2
│   ├── traditional methods
│   ├── deep models, including foundation models
│   └── Riemannian methods
└── P300
    ├── traditional methods
    ├── deep models, including foundation models
    └── Riemannian methods
```

The task axis answers “what is decoded?” The method axis answers “how is it decoded?” Earlier documentation split task-specific deep models and foundation models into two top-level categories. That was misleading: both are deep models, while the supplied benchmark reports establish traditional methods as the third top-level family.

## Tasks and data identities

### MI3

The current MI32 common-32 v4 release uses `left_upper`, `right_upper`, and `non_upper`. It contains 8 source datasets, 230 subjects, and 97,608 trials. The exact class counts are 24,402 / 24,402 / 48,804. Signals are stored as 32 channels × 750 samples at 250 Hz in volts.

Missing target electrodes are reconstructed with Perrin spherical-spline interpolation using template geometry. `measured_mask` retains measured-versus-interpolated provenance. Subject splits are frozen before model training.

The traditional-method progress report is older: it records 8 datasets, 225 subjects, and 94,856 trials. Its scores are preserved as a dated 2026-09-04 snapshot and are not attributed to common-32 v4.

### MI2

Two frozen releases are currently documented:

- Deep models use the left/right view derived from MI32 common-32 v4: 230 subjects and 48,804 trials. No second filtering, resampling, rereferencing, or interpolation step is applied.
- Traditional and Riemannian materials describe an 18-source Binary MI release: 389 subjects and 59,216 trials. The Riemannian archive records 84 full folds and a representative 36-fold subset.

These releases implement the same left-versus-right task but are not paired datasets. Scores must retain their release and split identity.

### P300

P300 is a stimulus-locked Target/NonTarget ERP task. The harmonized contract uses 16 channels, 200 Hz, and a 0–600 ms post-stimulus window. Missing channels are spatially interpolated and the harmonized data are common-average referenced.

Three evaluation scopes are recorded:

- traditional methods: full 873-fold LOGO report, 16 methods, 13,968 method-fold evaluations;
- deep models: a fixed sample of 102 folds from that 873-fold universe, 7 models, 714 runs;
- Riemannian methods: a reduced frozen 85-fold LOGO protocol.

These scopes belong to the same task but are not numerically paired.

## Method families

### Traditional methods

The traditional snapshot evaluates seven methods on each MI task and sixteen on P300. The pool includes band-power, CSP, regularized CSP, DWT, flattened ERP or EEG features, shrinkage LDA, linear and RBF SVMs, BLDA, SWLDA, logistic regression, feature selection, template matching, and xDAWN.

Traditional methods are not a single pipeline. Feature extraction, selection, templates, spatial filters, and classifiers must be trained inside each fold. The P300 benchmark is documented in [P300 traditional benchmark](traditional/P300_BENCHMARK.md).

### Deep models

The task-specific architectures are EEGNet, TSception, RGNN, and EEG-Conformer. LaBraM, EEGMamba, CodeBrain, and Uni-NTFM are foundation or long-sequence models within the same deep-model family.

Seven models have recorded results; Uni-NTFM remains a separately labelled protocol benchmark because the audited public release did not provide an acceptable official downstream weight and head contract. Deep models are sensitive to unit, sampling rate, patch layout, pooling, and downstream heads. Every run must therefore record source, checkpoint, and adapter identity.

### Riemannian methods

The archive covers MDRM, TSLDA, KernelSVM, FGMDRM, RA, SPDNet, MFACC, RPA, TSA, Manifold-SimCLR, and MEKT-R. They operate on covariance or SPD representations, tangent spaces, alignment, or transfer learning.

Their information regimes are part of the method definition:

| Information regime | Meaning | Examples |
|---|---|---|
| `SOURCE_ONLY` | fit only on training data and labels | MDRM, TSLDA, KernelSVM, FGMDRM, SPDNet, MFACC |
| `TRANSDUCTIVE_UNLABELED_TARGET` | may estimate target-domain statistics without target labels | RA, MEKT-R |
| `TARGET_CALIBRATION_LABELED` | may use a pre-frozen labelled target calibration subset | RPA, TSA |

Reading unlabelled target-domain statistics is not label leakage, but it must be disclosed. A labelled calibration method must keep calibration and evaluation trials separate.

## Current task × method matrix

| Task | Traditional methods | Deep models | Riemannian methods |
|---|---|---|---|
| MI3 | 7-method dated snapshot | EEGNet, TSception, RGNN, EEG-Conformer, LaBraM, EEGMamba, and CodeBrain have current results; fold coverage differs | representative 16-fold protocol; MFACC 13/16; MEKT-R binary-only |
| MI2 | 7-method dated snapshot on 18-source release | 7 models on fold 0 of the MI32-derived view | representative 36 folds with formal, partial, or blocked status by method |
| P300 | 16 methods on the reported 873-fold LOGO protocol | 7 models on the fixed 102-fold sample | reduced 85-fold protocol with real method failures retained |

## Data-to-method boundary

Public EEG datasets differ in channel layout, sampling rate, unit, event codes, windowing, and participant organization. The benchmark freezes task-level data first, then applies only method-required transformations:

```text
raw data and source evidence
→ task-level harmonized data
→ frozen subject or LOGO split
→ method-specific adapter
→ frozen method or model core
→ predictions
→ independently calculated metrics
→ manifest and coverage
```

An adapter may not search preprocessing choices against test performance. Any fitted mean, variance, reference, spatial filter, feature selector, or calibration statistic must use only the information permitted by the current fold and method regime.

## Leakage controls

For deep models, one participant cannot appear in training, validation, and test roles. Hyperparameters are selected on validation data; the test fold is opened only after configuration freeze. Learned normalization and calibration statistics come from training data.

For Riemannian transfer methods, leakage cannot be judged by asking only whether target EEG is read. The original method contract, use of target labels, and separation between calibration and evaluation must all be checked.

For traditional methods, feature extraction steps that learn from data—PCA, xDAWN, SelectKBest, templates, and similar operations—must be fitted within the training fold rather than before splitting.

## How to read a score

A score needs at least five identities: task, frozen data release, split, implementation, and coverage.

- Traditional results are report transcriptions. The 2026-09-04 tables are dated, and the full P300 threshold-metric table is kept separate from the Macro-F1 snapshot.
- Deep MI2 evidence is fold 0 only. Deep MI3 model coverage is unequal. Deep P300 covers the fixed 102-fold sample.
- Riemannian status is reported over 36, 16, and 85 representative folds for MI2, MI3, and P300 respectively.

Do not average the three tasks into one score. Their labels, trial units, folds, and primary metrics differ. Even within one task, scores are strictly comparable only when release and split identities match.

## Evidence layout

An auditable run normally includes a result table, trial-level predictions, a run manifest, dataset-manifest SHA-256, train/validation/test identities, random seed, hyperparameters, source commit, and checkpoint hash. A model name is not a substitute for this evidence.

```text
MI32-Benchmark/
├── benchmark/
│   ├── datasets/          task data contracts and metadata
│   ├── model_adapters/    deep-model input adapters
│   ├── runners/           training and evaluation programs
│   ├── tests/             structure, semantics, and leakage gates
│   └── results/           machine-readable results and run evidence
└── docs/
    ├── traditional/       conventional baselines
    ├── deep_learning/     deep and foundation models
    ├── riemannian/        manifold methods
    └── project/           access, reproduction, release, and licensing
```

Recommended reading order:

1. [Dataset index](../benchmark/datasets/README.md)
2. the relevant method-family page
3. the result table and its coverage/evidence label
4. [Reproducibility guide](project/REPRODUCIBILITY.md)

## Minimal repository check

```bash
git clone https://github.com/WDT-HNU/MI32-Benchmark.git
cd MI32-Benchmark/benchmark

python scripts/verify_repository.py
python scripts/verify_dataset.py datasets/mi32/metadata
make cpu-test
```

A formal run additionally requires legally obtained signal packages, pinned upstream source, and the specified checkpoints. Freeze the task, release, split, and adapter before training; never use test results to select them.
