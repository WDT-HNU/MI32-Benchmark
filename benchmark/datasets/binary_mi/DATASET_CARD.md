# Binary left-versus-right MI data contract

This card records the frozen 18-source release used by the Riemannian Binary MI archive and reflected in the traditional-method snapshot. It preserves source-dataset and subject identity while harmonizing comparable left- and right-hand motor-imagery trials. It does not redefine the original public datasets and is distinct from the MI2 view used by deep models.

## Processing tree

```text
18 public MI datasets
└── retain Left / Right motor-imagery trials
    └── labels: 0 = Left, 1 = Right
        └── time: 3 s × 250 Hz = 750 samples
            └── channels: ordered 32-channel representation
                └── signal: 8–30 Hz, volts
                    ├── full protocol: 84 subject-independent folds
                    └── representative protocol: 18 datasets × 2 existing folds = 36 folds
```

## Frozen statistics

| Item | Value |
|---|---:|
| component datasets | 18 |
| subjects | 389 |
| trials | 59,216 |
| EEG channels | 32 |
| sampling rate | 250 Hz |
| epoch duration | 3 s |
| samples per trial | 750 |
| trial tensor | `(n_trials, 32, 750)` |
| physical unit | V |
| label `0` | Left |
| label `1` | Right |

| Component dataset | Subjects | Trials | Frozen folds |
|---|---:|---:|---:|
| BNCI2014-001 | 9 | 2,538 | 5 |
| BNCI2014-004 | 9 | 6,520 | 5 |
| Beetl2021-A | 3 | 600 | 3 |
| Beetl2021-B | 2 | 320 | 2 |
| Brandl2020 | 16 | 8,058 | 5 |
| Chang2025 | 28 | 2,260 | 5 |
| Dreyer2023C | 6 | 1,440 | 5 |
| Forenzo2023 | 15 | 2,193 | 5 |
| GuttmannFlury2025-MI | 31 | 2,520 | 5 |
| HefmiIch2025 | 10 | 600 | 5 |
| Kumar2024 | 18 | 7,156 | 5 |
| Lee2019-MI | 54 | 5,400 | 5 |
| Liu2024 | 50 | 2,000 | 5 |
| PhysionetMotorImagery | 109 | 4,918 | 5 |
| Schirrmeister2017 | 7 | 3,248 | 5 |
| Weibo2014 | 10 | 1,580 | 5 |
| Zhou2016 | 4 | 1,199 | 4 |
| Zhou2020 | 8 | 6,666 | 5 |
| **Total** | **389** | **59,216** | **84** |

## Harmonization rules

### Task and labels

Only left- and right-hand imagery trials are retained. Other classes from an original task do not enter this binary benchmark. Every method shares `0 = Left` and `1 = Right`; mapping cannot change by method, fold, or observed performance.

### Time and sampling

Each trial is fixed to three seconds, or 750 samples at 250 Hz. Longer epochs are cropped to the common valid interval. Short epochs are not padded with zeros, because padding would change covariance rank and energy. Source rates are resampled with anti-aliasing during upstream construction.

The delivered summary seals the final `250 Hz / 750 samples` contract. It does not invent cue-relative endpoints or library parameters absent from the raw construction record.

### Channels and signal

All trials use one ordered 32-channel representation. The formal contract does not use the early exploratory 156-channel zero-filled layout. Signals are band-limited to 8–30 Hz with zero-phase Butterworth filtering and stored in volts.

### Shared data versus method adapter

```text
frozen epoch [32,750], V, 8–30 Hz
└── method-specific adapter
    ├── centred SCM
    ├── OAS covariance
    └── ERP-aware augmented covariance for P300
        └── frozen method core
```

The shared layer does not choose covariance for every method. Most SPD/AIRM methods use a time-centred SCM. A deterministic minimal PSD→SPD repair is applied only when a strict-SPD numerical gate fails. OAS and ERP-specific covariance remain method contracts rather than global replacements.

## Recorded file layout

```text
../pooled_mi_riemann_32interp/
├── config.json
├── subjects.csv
└── data/
    └── subject_###.npz
```

| Artifact | Record |
|---|---|
| `config.json` | global contract, including `n_channels=32` and `n_samples=750` |
| `subjects.csv` | subject identifier, source dataset, and relative trial-file path |
| `data/subject_###.npz` | `X` trial tensor and binary `y` labels |
| `evaluation/within_dataset_splits.json` | 84 frozen train/test subject splits |
| `evaluation/representative_2fold_subset.json` | 36 representative folds and count audit |

Dataset and subject ownership are linked through the subject file and `subjects.csv`; the loader does not assume an unused per-trial metadata CSV.

## Evaluation splits

### Full 84-fold protocol

Evaluation is within-dataset and cross-subject:

- Beetl2021-A: 3 subjects, 3 LOSO folds;
- Beetl2021-B: 2 subjects, 2 LOSO folds;
- Zhou2016: 4 subjects, 4 LOSO folds;
- all remaining datasets: subject-level five-fold GroupKFold.

The frozen structure covers 389 subjects. Every subject is a test subject exactly once within its dataset. Records contain no empty fold, train/test subject overlap, or single-class train/test fold.

### Representative 36-fold protocol

When compute is constrained, use `18 datasets × 2 already-existing folds`. Seed `2026` selects from the 84 frozen folds without resplitting, changing membership, or reading performance. The 36-fold set is representative; it is not an equivalent substitute for the full 84-fold evaluation.

See the [Riemannian methods](../../../docs/riemannian/METHODS.md) for method-specific inputs and information regimes.
