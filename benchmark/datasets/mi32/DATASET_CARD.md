# Dataset card: MI32 common-32 v4

## Summary

MI32 common-32 v4 harmonizes eight source datasets for three-class motor-imagery benchmarking. It contains 230 subjects and 97,608 trials. Each trial is a volt-scale `float32 [32,750]` tensor sampled at 250 Hz.

The sealed release has this `SHA256SUMS` file hash:

```text
1017a3dd50d11ff9d92d987a40996fc2fd61ace99d465213bdb0316ff52aad06
```

## Intended use

- cross-subject, three-class motor-imagery research;
- reproducible comparison of task-specific networks and EEG foundation models;
- audit of channel layout, rate, unit, split, and adapter behavior.

The dataset is not intended for diagnosis, treatment, identity inference, or medical-device deployment. The heterogeneous `non_upper` label is a broad benchmark superclass, not a clinical category.

## Composition

| Source ID | Subjects | Measured target channels | Interpolated target channels |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 11 | 21 |
| PhysionetMI | 109 | 32 | 0 |
| Schirrmeister2017 | 14 | 32 | 0 |
| Stieger2021 | 62 | 32 | 0 |
| Wairagkar2018 | 14 | 19 | 13 |
| Weibo2014 | 10 | 32 | 0 |
| Zhou2016 | 4 | 14 | 18 |
| Zhou2020 | 8 | 16 | 16 |

The exact counts for left-upper, right-upper, and non-upper imagery are `24,402 / 24,402 / 48,804`. Per-subject selection targets a `1:1:2` ratio.

## Signal contract

- Channel order: `Fp1, Fp2, AF3, AF4, F7, F3, Fz, F4, F8, FC5, FC3, FCz, FC4, FC6, T7, C3, Cz, C4, T8, CP5, CP3, CPz, CP4, CP6, P7, P3, Pz, P4, P8, O1, Oz, O2`.
- Sampling rate: 250 Hz.
- Epoch duration: 3 seconds, 750 samples.
- Storage unit: volts.
- No runtime normalization is embedded in the data.
- `channel_mask=True` means all 32 released channels are model-valid.
- `measured_mask=True` denotes a measured source channel; `False` denotes interpolation.

## Missing-channel handling

Missing target electrodes are estimated from all available measured source channels. Interpolation uses the Perrin spherical-spline matrix implemented by MNE-Python with `standard_1005` template coordinates and `alpha=1e-5`. Directly measured target channels are copied without numerical modification. Downstream models must not zero interpolated channels.

Template rather than participant-digitized coordinates are a limitation.

## Splits and leakage control

All splits are subject-level. Datasets with at most ten subjects use leave-one-subject-out; larger datasets use five-fold GroupKFold. For test fold `f`, the runner uses fold 1 as validation when `f=0`, otherwise fold 0; all remaining subjects are used for training.

A subject cannot occupy multiple roles. Learned adapter statistics are fitted on training subjects only and sealed with a split hash.

## Quality control

The release contains no NaN, Inf, or flat trials. Robust extreme-value screening marks 501 trials for review; those trials remain in the sealed dataset. Any exclusion or sensitivity analysis must be declared before training and cannot be selected from validation or test outcomes.

## Known limitations

- `non_upper` may represent feet, tongue, or rest depending on the source dataset.
- `hands`, `both_hands`, and `both_hand` boundary events were excluded upstream.
- Source protocols, hardware, references, and participant populations remain heterogeneous.
- Interpolation rates differ by source and may become a domain cue.
- The alpha repository does not include the full raw-to-curated precursor builder.

## File layout

```text
MI32_COMMON32_V4/
├── config.json
├── subjects.csv
├── trials.csv
├── splits.csv
├── provenance.csv
├── quality_flags.csv
├── electrode_adjacency.npy
├── release_audit.json
├── SHA256SUMS
└── data/subject_000.npz ... subject_229.npz
```

Each NPZ contains `X`, `y`, `channel_mask`, and `measured_mask`. The public metadata snapshot is under `metadata/`; signal NPZ files and `trials.csv` are distributed only through an approved data release.

## Access and licensing

This repository does not grant redistribution rights for the eight source datasets. Before publishing a merged package, maintainers must record the official access URL, version/date, required agreement, redistribution permission, and citation for every source. Until that review is complete, use an authorized local copy or rebuild from legally obtained sources.
