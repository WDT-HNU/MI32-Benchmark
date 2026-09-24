# Dataset card: MI32 common32 v4

## Summary

MI32 common32 v4 is a harmonized three-class motor-imagery EEG benchmark derived from eight
source datasets. It contains 230 subjects and 97,608 trials. Every trial is represented as
`float32 [32, 750]` at 250 Hz and is stored in volts.

This card describes the sealed release whose `SHA256SUMS` file hashes to:

```text
1017a3dd50d11ff9d92d987a40996fc2fd61ace99d465213bdb0316ff52aad06
```

## Intended use

- Cross-subject three-class motor-imagery classification.
- Reproducible comparison of task-specific networks and EEG foundation models.
- Audits of channel, sampling-rate, unit, split, and model-adapter behavior.

It is not intended for diagnosis, treatment, identity inference, or deployment as a medical
device. The heterogeneous `non_upper` class is a broad benchmark superclass, not a clinical
category.

## Composition

| Source identifier | Subjects | Measured target channels | Interpolated target channels |
|---|---:|---:|---:|
| BNCI2014_001 | 9 | 11 | 21 |
| PhysionetMI | 109 | 32 | 0 |
| Schirrmeister2017 | 14 | 32 | 0 |
| Stieger2021 | 62 | 32 | 0 |
| Wairagkar2018 | 14 | 19 | 13 |
| Weibo2014 | 10 | 32 | 0 |
| Zhou2016 | 4 | 14 | 18 |
| Zhou2020 | 8 | 16 | 16 |

Label counts are exactly `24,402 / 24,402 / 48,804` for left upper limb, right upper limb,
and non-upper-limb imagery. The per-subject selection target is `1:1:2`.

## Common signal contract

- Ordered channels: `Fp1, Fp2, AF3, AF4, F7, F3, Fz, F4, F8, FC5, FC3, FCz, FC4, FC6,
  T7, C3, Cz, C4, T8, CP5, CP3, CPz, CP4, CP6, P7, P3, Pz, P4, P8, O1, Oz, O2`.
- Sampling rate: 250 Hz.
- Epoch duration: 3 seconds, 750 samples.
- Storage unit: volt.
- No runtime normalization is embedded in the data.
- `channel_mask=True` means all 32 released channels are model-valid.
- `measured_mask=True` identifies source-measured values; `False` identifies interpolation.

## Missing-channel handling

Missing target electrodes are estimated from all available measured source channels using the
Perrin spherical-spline interpolation matrix implemented by MNE-Python with `standard_1005`
template coordinates and `alpha=1e-5`. Directly measured target channels are copied without
numerical change. Interpolated channels must not be zeroed by downstream models.

The template montage is a limitation: it uses generic coordinates rather than each participant's
digitized electrode positions.

## Splits and leakage controls

All splits are subject-level. Datasets with at most ten subjects use LOSO assignments; larger
datasets use five GroupKFold assignments. For a run with test fold `f`, the runner uses fold 1 as
validation when `f=0`, otherwise fold 0, and uses all remaining subjects for training.

No subject may occur in more than one role. Model-adapter statistics are fitted only from the
training subject IDs and are sealed with a split hash.

## Quality control

The release contains no NaN, Inf, or flat trials. A robust-extreme screen flags 501 trials for
review; those trials remain in the sealed dataset. Any exclusion or sensitivity analysis must be
declared before training and cannot use validation/test outcomes to choose the policy.

## Known limitations

- `non_upper` can represent feet, tongue, or rest depending on the source dataset.
- `hands`, `both_hands`, and `both_hand` boundary events were excluded upstream.
- Source protocols, hardware, references, and subject populations remain heterogeneous.
- Interpolation burden differs by source dataset and may be a domain cue.
- The exact raw-to-curated precursor builder is not yet part of this alpha repository.

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

Each NPZ contains `X`, `y`, `channel_mask`, and `measured_mask`. The public metadata snapshot is
under `metadata/`; signal NPZ files and `trials.csv` are distributed only through an approved
data release.

## Access and licensing

This repository does not grant redistribution rights for the eight source datasets. Before a
public combined archive is released, the maintainer must record for every source: official access
URL, version/date, required agreement, redistribution permission, and required citation. Until
that review is complete, use an authorized local sealed copy or rebuild from lawfully obtained
sources.
