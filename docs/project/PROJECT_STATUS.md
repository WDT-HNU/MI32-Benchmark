# Project status

This page summarizes material currently archived in the repository. Status describes evidence scope, not a claim that every task-method cell is directly comparable.

## Task × method matrix

| Task | Traditional methods | Deep models | Riemannian methods |
|---|---|---|---|
| MI3 | 7-method 2026-09-04 snapshot | 7 models with unequal fold coverage; Uni-NTFM protocol route | representative 16-fold status archived |
| MI2 | 7-method 2026-09-04 snapshot | 7 models on fold 0 of the MI32-derived view | representative 36-fold status on the Binary MI release |
| P300 | 16-method full LOGO report, 873 folds | 7 models on the fixed 102-fold sample | reduced frozen 85-fold status |

## Data records

- MI32 common-32 v4: 8 sources, 230 subjects, 97,608 three-class trials.
- MI2 derived view: the same 230 subjects, 48,804 left/right trials.
- Binary MI contract: 18 sources, 389 subjects, 59,216 trials, 84 full folds and 36 representative folds.
- P300 contract: 15 reported eligible datasets, 16 channels, 200 Hz, and 0–600 ms epochs.
- Traditional MI3 historical snapshot: 8 datasets, 225 subjects, and 94,856 trials as reported on 2026-09-04.

## Traditional methods

The new archive contains the reviewed P300 data contract, the 16-method full P300 report, the three-task 2026-09-04 progress snapshot, and two machine-readable result transcriptions. The traditional report values are labelled `report_transcription`; they are not presented as newly recomputed predictions.

## Deep models

EEGNet, TSception, RGNN, EEG-Conformer, LaBraM, EEGMamba, CodeBrain, and Uni-NTFM have explicit input adapters, model definitions, or structural tests. Current formal results cover the first seven. Uni-NTFM remains a separate protocol route.

MI3 run CSVs and manifests are under `benchmark/results/mi32-common32-v4/current-evidence-20260911/`. Cross-task summaries are in [deep-model results](../deep_learning/RESULTS.md).

## Riemannian methods

The archive records the input representation, information regime, task applicability, and coverage of MDRM, TSLDA, KernelSVM, FGMDRM, RA, SPDNet, MFACC, RPA, TSA, Manifold-SimCLR, and MEKT-R. Partial, binary-only, and implementation-blocked states are retained.

## Archiving rule

Git stores code, metadata, manifests, tests, and compact results. EEG signals, upstream source trees, and pretrained weights are obtained separately and checked against recorded identities. A result must include task, release, split, adapter, implementation, and coverage; the method name alone is insufficient.
