# Results status

## Current-contract evidence snapshot

`results/mi32-common32-v4/current-partial-20260911/` contains the runs that match the current
sealed foundation-model unit contracts. It is deliberately called **partial**: the classic models
have only fold 0, CodeBrain and EEGMamba have folds 0/2/3, and LaBraM has folds 0/2. Missing folds
must not be imputed, averaged away, or represented as a complete leaderboard.

Available test rows:

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

These rows are an evidence inventory, not a ranking table: fold coverage is unequal and no
multi-fold mean or uncertainty interval is yet available.

## Historical package

`results/mi32-common32-v4/historical-fold0-pre-unit-hardening/` preserves the earlier seven-model
fold-0 evidence package so that the audit trail is not destroyed. Its EEGMamba and CodeBrain
scores are superseded because that package predates the final `V -> µV/100` adapter contract.
Do not cite that directory as the current leaderboard. Classic-model rows are retained in the
current snapshot; the historical files exist only for provenance.

## Why Uni-NTFM is absent

The public Uni-NTFM repository supplies a backbone/training sketch but no official pretrained
checkpoint or downstream classification-head contract. This project has an explicit supervised
protocol adapter, but any resulting number is labeled `protocol_benchmark`, not
`formal_reproduction`, and is not mixed into the formal-model table.

## Required next result milestone

Run all predeclared folds on one frozen release commit, then publish per-fold and aggregate metrics
with confidence intervals and dataset-level sensitivity analyses. Test predictions are now emitted
as CSV so metrics can be independently recomputed.
