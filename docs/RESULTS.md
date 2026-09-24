# Results

The current files are in `results/mi32-common32-v4/current-evidence-20260911/`. They use the model
input definitions documented in the current model cards.

## Test rows on record

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

One caution matters here: the models do not yet have the same fold coverage. A ranking should use
matched folds and state its aggregation rule. Fold 0 by itself is a pipeline result, not the final
cross-subject estimate.

Each CSV has a paired run manifest with the dataset, source commit, checkpoint, adapter, split,
seed, and selected epoch. New runs also save per-trial labels, predictions, and class probabilities
so the metrics can be recomputed independently.

## Older runs

`results/mi32-common32-v4/historical-fold0-pre-unit-hardening/` keeps the earlier seven-model
fold-0 package. It is useful for tracing the work, but it is not part of the table above.

Uni-NTFM remains in a separate `protocol_benchmark` track because its public release does not
include the downstream checkpoint/head needed for a paper-level reproduction.
