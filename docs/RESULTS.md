# Results

## Current-contract evidence snapshot

`results/mi32-common32-v4/current-evidence-20260911/` contains runs matching the sealed
foundation-model unit contracts. The recorded fold coverage is:

- EEGNet, TSception, RGNN backbone, and EEG-Conformer: fold 0;
- CodeBrain and EEGMamba: folds 0, 2, and 3;
- LaBraM: folds 0 and 2.

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

Each row is interpreted at its recorded fold scope. Cross-model aggregation uses matched folds and
publishes its aggregation rule together with the resulting table.

## Historical audit package

`results/mi32-common32-v4/historical-fold0-pre-unit-hardening/` preserves the earlier seven-model
fold-0 package as an immutable provenance record. The current-contract reporting table is the
snapshot above.

## Uni-NTFM track

Uni-NTFM is published as a separately labeled `protocol_benchmark` implementation. Its model card,
adapter geometry, structure test, and explicit launch flag keep that track distinct from the seven
formal-reproduction models.

## Result artifacts

Published evidence contains metric CSVs and run manifests recording dataset identity, model source,
checkpoint identity, adapter contract, split, seed, and selected epoch. Current runners also emit
per-trial labels, predictions, and class probabilities for independent metric recomputation.
