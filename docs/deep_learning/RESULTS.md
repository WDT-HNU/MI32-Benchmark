# Deep-model results across tasks

Deep models have recorded runs on MI3, MI2, and P300. The protocols differ, so results are reported by task and no cross-task aggregate score is calculated.

## MI3

Current files are under `benchmark/results/mi32-common32-v4/current-evidence-20260911/`. Test-fold coverage differs by model; this table inventories current evidence and is not a final seven-model ranking.

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

## MI2

The reportable scope is fold 0. The test set contains 45 subjects excluded from training and 9,462 trials, with 4,731 trials per class.

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| EEGNet | 0.7106 | 0.7106 | 0.7105 |
| TSception | 0.6972 | 0.6972 | 0.6947 |
| RGNN backbone | 0.5677 | 0.5677 | 0.5624 |
| EEG-Conformer | 0.7317 | 0.7317 | 0.7308 |
| LaBraM | 0.5951 | 0.5951 | 0.5946 |
| EEGMamba | 0.6840 | 0.6840 | 0.6840 |
| CodeBrain | 0.6721 | 0.6721 | 0.6716 |

These values are fold-0 test results, not five-fold means.

## P300

The frozen protocol samples 102 within-subject LOGO folds. Seven models produced 714 runs. Metrics were aggregated from saved sampled-fold predictions.

| Model | PR-AUC | ROC-AUC | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| EEGNet | 0.3290 | 0.8186 | 0.7285 | 0.6256 |
| TSception | 0.4300 | 0.8395 | 0.7400 | 0.6476 |
| RGNN surrogate | 0.1229 | 0.5811 | 0.5583 | 0.4545 |
| EEG-Conformer | 0.2391 | 0.7462 | 0.6726 | 0.5446 |
| LaBraM | 0.4059 | 0.7930 | 0.6723 | 0.6728 |
| EEGMamba | 0.4972 | 0.8595 | 0.7435 | 0.6916 |
| CodeBrain | 0.4972 | 0.8584 | 0.7380 | 0.6940 |

RGNN uses a surrogate feature route and is not ranked as a complete official RGNN reproduction. Because P300 is highly imbalanced, ordinary accuracy cannot replace PR-AUC, balanced accuracy, or Macro-F1.

## Evidence boundaries

- MI2 covers fold 0 only and cannot establish cross-fold stability.
- MI3 model coverage differs, so a complete seven-model rank is not supported.
- P300 uses a frozen 102/873 LOGO sample, not all folds.
- Uni-NTFM has a structure and protocol route but no formal training result.
- Results from the superseded 32→60 EEGMamba adapter, 750→800 padding, LaBraM CLS pooling, linear resampling, per-trial z-scoring, or CodeBrain mean pooling / `Linear(200,classes)` are excluded.

The machine-readable cross-task summary is [`benchmark/results/cross-task/deep_model_task_summary.csv`](../../benchmark/results/cross-task/deep_model_task_summary.csv).
