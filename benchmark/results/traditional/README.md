# Traditional benchmark result transcriptions

This directory contains machine-readable versions of two supplied reports:

- `progress_snapshot_20260904.csv`: Accuracy, balanced accuracy, and Macro-F1 means and standard deviations for MI2, MI3, and P300.
- `p300_threshold_metrics.csv`: the full 16-method P300 ROC-AUC, PR-AUC, balanced accuracy, precision, recall, F1, and specificity table.

The CSV files are faithful transcriptions, not newly recomputed outputs. Their provenance class is `report_transcription`. The reports did not include fold-level predictions or the aggregation formula, so these values must not be relabelled as prediction-derived evidence.

Human-readable protocol and interpretation are in:

- [`docs/traditional/RESULTS_SNAPSHOT_20260904.md`](../../../docs/traditional/RESULTS_SNAPSHOT_20260904.md)
- [`docs/traditional/P300_BENCHMARK.md`](../../../docs/traditional/P300_BENCHMARK.md)
