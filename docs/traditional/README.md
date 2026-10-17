# Traditional methods

This section records the conventional signal-processing and machine-learning baselines used across the three tasks. Traditional methods are one of the three top-level method families in this benchmark; the other two are deep models and Riemannian methods.

| Task | Reported method set | Evaluation |
|---|---:|---|
| MI2: left vs right | 7 | five-fold cross-subject |
| MI3: left-upper / right-upper / non-upper | 7 | five-fold cross-subject |
| P300: Target / NonTarget | 16 | LOGO by complete session/run group |

The attached progress report is a dated 2026-09-04 snapshot. Its MI3 data line reports 8 datasets, 225 subjects, and 94,856 trials, which predates the current MI32 common-32 v4 release of 230 subjects and 97,608 trials. The snapshot is therefore preserved as historical evidence rather than relabelled as a result from the later release.

## Documents

| Topic | File |
|---|---|
| P300 data harmonization | [P300 dataset card](../../benchmark/datasets/p300/DATASET_CARD.md) |
| Full P300 traditional benchmark | [P300 benchmark](P300_BENCHMARK.md) |
| Three-task progress snapshot | [2026-09-04 results](RESULTS_SNAPSHOT_20260904.md) |
| Machine-readable transcriptions | [traditional results archive](../../benchmark/results/traditional/README.md) |

Deep architectures, including EEG foundation models, remain one method family and are documented under [deep learning](../deep_learning/README.md). Riemannian methods are documented separately under [Riemannian methods](../riemannian/README.md).
