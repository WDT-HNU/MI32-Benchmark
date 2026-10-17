# MI3 deep-model benchmark protocol

The public runnable protocol is cross-subject, three-class motor imagery. All models use the same data roles and test-opening rule.

## Train, validation, and test

For test fold `f`, subjects assigned to `f` form the test set. Fold 1 is the validation set when `f=0`; otherwise fold 0 is the validation set. All remaining subjects are used for training.

No subject appears in more than one role. Every learned statistic, including adapter calibration, is fitted on training subjects only. The sorted training-subject IDs are sealed with a SHA-256 hash.

## Hyperparameters

The alpha protocol searches learning rate only:

- supervised models: `1e-3` or `3e-4`;
- pretrained foundation models: `1e-4` or `3e-5`;
- Uni-NTFM protocol benchmark: `2e-4` or `5e-5`.

The choice is based on validation Macro-F1. Dropout, architecture, channel mapping, label rules, dataset filters, and test-time processing remain fixed.

Tuning commands omit `--final-test`, so the runner does not instantiate the test dataset during selection. After recording the chosen learning rate, the final command loads the validation-selected checkpoint and evaluates the test fold once.

## Metrics

Primary metrics are Macro-F1 and balanced accuracy. Ordinary accuracy is also reported, but it cannot summarize the approximate `1:1:2` class distribution on its own. Additional diagnostics include:

- per-class recall;
- upper-limb versus non-upper performance;
- left/right accuracy conditional on an upper-limb prediction;
- subject-mean metrics where available;
- the three-class confusion matrix.

A paper-level study should run every predeclared fold and report means, dispersion, source/subject sensitivity, and uncertainty or paired tests at a predeclared analysis unit. Trials pooled from eight source datasets are not independent and identically distributed observations.

## Result labels

- `formal_reproduction`: official or source-aligned model with pinned source and checkpoint identity;
- `protocol_benchmark`: explicitly declared downstream wrapper where no official task head or checkpoint exists;
- `ablation`: adapter comparison excluded from the primary result table;
- `invalidated`: historical run whose architecture or input definition was superseded.

MI2 and P300 results retain their own frozen protocols and are summarized in [RESULTS.md](RESULTS.md); they should not be described as runs of this MI3 split definition.
