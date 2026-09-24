# Benchmark protocol

The task is cross-subject, three-class motor-imagery classification. All models use the same data
roles and the same rule for opening the test set.

## Train, validation, and test

For test fold `f`, subjects assigned to `f` form the test set. Validation uses fold 1 when `f=0`
and fold 0 otherwise. The remaining subjects are used for training.

No subject appears in two roles. Anything that learns a statistic—including calibration inside an
adapter—is fitted from the training subjects. The sorted training IDs are saved as a SHA-256 hash.

## Hyperparameters

The alpha protocol searches only the learning rate:

- supervised models: `1e-3` or `3e-4`;
- pretrained foundation models: `1e-4` or `3e-5`;
- Uni-NTFM protocol track: `2e-4` or `5e-5`.

Validation macro-F1 chooses between the two. Dropout, architecture, channel mapping, label rules,
data filtering, and test-time processing stay fixed.

Tuning commands do not use `--final-test`, so the runner never instantiates the test dataset during
selection. Once the learning rate is written down, the final command loads the best validation
checkpoint and evaluates the test fold once.

## Metrics

The two primary metrics are macro-F1 and balanced accuracy. Plain accuracy is included, but the
`1:1:2` class distribution makes it a poor summary on its own. Reports also include:

- recall for each of the three classes;
- upper-limb versus non-upper-limb performance;
- left/right accuracy conditional on detecting an upper-limb trial;
- subject-mean metrics when available;
- the three-class confusion matrix.

For a paper, run all planned folds and report mean, dispersion, source/subject sensitivity, and an
uncertainty estimate or paired test based on a declared analysis unit. Trials from the eight source
datasets are not IID samples.

## Result labels used in this repository

- `formal_reproduction`: official or source-aligned model with fixed code/checkpoint identity;
- `protocol_benchmark`: declared downstream wrapper on a public backbone when no official task
  head or checkpoint is available;
- `ablation`: comparison of adapter choices, kept out of the main result table;
- `invalidated`: an older run whose structure or input definition has been replaced.
