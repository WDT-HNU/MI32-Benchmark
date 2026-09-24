# Benchmark protocol

## Scientific target

The benchmark estimates cross-subject generalization for a harmonized three-class motor-imagery
task while comparing distinct neural architectures under one leakage-resistant evaluation rule.

## Data roles

For test fold `f`, test subjects are exactly the rows with `fold=f`. Validation subjects use fold
1 when `f=0`, otherwise fold 0. All remaining subjects form training. Every transformation that
learns statistics is fitted on training subjects only and stores a SHA-256 of the sorted training
subject IDs.

## Hyperparameter policy

The alpha protocol searches learning rate only:

- supervised models: `1e-3` versus `3e-4`;
- pretrained foundation models: `1e-4` versus `3e-5`;
- Uni-NTFM protocol track: `2e-4` versus `5e-5`.

Selection uses validation macro-F1. Dropout, architecture, channel mapping, label ontology, data
filtering, and test-time processing are not adapted after viewing validation or test outcomes.

## Test isolation

Tuning commands omit `--final-test`, so the test dataset is not instantiated. After the selected
configuration is frozen, the final command evaluates validation and test from the same best
validation checkpoint. Test metrics cannot trigger another run or change.

## Metrics

Primary metrics are macro-F1 and balanced accuracy. Accuracy is reported but is insufficient on
the `1:1:2` label distribution. Reports also include per-class recall, a binary upper-vs-non-upper
score, conditional left/right accuracy after upper-limb detection, subject-mean metrics when
available, and the three-class confusion matrix.

## Statistical reporting

Fold 0 alone is a pipeline/evidence milestone, not the final scientific estimate. A paper should
report all planned folds, mean and dispersion, per-source/per-subject sensitivity, and confidence
intervals or paired tests based on predeclared units of analysis. The heterogeneous source
datasets should not be treated as IID trials.

## Result classes

- `formal_reproduction`: identity-sealed official or official-aligned model under MI32 protocol.
- `protocol_benchmark`: an explicit custom downstream protocol on an official backbone when the
  upstream release lacks the contract needed for paper reproduction.
- `ablation`: none/naive/official input-adapter comparisons, never merged into the main table.
- `invalidated`: historical result produced by a superseded structure or input contract.
