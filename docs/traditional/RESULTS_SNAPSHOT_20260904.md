# Traditional-method results snapshot — 2026-09-04

This page is the reviewed English transcription of the supplied three-page progress report. It is intentionally dated: the MI3 counts and scores belong to the reported 2026-09-04 artifact and are not relabelled as results from the later MI32 common-32 v4 release.

## Scope

| Task | Data line in the source report | Classes | Methods | Evaluation |
|---|---|---|---:|---|
| MI2 | 18 datasets; 389 subjects | Left / Right | 7 | five-fold cross-subject |
| MI3 | 8 datasets; 225 subjects; 94,856 trials | left-upper / right-upper / non-upper | 7 | five-fold cross-subject |
| P300 | 15 eligible datasets; 25 evaluable subjects | Target / NonTarget | 16 | LOGO |

## MI2

| Method | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Band-power + linear SVM | **0.564 ± 0.089** | **0.564 ± 0.089** | **0.536 ± 0.108** |
| Band-power + shrinkage LDA | 0.562 ± 0.088 | 0.562 ± 0.088 | 0.533 ± 0.108 |
| DWT + shrinkage LDA | 0.533 ± 0.071 | 0.533 ± 0.071 | 0.499 ± 0.098 |
| CSP + linear SVM | 0.524 ± 0.060 | 0.525 ± 0.060 | 0.446 ± 0.103 |
| CSP + shrinkage LDA | 0.520 ± 0.059 | 0.521 ± 0.059 | 0.433 ± 0.102 |
| Flatten + shrinkage LDA | 0.514 ± 0.059 | 0.514 ± 0.059 | 0.509 ± 0.061 |
| Regularized CSP + shrinkage LDA | 0.513 ± 0.046 | 0.513 ± 0.045 | 0.415 ± 0.090 |

## MI3

| Method | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Band-power + shrinkage LDA | **0.506 ± 0.017** | 0.405 ± 0.015 | 0.392 ± 0.023 |
| DWT + shrinkage LDA | 0.501 ± 0.014 | 0.376 ± 0.014 | 0.341 ± 0.030 |
| CSP + shrinkage LDA | 0.496 ± 0.010 | 0.343 ± 0.008 | 0.262 ± 0.027 |
| Regularized CSP + shrinkage LDA | 0.496 ± 0.009 | 0.343 ± 0.008 | 0.261 ± 0.026 |
| Band-power + linear SVM | 0.481 ± 0.026 | **0.436 ± 0.018** | **0.435 ± 0.018** |
| Flatten + shrinkage LDA | 0.471 ± 0.006 | 0.372 ± 0.004 | 0.360 ± 0.006 |
| CSP + linear SVM | 0.466 ± 0.028 | 0.382 ± 0.018 | 0.367 ± 0.030 |

The accuracy leader and the balanced-accuracy/Macro-F1 leader are different. This matters under the approximate `1:1:2` class distribution: ordinary accuracy gives the larger non-upper class more influence.

## P300

| Method | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Flatten + LDA | **0.882 ± 0.052** | 0.685 ± 0.116 | **0.702 ± 0.118** |
| xDAWN + LDA | 0.847 ± 0.065 | 0.585 ± 0.136 | 0.577 ± 0.153 |
| SWLDA | 0.880 ± 0.048 | 0.604 ± 0.102 | 0.614 ± 0.121 |
| Flatten + linear SVM | 0.792 ± 0.094 | 0.732 ± 0.086 | 0.676 ± 0.099 |
| PCA64 + LDA | 0.870 ± 0.050 | 0.591 ± 0.084 | 0.600 ± 0.106 |
| xDAWN + linear SVM | 0.680 ± 0.146 | 0.593 ± 0.136 | 0.551 ± 0.125 |
| DWT(db4) + linear SVM | 0.792 ± 0.094 | 0.729 ± 0.083 | 0.675 ± 0.096 |
| DWT(db4) + shrinkage LDA | **0.882 ± 0.052** | 0.677 ± 0.114 | 0.695 ± 0.118 |
| ERP statistical features + LDA | 0.874 ± 0.053 | 0.636 ± 0.123 | 0.642 ± 0.134 |
| SelectKBest60 + LDA | 0.867 ± 0.049 | 0.583 ± 0.089 | 0.583 ± 0.110 |
| L1 logistic | 0.794 ± 0.094 | **0.743 ± 0.090** | 0.682 ± 0.102 |
| Template correlation | 0.611 ± 0.057 | 0.652 ± 0.067 | 0.528 ± 0.064 |
| BLDA | 0.627 ± 0.059 | 0.723 ± 0.061 | 0.560 ± 0.083 |
| Flatten + RBF-SVM | 0.840 ± 0.083 | 0.709 ± 0.092 | 0.701 ± 0.104 |
| ERP statistical features + linear SVM | 0.756 ± 0.102 | 0.717 ± 0.092 | 0.644 ± 0.101 |
| Time-lagged template matching | 0.597 ± 0.070 | 0.637 ± 0.066 | 0.514 ± 0.066 |

The source overview names Flatten + LDA as the accuracy leader. The full table shows a tie at the displayed precision between Flatten + LDA and DWT(db4) + shrinkage LDA (`0.882 ± 0.052`), which is preserved here. L1 logistic regression leads balanced accuracy, while Flatten + LDA leads Macro-F1 by 0.001 over Flatten + RBF-SVM.

PR-AUC is not present in this three-page table, but the companion [P300 benchmark report](P300_BENCHMARK.md) includes it. The two documents also use different F1 labels: this page reports `Macro-F1`, whereas the companion table labels its column `F1`. They are kept separate rather than merged.

## Evidence status

All values above are transcribed from the supplied report. The matching machine-readable transcription is stored under [`benchmark/results/traditional/`](../../benchmark/results/traditional/README.md). The source report does not specify the exact aggregation unit behind each mean and standard deviation, so this page does not infer one.
