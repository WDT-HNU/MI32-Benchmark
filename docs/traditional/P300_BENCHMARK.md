# P300 traditional benchmark

This document is the reviewed English version of the supplied seven-page `P300 Traditional Benchmark` report. It preserves the reported protocol and values while separating two metric views that the source report presents independently.

## Objective

The benchmark compares traditional P300 classifiers under one preprocessing contract, one set of LOGO splits, and one metric suite. Keeping these conditions fixed avoids attributing a data-processing or evaluation difference to the classifier itself and provides conventional baselines for deep and Riemannian methods.

## Protocol

| Item | Frozen setting |
|---|---|
| Scope | 15 P300 datasets |
| Labels | `NonTarget = 0`, `Target = 1` |
| Montage | 16 channels; missing channels spatially interpolated, then common-average referenced |
| Channels | `Fz, Cz, Pz, Oz, P3, P4, PO7, PO8, F3, F4, FCz, C3, C4, CP3, CPz, CP4` |
| Filtering | 0.5–20 Hz band-pass |
| Sampling | 200 Hz |
| Epoch | 0–600 ms after stimulus |
| Evaluation | leave one complete session/run group out |
| Scale | 873 folds × 16 methods = 13,968 evaluations |

See the [P300 dataset card](../../benchmark/datasets/p300/DATASET_CARD.md) for the task-level data contract.

## Method pool

| Route | Methods |
|---|---|
| ERP and linear discriminants | Flatten + shrinkage LDA; BLDA; SWLDA |
| Margin-based classifiers | Flatten + linear SVM; Flatten + RBF-SVM; L1 logistic regression |
| Dimensionality reduction and selection | PCA64 + LDA; SelectKBest60 + LDA |
| Time-frequency and statistics | DWT(db4) + shrinkage LDA; DWT(db4) + linear SVM; ERP statistics + LDA; ERP statistics + linear SVM |
| Template matching | Template correlation; time-lagged template matching with a fixed ±40 ms lag bank |
| Spatial filtering | xDAWN + LDA; xDAWN + linear SVM |

The pool deliberately retains several method routes. It is not reduced to whichever classifier leads one metric.

## Metric policy

P300 epochs are strongly imbalanced because NonTarget events outnumber Target events. The report therefore uses:

- ROC-AUC for threshold-independent ranking ability;
- PR-AUC for Target-focused precision-recall performance;
- balanced accuracy for equal weighting of the two classes;
- precision, recall, F1, and specificity to expose fixed-threshold behavior.

Ordinary accuracy appears in the separate progress snapshot but is not used here as the sole ranking criterion.

## Full reported metric table

| Method | ROC-AUC | PR-AUC | Balanced accuracy | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLDA | 0.830 | 0.575 | 0.723 | 0.282 | 0.850 | 0.399 | 0.597 |
| SWLDA | 0.828 | 0.570 | 0.604 | 0.672 | 0.227 | 0.301 | 0.981 |
| Flatten + shrinkage LDA | 0.827 | 0.569 | 0.685 | 0.623 | 0.412 | 0.475 | 0.958 |
| DWT + shrinkage LDA | 0.824 | 0.564 | 0.677 | 0.628 | 0.395 | 0.463 | 0.959 |
| L1 logistic | 0.817 | 0.541 | 0.743 | 0.415 | 0.680 | 0.503 | 0.806 |
| Flatten + RBF-SVM | 0.811 | 0.540 | 0.710 | 0.504 | 0.538 | 0.505 | 0.881 |
| Flatten + linear SVM | 0.808 | 0.523 | 0.732 | 0.411 | 0.657 | 0.494 | 0.807 |
| DWT + linear SVM | 0.807 | 0.521 | 0.729 | 0.410 | 0.654 | 0.492 | 0.805 |
| ERP statistics + LDA | 0.793 | 0.506 | 0.636 | 0.559 | 0.306 | 0.362 | 0.965 |
| PCA64 + LDA | 0.791 | 0.493 | 0.591 | 0.605 | 0.212 | 0.284 | 0.969 |
| ERP statistics + linear SVM | 0.782 | 0.482 | 0.717 | 0.366 | 0.672 | 0.459 | 0.762 |
| SelectKBest60 + LDA | 0.760 | 0.437 | 0.583 | 0.495 | 0.196 | 0.249 | 0.970 |
| Template correlation | 0.716 | 0.380 | 0.652 | 0.251 | 0.694 | 0.342 | 0.609 |
| Time-lagged template | 0.698 | 0.354 | 0.637 | 0.242 | 0.676 | 0.327 | 0.598 |
| xDAWN + LDA | 0.629 | 0.346 | 0.585 | 0.330 | 0.218 | 0.246 | 0.952 |
| xDAWN + linear SVM | 0.619 | 0.315 | 0.593 | 0.262 | 0.479 | 0.327 | 0.707 |

These are report-level aggregate values. Because the report does not provide the per-fold prediction files or aggregation formula in the supplied material, the independently averaged columns should not be algebraically reconstructed from one another.

## Interpretation

BLDA, SWLDA, Flatten + shrinkage LDA, and DWT + shrinkage LDA form the leading ROC-AUC group. Their operational behavior is nevertheless different. BLDA reports high Target recall (0.850) with lower precision and specificity; SWLDA reports high precision and specificity with recall of 0.227. L1 logistic regression leads balanced accuracy (0.743), while Flatten + RBF-SVM has the highest reported F1 in this table (0.505).

The two xDAWN pipelines have the lowest reported ROC-AUC values in this benchmark. The report establishes that observation but does not support a causal explanation, so none is added here.

The defensible conclusion is not a single universal winner. BLDA, SWLDA, Flatten + shrinkage LDA, DWT + shrinkage LDA, and L1 logistic regression provide complementary core baselines, while SVM, template, and xDAWN routes retain methodological coverage.

## Literature anchors

- Rivet et al., “xDAWN Algorithm to Enhance Evoked Potentials: Application to Brain-Computer Interface,” *IEEE Transactions on Biomedical Engineering*, 2009. [doi:10.1109/TBME.2009.2012869](https://doi.org/10.1109/TBME.2009.2012869)
- Krusienski et al., “A comparison of classification techniques for the P300 Speller,” *Journal of Neural Engineering*, 2006. [doi:10.1088/1741-2560/3/4/007](https://doi.org/10.1088/1741-2560/3/4/007)
- Hoffmann et al., “An efficient P300-based brain-computer interface for disabled subjects,” *Journal of Neuroscience Methods*, 2008. [doi:10.1016/j.jneumeth.2007.03.005](https://doi.org/10.1016/j.jneumeth.2007.03.005)
- Rakotomamonjy and Guigue, “BCI Competition III: Dataset II—Ensemble of SVMs for BCI P300 Speller,” *IEEE Transactions on Biomedical Engineering*, 2008. [doi:10.1109/TBME.2008.915728](https://doi.org/10.1109/TBME.2008.915728)

The supplied note dated the last reference to 2005; the reviewed bibliography corrects it to the journal publication year, 2008.
